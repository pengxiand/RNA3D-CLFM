"""BridgeBind3D v3 — Graph-native + 3D-equivariant dual encoder.

Architecture changes vs v2
──────────────────────────
RNA tower:
  Transformer (flat tokens)  →  EGNN (E(3)-equivariant GNN, real 3D coords)
  Message passing propagates nucleotide identity + backbone/pairing/stacking
  interactions through real 3-D space; coordinate-equivariant updates.

Ligand tower:
  Transformer (SMILES tokens)  →  GINE (Graph Isomorphism + Edge features)
  Operates on the molecular graph with real bond types and hybridisation.

Interaction:
  Multi-round bidirectional cross-attention with proper padding masks.
  Queries attend over valid nodes only (padded zeros are ignored).

Scoring:
  rank_score  = cosine_similarity(z_rna, z_lig)   — L2-normalised dot product
                Fast dual-encoder score; pocket embeddings can be pre-computed
                offline for large-scale virtual screening.
  dock_score  = MLP( [z_rna, z_lig, z_rna*z_lig, |z_rna-z_lig|] )
                Slower, but richer re-ranking score.
  site_logits = per-residue MLP on cross-attended RNA features.

Input types:
  BatchedGraph (from graph_data.py) — plain tensors, no external library.

Node feature dims (must match featurizers.py):
  RNA  node: 7   (nt_onehot×5 + degree + pos_norm) or 8 if include_pocket_feat=True
  RNA  edge: 4   (backbone / pairing / stacking / other)
  Lig  node: 9   (atomic×6 + hybridisation×3)
  Lig  edge: 4   (bond type one-hot)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from bridgebind3d.egnn import EGNNEncoder
from bridgebind3d.gin_encoder import GINEncoder, FP2LigandEncoder
from bridgebind3d.graph_data import BatchedGraph
from bridgebind3d.optimol_encoder import OptiMolLigandEncoder

# ------------------------------------------------------------------
# Feature dimension constants (match featurizers.py)
# ------------------------------------------------------------------
RNA_NODE_DIM: int = 7   # nt_onehot(5) + degree(1) + pos_norm(1); +1 if include_pocket_feat
RNA_EDGE_DIM: int = 4   # edge_type_onehot
LIG_NODE_DIM: int = 9   # atom_features(6) + hybridization(3)
LIG_EDGE_DIM: int = 4   # bond_type_onehot


# ------------------------------------------------------------------
# Building blocks
# ------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention with optional key_padding_mask."""

    def __init__(self, dim: int, num_heads: int, dropout: float, ffn_mult: int = 4) -> None:
        super().__init__()
        self.norm_q  = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn    = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.attn_drop = nn.Dropout(dropout)
        self.norm_ffn  = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim), nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        nq  = self.norm_q(query)
        nkv = self.norm_kv(key_value)
        h, _ = self.attn(nq, nkv, nkv, key_padding_mask=key_padding_mask)
        query = query + self.attn_drop(h)
        query = query + self.ffn(self.norm_ffn(query))
        return query


class DistBiasedSelfAttnBlock(nn.Module):
    """Pre-norm self-attention with Gaussian-RBF pairwise 3D distance bias.

    For each pair (i, j) of RNA nucleotides, encodes their Euclidean distance
    d_ij via NUM_RBF radial basis functions (evenly spaced 0..MAX_DIST Å) and
    projects to one additive logit bias per attention head.  This lets the
    model leverage 3-D spatial proximity when refining residue representations
    between cross-attention rounds.

    Used only when interaction.use_dist_bias=true in model config (v9+).
    """

    NUM_RBF: int   = 16
    MAX_DIST: float = 20.0   # Å — covers typical RNA pocket diameter

    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        # Fixed RBF centres (not trained), learnable linear projection
        centers = torch.linspace(0.0, self.MAX_DIST, self.NUM_RBF)
        self.register_buffer("rbf_centers", centers)       # [R]
        self._sigma: float = self.MAX_DIST / self.NUM_RBF  # RBF bandwidth (Å)
        self.dist_proj  = nn.Linear(self.NUM_RBF, num_heads, bias=True)
        # Standard pre-norm self-attention + FFN
        self.norm       = nn.LayerNorm(dim)
        self.norm_ffn   = nn.LayerNorm(dim)
        self.attn       = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )
        self.attn_drop = nn.Dropout(dropout)

    def _build_bias(
        self, pos: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Return additive attention bias [B*H, L, L] from 3-D positions."""
        B, L, _ = pos.shape
        H = self.num_heads
        # Pairwise Euclidean distances [B, L, L]
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)         # [B, L, L, 3]
        dist = diff.norm(dim=-1)                           # [B, L, L]
        # Gaussian RBF encoding [B, L, L, R]
        d_c  = dist.unsqueeze(-1) - self.rbf_centers       # [B, L, L, R]
        rbf  = torch.exp(-(d_c ** 2) / (self._sigma ** 2 + 1e-8))
        # Project to H per-head biases: [B, L, L, H] → [B*H, L, L]
        bias = self.dist_proj(rbf)                             # [B, L, L, H]
        bias = bias.permute(0, 3, 1, 2).reshape(B * H, L, L)  # [B*H, L, L]
        # Mask padding key positions → -inf so they're ignored after softmax
        if mask is not None:
            pad_k = (~mask).unsqueeze(1).unsqueeze(1)          # [B, 1, 1, L]
            pad_k = pad_k.expand(B, H, L, L).reshape(B * H, L, L)
            bias  = bias.masked_fill(pad_k, float("-inf"))
        return bias

    def forward(
        self, x: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x    [B, L, D]  padded node features
            pos  [B, L, 3]  padded 3-D nucleotide positions (zeros for padding)
            mask [B, L]     True = valid nucleotide
        """
        bias = self._build_bias(pos, mask)              # [B*H, L, L]
        h    = self.norm(x)
        h, _ = self.attn(
            h, h, h,
            attn_mask        = bias,
            key_padding_mask = (~mask if mask is not None else None),
        )
        x = x + self.attn_drop(h)
        x = x + self.ffn(self.norm_ffn(x))
        return x


class AttentionPooling(nn.Module):
    """Learned attention pooling [B, L, D] → [B, D] with optional mask."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.key  = nn.Linear(dim, 1, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # mask: [B, L] True = valid positions
        w = self.key(x)                            # [B, L, 1]
        if mask is not None:
            w = w.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        w = torch.softmax(w, dim=1)
        return self.norm((w * x).sum(dim=1))       # [B, D]


def _mlp_head(in_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )


# ------------------------------------------------------------------
# Interaction module
# ------------------------------------------------------------------

class InteractionModuleV3(nn.Module):
    """Multi-round bidirectional cross-attention with optional distance-biased
    RNA self-attention interleaved before each cross-attention round.

    When use_dist_bias=True (v9+), each round becomes:
      1. DistBiasedSelfAttnBlock  — geometry-aware RNA self-refinement
      2. RNA cross-attends to Ligand
      3. Ligand cross-attends to RNA
    """

    def __init__(
        self,
        embed_dim:     int,
        num_heads:     int,
        dropout:       float,
        num_layers:    int  = 2,
        use_dist_bias: bool = False,
    ) -> None:
        super().__init__()
        self.use_dist_bias = use_dist_bias
        self.rna_xattn = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.lig_xattn = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        if use_dist_bias:
            # One geometry-aware self-attention block per cross-attn round
            self.rna_dist_attn = nn.ModuleList([
                DistBiasedSelfAttnBlock(embed_dim, num_heads, dropout)
                for _ in range(num_layers)
            ])

    def forward(
        self,
        rna:     torch.Tensor,               # [B, Lr, D]
        lig:     torch.Tensor,               # [B, Ll, D]
        rna_mask: torch.Tensor,              # [B, Lr] True = valid
        lig_mask: torch.Tensor,              # [B, Ll] True = valid
        rna_pos:  torch.Tensor | None = None,  # [B, Lr, 3] EGNN-updated coords
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # key_padding_mask for nn.MultiheadAttention: True = IGNORE
        rna_kpm = ~rna_mask    # [B, Lr]
        lig_kpm = ~lig_mask    # [B, Ll]
        dist_layers = self.rna_dist_attn if self.use_dist_bias else [None] * len(self.rna_xattn)
        for dist_layer, rna_layer, lig_layer in zip(dist_layers, self.rna_xattn, self.lig_xattn):
            # ① Geometry-aware RNA self-attention (v9+)
            if dist_layer is not None and rna_pos is not None:
                rna = dist_layer(rna, rna_pos, rna_mask)
                rna = rna.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
            # ② Bidirectional cross-attention
            new_rna = rna_layer(rna, lig, key_padding_mask=lig_kpm)
            new_lig = lig_layer(lig, rna, key_padding_mask=rna_kpm)
            rna, lig = new_rna, new_lig
        return rna, lig


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _unpack_to_padded(
    h: torch.Tensor,               # [total_N, D]  all-nodes-flat
    batch_index: torch.Tensor,     # [total_N]
    B: int,
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert flat node embeddings to padded [B, max_len, D] + boolean mask."""
    D      = h.shape[1]
    device = h.device
    out    = torch.zeros(B, max_len, D, device=device, dtype=h.dtype)
    mask   = torch.zeros(B, max_len, dtype=torch.bool, device=device)  # True = valid

    # Vectorised scatter for each batch entry.
    for b in range(B):
        node_idx = (batch_index == b).nonzero(as_tuple=True)[0]
        n = min(node_idx.shape[0], max_len)
        out[b, :n]  = h[node_idx[:n]]
        mask[b, :n] = True

    return out, mask


# ------------------------------------------------------------------
# Main model
# ------------------------------------------------------------------

class UnifiedInteractionModelV3(nn.Module):
    """BridgeBind3D v3.

    Config keys (all optional with sensible defaults)
    ─────────────────────────────────────────────────
    embed_dim              int    256
    num_heads              int    8
    dropout                float  0.1
    max_rna_tokens         int    128   # pad length after EGNN
    max_lig_tokens         int    64    # pad length after GINE
    rna_tower:
      num_layers           int    4
    ligand_tower:
      num_layers           int    4
    interaction:
      num_layers           int    2
    projection:
      dim                  int    embed_dim
    include_pocket_feat    bool   False  # if True: appends in_pocket flag to RNA node features (7→8)

    Outputs
    ───────
    rank_score     [B]        cosine_sim(z_rna, z_lig); fast dual-encoder score
    screen_score   [B]        alias for rank_score
    dock_score     [B]        MLP re-ranking score; uses cross-attended features
    site_logits    [B, Lr]    per-residue binding-site logits
    site_label_pad [B, Lr]    float32 ground-truth pocket labels (0/1), 0 for missing
    z_rna        [B, P]     L2-normalised RNA pocket embedding
    z_lig        [B, P]     L2-normalised ligand embedding
    z_pair       [B, 4P]    concatenated pair features
    """

    def __init__(self, model_cfg: dict[str, Any]) -> None:
        super().__init__()
        embed_dim   = int(model_cfg.get("embed_dim",   256))
        num_heads   = int(model_cfg.get("num_heads",   8))
        dropout     = float(model_cfg.get("dropout",   0.1))
        proj_dim    = int(model_cfg.get("projection",  {}).get("dim", embed_dim))

        rna_layers   = int(model_cfg.get("rna_tower",    {}).get("num_layers", 4))
        lig_layers   = int(model_cfg.get("ligand_tower", {}).get("num_layers", 4))
        cross_layers = int(model_cfg.get("interaction",  {}).get("num_layers", 2))

        self.max_rna_tokens = int(model_cfg.get("max_rna_tokens", 128))
        self.max_lig_tokens = int(model_cfg.get("max_lig_tokens",  64))

        # ── Ligand encoder type ──────────────────────────────────────
        self.use_pretrained_ligand = bool(model_cfg.get("use_pretrained_ligand", False))
        self.use_fp2_ligand        = bool(model_cfg.get("use_fp2_ligand", False))

        # ── Whether to include in_pocket flag as extra RNA node feature ─
        self.include_pocket_feat = bool(model_cfg.get("include_pocket_feat", False))
        _rna_node_dim = RNA_NODE_DIM + (1 if self.include_pocket_feat else 0)

        # ── Graph encoders ──────────────────────────────────────────
        self.rna_encoder = EGNNEncoder(
            in_dim=_rna_node_dim, hidden_dim=embed_dim, edge_dim=RNA_EDGE_DIM,
            num_layers=rna_layers, dropout=dropout,
        )
        if self.use_fp2_ligand:
            # Morgan FP2 (2048-bit) → MLP → single token [B, D]
            fp_dim = int(model_cfg.get("fp_dim", 2048))
            self.lig_encoder = FP2LigandEncoder(
                fp_dim=fp_dim, embed_dim=embed_dim, dropout=dropout,
            )
        elif self.use_pretrained_ligand:
            # Pretrained OptiMol RGCN (56-dim μ) + trainable projection
            _freeze   = bool(model_cfg.get("freeze_ligand",  True))
            _opt_dir  = model_cfg.get("optimol_dir")  or None
            _map_file = model_cfg.get("optimol_map_file") or None
            self.lig_encoder = OptiMolLigandEncoder(
                out_dim=embed_dim,
                freeze=_freeze,
                optimol_dir=_opt_dir,
                map_file=_map_file,
            )
        else:
            self.lig_encoder = GINEncoder(
                in_dim=LIG_NODE_DIM, hidden_dim=embed_dim, edge_dim=LIG_EDGE_DIM,
                num_layers=lig_layers, dropout=dropout,
            )

        # ── Cross-attention interaction ──────────────────────────────
        use_dist_bias = bool(
            model_cfg.get("interaction", {}).get("use_dist_bias", False)
        )
        self.use_dist_bias = use_dist_bias
        self.interaction = InteractionModuleV3(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, num_layers=cross_layers,
            use_dist_bias=use_dist_bias,
        )

        # ── Pooling ─────────────────────────────────────────────────
        self.rna_pool = AttentionPooling(embed_dim)
        self.lig_pool = AttentionPooling(embed_dim)

        # ── Dual-encoder projections (L2-normalised) ────────────────
        # Dropout before L2-norm injects noise during training and is the most
        # direct guard against embedding-space collapse in contrastive setups.
        self.z_rna_proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim), nn.LayerNorm(proj_dim), nn.Dropout(dropout)
        )
        self.z_lig_proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim), nn.LayerNorm(proj_dim), nn.Dropout(dropout)
        )

        # ── Prediction heads ────────────────────────────────────────
        # dock_score: slow re-ranking from pair features
        self.dock_head = _mlp_head(proj_dim * 4, embed_dim, dropout)
        # site_logits: per-residue from cross-attended RNA
        self.site_head = _mlp_head(embed_dim * 3, embed_dim, dropout)

    # ----------------------------------------------------------------
    def forward(
        self,
        rna_bg: BatchedGraph,
        lig_bg,   # BatchedGraph  OR  list[str] (SMILES) when use_pretrained_ligand=True
    ) -> dict[str, torch.Tensor]:
        device = rna_bg.node_feat.device
        B      = int(rna_bg.batch_index.max().item()) + 1

        # ── 1. Graph encoding ────────────────────────────────────────
        # Optionally prepend in_pocket flag to RNA node features before EGNN
        if self.include_pocket_feat and rna_bg.site_label is not None:
            rna_node_input = torch.cat(
                [rna_bg.node_feat, rna_bg.site_label.unsqueeze(-1)], dim=-1
            )
        else:
            rna_node_input = rna_bg.node_feat
        # EGNN: processes all RNA nodes in the batch together [total_N, D]
        h_rna, pos_out = self.rna_encoder(
            rna_node_input, rna_bg.pos,
            rna_bg.edge_index, rna_bg.edge_feat,
        )        # Guard: replace any residual NaN/inf in EGNN output (e.g., from
        # extreme coordinate inputs during the first steps of a new phase).
        h_rna = h_rna.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        # ── 2. Ligand encoding (three paths) ────────────────────────
        if self.use_fp2_ligand:
            # lig_bg is a [B, fp_dim] float32 tensor of Morgan FP2 fingerprints
            h_lig_fp2 = self.lig_encoder(lig_bg)               # [B, D]
            h_lig_pad = h_lig_fp2.unsqueeze(1)                 # [B, 1, D]
            lig_mask  = torch.ones(B, 1, dtype=torch.bool, device=device)
        elif self.use_pretrained_ligand:
            # lig_bg is a list[str] of SMILES
            # OptiMolLigandEncoder returns [B, embed_dim] directly
            pooled_lig_pretrained = self.lig_encoder(lig_bg)          # [B, D]
            # Represent ligand as a single token for cross-attention
            h_lig_pad = pooled_lig_pretrained.unsqueeze(1)            # [B, 1, D]
            lig_mask  = torch.ones(B, 1, dtype=torch.bool, device=device)  # [B, 1]
        else:
            # lig_bg is a BatchedGraph
            h_lig = self.lig_encoder(
                lig_bg.node_feat, lig_bg.edge_index, lig_bg.edge_feat,
            )
            h_lig_pad, lig_mask = _unpack_to_padded(
                h_lig, lig_bg.batch_index, B, self.max_lig_tokens
            )  # [B, Ll, D], [B, Ll]

        # ── 3a. Unpack RNA to padded [B, max_len, D] ─────────────────
        h_rna_pad, rna_mask = _unpack_to_padded(
            h_rna, rna_bg.batch_index, B, self.max_rna_tokens
        )  # [B, Lr, D], [B, Lr]

        # ── 3a'. Unpack EGNN-updated coordinates for distance bias ────
        # pos_out: [total_N, 3] → pos_pad: [B, max_rna, 3] (zeros for padding)
        # Only computed when use_dist_bias=True to avoid unnecessary work.
        if self.use_dist_bias:
            pos_pad, _ = _unpack_to_padded(
                pos_out.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0),
                rna_bg.batch_index, B, self.max_rna_tokens,
            )  # [B, Lr, 3]
        else:
            pos_pad = None

        # ── 3b. Cross-attention refinement (with masks) ──────────────
        h_rna_pad, h_lig_pad = self.interaction(
            h_rna_pad, h_lig_pad, rna_mask, lig_mask,
            rna_pos=pos_pad,
        )
        # Guard: NaN can emerge from zero-padded positions flowing through
        # LayerNorm inside cross-attention (0/0 or inf-inf edge cases).
        h_rna_pad = h_rna_pad.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        h_lig_pad = h_lig_pad.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        # ── 4. Pooling — use pocket mask when site_label is available ──
        # Unpack site_label to padded form [B, max_rna_tokens] for pocket masking
        # and for returning as supervised signal to the training loop.
        if rna_bg.site_label is not None:
            sl_pad = torch.zeros(B, self.max_rna_tokens, dtype=torch.float32, device=device)
            for _b in range(B):
                _node_idx = (rna_bg.batch_index == _b).nonzero(as_tuple=True)[0]
                _n = min(_node_idx.shape[0], self.max_rna_tokens)
                sl_pad[_b, :_n] = rna_bg.site_label[_node_idx[:_n]]
            # pocket_mask: in_pocket residues that are also valid (not padded)
            pocket_mask = rna_mask & (sl_pad > 0.5)
            # fallback: if no pocket residues annotated, use all valid residues
            _has_pocket = pocket_mask.any(dim=1, keepdim=True)  # [B, 1]
            pool_mask   = torch.where(_has_pocket, pocket_mask, rna_mask)
        else:
            sl_pad    = torch.zeros(B, self.max_rna_tokens, dtype=torch.float32, device=device)
            pool_mask = rna_mask
        pooled_rna = self.rna_pool(h_rna_pad, pool_mask)   # [B, D]
        pooled_lig = self.lig_pool(h_lig_pad, lig_mask)   # [B, D]
        pooled_rna = pooled_rna.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        pooled_lig = pooled_lig.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        # ── 5. Dual-encoder projections + L2 normalise ───────────────
        z_rna = F.normalize(self.z_rna_proj(pooled_rna), p=2, dim=-1)  # [B, P]
        z_lig = F.normalize(self.z_lig_proj(pooled_lig), p=2, dim=-1)  # [B, P]

        # Fast score: cosine similarity ∈ [-1, 1]
        rank_score = (z_rna * z_lig).sum(-1)               # [B]

        # ── 6. Slow re-ranking score ──────────────────────────────────
        z_pair     = torch.cat([z_rna, z_lig, z_rna * z_lig, torch.abs(z_rna - z_lig)], dim=-1)
        dock_score = self.dock_head(z_pair).squeeze(-1)     # [B]

        # ── 7. Per-residue site logits ────────────────────────────────
        # Normalise both components before the product term to bound the gradient
        # magnitude flowing back through h_rna_pad (pre-norm cross-attn output is
        # unbounded; multiplying two unbounded vectors can cause explosive grads).
        # Use eps > 0 so that zero-padded positions (h_rna_pad == 0) don't produce
        # 0/0 = NaN from F.normalize.
        lig_ctx    = pooled_lig.unsqueeze(1).expand(-1, self.max_rna_tokens, -1)
        h_rna_n    = h_rna_pad  / (h_rna_pad.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6))
        lig_ctx_n  = lig_ctx    / (lig_ctx.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6))
        site_feat  = torch.cat([h_rna_n, lig_ctx_n, h_rna_n * lig_ctx_n], dim=-1)
        site_logits = self.site_head(site_feat).squeeze(-1) # [B, Lr]

        return {
            "rank_score":     rank_score,
            "screen_score":   rank_score,
            "dock_score":     dock_score,
            "site_logits":    site_logits,
            "site_label_pad": sl_pad,      # [B, Lr] float32 pocket ground-truth
            "rna_mask":       rna_mask,    # [B, Lr] bool, True = valid (non-padding)
            "z_rna":          z_rna,
            "z_lig":          z_lig,
            "z_pair":         z_pair,
        }
