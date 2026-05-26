"""BridgeBind3D v4 — RNA-FM sequence features + EGNN 3D structure + GIN ligand.

Architecture changes vs v3
──────────────────────────
RNA tower:
  RNA-FM T12 (pretrained, last 2 layers finetunable) extracts per-nucleotide
  embeddings [N, 640]. These are projected to embed_dim and fused with the
  7-dim structural node features before EGNN message passing.
  Result: EGNN nodes carry both sequence context AND 3D geometry.

Ligand tower:
  GINEncoder (trained from scratch, no pretrained bottleneck).
  Replaces OptiMol's variational RGCN (56-dim μ), giving the ligand encoder
  full freedom to learn task-specific representations.

Everything else (cross-attention interaction, cosine scoring, dual projection,
site logits, OneCycleLR training) is identical to v3.

Node feature dims:
  RNA  node: 7   → RNA-FM proj (embed_dim) + 7 cat → Linear → embed_dim
  RNA  edge: 4
  Lig  node: 9
  Lig  edge: 4
"""

from __future__ import annotations

from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from bridgebind3d.egnn import EGNNEncoder
from bridgebind3d.gin_encoder import GINEncoder
from bridgebind3d.graph_data import BatchedGraph
from bridgebind3d.rna_fm_encoder import RNAFMNodeEmbedder
from bridgebind3d.featurizers import build_lig_graph_cached
from bridgebind3d.graph_data import batch_ligand_graphs

# Feature dimension constants (match featurizers.py)
RNA_NODE_DIM: int = 7
RNA_EDGE_DIM: int = 4
LIG_NODE_DIM: int = 9
LIG_EDGE_DIM: int = 4


# ── Reuse building blocks from v3 ────────────────────────────────────────────

class CrossAttentionBlock(nn.Module):
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

    def forward(self, query, key_value, key_padding_mask=None):
        nq  = self.norm_q(query)
        nkv = self.norm_kv(key_value)
        h, _ = self.attn(nq, nkv, nkv, key_padding_mask=key_padding_mask)
        query = query + self.attn_drop(h)
        query = query + self.ffn(self.norm_ffn(query))
        return query


class AttentionPooling(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.key  = nn.Linear(dim, 1, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        w = self.key(x)
        if mask is not None:
            w = w.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        w = torch.softmax(w, dim=1)
        return self.norm((w * x).sum(dim=1))


class InteractionModuleV4(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float, num_layers: int = 2) -> None:
        super().__init__()
        self.rna_xattn = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.lig_xattn = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])

    def forward(self, rna, lig, rna_mask, lig_mask):
        rna_kpm = ~rna_mask
        lig_kpm = ~lig_mask
        for rna_layer, lig_layer in zip(self.rna_xattn, self.lig_xattn):
            new_rna = rna_layer(rna, lig, key_padding_mask=lig_kpm)
            new_lig = lig_layer(lig, rna, key_padding_mask=rna_kpm)
            rna, lig = new_rna, new_lig
        return rna, lig


def _unpack_to_padded(h, batch_index, B, max_len):
    D, device = h.shape[1], h.device
    out  = torch.zeros(B, max_len, D, device=device, dtype=h.dtype)
    mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)
    for b in range(B):
        node_idx = (batch_index == b).nonzero(as_tuple=True)[0]
        n = min(node_idx.shape[0], max_len)
        out[b, :n]  = h[node_idx[:n]]
        mask[b, :n] = True
    return out, mask


def _mlp_head(in_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )


# ── Main model ────────────────────────────────────────────────────────────────

class UnifiedInteractionModelV4(nn.Module):
    """BridgeBind3D v4: RNA-FM + EGNN (RNA) × GINEncoder (ligand).

    Config keys
    ───────────
    embed_dim          int   256
    num_heads          int   8
    dropout            float 0.1
    max_rna_tokens     int   128
    max_lig_tokens     int   64
    rna_tower:
      num_layers       int   6
    ligand_tower:
      num_layers       int   4
    interaction:
      num_layers       int   3
    rna_fm:
      n_unfrozen_layers int  2      # how many RNA-FM tail layers to finetune
      model_location   str  null    # path to RNA-FM_pretrained.pth, null=auto
    projection:
      dim              int  embed_dim
    """

    def __init__(self, model_cfg: dict[str, Any]) -> None:
        super().__init__()
        embed_dim   = int(model_cfg.get("embed_dim",  256))
        num_heads   = int(model_cfg.get("num_heads",  8))
        dropout     = float(model_cfg.get("dropout",  0.1))
        proj_dim    = int(model_cfg.get("projection", {}).get("dim", embed_dim))

        rna_layers   = int(model_cfg.get("rna_tower",    {}).get("num_layers", 6))
        lig_layers   = int(model_cfg.get("ligand_tower", {}).get("num_layers", 4))
        cross_layers = int(model_cfg.get("interaction",  {}).get("num_layers", 3))

        self.max_rna_tokens = int(model_cfg.get("max_rna_tokens", 128))
        self.max_lig_tokens = int(model_cfg.get("max_lig_tokens",  64))

        # ── RNA-FM sequence embedder ─────────────────────────────────────────
        rna_fm_cfg = model_cfg.get("rna_fm", {})
        self.rna_fm = RNAFMNodeEmbedder(
            out_dim=embed_dim,
            n_unfrozen_layers=int(rna_fm_cfg.get("n_unfrozen_layers", 2)),
            model_location=rna_fm_cfg.get("model_location", None),
        )

        # ── Fuse RNA-FM proj (embed_dim) + structural feat (7) → embed_dim ──
        self.rna_node_fuse = nn.Sequential(
            nn.Linear(embed_dim + RNA_NODE_DIM, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # ── EGNN RNA encoder (in_dim = embed_dim after fusion) ───────────────
        self.rna_encoder = EGNNEncoder(
            in_dim=embed_dim, hidden_dim=embed_dim, edge_dim=RNA_EDGE_DIM,
            num_layers=rna_layers, dropout=dropout,
        )

        # ── GIN ligand encoder (no pretrained weights) ────────────────────────
        self.lig_encoder = GINEncoder(
            in_dim=LIG_NODE_DIM, hidden_dim=embed_dim, edge_dim=LIG_EDGE_DIM,
            num_layers=lig_layers, dropout=dropout,
        )

        # ── Cross-attention ───────────────────────────────────────────────────
        self.interaction = InteractionModuleV4(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, num_layers=cross_layers,
        )

        # ── Pooling ───────────────────────────────────────────────────────────
        self.rna_pool = AttentionPooling(embed_dim)
        self.lig_pool = AttentionPooling(embed_dim)

        # ── Projections ───────────────────────────────────────────────────────
        self.z_rna_proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim), nn.LayerNorm(proj_dim), nn.Dropout(dropout)
        )
        self.z_lig_proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim), nn.LayerNorm(proj_dim), nn.Dropout(dropout)
        )

        # ── Prediction heads ──────────────────────────────────────────────────
        self.dock_head = _mlp_head(proj_dim * 4, embed_dim, dropout)
        self.site_head = _mlp_head(embed_dim * 3, embed_dim, dropout)

    # ── Parameter groups for optimizer ──────────────────────────────────────

    def rna_fm_unfrozen_params(self):
        """Trainable parameters in RNA-FM backbone (last layers)."""
        yield from self.rna_fm._unfrozen_params()

    def other_params(self):
        """All other trainable parameters (everything except RNA-FM backbone)."""
        rna_fm_ids = {id(p) for p in self.rna_fm.fm_model.parameters()}
        for p in self.parameters():
            if id(p) not in rna_fm_ids and p.requires_grad:
                yield p

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        rna_bg: BatchedGraph,
        lig_input,    # list[str] of SMILES  OR  BatchedGraph
        rna_sequences: List[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        device = rna_bg.node_feat.device
        B      = int(rna_bg.batch_index.max().item()) + 1

        # ── Per-graph node counts for sequence alignment ────────────────────
        seq_lengths = [int((rna_bg.batch_index == b).sum().item()) for b in range(B)]

        # ── 1. RNA-FM sequence features ─────────────────────────────────────
        if rna_sequences is None:
            # Fallback: placeholder zeros (e.g. during inference without seqs)
            fm_emb = torch.zeros(
                rna_bg.node_feat.shape[0], self.rna_fm.proj[0].in_features,
                device=device, dtype=rna_bg.node_feat.dtype,
            )
            fm_emb = self.rna_fm.proj(fm_emb)
        else:
            fm_emb = self.rna_fm(rna_sequences, seq_lengths, device=device)

        # ── 2. Fuse FM embeddings with structural node features ─────────────
        fused = self.rna_node_fuse(
            torch.cat([fm_emb, rna_bg.node_feat.float()], dim=-1)
        )  # [total_N, embed_dim]

        # ── 3. EGNN with fused node features ────────────────────────────────
        h_rna, _ = self.rna_encoder(fused, rna_bg.pos, rna_bg.edge_index, rna_bg.edge_feat)
        h_rna = h_rna.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        # ── 4. Ligand encoding ───────────────────────────────────────────────
        if isinstance(lig_input, list):
            # SMILES list → build molecular graphs on-the-fly
            lig_graphs = [build_lig_graph_cached(s) for s in lig_input]
            _lig_bg = batch_ligand_graphs(lig_graphs)
            lig_bg = BatchedGraph(
                node_feat=_lig_bg.node_feat.to(device),
                edge_index=_lig_bg.edge_index.to(device),
                edge_feat=_lig_bg.edge_feat.to(device),
                batch_index=_lig_bg.batch_index.to(device),
            )
        else:
            lig_bg = lig_input

        h_lig = self.lig_encoder(lig_bg.node_feat, lig_bg.edge_index, lig_bg.edge_feat)

        # ── 5. Unpack to padded ──────────────────────────────────────────────
        h_rna_pad, rna_mask = _unpack_to_padded(h_rna, rna_bg.batch_index, B, self.max_rna_tokens)
        h_lig_pad, lig_mask = _unpack_to_padded(h_lig, lig_bg.batch_index, B, self.max_lig_tokens)

        # ── 6. Cross-attention ───────────────────────────────────────────────
        h_rna_pad, h_lig_pad = self.interaction(h_rna_pad, h_lig_pad, rna_mask, lig_mask)
        h_rna_pad = h_rna_pad.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        h_lig_pad = h_lig_pad.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        # ── 7. Pooling ───────────────────────────────────────────────────────
        pooled_rna = self.rna_pool(h_rna_pad, rna_mask).nan_to_num(nan=0.0)
        pooled_lig = self.lig_pool(h_lig_pad, lig_mask).nan_to_num(nan=0.0)

        # ── 8. Dual-encoder projections + L2 normalise ───────────────────────
        z_rna = F.normalize(self.z_rna_proj(pooled_rna), p=2, dim=-1)
        z_lig = F.normalize(self.z_lig_proj(pooled_lig), p=2, dim=-1)
        rank_score = (z_rna * z_lig).sum(-1)

        # ── 9. Re-ranking + site logits ──────────────────────────────────────
        z_pair    = torch.cat([z_rna, z_lig, z_rna * z_lig, torch.abs(z_rna - z_lig)], dim=-1)
        dock_score = self.dock_head(z_pair).squeeze(-1)

        lig_ctx   = pooled_lig.unsqueeze(1).expand(-1, self.max_rna_tokens, -1)
        h_rna_n   = h_rna_pad  / (h_rna_pad.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6))
        lig_ctx_n = lig_ctx    / (lig_ctx.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6))
        site_feat = torch.cat([h_rna_n, lig_ctx_n, h_rna_n * lig_ctx_n], dim=-1)
        site_logits = self.site_head(site_feat).squeeze(-1)

        return {
            "rank_score":   rank_score,
            "screen_score": rank_score,
            "dock_score":   dock_score,
            "site_logits":  site_logits,
            "z_rna":        z_rna,
            "z_lig":        z_lig,
            "z_pair":       z_pair,
        }
