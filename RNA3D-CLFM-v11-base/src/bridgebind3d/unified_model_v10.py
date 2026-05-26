"""BridgeBind3D v10 — RNA-FM + EGNN + GIN + 3D distance-biased cross-attention.

Architecture
────────────
  v9 (DistBiasedSelfAttn bidirectional cross-attention, EGNN 3-D geometry) +
  v4 (RNA-FM T12 per-nucleotide sequence embeddings, fused before EGNN).

RNA tower
  RNA-FM T12 (last n_unfrozen_layers fine-tunable, rest frozen)
    → token-level embeddings [N, 640] → Linear(640→embed_dim) + LayerNorm
  Fused with structural node features (7 or 8 dim, include_pocket_feat flag)
    → Linear(embed_dim + struct_dim, embed_dim) + LayerNorm + GELU
  EGNN-6L (E(3)-equivariant, updates both features and 3-D positions)

Ligand tower
  GINEncoder-4L (trained from scratch, real bond graph)

Interaction
  InteractionModuleV3 with use_dist_bias=True:
    each round = DistBiasedSelfAttnBlock (RNA geometry-aware self-attn)
                 + bidirectional cross-attention

Scoring
  rank_score  = cosine_similarity(z_rna, z_lig)   [B]
  dock_score  = MLP(pair features)                 [B]
  site_logits = per-residue MLP on cross-attended RNA [B, max_rna_tokens]

Config keys
───────────
  model:
    version: 10
    embed_dim: 256
    num_heads: 4
    dropout: 0.1
    max_rna_tokens: 80
    max_lig_tokens: 64
    include_pocket_feat: true
    rna_tower:
      num_layers: 6
    ligand_tower:
      num_layers: 4
    interaction:
      num_layers: 4
      use_dist_bias: true
    rna_fm:
      n_unfrozen_layers: 1      # last 1 layer fine-tunable (conservative)
      model_location: "multimolecule/rnafm"  # HF repo id
    projection:
      dim: 256
  train:
    lr_rna_fm: 5.0e-6   # small LR for RNA-FM fine-tuning
"""

from __future__ import annotations

from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from bridgebind3d.egnn import EGNNEncoder
from bridgebind3d.gin_encoder import GINEncoder
from bridgebind3d.graph_data import BatchedGraph, batch_ligand_graphs
from bridgebind3d.featurizers import build_lig_graph_cached
from bridgebind3d.rna_fm_encoder import RNAFMNodeEmbedder

# Reuse building blocks from v3 (no duplication)
from bridgebind3d.unified_model_v3 import (
    AttentionPooling,
    CrossAttentionBlock,          # noqa: F401 (re-exported for completeness)
    DistBiasedSelfAttnBlock,      # noqa: F401
    InteractionModuleV3,
    _mlp_head,
    _unpack_to_padded,
    RNA_NODE_DIM,
    RNA_EDGE_DIM,
    LIG_NODE_DIM,
    LIG_EDGE_DIM,
)


# ─────────────────────────────────────────────────────────────────────────────

class UnifiedInteractionModelV10(nn.Module):
    """BridgeBind3D v10: RNA-FM + EGNN + GIN + DistBiasedSelfAttn.

    Combines v4's RNA-FM sequence encoder with v9's 3D distance-biased
    cross-attention.  All other design choices (EGNN-6L, GINEncoder-4L,
    4-round bidirectional cross-attention, pocket-aware pooling, site-BCE
    auxiliary loss) carry over from v9.
    """

    def __init__(self, model_cfg: dict[str, Any]) -> None:
        super().__init__()

        embed_dim  = int(model_cfg.get("embed_dim",  256))
        num_heads  = int(model_cfg.get("num_heads",  4))
        dropout    = float(model_cfg.get("dropout",  0.1))
        proj_dim   = int(model_cfg.get("projection", {}).get("dim", embed_dim))

        rna_layers   = int(model_cfg.get("rna_tower",    {}).get("num_layers", 6))
        lig_layers   = int(model_cfg.get("ligand_tower", {}).get("num_layers", 4))
        cross_layers = int(model_cfg.get("interaction",  {}).get("num_layers", 4))
        use_dist_bias = bool(
            model_cfg.get("interaction", {}).get("use_dist_bias", True)
        )

        self.max_rna_tokens    = int(model_cfg.get("max_rna_tokens", 80))
        self.max_lig_tokens    = int(model_cfg.get("max_lig_tokens", 64))
        self.include_pocket_feat = bool(model_cfg.get("include_pocket_feat", False))
        self.use_dist_bias     = use_dist_bias

        # ── RNA-FM sequence embedder ─────────────────────────────────────────
        rna_fm_cfg = model_cfg.get("rna_fm", {})
        self.rna_fm = RNAFMNodeEmbedder(
            out_dim=embed_dim,
            n_unfrozen_layers=int(rna_fm_cfg.get("n_unfrozen_layers", 1)),
            model_location=rna_fm_cfg.get("model_location", None),
        )

        # ── Fuse: RNA-FM (embed_dim) + structural (7 or 8) → embed_dim ───────
        _struct_dim = RNA_NODE_DIM + (1 if self.include_pocket_feat else 0)
        self.rna_node_fuse = nn.Sequential(
            nn.Linear(embed_dim + _struct_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # ── EGNN RNA encoder (in_dim = embed_dim after fusion) ───────────────
        self.rna_encoder = EGNNEncoder(
            in_dim=embed_dim, hidden_dim=embed_dim, edge_dim=RNA_EDGE_DIM,
            num_layers=rna_layers, dropout=dropout,
        )

        # ── GIN ligand encoder ────────────────────────────────────────────────
        self.lig_encoder = GINEncoder(
            in_dim=LIG_NODE_DIM, hidden_dim=embed_dim, edge_dim=LIG_EDGE_DIM,
            num_layers=lig_layers, dropout=dropout,
        )

        # ── Interaction: bidirectional cross-attn with 3D dist-bias ──────────
        self.interaction = InteractionModuleV3(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, num_layers=cross_layers,
            use_dist_bias=use_dist_bias,
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

    # ── Parameter groups ─────────────────────────────────────────────────────

    def rna_fm_unfrozen_params(self):
        """Trainable parameters inside the RNA-FM backbone (last n layers)."""
        yield from self.rna_fm._unfrozen_params()

    def other_params(self):
        """All trainable parameters except the RNA-FM backbone."""
        rna_fm_ids = {id(p) for p in self.rna_fm.fm_model.parameters()}
        for p in self.parameters():
            if id(p) not in rna_fm_ids and p.requires_grad:
                yield p

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        rna_bg: BatchedGraph,
        lig_input,                        # list[str] SMILES  OR  BatchedGraph
        rna_sequences: List[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        device = rna_bg.node_feat.device
        B      = int(rna_bg.batch_index.max().item()) + 1

        # ── Per-graph node counts (needed by RNA-FM for flat → per-token) ─────
        seq_lengths = [int((rna_bg.batch_index == b).sum().item()) for b in range(B)]

        # ── 1. Structural node feature (optionally include in_pocket flag) ────
        if self.include_pocket_feat and rna_bg.site_label is not None:
            rna_struct = torch.cat(
                [rna_bg.node_feat, rna_bg.site_label.unsqueeze(-1)], dim=-1
            )   # [total_N, 8]
        else:
            rna_struct = rna_bg.node_feat   # [total_N, 7]

        # ── 2. RNA-FM per-nucleotide embeddings [total_N, embed_dim] ──────────
        if rna_sequences is None:
            # Fallback zeros (inference without sequences / dry-run)
            fm_emb = torch.zeros(
                rna_bg.node_feat.shape[0], self.rna_fm.proj[0].in_features,
                device=device, dtype=rna_bg.node_feat.dtype,
            )
            fm_emb = self.rna_fm.proj(fm_emb)
        else:
            fm_emb = self.rna_fm(rna_sequences, seq_lengths, device=device)

        # ── 3. Fuse FM embeddings with structural features ────────────────────
        fused = self.rna_node_fuse(
            torch.cat([fm_emb, rna_struct.float()], dim=-1)
        )   # [total_N, embed_dim]

        # ── 4. EGNN (fused features + 3-D coordinates) ───────────────────────
        h_rna, pos_out = self.rna_encoder(
            fused, rna_bg.pos, rna_bg.edge_index, rna_bg.edge_feat,
        )   # [total_N, D], [total_N, 3]
        h_rna = h_rna.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        # ── 5. Ligand encoding ────────────────────────────────────────────────
        if isinstance(lig_input, list):
            # SMILES list → build molecular graphs on-the-fly (same as v4)
            lig_graphs = [build_lig_graph_cached(s) for s in lig_input]
            _lig_bg    = batch_ligand_graphs(lig_graphs)
            lig_bg     = BatchedGraph(
                node_feat=_lig_bg.node_feat.to(device),
                edge_index=_lig_bg.edge_index.to(device),
                edge_feat=_lig_bg.edge_feat.to(device),
                batch_index=_lig_bg.batch_index.to(device),
            )
        else:
            lig_bg = lig_input

        h_lig = self.lig_encoder(lig_bg.node_feat, lig_bg.edge_index, lig_bg.edge_feat)

        # ── 6. Unpack RNA → padded [B, max_rna_tokens, D] + mask ─────────────
        h_rna_pad, rna_mask = _unpack_to_padded(
            h_rna, rna_bg.batch_index, B, self.max_rna_tokens
        )   # [B, Lr, D], [B, Lr]

        # ── 6a. Unpack EGNN-updated 3D coords for dist-bias ──────────────────
        if self.use_dist_bias:
            pos_pad, _ = _unpack_to_padded(
                pos_out.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0),
                rna_bg.batch_index, B, self.max_rna_tokens,
            )   # [B, Lr, 3]
        else:
            pos_pad = None

        # ── 7. Unpack ligand → padded [B, max_lig_tokens, D] + mask ──────────
        h_lig_pad, lig_mask = _unpack_to_padded(
            h_lig, lig_bg.batch_index, B, self.max_lig_tokens
        )   # [B, Ll, D], [B, Ll]

        # ── 8. Cross-attention with 3D distance bias ──────────────────────────
        h_rna_pad, h_lig_pad = self.interaction(
            h_rna_pad, h_lig_pad, rna_mask, lig_mask,
            rna_pos=pos_pad,
        )
        h_rna_pad = h_rna_pad.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        h_lig_pad = h_lig_pad.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        # ── 9. Pocket-aware pooling ───────────────────────────────────────────
        if rna_bg.site_label is not None:
            sl_pad = torch.zeros(B, self.max_rna_tokens, dtype=torch.float32, device=device)
            for _b in range(B):
                _idx = (rna_bg.batch_index == _b).nonzero(as_tuple=True)[0]
                _n   = min(_idx.shape[0], self.max_rna_tokens)
                sl_pad[_b, :_n] = rna_bg.site_label[_idx[:_n]]
            pocket_mask = rna_mask & (sl_pad > 0.5)
            _has_pocket = pocket_mask.any(dim=1, keepdim=True)
            pool_mask   = torch.where(_has_pocket, pocket_mask, rna_mask)
        else:
            sl_pad    = torch.zeros(B, self.max_rna_tokens, dtype=torch.float32, device=device)
            pool_mask = rna_mask

        pooled_rna = self.rna_pool(h_rna_pad, pool_mask).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        pooled_lig = self.lig_pool(h_lig_pad, lig_mask).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        # ── 10. Dual-encoder projections + L2 normalise ───────────────────────
        z_rna = F.normalize(self.z_rna_proj(pooled_rna), p=2, dim=-1)   # [B, P]
        z_lig = F.normalize(self.z_lig_proj(pooled_lig), p=2, dim=-1)   # [B, P]

        rank_score = (z_rna * z_lig).sum(-1)                             # [B]

        z_pair     = torch.cat([z_rna, z_lig, z_rna * z_lig, torch.abs(z_rna - z_lig)], dim=-1)
        dock_score = self.dock_head(z_pair).squeeze(-1)                  # [B]

        # ── 11. Per-residue site logits ───────────────────────────────────────
        lig_ctx   = pooled_lig.unsqueeze(1).expand(-1, self.max_rna_tokens, -1)
        h_rna_n   = h_rna_pad / (h_rna_pad.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6))
        lig_ctx_n = lig_ctx   / (lig_ctx.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6))
        site_feat = torch.cat([h_rna_n, lig_ctx_n, h_rna_n * lig_ctx_n], dim=-1)
        site_logits = self.site_head(site_feat).squeeze(-1)              # [B, Lr]

        return {
            "rank_score":     rank_score,
            "screen_score":   rank_score,
            "dock_score":     dock_score,
            "site_logits":    site_logits,
            "site_label_pad": sl_pad,
            "rna_mask":       rna_mask,
            "z_rna":          z_rna,
            "z_lig":          z_lig,
            "z_pair":         z_pair,
        }
