from __future__ import annotations

import torch
import torch.nn as nn

from bridgebind3d.egnn import scatter_sum


class GINELayer(nn.Module):
    """GIN layer with edge features (GINE-style)."""

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.eps = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_feat: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return h

        src, dst = edge_index[0], edge_index[1]
        m = h[src] + self.edge_proj(edge_feat)
        m = self.msg_mlp(m)
        agg = scatter_sum(m, dst, dim_size=h.shape[0])

        out = self.update_mlp((1.0 + self.eps) * h + agg)
        return self.norm(h + out)


class GINEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, edge_dim: int, num_layers: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([GINELayer(hidden_dim=hidden_dim, edge_dim=edge_dim, dropout=dropout) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_feat: torch.Tensor, edge_index: torch.Tensor, edge_feat: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(node_feat)
        for layer in self.layers:
            h = layer(h, edge_index, edge_feat)
        return self.out_norm(h)


class FP2LigandEncoder(nn.Module):
    """Morgan FP2 fingerprint (bit-vector) → MLP → single embedding token.

    Mirrors SMARTBind's ligand encoder: a pre-computed 2048-bit ECFP4
    fingerprint is projected through a two-layer MLP with LayerNorm.
    Returns [B, embed_dim]; caller unsqueezes to [B, 1, D] for cross-attention.
    """

    def __init__(self, fp_dim: int = 2048, embed_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fp_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, embed_dim),
        )

    def forward(self, fp_batch: torch.Tensor) -> torch.Tensor:
        """Args:
            fp_batch: [B, fp_dim] float32 tensor of FP2 bit-vectors.
        Returns:
            [B, embed_dim] float32 tensor.
        """
        return self.net(fp_batch)
