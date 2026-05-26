from __future__ import annotations

import torch
import torch.nn as nn


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    if src.numel() == 0:
        return torch.zeros((dim_size, src.shape[-1]), dtype=src.dtype, device=src.device)
    out = torch.zeros((dim_size, src.shape[-1]), dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    return out


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    if src.numel() == 0:
        return torch.zeros((dim_size, src.shape[-1]), dtype=src.dtype, device=src.device)
    out = scatter_sum(src, index, dim_size)
    ones = torch.ones((src.shape[0], 1), dtype=src.dtype, device=src.device)
    cnt = scatter_sum(ones, index, dim_size).clamp(min=1.0)
    return out / cnt


class EGNNLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_index.numel() == 0:
            return h, x

        src, dst = edge_index[0], edge_index[1]
        h_src = h[src]
        h_dst = h[dst]

        rel = x[src] - x[dst]
        d2 = (rel * rel).sum(dim=-1, keepdim=True)
        # Clamp d2 to prevent large Angstrom-scale coordinates from causing
        # inf values in edge_mlp (inf * negative_weight = -inf → inf-inf = NaN).
        d2 = d2.clamp(max=1e4)
        e_in = torch.cat([h_src, h_dst, edge_feat, d2], dim=-1)
        m = self.edge_mlp(e_in)

        agg = scatter_sum(m, dst, dim_size=h.shape[0])
        h_upd = self.node_mlp(torch.cat([h, agg], dim=-1))
        h = self.norm(h + h_upd)

        coord_scale = self.coord_mlp(m)
        dx = scatter_sum(coord_scale * rel, dst, dim_size=x.shape[0])
        # Clamp displacement to prevent coordinate explosion across EGNN layers.
        x = x + dx.clamp(-0.5, 0.5)
        return h, x


class EGNNEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        edge_dim: int,
        num_layers: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([EGNNLayer(hidden_dim=hidden_dim, edge_dim=edge_dim, dropout=dropout) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_feat: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.in_proj(node_feat)
        x = pos
        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_feat)
        return self.out_norm(h), x
