from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class RNAGraph:
    node_feat: torch.Tensor  # [N, Fr]
    pos: torch.Tensor  # [N, 3]
    edge_index: torch.Tensor  # [2, E]
    edge_feat: torch.Tensor  # [E, Er]
    site_label: torch.Tensor  # [N]
    node_id: Optional[list[str]] = None
    sequence: str = ""  # nt_code string for RNA-FM (one char per node, in node order)


@dataclass
class LigandGraph:
    node_feat: torch.Tensor  # [M, Fl]
    edge_index: torch.Tensor  # [2, E]
    edge_feat: torch.Tensor  # [E, El]


@dataclass
class BatchedGraph:
    node_feat: torch.Tensor
    edge_index: torch.Tensor
    edge_feat: torch.Tensor
    batch_index: torch.Tensor
    pos: Optional[torch.Tensor] = None
    site_label: Optional[torch.Tensor] = None


def _empty_edge_index() -> torch.Tensor:
    return torch.zeros((2, 0), dtype=torch.long)


def _empty_edge_feat(dim: int) -> torch.Tensor:
    return torch.zeros((0, dim), dtype=torch.float32)


def batch_rna_graphs(graphs: list[RNAGraph]) -> BatchedGraph:
    node_feat_parts = []
    pos_parts = []
    edge_idx_parts = []
    edge_feat_parts = []
    batch_idx_parts = []
    site_parts = []

    node_offset = 0
    edge_dim = graphs[0].edge_feat.shape[1] if graphs and graphs[0].edge_feat.ndim == 2 else 0

    for i, g in enumerate(graphs):
        n = int(g.node_feat.shape[0])
        node_feat_parts.append(g.node_feat)
        pos_parts.append(g.pos)
        site_parts.append(g.site_label)
        batch_idx_parts.append(torch.full((n,), i, dtype=torch.long))

        if g.edge_index.numel() > 0:
            edge_idx_parts.append(g.edge_index + node_offset)
            edge_feat_parts.append(g.edge_feat)

        node_offset += n

    node_feat = torch.cat(node_feat_parts, dim=0) if node_feat_parts else torch.zeros((0, 1), dtype=torch.float32)
    pos = torch.cat(pos_parts, dim=0) if pos_parts else torch.zeros((0, 3), dtype=torch.float32)
    site_label = torch.cat(site_parts, dim=0) if site_parts else torch.zeros((0,), dtype=torch.float32)
    batch_index = torch.cat(batch_idx_parts, dim=0) if batch_idx_parts else torch.zeros((0,), dtype=torch.long)

    if edge_idx_parts:
        edge_index = torch.cat(edge_idx_parts, dim=1)
        edge_feat = torch.cat(edge_feat_parts, dim=0)
    else:
        edge_index = _empty_edge_index()
        edge_feat = _empty_edge_feat(edge_dim if edge_dim > 0 else 1)

    return BatchedGraph(
        node_feat=node_feat,
        pos=pos,
        edge_index=edge_index,
        edge_feat=edge_feat,
        batch_index=batch_index,
        site_label=site_label,
    )


def batch_ligand_graphs(graphs: list[LigandGraph]) -> BatchedGraph:
    node_feat_parts = []
    edge_idx_parts = []
    edge_feat_parts = []
    batch_idx_parts = []

    node_offset = 0
    edge_dim = graphs[0].edge_feat.shape[1] if graphs and graphs[0].edge_feat.ndim == 2 else 0

    for i, g in enumerate(graphs):
        n = int(g.node_feat.shape[0])
        node_feat_parts.append(g.node_feat)
        batch_idx_parts.append(torch.full((n,), i, dtype=torch.long))

        if g.edge_index.numel() > 0:
            edge_idx_parts.append(g.edge_index + node_offset)
            edge_feat_parts.append(g.edge_feat)

        node_offset += n

    node_feat = torch.cat(node_feat_parts, dim=0) if node_feat_parts else torch.zeros((0, 1), dtype=torch.float32)
    batch_index = torch.cat(batch_idx_parts, dim=0) if batch_idx_parts else torch.zeros((0,), dtype=torch.long)

    if edge_idx_parts:
        edge_index = torch.cat(edge_idx_parts, dim=1)
        edge_feat = torch.cat(edge_feat_parts, dim=0)
    else:
        edge_index = _empty_edge_index()
        edge_feat = _empty_edge_feat(edge_dim if edge_dim > 0 else 1)

    return BatchedGraph(
        node_feat=node_feat,
        edge_index=edge_index,
        edge_feat=edge_feat,
        batch_index=batch_index,
    )
