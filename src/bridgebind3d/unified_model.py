from __future__ import annotations

import torch
import torch.nn as nn


class InteractionModule(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.rna_to_lig = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.lig_to_rna = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm_rna = nn.LayerNorm(embed_dim)
        self.norm_lig = nn.LayerNorm(embed_dim)

    def forward(self, rna_tokens: torch.Tensor, lig_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rna_ctx, _ = self.rna_to_lig(query=rna_tokens, key=lig_tokens, value=lig_tokens)
        lig_ctx, _ = self.lig_to_rna(query=lig_tokens, key=rna_tokens, value=rna_tokens)
        return self.norm_rna(rna_tokens + rna_ctx), self.norm_lig(lig_tokens + lig_ctx)


class UnifiedInteractionModel(nn.Module):
    """Shared backbone + 3 heads.

    Outputs:
    - ranking score (pair scalar)
    - decoy score/logit (pair scalar)
    - docking score (pair scalar)
    - site logits (per RNA residue/token)
    """

    def __init__(self, embed_dim: int = 128, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.rna_encoder = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout))
        self.lig_encoder = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout))
        self.interaction = InteractionModule(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

        pair_in = embed_dim * 2
        self.rank_head = nn.Sequential(nn.Linear(pair_in, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))
        self.decoy_head = nn.Sequential(nn.Linear(pair_in, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))
        self.dock_head = nn.Sequential(nn.Linear(pair_in, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))
        self.site_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))

    def forward(self, rna_tokens: torch.Tensor, lig_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        h_rna = self.rna_encoder(rna_tokens)
        h_lig = self.lig_encoder(lig_tokens)
        h_rna, h_lig = self.interaction(h_rna, h_lig)

        pooled_rna = h_rna.mean(dim=1)
        pooled_lig = h_lig.mean(dim=1)
        z_pair = torch.cat([pooled_rna, pooled_lig], dim=-1)

        rank_score = self.rank_head(z_pair).squeeze(-1)
        decoy_logit = self.decoy_head(z_pair).squeeze(-1)
        dock_score = self.dock_head(z_pair).squeeze(-1)
        site_logits = self.site_head(h_rna).squeeze(-1)

        return {
            "rank_score": rank_score,
            "decoy_logit": decoy_logit,
            "dock_score": dock_score,
            "site_logits": site_logits,
            "z_pair": z_pair,
            "z_rna": pooled_rna,
            "z_lig": pooled_lig,
            "h_rna": h_rna,
            "h_lig": h_lig,
        }
