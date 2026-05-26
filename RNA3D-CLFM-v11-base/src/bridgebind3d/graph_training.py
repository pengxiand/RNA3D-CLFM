from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from bridgebind3d.featurizers import build_pair_graphs
from bridgebind3d.graph_data import batch_ligand_graphs, batch_rna_graphs
from bridgebind3d.dual_encoder_model import DualEncoderModel


@dataclass
class BridgeData:
    docking: pd.DataFrame
    binary: pd.DataFrame
    decoy: pd.DataFrame


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_manifests(project_root: Path, cfg: dict[str, Any]) -> BridgeData:
    mdir = project_root / cfg["data"]["manifests_dir"]
    docking = pd.read_csv(mdir / cfg["data"]["docking_manifest"])
    binary = pd.read_csv(mdir / cfg["data"]["binary_manifest"])
    decoy = pd.read_csv(mdir / cfg["data"]["decoy_manifest"])
    return BridgeData(docking=docking, binary=binary, decoy=decoy)


def _clean_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v)


def _row_graphs(row: pd.Series):
    return build_pair_graphs(
        pocket_id=_clean_str(row.get("pocket_id", "")),
        ligand_smiles=_clean_str(row.get("ligand_smiles", "")),
        pocket_structure_path=_clean_str(row.get("target_structure_path", "")),
    )


def _batch_graphs(df: pd.DataFrame):
    rna_graphs = []
    lig_graphs = []
    for _, row in df.iterrows():
        rg, lg = _row_graphs(row)
        rna_graphs.append(rg)
        lig_graphs.append(lg)
    return batch_rna_graphs(rna_graphs), batch_ligand_graphs(lig_graphs)


def _sample_rank_batch(decoy_df: pd.DataFrame, batch_size: int, negatives_per_positive: int) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    pos = decoy_df[decoy_df["label_value"] == 1]
    neg = decoy_df[decoy_df["label_value"] == 0]
    pos_batch = pos.sample(n=min(batch_size, len(pos)), replace=len(pos) < batch_size).reset_index(drop=True)

    neg_groups = neg.groupby(neg["pocket_id"].astype(str))
    neg_batches = []
    for _, row in pos_batch.iterrows():
        pid = str(row.get("pocket_id", ""))
        pool = neg_groups.get_group(pid) if pid in neg_groups.groups else neg
        sampled = pool.sample(n=negatives_per_positive, replace=len(pool) < negatives_per_positive).reset_index(drop=True)
        neg_batches.append(sampled)
    return pos_batch, neg_batches


def _margin_rank_loss(model: DualEncoderModel, pos_batch: pd.DataFrame, neg_batches: list[pd.DataFrame], margin: float) -> torch.Tensor:
    losses = []
    for i in range(len(pos_batch)):
        pos_df = pos_batch.iloc[[i]]
        neg_df = neg_batches[i]

        rna_pos, lig_pos = _batch_graphs(pos_df)
        rna_neg, lig_neg = _batch_graphs(neg_df)

        s_pos = model(rna_pos, lig_pos)["screening_logit"].mean()
        s_neg = model(rna_neg, lig_neg)["screening_logit"].mean()
        target = torch.tensor([1.0], dtype=torch.float32)
        losses.append(F.margin_ranking_loss(s_pos.unsqueeze(0), s_neg.unsqueeze(0), target, margin=margin))

    return torch.stack(losses).mean() if losses else torch.tensor(0.0)


def _binary_batch_loss(model: DualEncoderModel, batch_df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    rna_b, lig_b = _batch_graphs(batch_df)
    out = model(rna_b, lig_b)
    y = torch.tensor(batch_df["label_value"].astype(float).values, dtype=torch.float32)
    decoy = F.binary_cross_entropy_with_logits(out["screening_logit"], y)
    affinity_bin = F.binary_cross_entropy_with_logits(out["affinity_score"], y)
    return decoy, affinity_bin


def _dock_batch_loss(model: DualEncoderModel, batch_df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    rna_b, lig_b = _batch_graphs(batch_df)
    out = model(rna_b, lig_b)

    y_dock = torch.tensor(batch_df["label_value"].astype(float).values, dtype=torch.float32)
    dock = F.mse_loss(out["affinity_score"], y_dock)

    site_target = rna_b.site_label
    site_loss = F.binary_cross_entropy_with_logits(out["site_logits"], site_target)
    return dock, site_loss


def train_dual_encoder(project_root: Path, config_path: Path) -> None:
    cfg = load_config(config_path)
    data = load_manifests(project_root, cfg)

    torch.manual_seed(int(cfg.get("seed", 42)))

    model = DualEncoderModel(
        rna_in_dim=int(cfg["model"]["rna_in_dim"]),
        rna_edge_dim=int(cfg["model"]["rna_edge_dim"]),
        lig_in_dim=int(cfg["model"]["lig_in_dim"]),
        lig_edge_dim=int(cfg["model"]["lig_edge_dim"]),
        embed_dim=int(cfg["model"]["embed_dim"]),
        rna_layers=int(cfg["model"]["rna_layers"]),
        lig_layers=int(cfg["model"]["lig_layers"]),
        dropout=float(cfg["model"].get("dropout", 0.1)),
    )

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))

    batch_size = int(cfg["train"]["batch_size"])
    epochs = int(cfg["train"]["epochs"])
    margin = float(cfg["loss"].get("ranking_margin", 0.2))

    stage1_epochs = int(cfg["schedule"].get("stage1_epochs", 4))
    stage2_epochs = int(cfg["schedule"].get("stage2_epochs", 4))
    stage3_epochs = int(cfg["schedule"].get("stage3_epochs", 4))

    neg_cfg = cfg["train"].get("negatives_per_positive", {})
    stage1_neg = int(neg_cfg.get("stage1", 20))
    stage2_neg = int(neg_cfg.get("stage2", 50))
    stage3_neg = int(neg_cfg.get("stage3", 100))

    w_rank = float(cfg["loss"].get("lambda_rank", 1.0))
    w_dock = float(cfg["loss"].get("lambda_dock", 0.5))
    w_site = float(cfg["loss"].get("lambda_site", 0.5))
    w_decoy = float(cfg["loss"].get("lambda_decoy", 1.0))

    for epoch in range(1, epochs + 1):
        if epoch <= stage1_epochs:
            phase = "phase1"
            n_neg = stage1_neg
            use_site = False
            use_decoy = False
        elif epoch <= stage1_epochs + stage2_epochs:
            phase = "phase2"
            n_neg = stage2_neg
            use_site = True
            use_decoy = False
        else:
            phase = "phase3"
            n_neg = stage3_neg
            use_site = True
            use_decoy = True

        pos_batch, neg_batches = _sample_rank_batch(data.decoy, batch_size=batch_size, negatives_per_positive=n_neg)
        dock_batch = data.docking.sample(n=min(batch_size, len(data.docking)), replace=len(data.docking) < batch_size)
        bin_batch = data.binary.sample(n=min(batch_size, len(data.binary)), replace=len(data.binary) < batch_size)

        rank_loss = _margin_rank_loss(model, pos_batch, neg_batches, margin=margin)
        dock_loss, site_loss = _dock_batch_loss(model, dock_batch)
        decoy_loss, affinity_bin_loss = _binary_batch_loss(model, bin_batch)

        total = w_rank * rank_loss + w_dock * (dock_loss + 0.2 * affinity_bin_loss)
        if use_site:
            total = total + w_site * site_loss
        if use_decoy:
            total = total + w_decoy * decoy_loss

        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"].get("grad_clip", 1.0)))
        opt.step()

        print(
            f"[epoch {epoch}/{epochs}] {phase} neg={n_neg} "
            f"rank={rank_loss.item():.4f} dock={dock_loss.item():.4f} site={site_loss.item():.4f} "
            f"decoy={decoy_loss.item():.4f} aff_bin={affinity_bin_loss.item():.4f} total={total.item():.4f}"
        )

    out_dir = project_root / cfg["output"]["checkpoint_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dual = out_dir / "dual_encoder_last.pt"
    ckpt_legacy = out_dir / "two_tower_last.pt"
    payload = {"model_state_dict": model.state_dict(), "config": cfg}
    torch.save(payload, ckpt_dual)
    torch.save(payload, ckpt_legacy)
    print(f"[OK] Saved checkpoint: {ckpt_dual}")
    print(f"[OK] Saved checkpoint (legacy name): {ckpt_legacy}")


def train_two_tower(project_root: Path, config_path: Path) -> None:
    """Backward-compatible alias for train_dual_encoder."""
    train_dual_encoder(project_root, config_path)
