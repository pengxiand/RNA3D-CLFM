from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from bridgebind3d.featurizers import build_pair_features
from bridgebind3d.unified_model import UnifiedInteractionModel


@dataclass
class BridgeData:
    docking: pd.DataFrame
    binary: pd.DataFrame
    decoy: pd.DataFrame
    site: pd.DataFrame


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_manifests(project_root: Path, cfg: dict[str, Any]) -> BridgeData:
    mdir = project_root / cfg["data"]["manifests_dir"]
    docking = pd.read_csv(mdir / cfg["data"]["docking_manifest"])
    binary = pd.read_csv(mdir / cfg["data"]["binary_manifest"])
    decoy = pd.read_csv(mdir / cfg["data"]["decoy_manifest"])
    site = pd.read_csv(mdir / cfg["data"]["site_manifest"])
    return BridgeData(docking=docking, binary=binary, decoy=decoy, site=site)


def _clean_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v)


def _batch_features(
    df: pd.DataFrame,
    max_rna_tokens: int,
    max_lig_tokens: int,
    embed_dim: int,
    featurizer_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    rna_rows = []
    lig_rows = []
    for _, row in df.iterrows():
        pocket_id = _clean_str(row.get("pocket_id", ""))
        ligand = _clean_str(row.get("ligand_smiles", ""))
        structure_path = _clean_str(row.get("target_structure_path", ""))
        rna_tokens, lig_tokens = build_pair_features(
            pocket_id,
            ligand,
            max_rna_tokens,
            max_lig_tokens,
            embed_dim,
            pocket_structure_path=structure_path,
            featurizer_mode=featurizer_mode,
        )
        rna_rows.append(rna_tokens)
        lig_rows.append(lig_tokens)
    return torch.stack(rna_rows, dim=0), torch.stack(lig_rows, dim=0)


def _sample_site_targets(site_df: pd.DataFrame, pocket_ids: list[str], max_rna_tokens: int) -> torch.Tensor:
    pocket_set = set(pocket_ids)
    selected = site_df[site_df["pocket_id"].astype(str).isin(pocket_set)]
    y = torch.zeros(len(pocket_ids), max_rna_tokens)
    if selected.empty:
        return y

    grouped = selected.groupby(selected["pocket_id"].astype(str))
    for i, pid in enumerate(pocket_ids):
        if pid not in grouped.groups:
            continue
        grp = grouped.get_group(pid)
        n = min(len(grp), max_rna_tokens)
        y[i, :n] = torch.tensor(grp["in_pocket"].astype(float).values[:n])
    return y


def build_rank_batch(decoy_df: pd.DataFrame, batch_size: int, negatives_per_positive: int) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    pos = decoy_df[decoy_df["label_value"] == 1]
    neg = decoy_df[decoy_df["label_value"] == 0]
    pos_batch = pos.sample(n=min(batch_size, len(pos)), replace=len(pos) < batch_size)

    neg_batches: list[pd.DataFrame] = []
    grouped_neg = neg.groupby(neg["pocket_id"].astype(str))
    for _, row in pos_batch.iterrows():
        pid = str(row["pocket_id"])
        this_neg_rows = []
        if pid in grouped_neg.groups:
            pool = grouped_neg.get_group(pid)
        else:
            pool = neg
        for _ in range(negatives_per_positive):
            this_neg_rows.append(pool.sample(n=1, replace=len(pool) < 1))
        neg_batches.append(pd.concat(this_neg_rows, ignore_index=True))
    return pos_batch.reset_index(drop=True), neg_batches


def contrastive_infonce_loss(
    model: UnifiedInteractionModel,
    pos_batch: pd.DataFrame,
    neg_batches: list[pd.DataFrame],
    max_rna_tokens: int,
    max_lig_tokens: int,
    embed_dim: int,
    featurizer_mode: str,
    temperature: float,
) -> torch.Tensor:
    losses = []
    for i in range(len(pos_batch)):
        pos_df = pos_batch.iloc[[i]].reset_index(drop=True)
        neg_df = neg_batches[i]
        cand_df = pd.concat([pos_df, neg_df], ignore_index=True)
        rna, lig = _batch_features(cand_df, max_rna_tokens, max_lig_tokens, embed_dim, featurizer_mode)
        scores = model(rna, lig)["rank_score"] / temperature
        target = torch.tensor([0], dtype=torch.long)
        losses.append(F.cross_entropy(scores.unsqueeze(0), target))
    return torch.stack(losses).mean()


def train_unified(project_root: Path, config_path: Path) -> None:
    cfg = load_config(config_path)
    data = load_manifests(project_root, cfg)

    torch.manual_seed(int(cfg.get("seed", 42)))
    model = UnifiedInteractionModel(
        embed_dim=cfg["model"]["embed_dim"],
        num_heads=cfg["model"]["num_heads"],
        dropout=cfg["model"]["dropout"],
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    max_rna_tokens = cfg["data"]["max_rna_tokens"]
    max_lig_tokens = cfg["data"]["max_lig_tokens"]
    embed_dim = cfg["model"]["embed_dim"]

    batch_size = cfg["train"]["batch_size"]
    epochs = cfg["train"]["epochs"]
    negatives_per_positive = cfg["train"].get("negatives_per_positive", 4)
    temperature = cfg["loss"]["contrastive_temperature"]
    featurizer_mode = cfg.get("featurizer", {}).get("mode", "real")

    lambda_rank = cfg["loss"]["lambda_rank"]
    lambda_dock = cfg["loss"]["lambda_dock"]
    lambda_site = cfg["loss"]["lambda_site"]
    lambda_decoy = cfg["loss"].get("lambda_decoy", 0.0)

    # Backward-compatible schedule parsing.
    phase1_epochs = cfg.get("schedule", {}).get("phase1_epochs", cfg.get("schedule", {}).get("stage1_epochs", 1))
    phase2_epochs = cfg.get("schedule", {}).get("phase2_epochs", cfg.get("schedule", {}).get("stage2_epochs", 1))
    phase3_epochs = cfg.get("schedule", {}).get("phase3_epochs", cfg.get("schedule", {}).get("stage3_epochs", 1))

    # Optional decoy curriculum (20 -> 50 -> 100 etc.).
    decoy_curriculum = cfg.get("train", {}).get("decoy_curriculum", {})
    stage1_neg = decoy_curriculum.get("stage1_negatives_per_positive", negatives_per_positive)
    stage2_neg = decoy_curriculum.get("stage2_negatives_per_positive", negatives_per_positive)
    stage3_neg = decoy_curriculum.get("stage3_negatives_per_positive", negatives_per_positive)

    for epoch in range(1, epochs + 1):
        # Phase schedule: progressively enable objectives.
        if epoch <= phase1_epochs:
            w_rank, w_dock, w_site = 1.0, 1.0, 0.0
            w_decoy = 1.0
            epoch_negatives = stage1_neg
            phase = "phase1-rank+dock"
        elif epoch <= phase1_epochs + phase2_epochs:
            w_rank, w_dock, w_site = 1.0, 0.5, 1.0
            w_decoy = 1.0
            epoch_negatives = stage2_neg
            phase = "phase2-add-site"
        else:
            w_rank, w_dock, w_site = 1.0, 1.0, 1.0
            w_decoy = 1.0
            epoch_negatives = stage3_neg
            phase = "phase3-joint"

        pos_batch, neg_batches = build_rank_batch(
            data.decoy,
            batch_size=batch_size,
            negatives_per_positive=epoch_negatives,
        )
        dock_batch = data.docking.sample(n=min(batch_size, len(data.docking)), replace=len(data.docking) < batch_size)
        binary_batch = data.binary.sample(n=min(batch_size, len(data.binary)), replace=len(data.binary) < batch_size)

        rna_dock, lig_dock = _batch_features(dock_batch, max_rna_tokens, max_lig_tokens, embed_dim, featurizer_mode)
        rna_bin, lig_bin = _batch_features(binary_batch, max_rna_tokens, max_lig_tokens, embed_dim, featurizer_mode)

        out_dock = model(rna_dock, lig_dock)
        out_bin = model(rna_bin, lig_bin)

        rank_loss = contrastive_infonce_loss(
            model=model,
            pos_batch=pos_batch,
            neg_batches=neg_batches,
            max_rna_tokens=max_rna_tokens,
            max_lig_tokens=max_lig_tokens,
            embed_dim=embed_dim,
            featurizer_mode=featurizer_mode,
            temperature=temperature,
        )

        target_dock = torch.tensor(dock_batch["label_value"].astype(float).values, dtype=torch.float32)
        dock_reg_loss = F.mse_loss(out_dock["dock_score"], target_dock)

        target_bin = torch.tensor(binary_batch["label_value"].astype(float).values, dtype=torch.float32)
        dock_bin_loss = F.binary_cross_entropy_with_logits(out_bin["dock_score"], target_bin)
        dock_loss = 0.5 * (dock_reg_loss + dock_bin_loss)

        decoy_loss = F.binary_cross_entropy_with_logits(out_bin["decoy_logit"], target_bin)

        pocket_ids = [str(x) for x in dock_batch["pocket_id"].values]
        site_target = _sample_site_targets(data.site, pocket_ids, max_rna_tokens=max_rna_tokens)
        site_loss = F.binary_cross_entropy_with_logits(out_dock["site_logits"], site_target)

        total = (
            w_rank * lambda_rank * rank_loss
            + w_dock * lambda_dock * dock_loss
            + w_site * lambda_site * site_loss
            + w_decoy * lambda_decoy * decoy_loss
        )

        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
        opt.step()

        print(
            f"[epoch {epoch}/{epochs}] {phase} "
            f"rank={rank_loss.item():.4f} dock={dock_loss.item():.4f} "
            f"(reg={dock_reg_loss.item():.4f}, bin={dock_bin_loss.item():.4f}) "
            f"decoy={decoy_loss.item():.4f} site={site_loss.item():.4f} "
            f"neg={epoch_negatives} total={total.item():.4f}"
        )

    ckpt_dir = project_root / cfg["output"]["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": cfg}, ckpt_dir / "unified_multitask_last.pt")
    print(f"[OK] Saved checkpoint: {ckpt_dir / 'unified_multitask_last.pt'}")
