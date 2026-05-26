from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from bridgebind3d.featurizers import build_pair_features, build_rna_tokens_from_pocket, build_rna_graph_cached, build_lig_graph_cached
from bridgebind3d.graph_data import BatchedGraph, batch_rna_graphs, batch_ligand_graphs
from bridgebind3d.unified_model import UnifiedInteractionModel
from bridgebind3d.unified_model_v2 import UnifiedInteractionModelV2
from bridgebind3d.unified_model_v3 import UnifiedInteractionModelV3


@dataclass
class BridgeData:
    docking: pd.DataFrame
    decoy: pd.DataFrame
    site: pd.DataFrame


def _read_manifest(path: Path, usecols: list[str], nrows: int | None) -> pd.DataFrame:
    """Read a manifest CSV, using a pre-built parquet cache when available.

    The parquet file (same name with .parquet extension) is loaded if it
    exists — it is typically 5-10x faster than reading the raw CSV from a
    network filesystem.  Run scripts/convert_manifests_to_parquet.py once to
    create the caches.
    """
    parquet_path = path.with_suffix(".parquet")
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path, columns=usecols)
        if nrows is not None and nrows > 0:
            df = df.iloc[:int(nrows)]
        return df
    kwargs: dict[str, Any] = {"usecols": usecols}
    if nrows is not None and nrows > 0:
        kwargs["nrows"] = int(nrows)
    return pd.read_csv(path, **kwargs)


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_manifests(project_root: Path, cfg: dict[str, Any]) -> BridgeData:
    import time
    mdir = project_root / cfg["data"]["manifests_dir"]
    data_cfg = cfg.get("data", {})

    def _load(name: str, manifest_key: str, cols: list[str], nrows_key: str) -> pd.DataFrame:
        path = mdir / cfg["data"][manifest_key]
        parquet = path.with_suffix(".parquet")
        src = str(parquet) if parquet.exists() else str(path)
        print(f"[load] {name}: reading {src} ...", flush=True)
        t0 = time.time()
        df = _read_manifest(path, usecols=cols, nrows=data_cfg.get(nrows_key))
        print(f"[load] {name}: {len(df):,} rows in {time.time()-t0:.1f}s", flush=True)
        return df

    def _load_with_split(name: str, manifest_key: str, cols: list[str], nrows_key: str) -> pd.DataFrame:
        path = mdir / cfg["data"][manifest_key]
        parquet = path.with_suffix(".parquet")
        src = str(parquet) if parquet.exists() else str(path)
        print(f"[load] {name}: reading {src} ...", flush=True)
        t0 = time.time()
        df = _read_manifest(path, usecols=cols + ["split"], nrows=data_cfg.get(nrows_key))
        before = len(df)
        df = df[df["split"].isin({"train", "validation"})].reset_index(drop=True)
        # Drop rows where label_value is NaN to prevent NaN losses
        if "label_value" in df.columns:
            n_nan = df["label_value"].isna().sum()
            if n_nan > 0:
                df = df.dropna(subset=["label_value"]).reset_index(drop=True)
                print(f"[load] {name}: dropped {n_nan} NaN label_value rows", flush=True)
        df = df.drop(columns=["split"])
        print(f"[load] {name}: {len(df):,} rows (filtered from {before:,}, test excluded) in {time.time()-t0:.1f}s", flush=True)
        return df

    docking = _load_with_split("docking", "docking_manifest",
                    ["pocket_id", "ligand_smiles", "target_structure_path", "label_value"],
                    "max_rows_docking")
    decoy   = _load_with_split("decoy",   "decoy_manifest",
                    ["pocket_id", "ligand_smiles", "target_structure_path", "label_value"],
                    "max_rows_decoy")
    site    = _load("site",    "site_manifest",
                    ["pocket_id", "in_pocket"],
                    "max_rows_site")

    # ── HARIBOSS hold-out split ───────────────────────────────────────────
    # Legacy: single-split CSV with 'hariboss_split' column ('train'/'test').
    split_file = data_cfg.get("hariboss_split_file")
    if split_file:
        split_path = mdir / split_file
        split_df   = pd.read_csv(split_path)
        test_pids  = set(split_df.loc[split_df["hariboss_split"] == "test", "pocket_id"])
        before_d, before_dec, before_s = len(docking), len(decoy), len(site)
        docking = docking[~docking["pocket_id"].isin(test_pids)].reset_index(drop=True)
        decoy   = decoy[~decoy["pocket_id"].isin(test_pids)].reset_index(drop=True)
        site    = site[~site["pocket_id"].isin(test_pids)].reset_index(drop=True)
        print(f"[split] hariboss hold-out: {len(test_pids)} test pockets excluded from training")
        print(f"[split] docking: {before_d:,} → {len(docking):,} | "
              f"decoy: {before_dec:,} → {len(decoy):,} | "
              f"site: {before_s:,} → {len(site):,}", flush=True)

    # ── K-fold split (SMARTBind-style) ────────────────────────────────────
    # kfold_split_file: hariboss_kfold_5fd_split.csv
    # kfold_fold_idx  : 0-4  (fold_N column; False = test for this fold)
    kfold_file = data_cfg.get("kfold_split_file")
    kfold_fold = data_cfg.get("kfold_fold_idx")
    if kfold_file is not None and kfold_fold is not None:
        kfold_fold = int(kfold_fold)
        kfold_path = mdir / kfold_file
        kf_df      = pd.read_csv(kfold_path)
        fold_col   = f"fold_{kfold_fold}"
        if fold_col not in kf_df.columns:
            raise ValueError(f"[kfold] column '{fold_col}' not found in {kfold_path}")
        # fold_col == False → this pocket is the TEST set for this fold
        test_pids_kf = set(kf_df.loc[~kf_df[fold_col].astype(bool), "pocket_id"])
        before_d, before_dec, before_s = len(docking), len(decoy), len(site)
        docking = docking[~docking["pocket_id"].isin(test_pids_kf)].reset_index(drop=True)
        decoy   = decoy[~decoy["pocket_id"].isin(test_pids_kf)].reset_index(drop=True)
        site    = site[~site["pocket_id"].isin(test_pids_kf)].reset_index(drop=True)
        print(f"[kfold] fold {kfold_fold}: {len(test_pids_kf)} test pockets excluded from training")
        print(f"[kfold] docking: {before_d:,} → {len(docking):,} | "
              f"decoy: {before_dec:,} → {len(decoy):,} | "
              f"site: {before_s:,} → {len(site):,}", flush=True)

    return BridgeData(docking=docking, decoy=decoy, site=site)


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
    device: torch.device | None = None,
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
    rna = torch.stack(rna_rows, dim=0)
    lig = torch.stack(lig_rows, dim=0)
    if device is not None:
        rna = rna.to(device)
        lig = lig.to(device)
    return rna, lig


# One-time diagnostic flag so we only print the overlap warning once.
_SITE_OVERLAP_CHECKED = False


def _sample_site_targets(
    site_df: pd.DataFrame,
    pocket_ids: list[str],
    max_rna_tokens: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    global _SITE_OVERLAP_CHECKED
    # Normalize both sides: strip whitespace + lowercase for robust matching.
    norm_ids = [p.strip().lower() for p in pocket_ids]
    norm_set = set(norm_ids)
    site_norm = site_df["pocket_id"].astype(str).str.strip().str.lower()

    selected = site_df[site_norm.isin(norm_set)]
    y = torch.zeros(len(pocket_ids), max_rna_tokens)
    if selected.empty:
        if not _SITE_OVERLAP_CHECKED:
            _SITE_OVERLAP_CHECKED = True
            site_sample = site_norm.unique()[:3].tolist()
            print(f"[site] WARNING: no pocket_id overlap between site manifest and docking batch."
                  f" Batch ids (first 3): {norm_ids[:3]}  Site ids (first 3): {site_sample}"
                  f" — site_loss will be 0 until pocket_ids match.", flush=True)
        return y.to(device) if device is not None else y

    _SITE_OVERLAP_CHECKED = True
    grouped = selected.groupby(site_norm[selected.index])
    for i, pid in enumerate(norm_ids):
        if pid not in grouped.groups:
            continue
        grp = grouped.get_group(pid)
        n = min(len(grp), max_rna_tokens)
        y[i, :n] = torch.tensor(grp["in_pocket"].astype(float).values[:n])
    return y.to(device) if device is not None else y


def _build_neg_index(
    decoy_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Pre-compute negative sample index per pocket. Call ONCE before training.

    Returns
    -------
    pos_df   : positive samples (reset index)
    neg_df   : negative samples (reset index)
    neg_index: mapping pocket_id -> integer positions in neg_df
    """
    pos_df = decoy_df[decoy_df["label_value"] == 1].reset_index(drop=True)
    neg_df = decoy_df[decoy_df["label_value"] == 0].reset_index(drop=True)
    neg_index: dict[str, np.ndarray] = {}
    for pid, group in neg_df.groupby(neg_df["pocket_id"].astype(str)):
        neg_index[str(pid)] = group.index.to_numpy()
    return pos_df, neg_df, neg_index


def _sample_rank_batch(
    pos_df: pd.DataFrame,
    neg_df: pd.DataFrame,
    neg_index: dict[str, np.ndarray],
    batch_size: int,
    negatives_per_positive: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fast per-step sampling that reuses the pre-computed neg_index.

    Returns
    -------
    pos_batch : B positive rows
    neg_flat  : B*K negative rows (rows 0..K-1 correspond to pos_batch[0], etc.)
    """
    B = min(batch_size, len(pos_df))
    K = negatives_per_positive
    pos_batch = pos_df.sample(n=B, replace=len(pos_df) < B)
    all_neg_idx = np.arange(len(neg_df))
    neg_locs: list[np.ndarray] = []
    for _, row in pos_batch.iterrows():
        pid = str(row["pocket_id"])
        pool = neg_index.get(pid, all_neg_idx)
        chosen = np.random.choice(pool, size=K, replace=len(pool) < K)
        neg_locs.append(chosen)
    neg_flat = neg_df.iloc[np.concatenate(neg_locs)].reset_index(drop=True)
    return pos_batch.reset_index(drop=True), neg_flat


def _rank_losses_batched(
    model: torch.nn.Module,
    pos_batch: pd.DataFrame,
    neg_flat: pd.DataFrame,
    K: int,
    max_rna_tokens: int,
    max_lig_tokens: int,
    embed_dim: int,
    featurizer_mode: str,
    temperature: float,
    margin: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute InfoNCE + margin-ranking losses with a SINGLE batched model forward pass.

    Instead of looping B times with separate forward calls (old code did B*(K+2) calls),
    we stack all B*(K+1) samples and run the model once.
    """
    B = len(pos_batch)

    # Featurize (hits in-memory cache after first epoch for RNA, first occurrence for ligand)
    pos_rna, pos_lig = _batch_features(
        pos_batch, max_rna_tokens, max_lig_tokens, embed_dim, featurizer_mode, device=None
    )  # [B, L_rna, D], [B, L_lig, D]
    neg_rna, neg_lig = _batch_features(
        neg_flat, max_rna_tokens, max_lig_tokens, embed_dim, featurizer_mode, device=None
    )  # [B*K, L_rna, D], [B*K, L_lig, D]

    L_rna, D_rna = pos_rna.shape[1], pos_rna.shape[2]
    L_lig, D_lig = pos_lig.shape[1], pos_lig.shape[2]

    # Reshape neg: [B, K, L, D]
    neg_rna = neg_rna.view(B, K, L_rna, D_rna)
    neg_lig = neg_lig.view(B, K, L_lig, D_lig)

    # Concatenate: pos at index 0, then K negs → [B, K+1, L, D]
    all_rna = torch.cat([pos_rna.unsqueeze(1), neg_rna], dim=1)
    all_lig = torch.cat([pos_lig.unsqueeze(1), neg_lig], dim=1)

    # Flatten to [B*(K+1), L, D] for model
    BK1 = B * (K + 1)
    all_rna_flat = all_rna.reshape(BK1, L_rna, D_rna).to(device)
    all_lig_flat = all_lig.reshape(BK1, L_lig, D_lig).to(device)

    # --- single forward pass ---
    out = model(all_rna_flat, all_lig_flat)
    scores = out["rank_score"].view(B, K + 1)  # [B, K+1]

    # InfoNCE: positive is at column 0
    contrast_loss = F.cross_entropy(
        scores / temperature, torch.zeros(B, dtype=torch.long, device=device)
    )

    # Margin ranking: each pos vs each of its K negatives
    pos_s = scores[:, 0].unsqueeze(1).expand(B, K).reshape(-1)  # [B*K]
    neg_s = scores[:, 1:].reshape(-1)                            # [B*K]
    rank_loss = F.margin_ranking_loss(
        pos_s, neg_s, torch.ones(B * K, device=device), margin=margin
    )

    return contrast_loss, rank_loss


# ------------------------------------------------------------------
# Graph helpers for v3
# ------------------------------------------------------------------

def _bg_to_device(bg: BatchedGraph, device: torch.device) -> BatchedGraph:
    """Move a BatchedGraph's tensors to device."""
    return BatchedGraph(
        node_feat=bg.node_feat.to(device),
        edge_index=bg.edge_index.to(device),
        edge_feat=bg.edge_feat.to(device),
        batch_index=bg.batch_index.to(device),
        pos=bg.pos.to(device) if bg.pos is not None else None,
        site_label=bg.site_label.to(device) if bg.site_label is not None else None,
    )


def _batch_graphs(
    df: pd.DataFrame,
    device: torch.device | None = None,
) -> tuple[BatchedGraph, BatchedGraph]:
    """Build batched RNA/ligand graphs from a DataFrame of pocket-ligand pairs."""
    rna_graphs, lig_graphs = [], []
    for _, row in df.iterrows():
        pid   = _clean_str(row.get("pocket_id", ""))
        spath = _clean_str(row.get("target_structure_path", ""))
        smi   = _clean_str(row.get("ligand_smiles", ""))
        rna_graphs.append(build_rna_graph_cached(pid, spath))
        lig_graphs.append(build_lig_graph_cached(smi))
    rna_bg = batch_rna_graphs(rna_graphs)
    lig_bg = batch_ligand_graphs(lig_graphs)
    if device is not None:
        rna_bg = _bg_to_device(rna_bg, device)
        lig_bg = _bg_to_device(lig_bg, device)
    return rna_bg, lig_bg


def _batch_rna_graphs(
    df: pd.DataFrame,
    device: torch.device | None = None,
) -> BatchedGraph:
    """Build batched RNA graphs only (used when ligand encoder is pretrained)."""
    rna_graphs = []
    for _, row in df.iterrows():
        pid   = _clean_str(row.get("pocket_id", ""))
        spath = _clean_str(row.get("target_structure_path", ""))
        rna_graphs.append(build_rna_graph_cached(pid, spath))
    rna_bg = batch_rna_graphs(rna_graphs)
    if device is not None:
        rna_bg = _bg_to_device(rna_bg, device)
    return rna_bg


def _rank_losses_batched_v3(
    model: torch.nn.Module,
    pos_batch: pd.DataFrame,
    neg_flat: pd.DataFrame,
    K: int,
    temperature: float,
    margin: float,
    device: torch.device,
    **_ignored: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """InfoNCE + margin-ranking for v3 (graph inputs).

    Ordering: pos_batch rows first, then neg_flat rows.
    After model forward: scores[:B] = pos, scores[B:].view(B,K) = per-pos negs.
    """
    B = len(pos_batch)
    all_df = pd.concat([pos_batch, neg_flat], ignore_index=True)  # B + B*K rows

    _pretrained = getattr(model, "use_pretrained_ligand", False)
    if _pretrained:
        rna_bg    = _batch_rna_graphs(all_df, device=device)
        smiles_in = all_df["ligand_smiles"].fillna("").tolist()
        out = model(rna_bg, smiles_in)
    else:
        rna_bg, lig_bg = _batch_graphs(all_df, device=device)
        out = model(rna_bg, lig_bg)
    scores_flat = out["rank_score"]           # [B*(K+1)]

    pos_s = scores_flat[:B]                   # [B]
    neg_s = scores_flat[B:].view(B, K)        # [B, K]
    all_scores = torch.cat([pos_s.unsqueeze(1), neg_s], dim=1)  # [B, K+1]

    contrast_loss = F.cross_entropy(
        all_scores / temperature,
        torch.zeros(B, dtype=torch.long, device=device),
    )
    rank_loss = F.margin_ranking_loss(
        pos_s.unsqueeze(1).expand(B, K).reshape(-1),
        neg_s.reshape(-1),
        torch.ones(B * K, device=device),
        margin=margin,
    )
    return contrast_loss, rank_loss


def _load_test_decoy_fold(project_root: Path, cfg: dict[str, Any]) -> "pd.DataFrame | None":
    """Return the TEST-fold rows of the decoy manifest for per-epoch validation.

    Returns None when no kfold split is configured.
    """
    data_cfg = cfg.get("data", {})
    kfold_file = data_cfg.get("kfold_split_file")
    kfold_fold = data_cfg.get("kfold_fold_idx")
    if not kfold_file or kfold_fold is None:
        return None
    mdir = project_root / data_cfg["manifests_dir"]
    path = mdir / data_cfg["decoy_manifest"]
    try:
        df = _read_manifest(
            path,
            usecols=["pocket_id", "ligand_smiles", "target_structure_path", "label_value"],
            nrows=None,
        )
    except Exception as e:
        print(f"[val] could not load decoy manifest for val: {e}", flush=True)
        return None

    kf_path = mdir / kfold_file
    kf_df = pd.read_csv(kf_path)
    fold_col = f"fold_{int(kfold_fold)}"
    if fold_col not in kf_df.columns:
        return None
    test_pids = set(kf_df.loc[~kf_df[fold_col].astype(bool), "pocket_id"])
    df = df[df["pocket_id"].isin(test_pids)].reset_index(drop=True)
    # Keep only rows with valid SMILES and label_value
    df = df.dropna(subset=["label_value", "ligand_smiles"]).reset_index(drop=True)
    print(f"[val] loaded {len(df)} test-fold decoy rows across {df['pocket_id'].nunique()} pockets", flush=True)
    return df


def _run_rank_val(
    model: torch.nn.Module,
    val_df: pd.DataFrame,
    device: torch.device,
    max_pockets: int = 40,
    decoys_per_pocket: int = 20,
    use_graphs: bool = True,
) -> tuple[float, int]:
    """Quick rank-percentile eval on held-out test pockets.

    Samples one native ligand per pocket and up to *decoys_per_pocket* decoys,
    scores them all, and computes beat-fraction rank_percentile.
    Returns (mean_rank_percentile, n_evaluated_pockets).
    """
    was_training = model.training
    model.eval()
    rp_list: list[float] = []

    pos_df = val_df[val_df["label_value"] == 1].copy()
    neg_df = val_df[val_df["label_value"] == 0].copy()
    pockets = pos_df["pocket_id"].unique()
    if len(pockets) > max_pockets:
        pockets = np.random.choice(pockets, max_pockets, replace=False)

    with torch.no_grad():
        for pid in pockets:
            p_rows = pos_df[pos_df["pocket_id"] == pid]
            n_rows = neg_df[neg_df["pocket_id"] == pid]
            if len(p_rows) == 0 or len(n_rows) == 0:
                continue
            native = p_rows.sample(1)
            decoys = n_rows.sample(min(decoys_per_pocket, len(n_rows)), replace=False)
            batch = pd.concat([native, decoys], ignore_index=True)
            try:
                if use_graphs:
                    _pretrained = getattr(model, "use_pretrained_ligand", False)
                    if _pretrained:
                        # OptiMol: RNA graph + SMILES list
                        rna_bg = _batch_rna_graphs(batch, device=device)
                        smiles_in = batch["ligand_smiles"].fillna("").tolist()
                        out = model(rna_bg, smiles_in)
                    else:
                        rna_bg, lig_bg = _batch_graphs(batch, device=device)
                        out = model(rna_bg, lig_bg)
                else:
                    rna_t, lig_t = _batch_features(
                        batch, 128, 64, 192, "real", device=device
                    )
                    out = model(rna_t, lig_t)
                scores = out["rank_score"].cpu().tolist()
                native_s = scores[0]
                decoy_s = scores[1:]
                beats = sum(1 for s in decoy_s if native_s >= s)
                rp_list.append(beats / max(1, len(decoy_s)))
            except Exception as exc:
                print(f"  [val] pocket {pid} skipped: {exc}", flush=True)

    if was_training:
        model.train()
    if rp_list:
        return float(np.mean(rp_list)), len(rp_list)
    return 0.0, 0


def train_unified(
    project_root: Path,
    config_path: Path,
    model_version: int = 1,
    resume_from: Path | None = None,
    kfold_fold_idx: int | None = None,
) -> None:
    cfg = load_config(config_path)
    # CLI-supplied kfold_fold_idx overrides the config value.
    if kfold_fold_idx is not None:
        cfg.setdefault("data", {})["kfold_fold_idx"] = kfold_fold_idx
        # Auto-update checkpoint_dir so each fold gets its own directory.
        # Replaces a trailing "fold0" (or any digit) with "fold{N}".
        import re as _re
        ckpt_dir_str = cfg.get("output", {}).get("checkpoint_dir", "")
        new_ckpt = _re.sub(r"fold\d+$", f"fold{kfold_fold_idx}", ckpt_dir_str)
        cfg.setdefault("output", {})["checkpoint_dir"] = new_ckpt
    data = load_manifests(project_root, cfg)

    # Load held-out test fold for per-epoch rank validation (read-only, never trained on).
    _val_decoy_df = _load_test_decoy_fold(project_root, cfg)

    torch.manual_seed(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] using {device} ({torch.cuda.device_count()} GPU(s) visible)", flush=True)

    _model_version = int(cfg.get("model", {}).get("version", model_version))
    if _model_version == 3:
        print("[model] using UnifiedInteractionModelV3 (EGNN + GINE + cross-attn)", flush=True)
        model = UnifiedInteractionModelV3(model_cfg=cfg["model"])
        _use_graphs = True
        _rank_fn = _rank_losses_batched_v3
    elif _model_version == 2:
        print("[model] using UnifiedInteractionModelV2", flush=True)
        model = UnifiedInteractionModelV2(model_cfg=cfg["model"])
        _use_graphs = False
        _rank_fn = _rank_losses_batched
    else:
        print("[model] using UnifiedInteractionModel (v1)", flush=True)
        model = UnifiedInteractionModel(model_cfg=cfg["model"])
        _use_graphs = False
        _rank_fn = _rank_losses_batched
    model = model.to(device)
    if torch.cuda.device_count() > 1 and not _use_graphs:
        print(f"[device] wrapping model with DataParallel across {torch.cuda.device_count()} GPUs", flush=True)
        model = torch.nn.DataParallel(model)
    elif _use_graphs:
        print(f"[device] v3 graph model — skipping DataParallel (BatchedGraph not scatter-compatible)", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    # ── LR Scheduler (optional) ──────────────────────────────────────────────
    _sched_cfg = cfg.get("train", {}).get("lr_scheduler", {})
    _sched_type = _sched_cfg.get("type", "none").lower()
    total_steps = int(cfg["train"]["epochs"]) * int(cfg["train"].get("steps_per_epoch", 1))
    warmup_steps = int(_sched_cfg.get("warmup_steps", 0))
    if _sched_type == "cosine":
        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            import math
            min_lr_ratio = float(_sched_cfg.get("min_lr_ratio", 0.01))
            return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)
        print(f"[scheduler] cosine with {warmup_steps} warmup steps, min_lr_ratio={_sched_cfg.get('min_lr_ratio', 0.01)}", flush=True)
    else:
        scheduler = None
        if _sched_type != "none":
            print(f"[scheduler] unknown type '{_sched_type}', no scheduler applied", flush=True)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    # Supports both explicit --resume path and a 'resume_from' key in the config.
    _resume_path = resume_from or cfg.get("train", {}).get("resume_from") or None
    start_epoch = 1
    if _resume_path is not None:
        _resume_path = Path(_resume_path)
        print(f"[resume] loading checkpoint: {_resume_path}", flush=True)
        ckpt = torch.load(_resume_path, map_location=device)
        # Support both bare state-dict and our standard {model_state_dict, epoch} format.
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
        else:
            model.load_state_dict(ckpt)
        if "optimizer_state_dict" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"[resume] resuming from epoch {start_epoch}/{cfg['train']['epochs']}", flush=True)
    else:
        start_epoch = 1

    max_rna_tokens = cfg["data"]["max_rna_tokens"]
    max_lig_tokens = cfg["data"]["max_lig_tokens"]
    embed_dim = cfg["model"]["embed_dim"]

    batch_size = cfg["train"]["batch_size"]
    epochs = cfg["train"]["epochs"]
    negatives_per_positive = cfg["train"].get("negatives_per_positive", 4)
    temperature = cfg["loss"]["contrastive_temperature"]
    featurizer_mode = cfg.get("featurizer", {}).get("mode", "real")

    lambda_rank = cfg["loss"].get("lambda_rank", 1.0)
    lambda_contrast = cfg["loss"].get("lambda_contrast", 1.0)
    lambda_dock = cfg["loss"]["lambda_dock"]
    lambda_site = cfg["loss"]["lambda_site"]
    site_pos_weight = float(cfg["loss"].get("site_pos_weight", 5.0))
    margin = float(cfg["loss"].get("ranking_margin", 0.2))
    # dock_loss_mode: 'bce' (default, soft binary) or 'regression' (Huber on normalized scores)
    dock_loss_mode = cfg["loss"].get("dock_loss_mode", "bce")

    steps_per_epoch = int(cfg["train"].get("steps_per_epoch", 1))
    log_every = int(cfg.get("output", {}).get("log_every", 20))

    # Backward-compatible schedule parsing.
    phase1_epochs = cfg.get("schedule", {}).get("phase1_epochs", cfg.get("schedule", {}).get("stage1_epochs", 1))
    phase2_epochs = cfg.get("schedule", {}).get("phase2_epochs", cfg.get("schedule", {}).get("stage2_epochs", 1))
    phase3_epochs = cfg.get("schedule", {}).get("phase3_epochs", cfg.get("schedule", {}).get("stage3_epochs", 1))

    # Optional decoy curriculum (20 -> 50 -> 100 etc.).
    decoy_curriculum = cfg.get("train", {}).get("decoy_curriculum", {})
    stage1_neg = decoy_curriculum.get("stage1_negatives_per_positive", negatives_per_positive)
    stage2_neg = decoy_curriculum.get("stage2_negatives_per_positive", negatives_per_positive)
    stage3_neg = decoy_curriculum.get("stage3_negatives_per_positive", negatives_per_positive)

    stage1_enable = cfg.get("schedule", {}).get("stage1_enable", {})
    stage2_enable = cfg.get("schedule", {}).get("stage2_enable", {})
    stage3_enable = cfg.get("schedule", {}).get("stage3_enable", {})

    def _is_enabled(stage_enable: dict[str, Any], key: str, default: bool) -> bool:
        if key not in stage_enable:
            return default
        return bool(stage_enable[key])

    # ---------------------------------------------------------------
    # One-time pre-computation (avoids repeated groupby in step loop)
    # ---------------------------------------------------------------
    print("[init] Building negative sample index (one-time groupby on decoy data)...", flush=True)
    pos_df, neg_df, neg_index = _build_neg_index(data.decoy)
    print(f"[init] Neg index built: {len(pos_df)} positives, {len(neg_df)} negatives, "
          f"{len(neg_index)} unique pockets", flush=True)

    # Pre-warm RNA feature/graph cache so first step isn't slow
    unique_pockets = data.decoy[["pocket_id", "target_structure_path"]].drop_duplicates()
    if _use_graphs:
        print("[init] Pre-warming RNA graph cache...", flush=True)
        for _, prow in unique_pockets.iterrows():
            build_rna_graph_cached(
                _clean_str(prow["pocket_id"]),
                _clean_str(prow["target_structure_path"]),
            )
        print(f"[init] RNA graph cache warmed: {len(unique_pockets)} unique pockets", flush=True)
    elif featurizer_mode == "real":
        print("[init] Pre-warming RNA token cache...", flush=True)
        for _, prow in unique_pockets.iterrows():
            build_rna_tokens_from_pocket(
                pocket_id=_clean_str(prow["pocket_id"]),
                pocket_structure_path=_clean_str(prow["target_structure_path"]),
                max_rna_tokens=max_rna_tokens,
                dim=embed_dim,
            )
        print(f"[init] RNA token cache warmed: {len(unique_pockets)} unique pockets", flush=True)

    global_step = (start_epoch - 1) * steps_per_epoch

    for epoch in range(start_epoch, epochs + 1):
        # Phase schedule: progressively enable objectives.
        if epoch <= phase1_epochs:
            w_rank, w_dock, w_site = 1.0, 1.0, 0.0
            epoch_negatives = stage1_neg
            phase = "phase1-rank+dock"
            stage_enable = stage1_enable
        elif epoch <= phase1_epochs + phase2_epochs:
            w_rank, w_dock, w_site = 1.0, 0.5, 1.0
            epoch_negatives = stage2_neg
            phase = "phase2-add-site"
            stage_enable = stage2_enable
        else:
            w_rank, w_dock, w_site = 1.0, 1.0, 1.0
            epoch_negatives = stage3_neg
            phase = "phase3-joint"
            stage_enable = stage3_enable

        need_contrast = _is_enabled(stage_enable, "contrastive", True)
        need_rank = _is_enabled(stage_enable, "rank", True)
        need_docking = _is_enabled(stage_enable, "docking", True)
        need_site = _is_enabled(stage_enable, "site", True)

        # Accumulate losses for epoch-level summary
        epoch_losses: dict[str, list[float]] = {
            "contrast": [], "rank": [], "dock": [], "site": [], "total": []
        }

        for step in range(steps_per_epoch):
            global_step += 1

            # Fast sampling: no groupby, uses pre-computed neg_index
            pos_batch, neg_flat = _sample_rank_batch(
                pos_df, neg_df, neg_index,
                batch_size=batch_size,
                negatives_per_positive=epoch_negatives,
            )

            dock_batch = None
            out_dock = None

            if need_docking or need_site:
                dock_batch = data.docking.sample(n=min(batch_size, len(data.docking)), replace=len(data.docking) < batch_size)
                if _use_graphs:
                    _pretrained = getattr(model, "use_pretrained_ligand", False)
                    if _pretrained:
                        rna_bg     = _batch_rna_graphs(dock_batch, device=device)
                        smiles_in  = dock_batch["ligand_smiles"].fillna("").tolist()
                        out_dock   = model(rna_bg, smiles_in)
                    else:
                        rna_bg, lig_bg = _batch_graphs(dock_batch, device=device)
                        out_dock = model(rna_bg, lig_bg)
                else:
                    rna_dock, lig_dock = _batch_features(dock_batch, max_rna_tokens, max_lig_tokens, embed_dim, featurizer_mode, device=device)
                    out_dock = model(rna_dock, lig_dock)

            # Batched ranking losses: single forward pass for all B*(K+1) samples
            contrast_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
            rank_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
            if need_contrast or need_rank:
                try:
                    _cl, _rl = _rank_fn(
                        model=model,
                        pos_batch=pos_batch,
                        neg_flat=neg_flat,
                        K=epoch_negatives,
                        max_rna_tokens=max_rna_tokens,
                        max_lig_tokens=max_lig_tokens,
                        embed_dim=embed_dim,
                        featurizer_mode=featurizer_mode,
                        temperature=temperature,
                        margin=margin,
                        device=device,
                    )
                    if need_contrast:
                        contrast_loss = _cl
                    if need_rank:
                        rank_loss = _rl
                except torch.OutOfMemoryError:
                    # Keep long-running Slurm jobs alive on occasional large graph batches.
                    print(
                        f"  [step {global_step}] WARNING: CUDA OOM in rank forward, skipping step",
                        flush=True,
                    )
                    opt.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

            # dock_loss_mode:
            #   'bce'        — BCEWithLogits (soft binary, works with 0/1 or normalized labels)
            #   'regression' — Huber loss on normalized docking scores (continuous 0-1)
            #                  uses sigmoid(dock_score) so output is bounded [0,1]
            dock_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
            if need_docking and dock_batch is not None and out_dock is not None:
                target_dock = torch.tensor(dock_batch["label_value"].astype(float).values, dtype=torch.float32, device=device)
                if dock_loss_mode == "regression":
                    pred_dock = torch.sigmoid(out_dock["dock_score"])
                    dock_loss = F.huber_loss(pred_dock, target_dock, delta=0.2)
                else:
                    dock_loss = F.binary_cross_entropy_with_logits(out_dock["dock_score"], target_dock)

            site_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
            if need_site and dock_batch is not None and out_dock is not None:
                pocket_ids = [str(x) for x in dock_batch["pocket_id"].values]
                site_len = int(out_dock["site_logits"].shape[1])
                site_target = _sample_site_targets(data.site, pocket_ids, max_rna_tokens=site_len, device=device)
                _site_pw = torch.tensor([site_pos_weight], dtype=torch.float32, device=device)
                site_loss = F.binary_cross_entropy_with_logits(
                    out_dock["site_logits"], site_target, pos_weight=_site_pw
                )

            total = torch.tensor(0.0, dtype=torch.float32, device=device)
            if need_contrast:
                total = total + lambda_contrast * contrast_loss
            if need_rank:
                total = total + w_rank * lambda_rank * rank_loss
            if need_docking and dock_loss.isfinite():
                total = total + w_dock * lambda_dock * dock_loss
            if need_site and site_loss.isfinite():
                total = total + w_site * lambda_site * site_loss

            opt.zero_grad()
            if not total.isfinite():
                # Loss is NaN/Inf — skip this step without updating weights.
                # This can happen in the first few phase-2 steps while the site
                # head is being pulled from its random initialisation.
                print(
                    f"  [step {global_step}] WARNING: non-finite loss ({total.item()}) — skipping update",
                    flush=True,
                )
            else:
                total.backward()
                # Sanitise any NaN/Inf gradients (secondary guard)
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                opt.step()
                if scheduler is not None:
                    scheduler.step()

            epoch_losses["contrast"].append(contrast_loss.item())
            epoch_losses["rank"].append(rank_loss.item())
            epoch_losses["dock"].append(dock_loss.item())
            epoch_losses["site"].append(site_loss.item())
            epoch_losses["total"].append(total.item())

            if global_step % log_every == 0:
                print(
                    f"  [step {global_step}] "
                    f"contrast={contrast_loss.item():.4f} rank={rank_loss.item():.4f} "
                    f"dock={dock_loss.item():.4f} site={site_loss.item():.4f} total={total.item():.4f}",
                    flush=True,
                )

        def _mean(lst: list[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        print(
            f"[epoch {epoch}/{epochs}] {phase} "
            f"contrast={_mean(epoch_losses['contrast']):.4f} rank={_mean(epoch_losses['rank']):.4f} "
            f"dock={_mean(epoch_losses['dock']):.4f} site={_mean(epoch_losses['site']):.4f} "
            f"neg={epoch_negatives} steps={steps_per_epoch} total={_mean(epoch_losses['total']):.4f}",
            flush=True,
        )

        # ── Per-epoch validation ─────────────────────────────────────
        # Reads `val_max_pockets` / `val_decoys_per_pocket` from config output
        # section; falls back to safe defaults.
        _val_cfg = cfg.get("output", {})
        if _val_decoy_df is not None and len(_val_decoy_df) > 0:
            val_rp, val_n = _run_rank_val(
                model=model,
                val_df=_val_decoy_df,
                device=device,
                max_pockets=int(_val_cfg.get("val_max_pockets", 40)),
                decoys_per_pocket=int(_val_cfg.get("val_decoys_per_pocket", 20)),
                use_graphs=_use_graphs,
            )
            print(
                f"[epoch {epoch}/{epochs}] val rank_percentile={val_rp:.4f} (n_pockets={val_n})",
                flush=True,
            )

        # ── Checkpoint saving (every epoch by default) ───────────────
        # Supports both `save_every_epoch: true` and `save_every_epochs: N`.
        _save_cfg = cfg.get("output", {}).get("save_every_epoch",
                    cfg.get("output", {}).get("save_every_epochs", 1))
        if _save_cfg is True:
            save_every_n = 1
        else:
            save_every_n = int(_save_cfg) if _save_cfg else 0
        if save_every_n > 0 and epoch % save_every_n == 0:
            ckpt_dir = project_root / cfg["output"]["checkpoint_dir"]
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ep_path = ckpt_dir / f"unified_multitask_ep{epoch:03d}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "config": cfg,
                "epoch": epoch,
            }, ep_path)
            print(f"[OK] Saved checkpoint: {ep_path}", flush=True)

    ckpt_dir = project_root / cfg["output"]["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "config": cfg,
        "epoch": epochs,
    }, ckpt_dir / "unified_multitask_last.pt")
    print(f"[OK] Saved checkpoint: {ckpt_dir / 'unified_multitask_last.pt'}", flush=True)
