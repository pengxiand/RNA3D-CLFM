#!/usr/bin/env python3
"""
eval_hariboss5fd.py — Standalone evaluation script.

Loads best.pt for each fold of a trained run and computes rank percentile
using the full cross-pocket decoy pool (no sub-sampling), matching SMARTBind's
validation protocol exactly.

Usage:
    python scripts/eval_hariboss5fd.py --config configs/hariboss5fd_rank_pkl_v11.yaml
    python scripts/eval_hariboss5fd.py --config configs/hariboss5fd_rank_pkl_v9.yaml
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# ── project path setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bridgebind3d.featurizers import build_rna_graph_cached, build_lig_graph_cached
from bridgebind3d.graph_data import batch_rna_graphs, batch_ligand_graphs
from bridgebind3d.unified_model_v3 import UnifiedInteractionModelV3
from bridgebind3d.unified_model_v4 import UnifiedInteractionModelV4
from bridgebind3d.unified_model_v10 import UnifiedInteractionModelV10

# Re-use helpers from the training script
_TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_hariboss5fd_rank_pkl.py"
sys.path.insert(0, str(_TRAIN_SCRIPT.parent))
from train_hariboss5fd_rank_pkl import (
    load_pkl_records,
    _to_device,
)

# ── defaults (same as train script) ──────────────────────────────────────────
_WORKSPACE = PROJECT_ROOT.parents[1]
_DEFAULT_PKL = _WORKSPACE / "new" / "hariboss_5fd_cov08_clustered_seqid_0.3_clean.pkl"
_DEFAULT_DECOY_LOOKUP = PROJECT_ROOT / "data" / "decoy_fp2_lookup.pkl"
_DEFAULT_RNA3D_DIRS = [
    str(_WORKSPACE / "rnamigos2" / "data" / "json_pockets_3d"),
    str(PROJECT_ROOT / "data" / "json_pockets_cif"),
]


# ═════════════════════════════════════════════════════════════════════════════
# Per-pocket PKL-decoy eval
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_rank_percentile_full(
    model,
    test_recs: list[dict],
    device: torch.device,
    use_pretrained: bool,
) -> float:
    """Mean per-pocket rank percentile using each pocket's own decoy_smiles.

    For every test pocket (not seen during training):
      - native  = rec["native_smiles"]
      - decoys  = rec["decoy_smiles"]  (~200 FP2-matched hard negatives from PKL)
      - RP      = mean(decoy_scores < native_score)
    All test pockets with decoys are evaluated.
    """
    model.eval()
    rp_list: list[float] = []

    for rec in test_recs:
        native = rec["native_smiles"]
        decoys = rec.get("decoy_smiles") or []
        if not native or not decoys:
            continue

        all_smiles = [native] + decoys
        rg = build_rna_graph_cached(rec["pkl_key"], rec["json_path"])
        rna_graphs = [rg] * len(all_smiles)
        rna_bg = _to_device(batch_rna_graphs(rna_graphs), device)

        try:
            if isinstance(model, (UnifiedInteractionModelV4, UnifiedInteractionModelV10)):
                rna_seqs = [rg.sequence for rg in rna_graphs]
                out = model(rna_bg, all_smiles, rna_sequences=rna_seqs)
            elif use_pretrained:
                out = model(rna_bg, all_smiles)
            else:
                lig_gs = [build_lig_graph_cached(s) for s in all_smiles]
                lig_bg = _to_device(batch_ligand_graphs(lig_gs), device)
                out = model(rna_bg, lig_bg)
        except Exception as e:
            print(f"  [skip] {rec['pkl_key']}: {e}", flush=True)
            continue

        scores = out["rank_score"].cpu().numpy()
        native_score = float(scores[0])
        decoy_scores = scores[1:]
        rp_list.append(float(np.mean(decoy_scores < native_score)))

    if not rp_list:
        return 0.5
    return float(np.mean(rp_list))


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--folds",  default="0,1,2,3,4",
                        help="Comma-separated fold indices to evaluate (default: 0,1,2,3,4)")
    parser.add_argument("--ckpt",   default="best.pt",
                        help="Checkpoint filename to load (default: best.pt)")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device}  config={args.config}", flush=True)

    # Load data
    data_cfg  = cfg["data"]
    pkl_path  = Path(data_cfg.get("pkl_path", str(_DEFAULT_PKL)))
    rna3d_dirs = data_cfg.get("rna3d_dirs", _DEFAULT_RNA3D_DIRS)
    decoy_lookup_path = Path(data_cfg.get("decoy_lookup_path", str(_DEFAULT_DECOY_LOOKUP)))

    print(f"[eval] loading PKL: {pkl_path}", flush=True)
    records = load_pkl_records(pkl_path, rna3d_dirs, decoy_lookup_path)
    print(f"[eval] {len(records)} records loaded", flush=True)

    model_cfg     = cfg["model"]
    model_version = int(model_cfg.get("version", 3))
    use_pretrained = bool(model_cfg.get("use_pretrained_ligand", True))

    fold_indices = [int(f) for f in args.folds.split(",")]
    ckpt_base    = cfg["output"]["checkpoint_dir"]   # ends in foldX

    fold_rps: list[float] = []

    for fold_idx in fold_indices:
        fold_dir = Path(ckpt_base.replace("fold0", f"fold{fold_idx}"))
        ckpt_path = fold_dir / args.ckpt

        if not ckpt_path.exists():
            print(f"[fold {fold_idx}] ✗ checkpoint not found: {ckpt_path}", flush=True)
            continue

        # Test split
        test_recs = [
            r for r in records
            if len(r["train_split"]) > fold_idx and not bool(r["train_split"][fold_idx])
        ]
        print(f"\n[fold {fold_idx}] test={len(test_recs)} pockets  ckpt={ckpt_path}", flush=True)

        # Build model
        if model_version == 4:
            model = UnifiedInteractionModelV4(model_cfg=model_cfg).to(device)
            use_pretrained = False
        elif model_version == 10:
            model = UnifiedInteractionModelV10(model_cfg=model_cfg).to(device)
            use_pretrained = False
        else:
            model = UnifiedInteractionModelV3(model_cfg=model_cfg).to(device)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        stored_rp = float(ckpt.get("best_rp", 0.0))
        print(f"[fold {fold_idx}] stored best_rp (100-decoy) = {stored_rp:.4f}", flush=True)

        rp = eval_rank_percentile_full(
            model, test_recs, device, use_pretrained,
        )
        n_decoys_avg = int(np.mean([len(r.get("decoy_smiles") or []) for r in test_recs if r.get("decoy_smiles")]))
        print(f"[fold {fold_idx}] PKL-decoy RP = {rp:.4f}  (n_test={len(test_recs)} pockets, avg decoys={n_decoys_avg})", flush=True)
        fold_rps.append(rp)

    if fold_rps:
        mean_rp = float(np.mean(fold_rps))
        std_rp  = float(np.std(fold_rps))
        print("\n" + "=" * 55, flush=True)
        print(f"  Folds evaluated : {fold_indices[:len(fold_rps)]}", flush=True)
        print(f"  Per-fold RP     : {[f'{r:.4f}' for r in fold_rps]}", flush=True)
        print(f"  Mean RP         : {mean_rp:.4f}  ±{std_rp:.4f}", flush=True)
        print(f"  Gap vs SMARTBind: {0.779 - mean_rp:.4f}", flush=True)
        print("=" * 55, flush=True)


if __name__ == "__main__":
    main()
