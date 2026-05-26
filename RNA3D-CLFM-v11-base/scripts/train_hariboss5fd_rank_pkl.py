#!/usr/bin/env python3
"""Train BridgeBind3D (EGNN + pretrained OptiMol SMILES encoder) on
hariboss_5fd_cov08_clustered_seqid_0.3_clean.pkl — ranking loss only.

Architecture
───────────
  RNA tower   : EGNN on 3-D pocket graph  (atom coords from json)
  Ligand tower: OptiMol pretrained RGCN   (56-dim μ → learned projection)
                  freeze_ligand=True by default; set False to fine-tune

Loss
────
  Per-step: one mini-batch of B positives, each paired with K decoys.
    InfoNCE contrastive  (pos vs K negs, in-batch)
  + Margin ranking       (pos vs each neg)
  No docking/affinity/site-prediction losses — rank metric only.

Validation
──────────
  Per epoch: rank percentile on the test fold using the pkl's decoy_smiles.
  At end: option to run external decoy eval via eval_unified_v3_external_decoy.py.

Usage
─────
  # Single fold
  python scripts/train_hariboss5fd_rank_pkl.py --fold 0

  # Resume
  python scripts/train_hariboss5fd_rank_pkl.py --fold 0 --resume outputs/egnn_optimol_fold0/last.pt

  # Override config
  python scripts/train_hariboss5fd_rank_pkl.py --config configs/hariboss5fd_rank_pkl.yaml --fold 2
"""
from __future__ import annotations

import argparse
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import math

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# ── project path setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bridgebind3d.featurizers import build_rna_graph_cached, build_lig_graph_cached
from bridgebind3d.graph_data import BatchedGraph, batch_rna_graphs, batch_ligand_graphs
from bridgebind3d.unified_model_v3 import UnifiedInteractionModelV3
from bridgebind3d.unified_model_v4 import UnifiedInteractionModelV4

# ── FP2 fingerprint utilities (lazy import rdkit so GIN runs unaffected) ─────
_fp2_cache: dict[str, np.ndarray] = {}          # smiles → float32 [fp_dim]

def smiles_to_fp2(smiles: str, fp_dim: int = 2048) -> np.ndarray | None:
    """Compute Morgan FP2 (radius=2, nBits=fp_dim) as a float32 bit-vector.

    Results are cached in _fp2_cache to avoid repeated RDKit calls.
    Returns None if the SMILES cannot be parsed.
    """
    global _fp2_cache
    if smiles in _fp2_cache:
        return _fp2_cache[smiles]
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=fp_dim)
        arr = np.zeros(fp_dim, dtype=np.float32)
        from rdkit.DataStructs import ConvertToNumpyArray
        ConvertToNumpyArray(fp, arr)
        _fp2_cache[smiles] = arr
        return arr
    except Exception:
        return None


def smiles_list_to_fp2_tensor(
    smiles_list: list[str],
    device: torch.device,
    fp_dim: int = 2048,
) -> torch.Tensor:
    """Stack FP2 fingerprints for a list of SMILES into a [N, fp_dim] tensor.

    Any SMILES that fails are replaced with a zero vector.
    """
    rows = []
    for s in smiles_list:
        fp = smiles_to_fp2(s, fp_dim=fp_dim)
        rows.append(fp if fp is not None else np.zeros(fp_dim, dtype=np.float32))
    return torch.tensor(np.stack(rows), dtype=torch.float32, device=device)
from bridgebind3d.unified_model_v10 import UnifiedInteractionModelV10

# ── workspace defaults ────────────────────────────────────────────────────────
_WORKSPACE = PROJECT_ROOT.parents[1]   # …/test_RNA/new/..  → …/test_RNA

# Hariboss PKL: proper seqid-0.3 clustering, correct 5-fold splits.
# Its decoy_smiles are FP2 binary vectors (computed by SMARTBind from
# decoy_library.smi).  We recover SMILES via a precomputed reverse-lookup
# pickle built by scripts/build_decoy_lookup.py.
_DEFAULT_PKL = (
    _WORKSPACE / "new" / "hariboss_5fd_cov08_clustered_seqid_0.3_clean.pkl"
)
_DEFAULT_DECOY_LOOKUP = (
    PROJECT_ROOT / "data" / "decoy_fp2_lookup.pkl"
)
_DEFAULT_RNA3D_DIRS = [
    # Only directories that contain 3-D coordinates (x/y/z per node).
    # json_pockets_expanded / json_pockets_annotated are 2-D topology-only
    # and have no coordinates — useless for EGNN.
    str(_WORKSPACE / "rnamigos2" / "data" / "json_pockets_3d"),
    str(PROJECT_ROOT / "data" / "json_pockets_cif"),   # built by build_pockets_from_cif.py
]


# ═════════════════════════════════════════════════════════════════════════════
# PKL loading + JSON resolution  (mirrors build_manifests_from_custom_pkl.py)
# ═════════════════════════════════════════════════════════════════════════════

def _candidate_names(pkl_key: str, rec: dict) -> list[str]:
    """JSON filename candidates for a SMARTBind pkl pocket key."""
    parts = str(pkl_key).split("_")
    names: list[str] = []
    if len(parts) >= 5:
        pdb        = parts[0].upper()
        rna_chain  = parts[1]
        lig_chain  = parts[2]
        ligand_id  = parts[3]
        residue    = "_".join(parts[4:]).replace("*", "")
        names.append(f"{pdb}_{rna_chain}_{ligand_id}_{residue}.json")
        names.append(f"{pdb}_{rna_chain}{lig_chain}_{ligand_id}_{residue}.json")
        names.append(f"{pdb}_{rna_chain.lower()}_{ligand_id}_{residue}.json")
        names.append(f"{pdb.lower()}_{rna_chain}_{ligand_id}_{residue}.json")
    # RNAmigos1 pickle names look like: 5dm7_#0:X:HGR:6178_BIND.nx_annot.p
    if len(parts) >= 2 and ":" in parts[1]:
        pdb = parts[0].upper()
        colon_parts = parts[1].split(":")
        if len(colon_parts) >= 4:
            rna_chain = colon_parts[1]
            ligand_id = colon_parts[2]
            residue = colon_parts[3].split("_")[0].replace("*", "")
            names.append(f"{pdb}_{rna_chain}_{ligand_id}_{residue}.json")
            names.append(f"{pdb.lower()}_{rna_chain}_{ligand_id}_{residue}.json")
    # Fallback from record metadata
    pdb       = str(rec.get("pdb_id") or rec.get("pdbid") or "").upper().strip()
    rna_chain = str(rec.get("rna_chain_id") or rec.get("chain") or "").strip()
    ligand_id = str(rec.get("ligand_id", "")).strip()
    resnum    = str(rec.get("ligand_resnum", "")).strip().replace("*", "")
    if pdb and rna_chain and ligand_id and resnum:
        names.append(f"{pdb}_{rna_chain}_{ligand_id}_{resnum}.json")
    return list(dict.fromkeys(n for n in names if n))


def _build_prefix_index(roots: list[Path]) -> dict[tuple[str, str], list[Path]]:
    idx: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for root in roots:
        if not root.exists():
            continue
        for p in root.glob("*.json"):
            parts = p.stem.split("_")
            if len(parts) >= 2:
                key = (parts[0].upper(), parts[1])
                idx[key].append(p)
    for k in idx:
        idx[k].sort(key=lambda x: (len(x.name), x.name))
    return idx


def _resolve_json(
    pkl_key: str, rec: dict, roots: list[Path], prefix_idx: dict
) -> Path | None:
    # 1) Exact match
    for root in roots:
        for name in _candidate_names(pkl_key, rec):
            p = root / name
            if p.exists():
                return p
    # 2) Prefix fallback
    parts = str(pkl_key).split("_")
    if len(parts) >= 2:
        key = (parts[0].upper(), parts[1])
        cands = prefix_idx.get(key, [])
        if cands:
            return cands[0]
    return None


def _load_fp2_lookup(lookup_path: Path) -> dict[tuple, str]:
    """Load precomputed FP2→SMILES lookup (built by build_decoy_lookup.py).

    Returns empty dict (silently) if the file does not exist yet — callers
    must handle the n_no_decoys count in that case.
    """
    if not lookup_path.exists():
        print(f"[pkl] WARNING: FP2 lookup not found at {lookup_path}. "
              "Run scripts/build_decoy_lookup.py first.", flush=True)
        return {}
    with lookup_path.open("rb") as f:
        return pickle.load(f)


def _fp2_to_smiles(fp2_vec, lookup: dict[tuple, str]) -> str | None:
    """Convert a 1024-dim FP2 list/array to SMILES via lookup table."""
    try:
        key = tuple(int(b) for b in fp2_vec)
        return lookup.get(key)
    except Exception:
        return None


def load_pkl_records(
    pkl_path: Path,
    rna3d_dirs: list[str],
    decoy_lookup_path: Path | None = None,
) -> list[dict]:
    """Load hariboss pkl and resolve 3-D JSON paths.

    The hariboss PKL stores decoy_smiles as 1024-dim FP2 binary vectors.
    Pass decoy_lookup_path (built by build_decoy_lookup.py) to convert
    them back to SMILES for OptiMol.

    Returns list of dicts with keys:
      pkl_key, pdb_id, json_path, native_smiles, decoy_smiles,
      train_split (list[bool]), contact_map_1d, ligand_id
    """
    roots = [Path(d) for d in rna3d_dirs]
    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    # Load FP2→SMILES reverse lookup (may be empty if not yet built)
    if decoy_lookup_path is None:
        decoy_lookup_path = _DEFAULT_DECOY_LOOKUP
    fp2_lookup = _load_fp2_lookup(decoy_lookup_path)

    prefix_idx = _build_prefix_index(roots)
    records: list[dict] = []
    n_missing_json  = 0
    n_no_native     = 0
    n_no_decoys     = 0

    def _is_record_dict(obj: object) -> bool:
        return isinstance(obj, dict) and any(
            k in obj for k in (
                "downloaded_ligand_smiles", "native_smiles", "train_split",
                "split_0_train", "decoy_smiles", "pickle_name",
            )
        )

    def _iter_raw_records(obj: object):
        if isinstance(obj, dict):
            for outer_key, outer_val in obj.items():
                if _is_record_dict(outer_val):
                    yield str(outer_val.get("pdb_id") or outer_val.get("pdbid") or outer_key), str(outer_key), outer_val
                elif isinstance(outer_val, dict):
                    for inner_key, rec in outer_val.items():
                        if isinstance(rec, dict):
                            yield str(outer_key), str(inner_key), rec
        elif isinstance(obj, list):
            for i, rec in enumerate(obj):
                if isinstance(rec, dict):
                    key = str(rec.get("pkl_key") or rec.get("pickle_name") or i)
                    pdb = str(rec.get("pdb_id") or rec.get("pdbid") or key.split("_")[0])
                    yield pdb, key, rec

    def _as_bool(value: object) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "t", "yes", "y"}
        return bool(value)

    for pdb_id, pkl_key, rec in _iter_raw_records(data):
        record_key = str(pkl_key)
        if rec.get("pickle_name"):
            record_key = str(rec["pickle_name"]).strip()
        native = str(
            rec.get("downloaded_ligand_smiles")
            or rec.get("native_smiles")
            or ""
        ).strip()
        if not native:
            n_no_native += 1
            continue

        raw_decoys = rec.get("decoy_smiles") or rec.get("decoys") or rec.get("decoy_smis") or []
        # Try SMILES strings first (future-proof); fall back to FP2 lookup.
        decoys: list[str] = []
        for d in raw_decoys:
            if isinstance(d, str) and d.strip():
                decoys.append(d.strip())
            elif fp2_lookup:
                smi = _fp2_to_smiles(d, fp2_lookup)
                if smi:
                    decoys.append(smi)

        if not decoys:
            n_no_decoys += 1
            continue

        json_p = _resolve_json(record_key, rec, roots, prefix_idx)
        if json_p is None:
            n_missing_json += 1
            continue
        train_split = list(rec.get("train_split") or [])
        if not train_split:
            train_split = [_as_bool(rec.get(f"split_{i}_train", False)) for i in range(10)]

        records.append({
            "pkl_key":        record_key,
            "pdb_id":         str(pdb_id),
            "json_path":      str(json_p),
            "native_smiles":  native,
            "decoy_smiles":   decoys,
            "train_split":    train_split,
            "contact_map_1d": list(rec.get("contact_map_1d") or []),
            "ligand_id":      str(rec.get("ligand_id", "")),
        })

    print(
        f"[pkl] loaded {len(records)} records "
        f"(skipped: {n_missing_json} missing-json, "
        f"{n_no_native} no-native, {n_no_decoys} no-decoys)",
        flush=True,
    )
    return records


# ═════════════════════════════════════════════════════════════════════════════
# Utility
# ═════════════════════════════════════════════════════════════════════════════

def _to_device(bg: BatchedGraph, device: torch.device) -> BatchedGraph:
    return BatchedGraph(
        node_feat=bg.node_feat.to(device),
        edge_index=bg.edge_index.to(device),
        edge_feat=bg.edge_feat.to(device),
        batch_index=bg.batch_index.to(device),
        pos=bg.pos.to(device) if bg.pos is not None else None,
        site_label=bg.site_label.to(device) if bg.site_label is not None else None,
    )


# ═════════════════════════════════════════════════════════════════════════════
# RNA–ligand map (false-negative filtering, mirrors SMARTBind's rna_smol_map)
# ═════════════════════════════════════════════════════════════════════════════

def _rna_key(rec: dict) -> str:
    """RNA system identifier derived from pkl_key (PDB_chain_…)."""
    parts = rec["pkl_key"].split("_")
    if len(parts) >= 2:
        return f"{parts[0].upper()}_{parts[1]}"
    return str(rec.get("pdb_id", "UNK"))


def build_rna_ligand_map(records: list[dict]) -> dict[str, set[str]]:
    """Map rna_key → set of known ligand_ids.

    Mirrors SMARTBind's ligands_for_rna_chain / rna_smol_map: when building
    the cross-pocket decoy pool, we filter out any molecule that is a known
    binder for the same RNA system (false-negative suppression).
    """
    rna_lig_map: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        rna_lig_map[_rna_key(rec)].add(str(rec.get("ligand_id", "")))
    return dict(rna_lig_map)


def build_tanimoto_hard_negs(
    train_recs: list[dict],
    rna_lig_map: dict[str, set[str]],
    n_top: int = 40,
    fp_dim: int = 2048,
) -> dict[str, list[str]]:
    """Pre-compute static Tanimoto hard negatives once before training.

    For each training pocket, find the n_top most Tanimoto-similar native
    ligands from *other* pockets (false-negative filtering applied via
    rna_lig_map).  Results are sorted hardest-first (highest similarity).

    Returns:
        pkl_key → list[smiles]  (up to n_top entries, sorted desc by sim)
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    from rdkit import DataStructs

    # Build per-key fingerprint lookup
    fps: dict[str, object] = {}
    for rec in train_recs:
        mol = Chem.MolFromSmiles(rec.get("native_smiles", ""))
        if mol is not None:
            fps[rec["pkl_key"]] = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=fp_dim
            )

    hard_negs: dict[str, list[str]] = {}
    for rec in train_recs:
        key = rec["pkl_key"]
        if key not in fps:
            continue
        my_fp = fps[key]
        rna_key    = _rna_key(rec)
        known_ligs = rna_lig_map.get(rna_key, set())

        sims: list[tuple[float, str]] = []
        for other in train_recs:
            ok = other["pkl_key"]
            if ok == key or ok not in fps:
                continue
            if str(other.get("ligand_id", "")) in known_ligs:
                continue
            sims.append((
                DataStructs.TanimotoSimilarity(my_fp, fps[ok]),
                other["native_smiles"],
            ))
        sims.sort(key=lambda x: x[0], reverse=True)
        hard_negs[key] = [s for _, s in sims[:n_top]]

    n_covered = len(hard_negs)
    avg_pool  = np.mean([len(v) for v in hard_negs.values()]) if hard_negs else 0.0
    print(
        f"[tanimoto_hn] built for {n_covered}/{len(train_recs)} pockets, "
        f"avg_pool={avg_pool:.1f}",
        flush=True,
    )
    return hard_negs


# ═════════════════════════════════════════════════════════════════════════════
# Ranking loss helpers
# ═════════════════════════════════════════════════════════════════════════════

def _circle_loss_pocket(
    dp: "torch.Tensor",
    margin: float = 0.25,
    gamma: float = 80.0,
) -> "torch.Tensor":
    """Circle Loss (Sun et al. 2020) for 1 positive + K negatives.

    dp : [1+K] rank_score tensor, dp[0] = positive score, dp[1:] = negatives.

    Each pair gets its own adaptive scale (alpha), making the loss more robust
    to hard negatives than vanilla InfoNCE:
      - easy positives (sp already > Delta_p) contribute little
      - easy negatives (sn already < Delta_n) contribute little
      - hard cases are automatically up-weighted

    gamma controls overall sharpness (paper default 80 for face recognition;
    use ~32-64 for molecular scores that live in [-1, 1]).
    """
    sp = dp[0]
    sn = dp[1:]
    O_p, O_n       = 1.0 + margin, -margin
    Delta_p, Delta_n = 1.0 - margin, margin

    alpha_p = F.relu(O_p - sp.detach())           # scalar
    alpha_n = F.relu(sn.detach() - O_n)           # [K]

    logit_n = gamma * alpha_n * (sn - Delta_n)    # [K]
    logit_p = -gamma * alpha_p * (sp - Delta_p)   # scalar

    return F.softplus(torch.logsumexp(logit_n, dim=0) + logit_p)


def _rank_loss_step(
    model: UnifiedInteractionModelV3,
    batch_recs: list[dict],
    device: torch.device,
    margin: float,
    temperature: float,
    hard_margin_topk: int,
    use_pretrained: bool,
    n_cross: int,
    n_hard: int,
    all_train_recs: list[dict],
    rna_lig_map: dict[str, set],
    hard_neg_cache: dict[str, list[str]] | None = None,
    lambda_site: float = 0.0,
    lambda_dock: float = 0.0,
    lambda_rna_contrast: float = 0.0,
    use_fp2_ligand: bool = False,
    use_circle_loss: bool = False,
    adaptive_margin: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One optimisation step — SMARTBind-style mixed negatives.

    For each pocket in the mini-batch:
      1) n_cross cross-pocket negatives: native SMILES from OTHER training
         pockets, filtered to exclude known binders for the same RNA system
         (mirrors SMARTBind's random_decoy_num cross-pocket FP2 sampling).
      2) n_hard hard negatives: per-pocket FP2 decoys from hariboss PKL
         (from decoy_library.smi, mirrors SMARTBind's extra_decoy_num).

    Loss: listwise InfoNCE + optional top-k hard margin ranking.
    """
    if not batch_recs:
        z = torch.tensor(0.0, device=device, requires_grad=True)
        return z, z, z

    _is_v4 = isinstance(model, (UnifiedInteractionModelV4, UnifiedInteractionModelV10))

    rna_graphs: list = []
    all_smiles: list[str] = []
    pocket_sizes: list[int] = []

    for rec in batch_recs:
        rg = build_rna_graph_cached(rec["pkl_key"], rec["json_path"])
        negs: list[str] = []

        # ── 1) Cross-pocket negatives: native SMILES of other train pockets ──
        if n_cross > 0:
            rna_key    = _rna_key(rec)
            known_ligs = rna_lig_map.get(rna_key, set())
            pool = [
                r for r in all_train_recs
                if r["pkl_key"] != rec["pkl_key"]
                and str(r.get("ligand_id", "")) not in known_ligs
                and r["native_smiles"]
            ]
            if pool:
                sampled = random.sample(pool, min(n_cross, len(pool)))
                negs.extend(r["native_smiles"] for r in sampled)

        # ── 2) Hard negatives: from cache (mined) or fallback to random decoys ─
        if n_hard > 0:
            cached_pool = hard_neg_cache.get(rec["pkl_key"], []) if hard_neg_cache else []
            if cached_pool:
                # Sample from precomputed hard negatives (already sorted hardest-first)
                n_sample = min(n_hard, len(cached_pool))
                negs.extend(random.sample(cached_pool[:max(n_sample * 2, len(cached_pool))], n_sample))
            elif rec["decoy_smiles"]:
                # Fallback: random decoys from hariboss PKL
                n_sample = min(n_hard, len(rec["decoy_smiles"]))
                hard_idx = np.random.choice(len(rec["decoy_smiles"]), n_sample, replace=False)
                negs.extend(rec["decoy_smiles"][i] for i in hard_idx)

        # ── Fallback if both pools empty ──────────────────────────────────────
        if not negs and rec["decoy_smiles"]:
            n_fb   = min(n_cross + n_hard, len(rec["decoy_smiles"]))
            fb_idx = np.random.choice(len(rec["decoy_smiles"]), n_fb, replace=False)
            negs   = [rec["decoy_smiles"][i] for i in fb_idx]

        if not negs:
            continue

        pocket_sizes.append(1 + len(negs))
        rna_graphs.extend([rg] * (1 + len(negs)))
        all_smiles.append(rec["native_smiles"])
        all_smiles.extend(negs)

    if not pocket_sizes:
        z = torch.tensor(0.0, device=device, requires_grad=True)
        return z, z, z

    rna_bg = _to_device(batch_rna_graphs(rna_graphs), device)

    if _is_v4:
        # v4: GIN ligand (pass SMILES list) + RNA-FM sequences
        rna_seqs = [rg.sequence for rg in rna_graphs]
        out = model(rna_bg, all_smiles, rna_sequences=rna_seqs)
    elif use_fp2_ligand:
        lig_bg = smiles_list_to_fp2_tensor(all_smiles, device)
        out = model(rna_bg, lig_bg)
    elif use_pretrained:
        out = model(rna_bg, all_smiles)
    else:
        lig_graphs = [build_lig_graph_cached(s) for s in all_smiles]
        lig_bg     = _to_device(batch_ligand_graphs(lig_graphs), device)
        out = model(rna_bg, lig_bg)

    scores      = out["rank_score"]
    dock_scores = out["dock_score"]

    # ── Primary loss (Circle or InfoNCE) + margin ranking on rank_score ──────
    # For each pocket: dp = [native_score, neg1_score, ..., negK_score]
    # Circle: adaptive per-pair scale — better for hard negatives (Sun 2020)
    # InfoNCE: cross_entropy(dp / T, 0)  ← native is class-0
    # Margin:  relu(m_i - (pos - neg_i)).mean(), m_i adaptive if adaptive_margin=True
    z_lig_all = out["z_lig"]   # [total, D] — needed for rna_cts sim-matrix
    z_rna_all = out["z_rna"]   # [total, D]

    infonce_list: list[torch.Tensor] = []
    margin_list:  list[torch.Tensor] = []
    native_indices: list[int] = []
    offset = 0

    for n_total in pocket_sizes:
        native_indices.append(offset)
        dp = scores[offset : offset + n_total]   # [1+K] rank_score: native first

        # ── Primary contrastive loss ──────────────────────────────────────
        if use_circle_loss:
            infonce_list.append(_circle_loss_pocket(dp, margin=margin))
        else:
            target = torch.zeros(1, dtype=torch.long, device=device)
            infonce_list.append(F.cross_entropy((dp / temperature).unsqueeze(0), target))

        # ── Margin ranking loss ───────────────────────────────────────────
        neg_scores = dp[1:]
        if neg_scores.numel() > 0:
            topk_idx: "torch.Tensor | None" = None
            if hard_margin_topk > 0 and neg_scores.numel() > hard_margin_topk:
                # keep only the hardest k negatives for margin loss
                topk_idx   = torch.topk(neg_scores, k=hard_margin_topk).indices
                neg_scores = neg_scores[topk_idx]

            if adaptive_margin:
                # Adaptive margin: harder negatives (similar to positive in
                # ligand embedding space) get a smaller margin, so the model
                # is not over-penalised for not separating truly similar pairs.
                z_pos      = z_lig_all[offset]                               # [D]
                z_negs_all = z_lig_all[offset + 1 : offset + n_total]        # [K, D]
                z_negs     = z_negs_all[topk_idx] if topk_idx is not None else z_negs_all
                cos_sim    = F.cosine_similarity(z_pos.unsqueeze(0), z_negs, dim=-1)  # [k]
                margin_vec = (margin * (1.0 - cos_sim)).clamp(min=0.05)              # [k]
                pair_loss  = F.relu(margin_vec - (dp[0] - neg_scores)).mean()
            else:
                pair_loss = F.margin_ranking_loss(
                    dp[0].expand_as(neg_scores), neg_scores,
                    torch.ones_like(neg_scores), margin=margin,
                )
            margin_list.append(pair_loss)

        offset += n_total

    smol_cts = torch.stack(infonce_list).mean() if infonce_list else torch.tensor(0.0, device=device, requires_grad=True)
    margin_l  = torch.stack(margin_list).mean()  if margin_list  else torch.tensor(0.0, device=device, requires_grad=True)

    # ── Auxiliary loss: site BCE (native ligand only) + dock ranking ─
    aux_loss = torch.tensor(0.0, device=device)

    if lambda_site > 0.0 and "site_logits" in out and "site_label_pad" in out:
        site_logits    = out["site_logits"]     # [total_B, Lr]
        site_label_pad = out["site_label_pad"]  # [total_B, Lr]
        rna_mask_pad   = out.get("rna_mask")    # [total_B, Lr] bool, True = valid node

        # Gather only native-ligand rows
        native_t   = torch.tensor(native_indices, dtype=torch.long, device=device)
        nat_logits = site_logits[native_t]     # [n_pockets, Lr]
        nat_labels = site_label_pad[native_t]  # [n_pockets, Lr]

        # Build valid-position mask: exclude padding (only compute BCE on real nodes)
        if rna_mask_pad is not None:
            valid_mask = rna_mask_pad[native_t].float()  # [n_pockets, Lr] 1=valid, 0=padding
        else:
            # Fallback: treat all positions as valid (no padding discrimination)
            valid_mask = torch.ones_like(nat_labels)

        n_valid = valid_mask.sum().clamp(min=1.0)
        n_pos   = (nat_labels * valid_mask).sum().clamp(min=1.0)
        n_neg   = (valid_mask - nat_labels * valid_mask).sum().clamp(min=1.0)
        # pos_weight balances the positive/negative class ratio among valid nodes
        pos_weight_val = (n_neg / n_pos).clamp(max=20.0)
        # Expand pos_weight to per-position tensor for masked BCE
        pos_weight_t = torch.full_like(nat_logits, pos_weight_val.item())

        # Compute elementwise BCE, then mask out padding and take mean over valid
        site_bce_elem = F.binary_cross_entropy_with_logits(
            nat_logits, nat_labels, weight=valid_mask,
            pos_weight=pos_weight_t, reduction="sum"
        ) / n_valid
        aux_loss = aux_loss + lambda_site * site_bce_elem

    # ── RNA-side InfoNCE: sim_matrix[i,j] = z_lig_i · z_rna_j / T  (P×P) ──
    # target = arange(P): diagonal = positive (each lig matches its own RNA)
    if lambda_rna_contrast > 0.0 and len(native_indices) >= 2:
        nat_t     = torch.tensor(native_indices, dtype=torch.long, device=device)
        z_lig_nat = z_lig_all[nat_t]   # [P, D]
        z_rna_nat = z_rna_all[nat_t]   # [P, D]
        sim_matrix = (z_lig_nat @ z_rna_nat.T) / temperature   # [P, P]
        target_rna = torch.arange(len(native_indices), device=device)
        rna_cts    = F.cross_entropy(sim_matrix, target_rna)
        aux_loss   = aux_loss + lambda_rna_contrast * rna_cts

    return smol_cts, margin_l, aux_loss


# ═════════════════════════════════════════════════════════════════════════════
# Validation — rank percentile, SMARTBind cross-pocket protocol
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_rank_percentile(
    model: UnifiedInteractionModelV3,
    test_recs: list[dict],
    device: torch.device,
    use_pretrained: bool,
    rna_lig_map: dict | None = None,
    use_fp2_ligand: bool = False,
) -> float:
    """Mean per-pocket rank percentile, SMARTBind cross-pocket protocol.

    For every test pocket:
      - native  = rec["native_smiles"]
      - decoys  = ALL other test pockets' native SMILES, excluding known
                  binders for the same RNA (mirrors SMARTBind is_val=True:
                  full cross-pocket pool, no sub-sampling)
      - RP      = mean(decoy_scores < native_score)
    """
    model.eval()
    rp_list: list[float] = []

    # Pre-build cross-pocket pool: (native_smiles, ligand_id, rna_key)
    pool = [
        (rec["native_smiles"], str(rec.get("ligand_id", "")), _rna_key(rec))
        for rec in test_recs
        if rec.get("native_smiles")
    ]

    for rec in test_recs:
        native = rec["native_smiles"]
        if not native:
            continue

        # All cross-pocket natives, minus known binders for the same RNA
        rna_key = _rna_key(rec)
        known_ligs = (rna_lig_map or {}).get(rna_key, set())
        decoys = [
            s for s, lig_id, rk in pool
            if s != native and lig_id not in known_ligs
        ]
        if not decoys:
            continue

        all_smiles = [native] + decoys
        rg = build_rna_graph_cached(rec["pkl_key"], rec["json_path"])
        rna_graphs = [rg] * len(all_smiles)
        rna_bg = _to_device(batch_rna_graphs(rna_graphs), device)

        try:
            if isinstance(model, (UnifiedInteractionModelV4, UnifiedInteractionModelV10)):
                rna_seqs = [rg.sequence for rg in rna_graphs]
                out = model(rna_bg, all_smiles, rna_sequences=rna_seqs)
            elif use_fp2_ligand:
                lig_bg = smiles_list_to_fp2_tensor(all_smiles, device)
                out = model(rna_bg, lig_bg)
            elif use_pretrained:
                out = model(rna_bg, all_smiles)
            else:
                lig_gs = [build_lig_graph_cached(s) for s in all_smiles]
                lig_bg = _to_device(batch_ligand_graphs(lig_gs), device)
                out = model(rna_bg, lig_bg)
        except Exception as e:
            print(f"[eval] skipping {rec['pkl_key']}: {e}", flush=True)
            continue

        scores = out["rank_score"].cpu().numpy()
        native_score = float(scores[0])
        decoy_scores = scores[1:]
        rp_list.append(float(np.mean(decoy_scores < native_score)))

    if not rp_list:
        return 0.5
    mean_rp = float(np.mean(rp_list))
    pool_size = len(pool) - 1
    print(f"[eval] rank_percentile={mean_rp:.4f}  (n={len(rp_list)} pockets, cross-pocket pool≈{pool_size})", flush=True)
    return mean_rp


# ═════════════════════════════════════════════════════════════════════════════
# Hard negative mining cache
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _refresh_hard_neg_cache(
    model,
    train_recs: list[dict],
    device: torch.device,
    use_pretrained: bool,
    n_candidates: int = 40,
    keep_frac: float = 0.5,
    batch_size: int = 16,
    rna_lig_map: dict | None = None,
) -> dict[str, list[str]]:
    """Score decoy candidates per pocket and cache hardest negatives.

    For each training pocket, samples *n_candidates* decoys (from decoy_smiles),
    scores them against the pocket using the current model, and keeps the
    highest-scoring (= most confusing) ``keep_frac`` fraction.  These become
    the hard-negative pool for the *next* training phase.

    Returns:
        dict[pkl_key -> list[smiles]]  — hardest decoys, sorted desc by score.
    """
    model.eval()
    cache: dict[str, list[str]] = {}

    pockets_with_decoys = [r for r in train_recs if r.get("decoy_smiles")]
    print(
        f"[hard_neg] refreshing cache for {len(pockets_with_decoys)} pockets "
        f"(n_cand={n_candidates}, keep={keep_frac:.0%}, bsz={batch_size})",
        flush=True,
    )

    for rec in pockets_with_decoys:
        n_sample = min(n_candidates, len(rec["decoy_smiles"]))
        sampled_idx = np.random.choice(len(rec["decoy_smiles"]), n_sample, replace=False)
        sampled_decoys = [rec["decoy_smiles"][i] for i in sampled_idx]

        scores: list[float] = []
        rg = build_rna_graph_cached(rec["pkl_key"], rec["json_path"])

        try:
            for start in range(0, len(sampled_decoys), batch_size):
                chunk = sampled_decoys[start : start + batch_size]
                rna_bg = _to_device(batch_rna_graphs([rg] * len(chunk)), device)
                if isinstance(model, (UnifiedInteractionModelV4, UnifiedInteractionModelV10)):
                    rna_seqs = [rg.sequence] * len(chunk)
                    out = model(rna_bg, chunk, rna_sequences=rna_seqs)
                elif use_pretrained:
                    out = model(rna_bg, chunk)
                else:
                    lig_gs = [build_lig_graph_cached(s) for s in chunk]
                    lig_bg = _to_device(batch_ligand_graphs(lig_gs), device)
                    out = model(rna_bg, lig_bg)
                scores.extend(out["rank_score"].cpu().tolist())
        except Exception as e:
            print(f"[hard_neg] skipping {rec['pkl_key']}: {e}", flush=True)
            continue

        # Sort by score desc; keep hardest keep_frac
        sorted_pairs = sorted(zip(scores, sampled_decoys), key=lambda x: x[0], reverse=True)
        n_keep = max(1, int(len(sorted_pairs) * keep_frac))
        cache[rec["pkl_key"]] = [s for _, s in sorted_pairs[:n_keep]]

    model.train()
    print(
        f"[hard_neg] cache built: {len(cache)} pockets, "
        f"avg pool={np.mean([len(v) for v in cache.values()]):.1f}",
        flush=True,
    )
    return cache


# ═════════════════════════════════════════════════════════════════════════════
# Main training loop
# ═════════════════════════════════════════════════════════════════════════════

def train(cfg: dict[str, Any], fold_idx: int, resume: Path | None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] fold={fold_idx}  device={device}", flush=True)

    # ── Load data ──────────────────────────────────────────────────────────
    data_cfg  = cfg["data"]
    pkl_path  = Path(data_cfg.get("pkl_path", str(_DEFAULT_PKL)))
    rna3d_dirs = data_cfg.get("rna3d_dirs", _DEFAULT_RNA3D_DIRS)
    decoy_lookup_path = Path(data_cfg.get("decoy_lookup_path", str(_DEFAULT_DECOY_LOOKUP)))

    records   = load_pkl_records(pkl_path, rna3d_dirs, decoy_lookup_path)
    rna_lig_map = build_rna_ligand_map(records)
    print(f"[data] rna_lig_map: {len(rna_lig_map)} RNA systems", flush=True)
    train_recs = [
        r for r in records
        if len(r["train_split"]) > fold_idx and bool(r["train_split"][fold_idx])
    ]
    test_recs = [
        r for r in records
        if len(r["train_split"]) > fold_idx and not bool(r["train_split"][fold_idx])
    ]
    print(f"[split] fold {fold_idx}: train={len(train_recs)}, test={len(test_recs)}", flush=True)

    if not train_recs:
        raise RuntimeError(f"No training records for fold {fold_idx}. Check pkl and fold index.")

    # ── Model ──────────────────────────────────────────────────────────────
    model_cfg      = cfg["model"]
    model_version  = int(model_cfg.get("version", 3))
    use_pretrained = bool(model_cfg.get("use_pretrained_ligand", True))
    use_fp2_ligand = bool(model_cfg.get("use_fp2_ligand", False))
    if model_version == 4:
        model = UnifiedInteractionModelV4(model_cfg=model_cfg).to(device)
        use_pretrained = False  # v4 uses GIN, not OptiMol
    elif model_version == 10:
        model = UnifiedInteractionModelV10(model_cfg=model_cfg).to(device)
        use_pretrained = False  # v10 uses GIN, not OptiMol
    else:
        model = UnifiedInteractionModelV3(model_cfg=model_cfg).to(device)
    if use_fp2_ligand:
        use_pretrained = False  # FP2 path is mutually exclusive

    start_epoch = 0
    best_rp_from_ckpt = 0.0
    if resume is not None and Path(resume).exists():
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_rp_from_ckpt = float(ckpt.get("best_rp", 0.0))  # restore true best
        print(f"[resume] loaded {resume}, continuing from epoch {start_epoch}, best_rp={best_rp_from_ckpt:.4f}", flush=True)

    train_cfg = cfg["train"]
    lr        = float(train_cfg["lr"])
    wd        = float(train_cfg.get("weight_decay", 1e-4))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    # ── Optimizer: single / dual / triple-lr depending on model version ─────
    lr_ligand = train_cfg.get("lr_ligand", None)
    lr_rna_fm = train_cfg.get("lr_rna_fm", None)
    model_cfg_inner = cfg.get("model", {})

    if model_version in (4, 10) and lr_rna_fm is not None:
        # v4: 2-group lr — RNA-FM unfrozen layers (small), everything else (main)
        lr_rna_fm = float(lr_rna_fm)
        rna_fm_params = list(model.rna_fm_unfrozen_params())
        rna_fm_ids    = {id(p) for p in rna_fm_params}
        other_params  = [p for p in model.parameters()
                         if id(p) not in rna_fm_ids and p.requires_grad]
        optimizer = torch.optim.AdamW([
            {"params": other_params,  "lr": lr},
            {"params": rna_fm_params, "lr": lr_rna_fm},
        ], weight_decay=wd)
        print(f"[optimizer] v4 dual-lr: main={lr:.2e}, rna_fm_last_layers={lr_rna_fm:.2e}  "
              f"(rna_fm trainable params={len(rna_fm_params)})", flush=True)
        use_dual_lr = True
        lr_ligand = None  # not used in v4
    else:
        use_dual_lr = (
            lr_ligand is not None
            and model_cfg_inner.get("use_pretrained_ligand", False)
            and not model_cfg_inner.get("freeze_ligand", True)
            and hasattr(model, "lig_encoder")
            and hasattr(model.lig_encoder, "backbone")
        )
        if use_dual_lr:
            lr_ligand = float(lr_ligand)
            backbone_ids = {id(p) for p in model.lig_encoder.backbone.parameters()}
            backbone_params = [p for p in model.lig_encoder.backbone.parameters() if p.requires_grad]
            other_params    = [p for p in model.parameters() if id(p) not in backbone_ids]
            optimizer = torch.optim.AdamW([
                {"params": other_params,    "lr": lr},
                {"params": backbone_params, "lr": lr_ligand},
            ], weight_decay=wd)
            print(f"[optimizer] dual-lr: main={lr:.2e}, ligand_backbone={lr_ligand:.2e}  "
                  f"(backbone params={len(backbone_params)})", flush=True)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # LR scheduler (cosine)
    epochs         = int(train_cfg["epochs"])
    steps_per_epoch = int(train_cfg["steps_per_epoch"])
    total_steps    = epochs * steps_per_epoch
    warmup_steps   = int(train_cfg.get("lr_scheduler", {}).get("warmup_steps", 500))
    min_lr_ratio   = float(train_cfg.get("lr_scheduler", {}).get("min_lr_ratio", 0.02))
    if use_dual_lr and model_version in (4, 10):
        max_lr_list = [lr, float(train_cfg.get("lr_rna_fm", lr))]
    elif use_dual_lr:
        max_lr_list = [lr, float(train_cfg.get("lr_ligand", lr))]
    else:
        max_lr_list = lr
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr_list,
        total_steps=total_steps,
        pct_start=warmup_steps / max(total_steps, 1),
        final_div_factor=1.0 / max(min_lr_ratio, 1e-9),
        anneal_strategy="cos",
    )
    # Fast-forward scheduler if resuming (suppress spurious PyTorch UserWarning
    # about step order — we deliberately skip optimizer.step() here).
    if start_epoch > 0:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for _ in range(start_epoch * steps_per_epoch):
                scheduler.step()

    # ── Loss config ────────────────────────────────────────────────────────
    loss_cfg       = cfg.get("loss", {})
    margin_start   = float(loss_cfg.get("ranking_margin", 0.6))
    # Temperature annealing: tau decays exponentially from tau_start → tau_end
    # If only contrastive_temperature is set, both start and end equal that value.
    tau_start      = float(loss_cfg.get("contrastive_temperature_start",
                            loss_cfg.get("contrastive_temperature", 0.2)))
    tau_end        = float(loss_cfg.get("contrastive_temperature_end",
                            loss_cfg.get("contrastive_temperature", tau_start)))
    temperature    = tau_start   # will be updated each epoch
    hard_margin_topk = int(loss_cfg.get("hard_margin_topk", 0))
    lambda_contrast = float(loss_cfg.get("lambda_contrast", 0.5))
    lambda_rank    = float(loss_cfg.get("lambda_rank", 0.0))
    lambda_site    = float(loss_cfg.get("lambda_site", 0.0))   # site BCE auxiliary
    lambda_dock    = float(loss_cfg.get("lambda_dock", 0.0))   # dock ranking auxiliary
    lambda_rna_contrast = float(loss_cfg.get("lambda_rna_contrast", 0.5))  # rna_cts triplet
    site_start_ep  = int(loss_cfg.get("site_start_epoch", 0))  # delay site loss onset
    use_circle_loss = bool(loss_cfg.get("use_circle_loss", False))
    adaptive_margin = bool(loss_cfg.get("adaptive_margin", False))
    circle_gamma    = float(loss_cfg.get("circle_gamma", 64.0))
    # Margin schedule: tanh decay with periodic restart (SMARTBind-style)
    # M(t) = M_0 * (1 - tanh(2 * t_in_cycle / N_restart)), t_in_cycle = epoch % N_restart
    # Set margin_restart_epochs=0 to fall back to linear decay (old behaviour).
    margin_restart_epochs = int(loss_cfg.get("margin_restart_epochs", 8))
    margin_end     = float(loss_cfg.get("ranking_margin_end", margin_start))

    # ── Early stopping + reload-best config ───────────────────────────────────
    es_cfg          = cfg.get("early_stopping", {})
    patience        = int(es_cfg.get("patience", 999))          # final stop: no improve for N epochs
    reload_patience = int(es_cfg.get("reload_patience", 999))   # reload best.pt after M epochs no improve
    reload_lr_scale = float(es_cfg.get("reload_lr_scale", 0.5)) # scale LR by this factor on reload

    # ── Hard negative mining config ─────────────────────────────────────────
    hn_cfg         = cfg.get("hard_negative_mining", {})
    hn_enabled     = bool(hn_cfg.get("enabled", False))
    hn_start_ep    = int(hn_cfg.get("start_epoch", 0))        # NEW: delay hard neg mining
    hn_refresh_ev  = int(hn_cfg.get("refresh_every", 2))      # refresh every N epochs
    hn_n_cand      = int(hn_cfg.get("n_candidates", 40))
    hn_keep_frac   = float(hn_cfg.get("keep_frac", 0.5))

    # ── Curriculum ─────────────────────────────────────────────────────────
    curriculum = train_cfg.get("decoy_curriculum", {})
    K_s1 = int(curriculum.get("stage1_negatives_per_positive", 4))
    K_s2 = int(curriculum.get("stage2_negatives_per_positive", 10))
    K_s3 = int(curriculum.get("stage3_negatives_per_positive", 20))
    # SMARTBind ratio: 2/3 cross-pocket (random_decoy_num) + 1/3 hard (extra_decoy_num)
    cross_ratio = float(curriculum.get("cross_pocket_ratio", 0.67))
    sched_cfg   = cfg.get("schedule", {})
    stage1_ep   = int(sched_cfg.get("stage1_epochs", epochs))
    stage2_ep   = int(sched_cfg.get("stage2_epochs", 0))

    batch_size   = int(train_cfg["batch_size"])
    val_decoy_k  = int(train_cfg.get("val_decoy_k", 100))
    log_every    = int(cfg.get("output", {}).get("log_every", 50))
    val_max      = int(cfg.get("output", {}).get("val_max_pockets", 200))

    out_dir = Path(
        cfg["output"]["checkpoint_dir"].replace("fold0", f"fold{fold_idx}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] checkpoints → {out_dir}", flush=True)

    best_rp = best_rp_from_ckpt   # restored from checkpoint (not reset to 0)
    no_improve_count = 0           # consecutive epochs without improvement
    n_reloads = 0                  # how many times we've reloaded best.pt
    hard_neg_cache: dict[str, list[str]] = {}   # populated at epoch end if hn_enabled

    for epoch in range(start_epoch, epochs):
        # ── Dynamic margin ────────────────────────────────────────────────────
        if margin_restart_epochs > 0:
            # SMARTBind tanh decay with periodic restart
            # M(t) = M_0 * (1 - tanh(2 * t_in_cycle / N_restart))
            t_in_cycle = epoch % margin_restart_epochs
            margin = margin_start * (1.0 - math.tanh(2.0 * t_in_cycle / max(margin_restart_epochs, 1)))
        else:
            # Linear decay (legacy)
            if epochs > 1:
                t = epoch / (epochs - 1)
            else:
                t = 0.0
            margin = margin_start + (margin_end - margin_start) * t

        # ── Temperature annealing: exponential τ_start → τ_end ───────────────
        if epochs > 1 and tau_end != tau_start:
            tau_frac  = epoch / (epochs - 1)   # 0 → 1 over training
            temperature = tau_start * ((tau_end / tau_start) ** tau_frac)
        else:
            temperature = tau_start

        # Choose K for this epoch (curriculum) → split into cross-pocket + hard
        if epoch < stage1_ep:
            K = K_s1
        elif epoch < stage1_ep + stage2_ep:
            K = K_s2
        else:
            K = K_s3
        n_cross = max(1, int(K * cross_ratio))
        n_hard  = K - n_cross   # 0 when cross_pocket_ratio=1.0 (pure cross-pocket)

        model.train()
        running_loss = 0.0
        n_steps_ok   = 0

        for step in range(steps_per_epoch):
            idx = np.random.choice(len(train_recs), batch_size,
                                   replace=len(train_recs) < batch_size)
            batch = [train_recs[i] for i in idx]

            optimizer.zero_grad()
            try:
                # Hard-negative pool:
                #   FP2 mode  → None: falls back to random rec["decoy_smiles"] per pocket
                #                     (same mix as v12b: 2/3 cross-pocket + 1/3 FP2-library decoys)
                #   GIN/pretrained → dynamic model-mined HN (refreshed every N epochs)
                if (not use_fp2_ligand) and hn_enabled and epoch >= hn_start_ep:
                    _hn_cache = hard_neg_cache
                else:
                    _hn_cache = None
                infonce, margin_l, aux_loss = _rank_loss_step(
                    model, batch, device, margin, temperature, hard_margin_topk, use_pretrained,
                    n_cross=n_cross, n_hard=n_hard,
                    all_train_recs=train_recs, rna_lig_map=rna_lig_map,
                    hard_neg_cache=_hn_cache,
                    lambda_site=lambda_site if epoch >= site_start_ep else 0.0,
                    lambda_dock=lambda_dock,
                    lambda_rna_contrast=lambda_rna_contrast,
                    use_fp2_ligand=use_fp2_ligand,
                    use_circle_loss=use_circle_loss,
                    adaptive_margin=adaptive_margin,
                )
                loss = lambda_contrast * infonce + lambda_rank * margin_l + aux_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
                running_loss += loss.item()
                n_steps_ok   += 1
            except Exception as e:
                print(f"[ep{epoch:02d} step{step+1:04d}] error: {e}", flush=True)
                continue

            if (step + 1) % log_every == 0:
                avg = running_loss / max(n_steps_ok, 1)
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"[ep{epoch:02d} steτ={temperature:.3f}  p{step+1:04d}/{steps_per_epoch}]"
                    f"  loss={avg:.4f}  K={K}(cross={n_cross}+hard={n_hard})"
                    f"  M={margin:.3f}  lr={lr_now:.2e}",
                    flush=True,
                )

        # ── Epoch-end validation ──────────────────────────────────────────
        # val_max_pockets: 0 = all pockets; >0 = randomly sample a subset
        # (smaller subset = faster but noisier checkpoint selection)
        if val_max > 0 and len(test_recs) > val_max:
            val_recs = random.sample(test_recs, val_max)
        else:
            val_recs = test_recs
        rp = eval_rank_percentile(
            model, val_recs, device, use_pretrained,
            rna_lig_map=rna_lig_map,
            use_fp2_ligand=use_fp2_ligand,
        )
        avg_loss = running_loss / max(n_steps_ok, 1)
        print(
            f"[ep{epoch:02d}]  avg_loss={avg_loss:.4f}"
            f"  val_rank_percentile={rp:.4f}  K={K}  margin={margin:.3f}",
            flush=True,
        )

        # ── Hard negative mining: refresh cache every N epochs (after start_epoch) ─
        # Dynamic HN refresh: only for GIN/pretrained encoders (not FP2 — uses static Tanimoto HN)
        if (not use_fp2_ligand) and hn_enabled and epoch >= hn_start_ep and (epoch + 1) % hn_refresh_ev == 0:
            hard_neg_cache = _refresh_hard_neg_cache(
                model, train_recs, device, use_pretrained,
                n_candidates=hn_n_cand, keep_frac=hn_keep_frac,
                rna_lig_map=rna_lig_map,
            )

        # ── Checkpoint ───────────────────────────────────────────────────
        ckpt_data = {
            "epoch":                epoch,
            "fold":                 fold_idx,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_rank_percentile":  rp,
            "avg_loss":             avg_loss,
            "best_rp":              best_rp,   # persist so resume restores correct baseline
        }
        torch.save(ckpt_data, out_dir / f"ep{epoch:03d}.pt")
        torch.save(ckpt_data, out_dir / "last.pt")
        if rp > best_rp:
            best_rp = rp
            no_improve_count = 0
            ckpt_data["best_rp"] = best_rp   # update after possibly raising best_rp
            torch.save(ckpt_data, out_dir / "best.pt")
            print(f"[ep{epoch:02d}] ★ new best RP={best_rp:.4f} → best.pt", flush=True)
        else:
            no_improve_count += 1
            print(f"[ep{epoch:02d}] no improvement ({no_improve_count}/{patience})", flush=True)
            # ── Reload best weights after reload_patience epochs of no improvement ──
            if no_improve_count >= reload_patience and (out_dir / "best.pt").exists():
                best_ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
                model.load_state_dict(best_ckpt["model_state_dict"])
                # Scale LR to do fine-tuning from the best point
                for pg in optimizer.param_groups:
                    pg["lr"] = pg["lr"] * reload_lr_scale
                no_improve_count = 0
                n_reloads += 1
                new_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[reload #{n_reloads}] restored best.pt (RP={best_rp:.4f}), "
                    f"LR scaled ×{reload_lr_scale} → {new_lr:.2e}",
                    flush=True,
                )
            if no_improve_count >= patience:
                print(f"[early stop] no improvement for {patience} epochs → stopping", flush=True)
                break

    print(f"\n[done] fold={fold_idx}  best_val_RP={best_rp:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train EGNN+OptiMol ranking model on hariboss 5-fold pkl"
    )
    p.add_argument(
        "--config", type=Path,
        help="YAML config file",
    )
    p.add_argument(
        "--fold", type=int, default=0,
        help="K-fold index; HARIBOSS uses 0-4, RNAmigos1 uses 0-9",
    )
    p.add_argument(
        "--resume", type=Path, default=None,
        metavar="CHECKPOINT",
        help="Resume from a .pt checkpoint saved by this script",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Load data and model only; skip training (for environment check)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.dry_run:
        # Sanity check: just load data + model and exit
        data_cfg = cfg["data"]
        pkl_path = Path(data_cfg.get("pkl_path", str(_DEFAULT_PKL)))
        rna3d_dirs = data_cfg.get("rna3d_dirs", _DEFAULT_RNA3D_DIRS)
        decoy_lp = Path(data_cfg.get("decoy_lookup_path", str(_DEFAULT_DECOY_LOOKUP)))
        records = load_pkl_records(pkl_path, rna3d_dirs, decoy_lp)
        rna_lig_map = build_rna_ligand_map(records)
        train_recs = [r for r in records if len(r["train_split"]) > args.fold and r["train_split"][args.fold]]
        test_recs  = [r for r in records if len(r["train_split"]) > args.fold and not r["train_split"][args.fold]]
        print(f"[dry-run] fold={args.fold}: train={len(train_recs)}, test={len(test_recs)}, rna_systems={len(rna_lig_map)}")
        mv = int(cfg["model"].get("version", 3))
        if mv == 4:
            model = UnifiedInteractionModelV4(cfg["model"])
        elif mv == 10:
            model = UnifiedInteractionModelV10(cfg["model"])
        else:
            model = UnifiedInteractionModelV3(cfg["model"])
        print(f"[dry-run] model v{mv} params: {sum(p.numel() for p in model.parameters()):,}")
        print("[dry-run] OK — all checks passed")
        return
    train(cfg, fold_idx=args.fold, resume=args.resume)


if __name__ == "__main__":
    main()
