#!/usr/bin/env python3
"""Build unified manifests from rnamigos2 and GerNA-Bind data.

This script does not modify upstream repositories. It creates symlinks (or copies)
under BridgeBind3D/data/raw and writes normalized CSV manifests under
BridgeBind3D/data/processed/manifests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_COLUMNS = [
    "source",
    "source_dataset",
    "task",
    "target_type",
    "target_id",
    "target_sequence",
    "target_structure_path",
    "pocket_id",
    "ligand_smiles",
    "label_type",
    "label_value",
    "split",
    "extra",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare unified BridgeBind3D manifests")
    parser.add_argument(
        "--rnamigos-root",
        type=Path,
        default=PROJECT_ROOT.parent / "rnamigos2",
        help="Path to local rnamigos2 repository",
    )
    parser.add_argument(
        "--gerna-root",
        type=Path,
        default=PROJECT_ROOT.parent / "GerNA-Bind",
        help="Path to local GerNA-Bind repository",
    )
    parser.add_argument(
        "--include-gerna",
        action="store_true",
        help="Include GerNA-Bind manifests (disabled by default for RNAmigos2-only workflow)",
    )
    parser.add_argument(
        "--link-mode",
        choices=["symlink", "copy", "none"],
        default="symlink",
        help="How to expose source datasets under data/raw",
    )
    parser.add_argument(
        "--build-pocket-node-manifest",
        action="store_true",
        help="Also export one-row-per-pocket-node manifest for binding-site tasks",
    )
    parser.add_argument(
        "--build-simulated-manifest",
        action="store_true",
        help="Export RNAmigos2 pretrain graph manifest for simulated binding-site pretraining",
    )
    parser.add_argument(
        "--build-ligand-decoy-manifest",
        action="store_true",
        help="Export RNAmigos2 ligand_db actives/decoys manifest for affinity pretraining",
    )
    parser.add_argument(
        "--augment-hariboss-actives",
        action="store_true",
        help=(
            "Append Hariboss positive actives into ligand-decoy manifest. "
            "RNAmigos2 decoys remain unchanged. Requires --gerna-root to point to GerNA-Bind."
        ),
    )
    parser.add_argument(
        "--materialize-augmented-ligand-db",
        action="store_true",
        help=(
            "Write an augmented ligand_db-like directory (separate output) where actives.txt "
            "contains RNAmigos2 actives + Hariboss positives for matched pockets."
        ),
    )
    parser.add_argument(
        "--augmented-ligand-db-mode",
        type=str,
        default="pdb_chembl",
        help="Target ligand_db mode subfolder for writing augmented actives.txt (default: pdb_chembl)",
    )
    parser.add_argument(
        "--augmented-ligand-db-out",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "augmented_ligand_db",
        help="Output folder for augmented ligand_db-style files",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str) -> str:
    if mode == "none":
        return "skip"

    if dst.exists() or dst.is_symlink():
        return "exists"

    if mode == "symlink":
        try:
            dst.symlink_to(src.resolve())
            return "symlink"
        except OSError:
            shutil.copytree(src, dst)
            return "copy-fallback"

    shutil.copytree(src, dst)
    return "copy"


def parse_fasta_map(fasta_path: Path) -> Dict[str, str]:
    seq_to_id: Dict[str, str] = {}
    if not fasta_path.exists():
        return seq_to_id

    current_id = None
    current_seq: List[str] = []

    with fasta_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    seq_to_id["".join(current_seq)] = current_id
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

    if current_id is not None:
        seq_to_id["".join(current_seq)] = current_id

    return seq_to_id


def stable_seq_id(seq: str, prefix: str) -> str:
    digest = hashlib.sha1(seq.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def build_gerna_manifest(gerna_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    data_root = gerna_root / "data"
    datasets = ["Robin", "Biosensor"]

    for ds in datasets:
        csv_path = data_root / ds / f"{ds}_random.csv"
        fasta_path = data_root / ds / "sequences.fasta"
        seq_to_name = parse_fasta_map(fasta_path)

        if not csv_path.exists():
            print(f"[WARN] Missing {csv_path}")
            continue

        frame = pd.read_csv(csv_path)
        required = {"rna", "ligand", "label", "split"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")

        for _, rec in frame.iterrows():
            seq = str(rec["rna"])
            target_name = seq_to_name.get(seq, stable_seq_id(seq, ds.lower()))
            structure_path = data_root / ds / "3d" / target_name / "relaxed_1000_model.pdb"

            rows.append(
                {
                    "source": "GerNA-Bind",
                    "source_dataset": ds,
                    "task": "affinity_classification",
                    "target_type": "RNA_sequence",
                    "target_id": target_name,
                    "target_sequence": seq,
                    "target_structure_path": str(structure_path) if structure_path.exists() else "",
                    "pocket_id": "",
                    "ligand_smiles": str(rec["ligand"]),
                    "label_type": "binary",
                    "label_value": float(rec["label"]),
                    "split": str(rec["split"]).lower(),
                    "extra": "{}",
                }
            )

    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def build_rnamigos_manifest(rnamigos_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    csv_root = rnamigos_root / "data" / "csvs"
    json_pockets = rnamigos_root / "data" / "json_pockets_expanded"

    docking_path = csv_root / "docking_data.csv"
    binary_path = csv_root / "binary_data.csv"

    if not docking_path.exists() or not binary_path.exists():
        raise FileNotFoundError("Expected docking_data.csv and binary_data.csv in rnamigos2/data/csvs")

    docking = pd.read_csv(docking_path, low_memory=False)
    binary = pd.read_csv(binary_path, low_memory=False)

    docking_rows: List[Dict[str, str]] = []
    binary_rows: List[Dict[str, str]] = []

    for _, rec in docking.iterrows():
        pocket = str(rec["PDB_ID_POCKET"])
        pocket_json = json_pockets / f"{pocket}.json"
        if "normalized_values" in docking.columns:
            label_value = float(rec["normalized_values"])
            label_type = "normalized_docking_score"
        else:
            label_value = float(rec["INTER"])
            label_type = "raw_docking_score"

        docking_rows.append(
            {
                "source": "RNAmigos2",
                "source_dataset": "docking_data",
                "task": "affinity_ranking",
                "target_type": "RNA_3D_pocket",
                "target_id": pocket,
                "target_sequence": "",
                "target_structure_path": str(pocket_json) if pocket_json.exists() else "",
                "pocket_id": pocket,
                "ligand_smiles": str(rec["LIGAND_SMILES"]),
                "label_type": label_type,
                "label_value": label_value,
                "split": str(rec["SPLIT"]).lower(),
                "extra": "{}",
            }
        )

    for _, rec in binary.iterrows():
        pocket = str(rec["PDB_ID_POCKET"])
        pocket_json = json_pockets / f"{pocket}.json"
        binary_rows.append(
            {
                "source": "RNAmigos2",
                "source_dataset": "binary_data",
                "task": "affinity_classification",
                "target_type": "RNA_3D_pocket",
                "target_id": pocket,
                "target_sequence": "",
                "target_structure_path": str(pocket_json) if pocket_json.exists() else "",
                "pocket_id": pocket,
                "ligand_smiles": str(rec["LIGAND_SMILES"]),
                "label_type": "binary",
                "label_value": int(rec["IS_NATIVE"]),
                "split": str(rec["SPLIT"]).lower(),
                "extra": json.dumps({"ligand_source": rec.get("LIGAND_SOURCE", "")}),
            }
        )

    return pd.DataFrame(docking_rows, columns=MANIFEST_COLUMNS), pd.DataFrame(binary_rows, columns=MANIFEST_COLUMNS)


def build_pocket_node_manifest(rnamigos_root: Path) -> pd.DataFrame:
    pockets_dir = rnamigos_root / "data" / "json_pockets_expanded"
    if not pockets_dir.exists():
        raise FileNotFoundError(f"Missing pocket directory: {pockets_dir}")

    rows: List[Dict[str, str]] = []
    for json_path in sorted(pockets_dir.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        nodes = payload.get("nodes", [])
        pocket_id = json_path.stem
        for node in nodes:
            if isinstance(node, dict):
                node_id = str(node.get("id", ""))
                node_attr = node
            elif isinstance(node, (list, tuple)):
                node_id = str(node[0]) if len(node) > 0 else ""
                node_attr = node[1] if len(node) > 1 and isinstance(node[1], dict) else {}
            else:
                continue
            rows.append(
                {
                    "source": "RNAmigos2",
                    "task": "binding_site_pretraining",
                    "target_id": pocket_id,
                    "pocket_id": pocket_id,
                    "node_id": node_id,
                    "nt_code": str(node_attr.get("nt_code", "")),
                    "in_pocket": int(bool(node_attr.get("in_pocket", False))),
                }
            )

    return pd.DataFrame(rows)


def build_simulated_pretrain_manifest(rnamigos_root: Path) -> pd.DataFrame:
    """Build one-row-per-graph manifest from RNAmigos2 simulated pretrain data."""
    graph_root = rnamigos_root / "data" / "pretrain_data" / "NR_chops"
    annotated_root = rnamigos_root / "data" / "pretrain_data" / "nr-graphs_annotated"
    if not graph_root.exists():
        raise FileNotFoundError(f"Missing simulated graph dir: {graph_root}")

    rows: List[Dict[str, str]] = []
    for nx_path in sorted(graph_root.glob("*.nx")):
        graph_id = nx_path.stem
        annot_path = annotated_root / f"{graph_id}_annot.p"
        rows.append(
            {
                "source": "RNAmigos2",
                "source_dataset": "pretrain_data_NR_chops",
                "task": "binding_site_self_supervised",
                "target_type": "RNA_3D_graph",
                "target_id": graph_id,
                "target_sequence": "",
                "target_structure_path": str(nx_path),
                "pocket_id": "",
                "ligand_smiles": "",
                "label_type": "none",
                "label_value": "",
                "split": "pretrain",
                "extra": json.dumps({"annotated_graph_path": str(annot_path) if annot_path.exists() else ""}),
            }
        )

    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def _read_smiles_file(smiles_path: Path) -> List[str]:
    if not smiles_path.exists():
        return []
    with smiles_path.open("r", encoding="utf-8") as handle:
        smiles = [line.strip() for line in handle if line.strip()]
    return smiles


def _normalize_split(v: str) -> str:
    x = str(v).strip().lower()
    if x in {"val", "valid", "validation", "dev"}:
        return "valid"
    if x in {"test", "holdout"}:
        return "test"
    return "train"


def _load_hariboss_split_map(split_json_path: Path) -> Dict[str, str]:
    if not split_json_path.exists():
        return {}
    payload = json.loads(split_json_path.read_text(encoding="utf-8"))
    out: Dict[str, str] = {}
    for split_name, keys in payload.items():
        norm = _normalize_split(split_name)
        if not isinstance(keys, list):
            continue
        for key in keys:
            out[str(key).strip().lower()] = norm
    return out


def _parse_hariboss_ligand_code(sm_ligand: str) -> Tuple[str, str]:
    # Example: ARG_.:B/47:A -> ligand code ARG, residue id 47
    text = str(sm_ligand)
    ligand_code = text.split("_")[0].strip().upper()
    match = re.search(r"/([^/:]+)", text)
    residue_id = match.group(1).strip() if match else ""
    return ligand_code, residue_id


def _parse_hariboss_rna_chain(sm_ligand_rna_chain: str) -> str:
    # Example: 1aju-A/A -> RNA chain A
    text = str(sm_ligand_rna_chain)
    if "-" not in text:
        return ""
    right = text.split("-", 1)[1]
    return right.split("/", 1)[0].strip().upper()


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v)


def build_ligand_decoy_manifest(rnamigos_root: Path) -> pd.DataFrame:
    """Build affinity-style manifest from RNAmigos2 ligand_db actives/decoys."""
    ligand_db_root = rnamigos_root / "data" / "ligand_db"
    pockets_root = rnamigos_root / "data" / "json_pockets_expanded"
    if not ligand_db_root.exists():
        raise FileNotFoundError(f"Missing ligand db dir: {ligand_db_root}")

    rows: List[Dict[str, str]] = []
    for pocket_dir in sorted(p for p in ligand_db_root.iterdir() if p.is_dir()):
        pocket_id = pocket_dir.name
        pocket_json = pockets_root / f"{pocket_id}.json"

        for mode_dir in sorted(p for p in pocket_dir.iterdir() if p.is_dir()):
            mode = mode_dir.name
            actives = _read_smiles_file(mode_dir / "actives.txt")
            decoys = _read_smiles_file(mode_dir / "decoys.txt")

            for smi in actives:
                rows.append(
                    {
                        "source": "RNAmigos2",
                        "source_dataset": f"ligand_db_{mode}",
                        "task": "affinity_classification",
                        "target_type": "RNA_3D_pocket",
                        "target_id": pocket_id,
                        "target_sequence": "",
                        "target_structure_path": str(pocket_json) if pocket_json.exists() else "",
                        "pocket_id": pocket_id,
                        "ligand_smiles": smi,
                        "label_type": "binary",
                        "label_value": 1,
                        "split": "pretrain",
                        "extra": json.dumps({"decoy_mode": mode, "sample_type": "active"}),
                    }
                )

            for smi in decoys:
                rows.append(
                    {
                        "source": "RNAmigos2",
                        "source_dataset": f"ligand_db_{mode}",
                        "task": "affinity_classification",
                        "target_type": "RNA_3D_pocket",
                        "target_id": pocket_id,
                        "target_sequence": "",
                        "target_structure_path": str(pocket_json) if pocket_json.exists() else "",
                        "pocket_id": pocket_id,
                        "ligand_smiles": smi,
                        "label_type": "binary",
                        "label_value": 0,
                        "split": "pretrain",
                        "extra": json.dumps({"decoy_mode": mode, "sample_type": "decoy"}),
                    }
                )

    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def build_hariboss_active_manifest(gerna_root: Path, rnamigos_root: Path) -> pd.DataFrame:
    """Build positive-only active rows from Hariboss matched to RNAmigos2 pocket IDs.

    Matching rule (strict):
    pocket_id = {PDB}_{RNA_CHAIN}_{LIGAND_CODE}_{RESIDUE_ID}
    Example: 1aju + chain A + ARG + 47 -> 1AJU_A_ARG_47
    """
    hariboss_csv = gerna_root / "data" / "Hariboss" / "hariboss.csv"
    split_json = gerna_root / "data" / "Hariboss" / "data_split.json"
    ligand_db_root = rnamigos_root / "data" / "ligand_db"
    pockets_root = rnamigos_root / "data" / "json_pockets_expanded"

    if not hariboss_csv.exists():
        raise FileNotFoundError(f"Missing Hariboss csv: {hariboss_csv}")
    if not ligand_db_root.exists():
        raise FileNotFoundError(f"Missing RNAmigos2 ligand_db dir: {ligand_db_root}")

    split_map = _load_hariboss_split_map(split_json)
    frame = pd.read_csv(hariboss_csv, low_memory=False)
    required = {"id", "sm_ligand", "sm_smiles", "sm_ligand_rna_chain"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{hariboss_csv} missing columns: {sorted(missing)}")

    # RNAmigos2 pocket lookup by lowercase for robust matching.
    pocket_lookup: Dict[str, str] = {}
    for p in ligand_db_root.iterdir():
        if p.is_dir():
            pocket_lookup[p.name.lower()] = p.name

    rows: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for _, rec in frame.iterrows():
        pdb_id = _safe_str(rec.get("id", "")).strip().upper()
        chain = _parse_hariboss_rna_chain(_safe_str(rec.get("sm_ligand_rna_chain", "")))
        lig_code, resid = _parse_hariboss_ligand_code(_safe_str(rec.get("sm_ligand", "")))
        smiles = _safe_str(rec.get("sm_smiles", "")).strip()
        if not pdb_id or not chain or not lig_code or not resid or not smiles:
            continue

        candidate = f"{pdb_id}_{chain}_{lig_code}_{resid}"
        pocket_id = pocket_lookup.get(candidate.lower())
        if not pocket_id:
            continue

        dedup_key = (pocket_id.lower(), smiles)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        split_key = f"{pdb_id.lower()}_{chain.lower()}"
        split = split_map.get(split_key, "pretrain")
        pocket_json = pockets_root / f"{pocket_id}.json"

        rows.append(
            {
                "source": "Hariboss",
                "source_dataset": "hariboss_positive",
                "task": "affinity_classification",
                "target_type": "RNA_3D_pocket",
                "target_id": pocket_id,
                "target_sequence": "",
                "target_structure_path": str(pocket_json) if pocket_json.exists() else "",
                "pocket_id": pocket_id,
                "ligand_smiles": smiles,
                "label_type": "binary",
                "label_value": 1,
                "split": split,
                "extra": json.dumps({"sample_type": "active", "source": "Hariboss"}),
            }
        )

    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def build_hariboss_active_map(gerna_root: Path, rnamigos_root: Path) -> Dict[str, set[str]]:
    """Return mapping: pocket_id -> set(smiles) for Hariboss matched positives."""
    df = build_hariboss_active_manifest(gerna_root, rnamigos_root)
    pocket_to_smiles: Dict[str, set[str]] = {}
    for _, rec in df.iterrows():
        pid = str(rec["pocket_id"])
        smi = str(rec["ligand_smiles"])
        if not pid or not smi:
            continue
        pocket_to_smiles.setdefault(pid, set()).add(smi)
    return pocket_to_smiles


def _write_lines(path: Path, lines: List[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def materialize_augmented_ligand_db_actives(
    rnamigos_root: Path,
    out_root: Path,
    hariboss_active_map: Dict[str, set[str]],
    mode: str = "pdb_chembl",
) -> Dict[str, int]:
    """Write augmented ligand_db files without modifying upstream rnamigos2.

    For each matched pocket, write:
    - {out_root}/{pocket_id}/{mode}/actives.txt  (merged actives)
    - {out_root}/{pocket_id}/{mode}/decoys.txt   (copied from upstream if present)
    """
    ligand_db_root = rnamigos_root / "data" / "ligand_db"
    ensure_dir(out_root)

    pockets_written = 0
    added_smiles = 0

    for pocket_id, extra_smiles in sorted(hariboss_active_map.items()):
        src_mode_dir = ligand_db_root / pocket_id / mode
        src_actives = _read_smiles_file(src_mode_dir / "actives.txt")
        src_decoys = _read_smiles_file(src_mode_dir / "decoys.txt")

        merged = sorted(set(src_actives).union(extra_smiles))
        added_smiles += max(0, len(merged) - len(set(src_actives)))

        dst_mode_dir = out_root / pocket_id / mode
        _write_lines(dst_mode_dir / "actives.txt", merged)
        if src_decoys:
            _write_lines(dst_mode_dir / "decoys.txt", src_decoys)

        pockets_written += 1

    return {
        "augmented_pockets_written": pockets_written,
        "augmented_added_active_smiles": added_smiles,
    }


def write_csv(df: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)


def main() -> None:
    args = parse_args()

    data_raw = PROJECT_ROOT / "data" / "raw"
    manifests_dir = PROJECT_ROOT / "data" / "processed" / "manifests"
    ensure_dir(data_raw)
    ensure_dir(manifests_dir)

    status_rnamigos = link_or_copy(args.rnamigos_root, data_raw / "rnamigos2", args.link_mode)
    status_gerna = "disabled"
    gerna_manifest = pd.DataFrame(columns=MANIFEST_COLUMNS)
    if args.include_gerna:
        status_gerna = link_or_copy(args.gerna_root, data_raw / "gerna_bind", args.link_mode)
        gerna_manifest = build_gerna_manifest(args.gerna_root)

    docking_manifest, binary_manifest = build_rnamigos_manifest(args.rnamigos_root)
    manifest_parts = [docking_manifest, binary_manifest]
    if args.include_gerna:
        manifest_parts.insert(0, gerna_manifest)

    write_csv(gerna_manifest, manifests_dir / "gerna_affinity_manifest.csv")
    write_csv(docking_manifest, manifests_dir / "rnamigos_docking_manifest.csv")
    write_csv(binary_manifest, manifests_dir / "rnamigos_binary_manifest.csv")

    summary = {
        "link_mode": args.link_mode,
        "link_status": {
            "rnamigos2": status_rnamigos,
            "gerna_bind": status_gerna,
        },
        "counts": {
            "gerna_affinity_rows": int(len(gerna_manifest)),
            "rnamigos_docking_rows": int(len(docking_manifest)),
            "rnamigos_binary_rows": int(len(binary_manifest)),
        },
    }

    if args.build_simulated_manifest:
        simulated_manifest = build_simulated_pretrain_manifest(args.rnamigos_root)
        write_csv(simulated_manifest, manifests_dir / "rnamigos_simulated_pretrain_manifest.csv")
        manifest_parts.append(simulated_manifest)
        summary["counts"]["rnamigos_simulated_pretrain_rows"] = int(len(simulated_manifest))

    if args.build_ligand_decoy_manifest:
        decoy_manifest = build_ligand_decoy_manifest(args.rnamigos_root)
        hariboss_active_manifest = pd.DataFrame(columns=MANIFEST_COLUMNS)
        hariboss_active_map: Dict[str, set[str]] = {}
        if args.augment_hariboss_actives:
            hariboss_active_manifest = build_hariboss_active_manifest(args.gerna_root, args.rnamigos_root)
            write_csv(hariboss_active_manifest, manifests_dir / "hariboss_active_manifest.csv")
            hariboss_active_map = build_hariboss_active_map(args.gerna_root, args.rnamigos_root)

        merged_decoy_manifest = pd.concat([decoy_manifest, hariboss_active_manifest], ignore_index=True)
        merged_decoy_manifest = merged_decoy_manifest.drop_duplicates(
            subset=["pocket_id", "ligand_smiles", "label_value"], keep="first"
        ).reset_index(drop=True)

        write_csv(merged_decoy_manifest, manifests_dir / "rnamigos_ligand_decoy_manifest.csv")
        manifest_parts.append(merged_decoy_manifest)
        summary["counts"]["rnamigos_ligand_decoy_rows"] = int(len(merged_decoy_manifest))
        summary["counts"]["hariboss_active_rows"] = int(len(hariboss_active_manifest))

        if args.materialize_augmented_ligand_db:
            stats = materialize_augmented_ligand_db_actives(
                rnamigos_root=args.rnamigos_root,
                out_root=args.augmented_ligand_db_out,
                hariboss_active_map=hariboss_active_map,
                mode=args.augmented_ligand_db_mode,
            )
            summary["augmented_ligand_db"] = {
                "mode": args.augmented_ligand_db_mode,
                "output_root": str(args.augmented_ligand_db_out),
                **stats,
            }

    bridge_manifest = pd.concat(manifest_parts, ignore_index=True)
    write_csv(bridge_manifest, manifests_dir / "bridge_manifest.csv")

    if args.build_pocket_node_manifest:
        pocket_nodes = build_pocket_node_manifest(args.rnamigos_root)
        write_csv(pocket_nodes, manifests_dir / "rnamigos_pocket_nodes_manifest.csv")
        summary["counts"]["pocket_node_rows"] = int(len(pocket_nodes))

    summary["counts"]["bridge_total_rows"] = int(len(bridge_manifest))

    summary_path = manifests_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"[OK] Wrote manifests to: {manifests_dir}")


if __name__ == "__main__":
    main()
