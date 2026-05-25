#!/usr/bin/env python3
"""Build a FP2-fingerprint → SMILES reverse-lookup pickle from decoy_library.smi.

The hariboss_5fd_cov08_clustered_seqid_0.3_clean.pkl stores `decoy_smiles` as
1024-dim FP2 binary vectors (computed by SMARTBind's convert_smiles_to_pf2 with
pybel/openbabel).  This script inverts that mapping so the training script can
recover the original SMILES for use with OptiMol.

Output: data/decoy_fp2_lookup.pkl  — dict mapping tuple(fp2_list) → smiles_str

Usage:
    python scripts/build_decoy_lookup.py \\
        --smi  /path/to/decoy_library.smi \\
        --out  data/decoy_fp2_lookup.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

try:
    from openbabel import pybel
except ImportError:
    try:
        import pybel  # older openbabel python binding
    except ImportError:
        print("ERROR: openbabel/pybel not found.  "
              "Activate your conda env and install openbabel:\n"
              "  conda install -c conda-forge openbabel", file=sys.stderr)
        sys.exit(1)


def convert_smiles_to_fp2(smiles: str) -> list[int] | None:
    """Replicate SMARTBind's convert_smiles_to_pf2 exactly."""
    try:
        mol = pybel.readstring("smi", smiles)
        fp2_bits = mol.calcfp(fptype="FP2").bits
        fp2_one_hot = [0] * 1024
        for bit in fp2_bits:
            fp2_one_hot[bit] = 1
        return fp2_one_hot
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FP2→SMILES lookup")
    parser.add_argument(
        "--smi",
        default=str(
            Path(__file__).resolve().parents[3]
            / "new" / "SMARTBind" / "notebook" / "decoy_library.smi"
        ),
        help="Path to decoy_library.smi (one SMILES per line)",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "data" / "decoy_fp2_lookup.pkl"),
        help="Output pickle path",
    )
    args = parser.parse_args()

    smi_path = Path(args.smi)
    out_path = Path(args.out)

    if not smi_path.exists():
        print(f"ERROR: {smi_path} not found", file=sys.stderr)
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading SMILES from {smi_path} ...")
    with smi_path.open() as f:
        all_smiles = [line.strip().split()[0] for line in f if line.strip()]
    print(f"  {len(all_smiles):,} SMILES loaded")

    lookup: dict[tuple, str] = {}
    n_fail = 0
    n_collision = 0

    for i, smi in enumerate(all_smiles):
        if i % 5000 == 0:
            print(f"  [{i:>7}/{len(all_smiles)}]  lookup size={len(lookup):,}  "
                  f"fail={n_fail}  collision={n_collision}", flush=True)
        fp2 = convert_smiles_to_fp2(smi)
        if fp2 is None:
            n_fail += 1
            continue
        key = tuple(fp2)
        if key in lookup:
            n_collision += 1
            # Keep first occurrence (same logic as original PKL build)
        else:
            lookup[key] = smi

    print(f"\nDone. {len(lookup):,} unique FP2 keys  "
          f"({n_fail} failed,  {n_collision} collisions ignored)")
    print(f"Saving to {out_path} ...")
    with out_path.open("wb") as f:
        pickle.dump(lookup, f, protocol=4)
    print("Saved.")


if __name__ == "__main__":
    main()
