from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Tuple

import torch

try:
    from rdkit import Chem
except Exception:
    Chem = None


NT_ORDER = ["A", "C", "G", "U", "N"]


def _seed_from_text(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def token_features_from_text(text: str, n_tokens: int, dim: int) -> torch.Tensor:
    """Deterministic placeholder featurizer.

    This is intentionally simple so the unified training scaffold can run end-to-end.
    Replace this with real RNA/ligand graph featurization in the next step.
    """
    g = torch.Generator()
    g.manual_seed(_seed_from_text(text))
    return torch.randn(n_tokens, dim, generator=g)


def _pad_or_trim(tokens: torch.Tensor, n_tokens: int, dim: int) -> torch.Tensor:
    if tokens.ndim != 2:
        tokens = tokens.view(-1, dim)
    feat_dim = tokens.shape[1]
    if feat_dim > dim:
        tokens = tokens[:, :dim]
    elif feat_dim < dim:
        tokens = torch.cat([tokens, torch.zeros(tokens.shape[0], dim - feat_dim)], dim=1)

    if tokens.shape[0] >= n_tokens:
        return tokens[:n_tokens]
    return torch.cat([tokens, torch.zeros(n_tokens - tokens.shape[0], dim)], dim=0)


def _nt_one_hot(nt_code: str) -> torch.Tensor:
    nt_code = (nt_code or "N").upper()
    if nt_code not in NT_ORDER:
        nt_code = "N"
    vec = torch.zeros(len(NT_ORDER))
    vec[NT_ORDER.index(nt_code)] = 1.0
    return vec


def _hash_vec(text: str, dim: int) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(_seed_from_text(text))
    return torch.randn(dim, generator=g)


def _read_pocket_nodes(pocket_structure_path: str) -> list[dict]:
    if not pocket_structure_path:
        return []
    path = Path(pocket_structure_path)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    nodes = payload.get("nodes", [])
    links = payload.get("links", [])

    degree = {}
    for edge in links:
        src = str(edge.get("source", ""))
        dst = str(edge.get("target", ""))
        degree[src] = degree.get(src, 0) + 1
        degree[dst] = degree.get(dst, 0) + 1

    out = []
    for n in nodes:
        if isinstance(n, dict):
            nid = str(n.get("id", ""))
            out.append(
                {
                    "id": nid,
                    "nt_code": str(n.get("nt_code", "N")),
                    "in_pocket": float(bool(n.get("in_pocket", False))),
                    "degree": float(degree.get(nid, 0)),
                }
            )
        elif isinstance(n, (list, tuple)) and len(n) >= 2:
            nid = str(n[0])
            attrs = n[1] if isinstance(n[1], dict) else {}
            out.append(
                {
                    "id": nid,
                    "nt_code": str(attrs.get("nt_code", "N")),
                    "in_pocket": float(bool(attrs.get("in_pocket", False))),
                    "degree": float(degree.get(nid, 0)),
                }
            )
    return out


def build_rna_tokens_from_pocket(
    pocket_id: str,
    pocket_structure_path: str,
    max_rna_tokens: int,
    dim: int,
) -> torch.Tensor:
    nodes = _read_pocket_nodes(pocket_structure_path)
    if not nodes:
        return token_features_from_text(f"rna::{pocket_id}", max_rna_tokens, dim)

    rows = []
    denom = float(max(1, len(nodes) - 1))
    for i, node in enumerate(nodes):
        base = torch.cat(
            [
                _nt_one_hot(node["nt_code"]),
                torch.tensor(
                    [
                        node["in_pocket"],
                        min(node["degree"], 20.0) / 20.0,
                        float(i) / denom,
                    ]
                ),
                _hash_vec(node["id"], 8),
            ]
        )
        rows.append(base)

    return _pad_or_trim(torch.stack(rows, dim=0), max_rna_tokens, dim)


def _atom_features(atom) -> torch.Tensor:
    return torch.tensor(
        [
            float(atom.GetAtomicNum()) / 100.0,
            float(atom.GetTotalDegree()) / 6.0,
            float(atom.GetFormalCharge()) / 4.0,
            float(atom.GetIsAromatic()),
            float(atom.GetTotalNumHs()) / 4.0,
            float(atom.IsInRing()),
        ]
    )


def build_ligand_tokens(
    ligand_smiles: str,
    max_lig_tokens: int,
    dim: int,
) -> torch.Tensor:
    smiles = ligand_smiles or ""
    if Chem is None:
        return token_features_from_text(f"lig::{smiles}", max_lig_tokens, dim)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return token_features_from_text(f"lig::{smiles}", max_lig_tokens, dim)

    rows = []
    for atom in mol.GetAtoms():
        rows.append(torch.cat([_atom_features(atom), _hash_vec(f"{smiles}:{atom.GetIdx()}", 8)]))
    if not rows:
        return token_features_from_text(f"lig::{smiles}", max_lig_tokens, dim)

    return _pad_or_trim(torch.stack(rows, dim=0), max_lig_tokens, dim)


def build_pair_features(
    pocket_id: str,
    ligand_smiles: str,
    max_rna_tokens: int,
    max_lig_tokens: int,
    dim: int,
    pocket_structure_path: str = "",
    featurizer_mode: str = "real",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if featurizer_mode == "placeholder":
        rna_tokens = token_features_from_text(f"rna::{pocket_id}", max_rna_tokens, dim)
        lig_tokens = token_features_from_text(f"lig::{ligand_smiles}", max_lig_tokens, dim)
        return rna_tokens, lig_tokens

    rna_tokens = build_rna_tokens_from_pocket(
        pocket_id=pocket_id,
        pocket_structure_path=pocket_structure_path,
        max_rna_tokens=max_rna_tokens,
        dim=dim,
    )
    lig_tokens = build_ligand_tokens(
        ligand_smiles=ligand_smiles,
        max_lig_tokens=max_lig_tokens,
        dim=dim,
    )
    return rna_tokens, lig_tokens
