from __future__ import annotations

import functools
import hashlib
import json
import re
from pathlib import Path
from typing import Tuple

import torch

from bridgebind3d.graph_data import LigandGraph, RNAGraph

try:
    from rdkit import Chem
except Exception:
    Chem = None


NT_ORDER = ["A", "C", "G", "U", "N"]
BOND_ORDER = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]

# Per-nucleotide pharmacophore properties (Watson-Crick base chemistry)
# purine: A/G=1, C/U=0; hbd: H-bond donors/2; hba: H-bond acceptors/3
_NT_PURINE   = {"A": 1.0, "G": 1.0, "C": 0.0, "U": 0.0, "N": 0.5}
_NT_HBD_NORM = {"A": 0.5, "G": 1.0, "C": 0.5, "U": 1.0, "N": 0.75}
_NT_HBA_NORM = {"A": 0.67, "G": 0.67, "C": 1.0, "U": 0.67, "N": 0.75}


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


def _to_float(v, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _extract_xyz(node: dict) -> torch.Tensor:
    # RNAmigos2 json node formats vary across preprocessing versions.
    direct = [
        ("x", "y", "z"),
        ("X", "Y", "Z"),
        ("coord_x", "coord_y", "coord_z"),
        ("cx", "cy", "cz"),
    ]
    for kx, ky, kz in direct:
        if kx in node and ky in node and kz in node:
            return torch.tensor([_to_float(node.get(kx)), _to_float(node.get(ky)), _to_float(node.get(kz))])

    for key in ["coord", "coords", "position", "xyz", "center", "centroid"]:
        val = node.get(key)
        if isinstance(val, (list, tuple)) and len(val) >= 3:
            return torch.tensor([_to_float(val[0]), _to_float(val[1]), _to_float(val[2])])

    for prefix in ["C4prime", "C4'", "P", "base"]:
        kx, ky, kz = f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"
        if kx in node and ky in node and kz in node:
            return torch.tensor([_to_float(node.get(kx)), _to_float(node.get(ky)), _to_float(node.get(kz))])

    return torch.zeros(3)


@functools.lru_cache(maxsize=None)
def _read_pocket_nodes(pocket_structure_path: str) -> tuple[dict, ...]:
    if not pocket_structure_path:
        return ()
    path = Path(pocket_structure_path)
    if not path.exists():
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ()
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
            rec = dict(n)
            rec.update(
                {
                    "id": nid,
                    "nt_code": str(n.get("nt_code", "N")),
                    "in_pocket": float(bool(n.get("in_pocket", False))),
                    "degree": float(degree.get(nid, 0)),
                }
            )
            out.append(rec)
        elif isinstance(n, (list, tuple)) and len(n) >= 2:
            nid = str(n[0])
            attrs = n[1] if isinstance(n[1], dict) else {}
            rec = dict(attrs)
            rec.update(
                {
                    "id": nid,
                    "nt_code": str(attrs.get("nt_code", "N")),
                    "in_pocket": float(bool(attrs.get("in_pocket", False))),
                    "degree": float(degree.get(nid, 0)),
                }
            )
            out.append(rec)
    return tuple(out)  # must be hashable for lru_cache


_POCKET_PAYLOAD_CACHE: dict[str, dict] = {}


def _read_pocket_payload(pocket_structure_path: str) -> dict:
    if not pocket_structure_path:
        return {}
    cached = _POCKET_PAYLOAD_CACHE.get(pocket_structure_path)
    if cached is not None:
        return cached
    path = Path(pocket_structure_path)
    if not path.exists():
        _POCKET_PAYLOAD_CACHE[pocket_structure_path] = {}
        return {}
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        d = {}
    _POCKET_PAYLOAD_CACHE[pocket_structure_path] = d
    return d


def _parse_residue_pos(node_id: str) -> float:
    m = re.search(r"(-?\d+)(?:[A-Za-z]?)$", str(node_id))
    if not m:
        return 0.0
    return float(m.group(1))


def _norm_pos_tensor(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    mn = x.min()
    mx = x.max()
    if float(mx - mn) < 1e-6:
        return torch.zeros_like(x)
    return (x - mn) / (mx - mn)


def _edge_type_onehot(edge_type: str) -> torch.Tensor:
    t = str(edge_type or "").upper()
    # [generic_json, backbone, pairing_like, other]
    vec = torch.zeros(4)
    if not t:
        vec[3] = 1.0
    elif t in {"B53", "B35", "BACKBONE"}:
        vec[1] = 1.0
    elif any(k in t for k in ["WW", "WH", "WS", "SH", "HS", "PAIR", "STACK"]):
        vec[2] = 1.0
    else:
        vec[0] = 1.0
    return vec


def build_rna_graph(pocket_id: str, pocket_structure_path: str) -> RNAGraph:
    payload = _read_pocket_payload(pocket_structure_path)
    nodes = _read_pocket_nodes(pocket_structure_path)
    if not nodes:
        # Deterministic fallback keeps pipeline runnable for missing structures.
        n = 16
        node_feat = torch.zeros((n, 10), dtype=torch.float32)
        node_feat[:, -1] = torch.linspace(0, 1, steps=n)
        pos = torch.stack([torch.linspace(0, 1, steps=n), torch.zeros(n), torch.zeros(n)], dim=-1)
        edge_src = torch.arange(0, n - 1, dtype=torch.long)
        edge_dst = torch.arange(1, n, dtype=torch.long)
        edge_index = torch.cat([torch.stack([edge_src, edge_dst]), torch.stack([edge_dst, edge_src])], dim=1)
        edge_feat = torch.zeros((edge_index.shape[1], 4), dtype=torch.float32)
        edge_feat[:, 1] = 1.0
        site_label = torch.zeros((n,), dtype=torch.float32)
        return RNAGraph(node_feat=node_feat, pos=pos, edge_index=edge_index, edge_feat=edge_feat, site_label=site_label, sequence="")

    node_ids = [str(n.get("id", "")) for n in nodes]
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    deg = torch.tensor([float(n.get("degree", 0.0)) for n in nodes], dtype=torch.float32)
    deg_norm = (deg.clamp(max=20.0) / 20.0).view(-1, 1)

    raw_pos = torch.tensor([_parse_residue_pos(nid) for nid in node_ids], dtype=torch.float32)
    pos_norm = _norm_pos_tensor(raw_pos).view(-1, 1)

    node_feat = []
    xyz = []
    site_label = []
    for i, n in enumerate(nodes):
        nt_code = str(n.get("nt_code", "N")).upper()
        if nt_code not in _NT_PURINE:
            nt_code = "N"
        nt = _nt_one_hot(nt_code)
        pharma = torch.tensor([
            _NT_PURINE[nt_code],
            _NT_HBD_NORM[nt_code],
            _NT_HBA_NORM[nt_code],
        ], dtype=torch.float32)
        node_feat.append(torch.cat([nt, deg_norm[i], pos_norm[i], pharma], dim=0))
        xyz.append(_extract_xyz(n))
        site_label.append(float(n.get("in_pocket", 0.0)))
    node_feat_t = torch.stack(node_feat, dim=0).to(torch.float32)
    pos_t = torch.stack(xyz, dim=0).to(torch.float32)

    # If all coordinates are missing, use pseudo coordinates on a line.
    if float(pos_t.abs().sum()) < 1e-8:
        n = len(nodes)
        pos_t = torch.stack([torch.linspace(0, 1, steps=n), torch.zeros(n), torch.zeros(n)], dim=-1)

    edge_pairs = []
    edge_feats = []
    for e in payload.get("links", []):
        src = str(e.get("source", ""))
        dst = str(e.get("target", ""))
        if src not in id_to_idx or dst not in id_to_idx:
            continue
        et = str(e.get("LW", e.get("type", e.get("interaction", ""))))
        s, d = id_to_idx[src], id_to_idx[dst]
        edge_pairs.append((s, d))
        edge_pairs.append((d, s))
        ef = _edge_type_onehot(et)
        edge_feats.append(ef)
        edge_feats.append(ef)

    # Add backbone adjacency by residue order.
    sorted_idx = sorted(range(len(node_ids)), key=lambda i: (_parse_residue_pos(node_ids[i]), node_ids[i]))
    for i in range(len(sorted_idx) - 1):
        s = sorted_idx[i]
        d = sorted_idx[i + 1]
        edge_pairs.append((s, d))
        edge_pairs.append((d, s))
        ef = _edge_type_onehot("backbone")
        edge_feats.append(ef)
        edge_feats.append(ef)

    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_feat = torch.stack(edge_feats, dim=0).to(torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_feat = torch.zeros((0, 4), dtype=torch.float32)

    # Build nt_code sequence string in node order (for RNA-FM input)
    sequence = "".join(
        str(n.get("nt_code", "N")).upper()[:1] for n in nodes
    )
    # Replace any non-ACGU character with N (RNA-FM expects ACGUN)
    sequence = "".join(c if c in "ACGU" else "N" for c in sequence)

    return RNAGraph(
        node_feat=node_feat_t,
        pos=pos_t,
        edge_index=edge_index,
        edge_feat=edge_feat,
        site_label=torch.tensor(site_label, dtype=torch.float32),
        node_id=node_ids,
        sequence=sequence,
    )


def _bond_one_hot(bond) -> torch.Tensor:
    if bond is None:
        return torch.zeros(4)
    bt = str(bond.GetBondType()).upper()
    vec = torch.zeros(4)
    if bt in BOND_ORDER:
        vec[BOND_ORDER.index(bt)] = 1.0
    return vec


def _hybrid_three(atom) -> torch.Tensor:
    hyb = str(atom.GetHybridization()).upper()
    return torch.tensor([
        float("SP" in hyb and "SP2" not in hyb and "SP3" not in hyb),
        float("SP2" in hyb),
        float("SP3" in hyb),
    ])


def build_lig_graph(ligand_smiles: str) -> LigandGraph:
    smiles = ligand_smiles or ""
    if Chem is None:
        node_feat = torch.zeros((1, 11), dtype=torch.float32)
        return LigandGraph(node_feat=node_feat, edge_index=torch.zeros((2, 0), dtype=torch.long), edge_feat=torch.zeros((0, 4)))

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        node_feat = torch.zeros((1, 11), dtype=torch.float32)
        return LigandGraph(node_feat=node_feat, edge_index=torch.zeros((2, 0), dtype=torch.long), edge_feat=torch.zeros((0, 4)))

    nfeat = []
    for atom in mol.GetAtoms():
        is_donor    = float(atom.GetAtomicNum() in {7, 8} and atom.GetTotalNumHs() > 0)
        is_acceptor = float(atom.GetAtomicNum() in {7, 8, 9, 16})
        nfeat.append(
            torch.tensor(
                [
                    float(atom.GetAtomicNum()) / 100.0,
                    float(atom.GetTotalDegree()) / 6.0,
                    float(atom.GetFormalCharge()) / 4.0,
                    float(atom.GetIsAromatic()),
                    float(atom.GetTotalNumHs()) / 4.0,
                    float(atom.IsInRing()),
                    is_donor,
                    is_acceptor,
                ]
            )
        )
        nfeat[-1] = torch.cat([nfeat[-1], _hybrid_three(atom)], dim=0)
    node_feat = torch.stack(nfeat, dim=0).to(torch.float32)

    edges = []
    efeat = []
    for bond in mol.GetBonds():
        s, d = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = _bond_one_hot(bond)
        edges.append((s, d))
        edges.append((d, s))
        efeat.append(bf)
        efeat.append(bf)

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_feat = torch.stack(efeat, dim=0).to(torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_feat = torch.zeros((0, 4), dtype=torch.float32)

    return LigandGraph(node_feat=node_feat, edge_index=edge_index, edge_feat=edge_feat)


def build_pair_graphs(pocket_id: str, ligand_smiles: str, pocket_structure_path: str = "") -> Tuple[RNAGraph, LigandGraph]:
    return (
        build_rna_graph(pocket_id=pocket_id, pocket_structure_path=pocket_structure_path),
        build_lig_graph(ligand_smiles=ligand_smiles),
    )


_RNA_TOKEN_CACHE: dict[tuple, torch.Tensor] = {}


def build_rna_tokens_from_pocket(
    pocket_id: str,
    pocket_structure_path: str,
    max_rna_tokens: int,
    dim: int,
) -> torch.Tensor:
    cache_key = (pocket_structure_path or pocket_id, max_rna_tokens, dim)
    if cache_key in _RNA_TOKEN_CACHE:
        return _RNA_TOKEN_CACHE[cache_key]

    nodes = _read_pocket_nodes(pocket_structure_path)
    if not nodes:
        result = token_features_from_text(f"rna::{pocket_id}", max_rna_tokens, dim)
        _RNA_TOKEN_CACHE[cache_key] = result
        return result

    rows = []
    denom = float(max(1, len(nodes) - 1))
    for i, node in enumerate(nodes):
        xyz = _extract_xyz(node)
        base = torch.cat(
            [
                xyz,
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

    result = _pad_or_trim(torch.stack(rows, dim=0), max_rna_tokens, dim)
    _RNA_TOKEN_CACHE[cache_key] = result
    return result


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


_LIG_TOKEN_CACHE: dict[tuple, torch.Tensor] = {}


def build_ligand_tokens(
    ligand_smiles: str,
    max_lig_tokens: int,
    dim: int,
) -> torch.Tensor:
    smiles = ligand_smiles or ""
    cache_key = (smiles, max_lig_tokens, dim)
    if cache_key in _LIG_TOKEN_CACHE:
        return _LIG_TOKEN_CACHE[cache_key]

    if Chem is None:
        result = token_features_from_text(f"lig::{smiles}", max_lig_tokens, dim)
        _LIG_TOKEN_CACHE[cache_key] = result
        return result

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        result = token_features_from_text(f"lig::{smiles}", max_lig_tokens, dim)
        _LIG_TOKEN_CACHE[cache_key] = result
        return result

    rows = []
    for atom in mol.GetAtoms():
        rows.append(torch.cat([_atom_features(atom), _hash_vec(f"{smiles}:{atom.GetIdx()}", 8)]))
    if not rows:
        result = token_features_from_text(f"lig::{smiles}", max_lig_tokens, dim)
        _LIG_TOKEN_CACHE[cache_key] = result
        return result

    result = _pad_or_trim(torch.stack(rows, dim=0), max_lig_tokens, dim)
    _LIG_TOKEN_CACHE[cache_key] = result
    return result


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


# ------------------------------------------------------------------
# Graph caches (for v3 graph-native training)
# ------------------------------------------------------------------

_RNA_GRAPH_CACHE: dict[str, "RNAGraph"] = {}
_LIG_GRAPH_CACHE: dict[str, "LigandGraph"] = {}


def build_rna_graph_cached(pocket_id: str, pocket_structure_path: str) -> "RNAGraph":
    """Return an RNAGraph, building and caching it on first call per path."""
    key = pocket_structure_path or pocket_id
    if key not in _RNA_GRAPH_CACHE:
        _RNA_GRAPH_CACHE[key] = build_rna_graph(pocket_id, pocket_structure_path)
    return _RNA_GRAPH_CACHE[key]


def build_lig_graph_cached(ligand_smiles: str) -> "LigandGraph":
    """Return a LigandGraph, building and caching it on first call per SMILES."""
    smiles = ligand_smiles or ""
    if smiles not in _LIG_GRAPH_CACHE:
        _LIG_GRAPH_CACHE[smiles] = build_lig_graph(smiles)
    return _LIG_GRAPH_CACHE[smiles]
