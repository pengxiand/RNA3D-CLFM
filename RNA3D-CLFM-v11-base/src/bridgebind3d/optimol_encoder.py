"""Pure-PyTorch re-implementation of the OptiMol DGL-RGCN ligand encoder.

Requires only rdkit and standard PyTorch — no DGL dependency.

Architecture (from rnamigos2/pretrained/optimol/params.json)
────────────────────────────────────────────────────────────
  RGCN backbone:
    3 relation-GCN layers,  hidden_dim = 32,  num_rels = 4
    JumpingKnowledge: concat all layer outputs → [B, 96]
  encoder_mean: Linear(96 → 56)  → molecule-level 56-dim μ
  trainable projection: Linear(56 → out_dim) + LayerNorm(out_dim)

The backbone weights are loaded directly from weights.pth (old DGL key
naming: ``encoder.layers.X.weight``), so DGL is never imported.

Atom/bond featurisation replicates rnamigos2/rnamigos/learning/ligand_encoding.py:
  • Maps loaded from  rnamigos2/data/map_files/edges_and_nodes_map.pickle
  • Node features: one-hot(atomic_num) + one_hot(formal_charge)
                   + one_hot(num_explicit_hs) + one_hot(is_aromatic)
                   + one_hot(chiral_tag)   → 16-dim
  • At forward time the first 10 dims are used (cut_embeddings=True)
  • Edge type: edge_map[bond_type]  → integer 0-3
  • Directed graph (both directions per bond)
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

# ── Path resolution ────────────────────────────────────────────────────────────
_HERE           = Path(__file__).resolve().parent          # …/new/neq/src/bridgebind3d/
_PROJECT_ROOT   = _HERE.parents[1]                         # …/new/neq/
_WORKSPACE_ROOT = _HERE.parents[3]                         # …/test_RNA/

OPTIMOL_DIR = _WORKSPACE_ROOT / "rnamigos2" / "pretrained" / "optimol"

# The map file ships with rnamigos2 but may also be mirrored inside BridgeBind3D.
_MAP_CANDIDATES = [
    _WORKSPACE_ROOT / "rnamigos2" / "data" / "map_files" / "edges_and_nodes_map.pickle",
    _PROJECT_ROOT   / "data" / "data" / "map_files" / "edges_and_nodes_map.pickle",
    _PROJECT_ROOT   / "data" / "map_files" / "edges_and_nodes_map.pickle",
]
MAP_FILE: Path = next((p for p in _MAP_CANDIDATES if p.exists()), _MAP_CANDIDATES[0])


# ── Featuriser ─────────────────────────────────────────────────────────────────

class _OptiMolFeaturiser:
    """Converts a SMILES string to (node_feat, src, dst, edge_type) tensors.

    Replicates MolGraphEncoder from rnamigos2 without DGL.
    """

    def __init__(self, map_file: Path = MAP_FILE) -> None:
        with open(map_file, "rb") as fh:
            self.edge_map     = pickle.load(fh)   # bond_type → int
            self.at_map       = pickle.load(fh)   # atomic_num → int
            self.chi_map      = pickle.load(fh)   # used for hs / aromaticity / chirality
            self.charges_map  = pickle.load(fh)   # formal_charge → int

        self._n_at   = len(self.at_map)
        self._n_chg  = len(self.charges_map)
        self._n_chi  = len(self.chi_map)
        self._feat_dim = self._n_at + self._n_chg + 3 * self._n_chi  # must equal 16

    def _oh(self, val: int, mapping: dict, default: int) -> torch.Tensor:
        n  = len(mapping)
        t  = torch.zeros(n, dtype=torch.float32)
        idx = mapping.get(val, default)
        if 0 <= idx < n:
            t[idx] = 1.0
        return t

    def encode(self, smiles: str) -> Tuple[
        torch.Tensor,   # node_feat  [N, feat_dim]
        torch.Tensor,   # src        [E] long
        torch.Tensor,   # dst        [E] long
        torch.Tensor,   # edge_type  [E] long
    ]:
        try:
            from rdkit import Chem, RDLogger
        except ImportError as e:
            raise ImportError("rdkit is required for OptiMolLigandEncoder featurisation") from e

        RDLogger.DisableLog('rdApp.*')   # suppress valence / sanitization warnings
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

        # ── Node features ──────────────────────────────────────────
        node_feats: List[torch.Tensor] = []
        for atom in mol.GetAtoms():
            f = torch.cat([
                self._oh(atom.GetAtomicNum(),        self.at_map,       6),
                self._oh(atom.GetFormalCharge(),     self.charges_map,  0),
                self._oh(atom.GetNumExplicitHs(),    self.chi_map,      0),
                self._oh(int(atom.GetIsAromatic()),  self.chi_map,      0),
                self._oh(int(atom.GetChiralTag()),   self.chi_map,      0),
            ])
            node_feats.append(f)
        node_feat = torch.stack(node_feats)   # [N, feat_dim]

        # ── Edges (directed: both directions per bond) ──────────────
        srcs, dsts, etypes = [], [], []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bt = bond.GetBondType()
            et = self.edge_map.get(bt, 0)
            srcs.extend([i, j])
            dsts.extend([j, i])
            etypes.extend([et, et])

        if len(srcs) == 0:
            # Single-atom molecule: add self-loop so RGCN message-pass is a no-op
            srcs  = [0]
            dsts  = [0]
            etypes = [0]

        src_t   = torch.tensor(srcs,   dtype=torch.long)
        dst_t   = torch.tensor(dsts,   dtype=torch.long)
        etype_t = torch.tensor(etypes, dtype=torch.long)
        return node_feat, src_t, dst_t, etype_t


# ── Pure-PyTorch RGCN layer ───────────────────────────────────────────────────

class _RGCNLayer(nn.Module):
    """Relation-specific GCN layer (sum aggregation, no self-loop, ReLU).

    Parameter ``weight`` has shape [num_rels, in_feat, out_feat], matching
    the key names in OptiMol's weights.pth.
    """

    def __init__(self, in_feat: int, out_feat: int, num_rels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_rels, in_feat, out_feat))
        self.h_bias = nn.Parameter(torch.zeros(out_feat))   # matches DGL RelGraphConv h_bias
        nn.init.xavier_uniform_(self.weight.view(num_rels, -1))

    def forward(
        self,
        h:         torch.Tensor,   # [N, in_feat]
        src:       torch.Tensor,   # [E] long
        dst:       torch.Tensor,   # [E] long
        edge_type: torch.Tensor,   # [E] long
    ) -> torch.Tensor:             # [N, out_feat]
        h_src = h[src]                                           # [E, in_feat]
        W     = self.weight[edge_type]                           # [E, in_feat, out_feat]
        msg   = torch.bmm(h_src.unsqueeze(1), W).squeeze(1)     # [E, out_feat]
        out   = torch.zeros(h.shape[0], self.weight.shape[2],
                            device=h.device, dtype=h.dtype)
        out.index_add_(0, dst, msg)
        return torch.relu(out + self.h_bias)


# ── OptiMol backbone ──────────────────────────────────────────────────────────

class _OptiMolBackbone(nn.Module):
    """Three-layer RGCN with JumpingKnowledge → encoder_mean.

    Output: [B, l_size=56] molecule-level embeddings.
    """

    # Default architectural hyper-parameters (from params.json)
    FEATURES_DIM = 16
    GCN_HDIM     = 32
    GCN_LAYERS   = 3
    NUM_RELS     = 4
    L_SIZE       = 56

    def __init__(self) -> None:
        super().__init__()
        dims = [self.FEATURES_DIM] + [self.GCN_HDIM] * self.GCN_LAYERS
        self.layers = nn.ModuleList(
            [_RGCNLayer(dims[i], dims[i + 1], self.NUM_RELS)
             for i in range(self.GCN_LAYERS)]
        )
        self.encoder_mean = nn.Linear(self.GCN_HDIM * self.GCN_LAYERS, self.L_SIZE)

    def forward(
        self,
        node_feat:  torch.Tensor,   # [N_total, FEATURES_DIM]
        src:        torch.Tensor,   # [E_total] long
        dst:        torch.Tensor,   # [E_total] long
        edge_type:  torch.Tensor,   # [E_total] long
        batch_idx:  torch.Tensor,   # [N_total] long  (0-based graph id per node)
    ) -> torch.Tensor:              # [B, L_SIZE]
        # cut_embeddings=True: discard last 6 atom features (use first 10)
        h = node_feat[:, :-6].float()   # [N, 10]

        sequence: List[torch.Tensor] = []
        for layer in self.layers:
            h = layer(h, src, dst, edge_type)
            sequence.append(h)

        # JumpingKnowledge: concatenate all layer outputs
        h = torch.cat(sequence, dim=-1)              # [N, GCN_HDIM * GCN_LAYERS]

        # Sum-pooling per graph
        B      = int(batch_idx.max().item()) + 1
        pooled = torch.zeros(B, h.shape[-1], device=h.device, dtype=h.dtype)
        pooled.index_add_(0, batch_idx, h)           # [B, GCN_HDIM * GCN_LAYERS]

        return self.encoder_mean(pooled)             # [B, L_SIZE]

    def load_optimol_weights(self, weights_path: Path) -> None:
        """Load directly from OptiMol's weights.pth (old DGL key naming).

        Key remapping:
          ``encoder.layers.X.weight`` → ``layers.X.weight``
          ``encoder_mean.weight``     → ``encoder_mean.weight``  (kept)
          ``encoder_mean.bias``       → ``encoder_mean.bias``    (kept)
          ``encoder_logv.*``          → ignored (not used)
        """
        raw = torch.load(weights_path, map_location="cpu")
        state: dict[str, torch.Tensor] = {}
        for k, v in raw.items():
            if k.startswith("encoder.layers."):
                # e.g. "encoder.layers.0.weight" → "layers.0.weight"
                new_k = k[len("encoder."):]
                state[new_k] = v
            elif k in ("encoder_mean.weight", "encoder_mean.bias"):
                state[k] = v
            # all other keys (encoder_logv, decoder, etc.) are ignored
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"[OptiMol] Unexpected keys in state dict: {unexpected}")
        # "encoder_mean.*" legitimately present; "layers.*" are the RGCN weights.
        expected_keys = {f"layers.{i}.weight" for i in range(self.GCN_LAYERS)} | \
                        {f"layers.{i}.h_bias" for i in range(self.GCN_LAYERS)} | \
                        {"encoder_mean.weight", "encoder_mean.bias"}
        loaded_keys   = set(state.keys())
        if not expected_keys.issubset(loaded_keys):
            raise RuntimeError(
                f"[OptiMol] Missing expected keys: {expected_keys - loaded_keys}"
            )


# ── Public module ─────────────────────────────────────────────────────────────

class OptiMolLigandEncoder(nn.Module):
    """Frozen pretrained OptiMol RGCN + trainable projection head.

    Args:
        out_dim:     Output embedding dimension (matched to model embed_dim).
        freeze:      If True, freeze all backbone weights (default True).
        optimol_dir: Override path to the pretrained directory.
        map_file:    Override path to the edges_and_nodes_map.pickle.
    """

    OPTIMOL_DIM = _OptiMolBackbone.L_SIZE  # 56

    def __init__(
        self,
        out_dim: int,
        freeze: bool = True,
        optimol_dir: Path | str | None = None,
        map_file: Path | str | None = None,
    ) -> None:
        super().__init__()
        _opt_dir  = Path(optimol_dir) if optimol_dir else OPTIMOL_DIR
        _map_file = Path(map_file)    if map_file    else MAP_FILE

        # ── Backbone (pretrained, optionally frozen) ─────────────────
        self.backbone = _OptiMolBackbone()
        self.backbone.load_optimol_weights(_opt_dir / "weights.pth")
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        self._freeze = freeze

        # ── Featuriser (stateless after init) ────────────────────────
        self._featuriser = _OptiMolFeaturiser(_map_file)

        # ── Trainable projection ──────────────────────────────────────
        self.proj = nn.Sequential(
            nn.Linear(self.OPTIMOL_DIM, out_dim),
            nn.LayerNorm(out_dim),
        )

    # ─────────────────────────────────────────────────────────────────
    def _featurise_batch(
        self,
        smiles_list: Sequence[str],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert a list of SMILES to batched flat tensors.

        Returns (node_feat, src, dst, edge_type, batch_idx) all on *device*.
        Invalid SMILES are replaced by a single dummy node (zero features).
        """
        all_nf, all_src, all_dst, all_et, all_bi = [], [], [], [], []
        offset = 0
        for b_idx, smi in enumerate(smiles_list):
            try:
                nf, src, dst, et = self._featuriser.encode(smi)
            except Exception:
                # Fallback: single zero-feature node, no edges
                nf  = torch.zeros(1, self._featuriser._feat_dim, dtype=torch.float32)
                src = torch.tensor([0], dtype=torch.long)
                dst = torch.tensor([0], dtype=torch.long)
                et  = torch.tensor([0], dtype=torch.long)
            N = nf.shape[0]
            all_nf.append(nf)
            all_src.append(src + offset)
            all_dst.append(dst + offset)
            all_et.append(et)
            all_bi.append(torch.full((N,), b_idx, dtype=torch.long))
            offset += N

        node_feat  = torch.cat(all_nf,  dim=0).to(device)
        src_t      = torch.cat(all_src, dim=0).to(device)
        dst_t      = torch.cat(all_dst, dim=0).to(device)
        edge_type  = torch.cat(all_et,  dim=0).to(device)
        batch_idx  = torch.cat(all_bi,  dim=0).to(device)
        return node_feat, src_t, dst_t, edge_type, batch_idx

    # ─────────────────────────────────────────────────────────────────
    def forward(self, smiles_list: Sequence[str]) -> torch.Tensor:
        """
        Args:
            smiles_list: B SMILES strings.

        Returns:
            Tensor of shape [B, out_dim] on the same device as self.proj.
        """
        device = next(self.proj.parameters()).device
        node_feat, src, dst, edge_type, batch_idx = self._featurise_batch(
            smiles_list, device
        )

        if self._freeze:
            with torch.no_grad():
                mu = self.backbone(node_feat, src, dst, edge_type, batch_idx)
        else:
            mu = self.backbone(node_feat, src, dst, edge_type, batch_idx)

        return self.proj(mu)    # [B, out_dim]
