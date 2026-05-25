"""Stub: OptiMol pretrained encoder. Only needed when use_pretrained_ligand=True."""
import torch.nn as nn


class OptiMolLigandEncoder(nn.Module):
    def __init__(self, out_dim=256, freeze=True, optimol_dir=None, map_file=None):
        super().__init__()
        raise RuntimeError(
            "OptiMol encoder not available. Set use_pretrained_ligand: false in config."
        )

    def forward(self, smiles_list):
        pass
