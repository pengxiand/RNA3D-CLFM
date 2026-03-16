from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BridgeSample:
    source: str
    source_dataset: str
    task: str
    target_type: str
    target_id: str
    target_sequence: str
    target_structure_path: str
    pocket_id: str
    ligand_smiles: str
    label_type: str
    label_value: float
    split: str
    extra: str
