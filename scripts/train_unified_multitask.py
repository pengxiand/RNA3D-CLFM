#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bridgebind3d.unified_training import train_unified


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train unified interaction backbone with ranking/docking/site heads")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "unified_multitask.yaml",
        help="Path to unified training config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_unified(PROJECT_ROOT, args.config)


if __name__ == "__main__":
    main()
