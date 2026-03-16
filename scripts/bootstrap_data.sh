#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EXTRA_ARGS=()
if [[ "${INCLUDE_GERNA:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--include-gerna --gerna-root ../GerNA-Bind)
fi

python scripts/prepare_data_bridge.py \
  --rnamigos-root ../rnamigos2 \
  --link-mode symlink \
  --build-pocket-node-manifest \
  --build-simulated-manifest \
  --build-ligand-decoy-manifest \
  "${EXTRA_ARGS[@]}"

echo "[OK] BridgeBind3D manifests generated."
