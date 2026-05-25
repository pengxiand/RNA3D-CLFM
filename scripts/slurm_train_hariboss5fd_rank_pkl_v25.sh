#!/usr/bin/env bash
# ============================================================================
# BridgeBind3D v25 — Eval-aligned cross-heavy negatives (cross:hard ≈ 5.7:1)
# Dataset: hariboss_5fd_cov08_clustered_seqid_0.3_clean.pkl
#
# Key changes vs v24: K_s2 36→48, cross_ratio 0.75→0.85, γ 64→32, m 0.30→0.25, τ_end 0.07→0.15
#
# Submit all 5 folds:
#   sbatch --array=0-4 scripts/slurm_train_hariboss5fd_rank_pkl_v25.sh
# ============================================================================
#SBATCH --job-name=bb3d_v25
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=6:00:00
#SBATCH --output=/blue/qsong1/liangjialu/bb3d_v25_fold%a_%j.out
#SBATCH --error=/blue/qsong1/liangjialu/bb3d_v25_fold%a_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=liangjialu@ufl.edu
#SBATCH --account=qsong1

set -euo pipefail

FOLD=${SLURM_ARRAY_TASK_ID}
ROOT=/home/liangjialu/blue_qsong1/liangjialu/data/test_RNA/new/BridgeBind3D
CONFIG=configs/hariboss5fd_rank_pkl_v25.yaml

echo "========================================"
echo " BridgeBind3D v25 x hariboss_5fd (Cross-heavy + tamed Circle)"
echo " Fold      : ${FOLD} / 4"
echo " Job ID    : ${SLURM_JOB_ID}  Array: ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}]"
echo " Node      : $(hostname)"
echo " GPUs      : ${CUDA_VISIBLE_DEVICES:-unset}"
echo " Start     : $(date)"
echo "========================================"

module load conda
conda activate b200

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

python -c "import torch; print(f'[torch] {torch.__version__} CUDA:{torch.cuda.is_available()} GPUs:{torch.cuda.device_count()}')"
python -c "from rdkit import Chem; print('[rdkit] OK')"

cd "${ROOT}"

# ── Pre-flight: build FP2→SMILES lookup if not present ───────────────────────
DECOY_LOOKUP="${ROOT}/data/decoy_fp2_lookup.pkl"
DECOY_SMI="/home/liangjialu/blue_qsong1/liangjialu/data/test_RNA/new/SMARTBind/notebook/decoy_library.smi"
if [ ! -f "${DECOY_LOOKUP}" ]; then
    echo "[setup] Building FP2→SMILES lookup (one-time, may take ~5 min)..."
    python scripts/build_decoy_lookup.py \
        --smi "${DECOY_SMI}" \
        --out "${DECOY_LOOKUP}"
fi

# ── Dry-run sanity check ──────────────────────────────────────────────────────
python -u scripts/train_hariboss5fd_rank_pkl.py \
    --config "${CONFIG}" \
    --fold "${FOLD}" \
    --dry-run

CKPT_DIR="/blue/qsong1/liangjialu/bb3d_v25/crossheavy_g32_m25_fold${FOLD}"
mkdir -p "${CKPT_DIR}"

# Auto-resume from last checkpoint if present
RESUME_ARG=""
LATEST=$(ls -1 "${CKPT_DIR}"/ep*.pt 2>/dev/null | sort | tail -n 1 || true)
if [ -n "${LATEST}" ]; then
    echo "[resume] found checkpoint: ${LATEST}"
    RESUME_ARG="--resume ${LATEST}"
fi

python -u scripts/train_hariboss5fd_rank_pkl.py \
    --config "${CONFIG}" \
    --fold "${FOLD}" \
    ${RESUME_ARG}

echo "========================================"
echo " Fold ${FOLD} done: $(date)"
echo "========================================"
