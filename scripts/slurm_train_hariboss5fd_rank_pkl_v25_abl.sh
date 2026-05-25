#!/usr/bin/env bash
# ============================================================================
# BridgeBind3D v25 ABLATION runner — cross/hard ratio sweep at fixed K
#
# Usage (always fold 0 unless overridden):
#   sbatch scripts/slurm_train_hariboss5fd_rank_pkl_v25_abl.sh ablA_pureCross
#   sbatch scripts/slurm_train_hariboss5fd_rank_pkl_v25_abl.sh ablB_heavyHard
#   sbatch scripts/slurm_train_hariboss5fd_rank_pkl_v25_abl.sh ablC_recommended
#
# Or submit all three at once:
#   for v in ablA_pureCross ablB_heavyHard ablC_recommended; do
#       sbatch scripts/slurm_train_hariboss5fd_rank_pkl_v25_abl.sh $v
#   done
# ============================================================================
#SBATCH --job-name=bb3d_v25_abl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=4:00:00
#SBATCH --output=/blue/qsong1/liangjialu/bb3d_v25_abl_%x_%j.out
#SBATCH --error=/blue/qsong1/liangjialu/bb3d_v25_abl_%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=liangjialu@ufl.edu
#SBATCH --account=qsong1

set -euo pipefail

VARIANT=${1:?"usage: sbatch $0 <ablA_pureCross|ablB_heavyHard|ablC_recommended>"}
FOLD=${FOLD:-0}
ROOT=/home/liangjialu/blue_qsong1/liangjialu/data/test_RNA/new/BridgeBind3D
CONFIG="configs/hariboss5fd_rank_pkl_v25_${VARIANT}.yaml"

if [ ! -f "${ROOT}/${CONFIG}" ]; then
    echo "ERROR: config not found: ${CONFIG}"
    exit 1
fi

echo "========================================"
echo " BridgeBind3D v25 ABLATION — ${VARIANT}"
echo " Fold      : ${FOLD}"
echo " Config    : ${CONFIG}"
echo " Job ID    : ${SLURM_JOB_ID}"
echo " Node      : $(hostname)"
echo " Start     : $(date)"
echo "========================================"

module load conda
conda activate b200
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

cd "${ROOT}"

# Sanity
python -c "import torch; print(f'[torch] {torch.__version__} CUDA:{torch.cuda.is_available()}')"

DECOY_LOOKUP="${ROOT}/data/decoy_fp2_lookup.pkl"
DECOY_SMI="/home/liangjialu/blue_qsong1/liangjialu/data/test_RNA/new/SMARTBind/notebook/decoy_library.smi"
if [ ! -f "${DECOY_LOOKUP}" ]; then
    echo "[setup] Building FP2→SMILES lookup..."
    python scripts/build_decoy_lookup.py --smi "${DECOY_SMI}" --out "${DECOY_LOOKUP}"
fi

python -u scripts/train_hariboss5fd_rank_pkl.py --config "${CONFIG}" --fold "${FOLD}" --dry-run

CKPT_DIR="/blue/qsong1/liangjialu/bb3d_v25_abl/${VARIANT}_fold${FOLD}"
mkdir -p "${CKPT_DIR}"

RESUME_ARG=""
LATEST=$(ls -1 "${CKPT_DIR}"/ep*.pt 2>/dev/null | sort | tail -n 1 || true)
if [ -n "${LATEST}" ]; then
    echo "[resume] found: ${LATEST}"
    RESUME_ARG="--resume ${LATEST}"
fi

python -u scripts/train_hariboss5fd_rank_pkl.py \
    --config "${CONFIG}" \
    --fold "${FOLD}" \
    ${RESUME_ARG}

echo "========================================"
echo " ${VARIANT} fold ${FOLD} done: $(date)"
echo "========================================"
