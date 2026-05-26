#!/usr/bin/env bash
# ============================================================================
# BridgeBind3D v11 — v9 + Dual Contrastive Loss (RNA-side InfoNCE)
#
# 核心改动 vs v9:
#   ① loss.lambda_rna_contrast: 1.0
#      RNA-side InfoNCE: mini-batch 内对称 dual contrastive
#      lig_i 与 rna_i 相似度 > lig_i 与所有其他 rna_j 相似度
#      复用 z_rna / z_lig, 无额外 forward pass
#   ② 架构同 v9 (EGNN-6L + GIN-4L + DistBiasedSelfAttn)
#
# Submit all 5 folds:
#   sbatch scripts/slurm_train_hariboss5fd_rank_pkl_v11.sh
# ============================================================================
#SBATCH --job-name=bb3d_v11_rank
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80gb
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=/blue/qsong1/liangjialu/bb3d_v11_base_rank_fold%a_%j.out
#SBATCH --error=/blue/qsong1/liangjialu/bb3d_v11_base_rank_fold%a_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=liangjialu@ufl.edu
#SBATCH --account=qsong1

set -euo pipefail

FOLD=${SLURM_ARRAY_TASK_ID}
ROOT=/home/liangjialu/blue_qsong1/liangjialu/data/test_RNA/new/RNA3D-CLFM-v11-base
CONFIG=configs/hariboss5fd_rank_pkl_v11.yaml

echo "========================================"
echo " BridgeBind3D v11 x hariboss_5fd (rank)"
echo " v9 + Dual Contrastive Loss (RNA-side InfoNCE)"
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
export TORCH_HOME="/blue/qsong1/liangjialu/.torch_hub"

mkdir -p "${TORCH_HOME}/hub/checkpoints"
echo "[setup] TORCH_HOME=${TORCH_HOME}"

python -c "import torch; print(f'[torch] {torch.__version__} CUDA:{torch.cuda.is_available()} GPUs:{torch.cuda.device_count()}')"
python -c "from rdkit import Chem; print('[rdkit] OK')"

cd "${ROOT}"

DECOY_LOOKUP="${ROOT}/data/decoy_fp2_lookup.pkl"
DECOY_SMI="/home/liangjialu/blue_qsong1/liangjialu/data/test_RNA/new/SMARTBind/notebook/decoy_library.smi"
if [ ! -f "${DECOY_LOOKUP}" ]; then
    echo "[setup] Building FP2→SMILES lookup..."
    python scripts/build_decoy_lookup.py \
        --smi "${DECOY_SMI}" \
        --out "${DECOY_LOOKUP}"
fi

echo "[setup] Running dry-run..."
python -u scripts/train_hariboss5fd_rank_pkl.py \
    --config "${CONFIG}" \
    --fold "${FOLD}" \
    --dry-run

CKPT_DIR="/blue/qsong1/liangjialu/bb3d_v11_base/dual_contrast_v11_fold${FOLD}"
mkdir -p "${CKPT_DIR}"

RESUME_ARG=""
LATEST=$(ls -1 "${CKPT_DIR}"/ep*.pt 2>/dev/null | sort | tail -n 1 || true)
if [ -n "${LATEST}" ]; then
    echo "[resume] found checkpoint: ${LATEST}"
    RESUME_ARG="--resume ${LATEST}"
else
    echo "[fresh] starting from scratch"
fi

python -u scripts/train_hariboss5fd_rank_pkl.py \
    --config "${CONFIG}" \
    --fold "${FOLD}" \
    ${RESUME_ARG}

echo "========================================"
echo " v11 fold ${FOLD} done: $(date)"
echo "========================================"
