#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48GB
#SBATCH --time=48:00:00
#SBATCH --job-name=transformer_train
#SBATCH --output=slurm-transformer-train-%j.out

set -euo pipefail

module load anaconda/3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate researchlab

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/researchlab}"
DATASET="${DATASET:-$PROJECT_ROOT/data/phase1_athaliana/processed/dataset.parquet}"
MODEL_NAME="${MODEL_NAME:-zhihan1996/DNABERT-2-117M}"
OUTDIR="${OUTDIR:-$PROJECT_ROOT/runs/transformer_athaliana}"
TRAIN_MODE="${TRAIN_MODE:-frozen}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_LENGTH="${MAX_LENGTH:-256}"

cd "$PROJECT_ROOT"

python scripts/train_transformer.py \
  --dataset "$DATASET" \
  --outdir "$OUTDIR" \
  --model-name "$MODEL_NAME" \
  --train-mode "$TRAIN_MODE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --max-length "$MAX_LENGTH" \
  --device auto
