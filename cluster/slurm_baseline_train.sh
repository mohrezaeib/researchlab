#!/bin/bash
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --job-name=baseline_train
#SBATCH --output=slurm-baseline-train-%j.out

set -euo pipefail

module load anaconda/3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate researchlab

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/researchlab}"
DATASET="${DATASET:-$PROJECT_ROOT/data/phase1_athaliana/processed/dataset.parquet}"
OUTDIR="${OUTDIR:-$PROJECT_ROOT/runs/baseline_athaliana}"

cd "$PROJECT_ROOT"

python scripts/train_baseline.py \
  --dataset "$DATASET" \
  --outdir "$OUTDIR"
