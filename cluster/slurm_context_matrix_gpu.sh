#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48GB
#SBATCH --time=24:00:00
#SBATCH --job-name=ctx_matrix
#SBATCH --output=slurm-ctx-matrix-%j.out

set -euo pipefail

module load anaconda/3
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi
CONDA_ENV="${CONDA_ENV:-researchlab}"
conda activate "$CONDA_ENV"

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/researchlab}"
DATASET="${DATASET:-$PROJECT_ROOT/data/phase1_athaliana_5pct_w800/processed/dataset.parquet}"
MODEL_NAME="${MODEL_NAME:-zhihan1996/DNA_bert_6}"
OUTROOT="${OUTROOT:-$PROJECT_ROOT/runs/context_matrix_5pct}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-512}"
DEBUG_MAX_ROWS="${DEBUG_MAX_ROWS:-128}"
TRAIN_MODE="${TRAIN_MODE:-frozen}"

cd "$PROJECT_ROOT"
mkdir -p "$OUTROOT"

for mode in upstream downstream both; do
  for bp in 50 100 200 300 400; do
    outdir="$OUTROOT/${mode}_${bp}"
    echo "=== Running mode=$mode bp=$bp out=$outdir ==="
    python scripts/train_transformer.py \
      --dataset "$DATASET" \
      --outdir "$outdir" \
      --model-name "$MODEL_NAME" \
      --train-mode "$TRAIN_MODE" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --max-length "$MAX_LENGTH" \
      --debug-max-rows "$DEBUG_MAX_ROWS" \
      --context-bp "$bp" \
      --context-mode "$mode" \
      --device auto
  done
done

python scripts/analyze_context_runs.py \
  --root "$OUTROOT" \
  --out-csv "$OUTROOT/context_matrix_summary.csv" \
  --out-md "$OUTROOT/context_matrix_summary.md"
