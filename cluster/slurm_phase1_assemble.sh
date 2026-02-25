#!/bin/bash
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --job-name=phase1_assemble
#SBATCH --output=slurm-phase1-assemble-%j.out

set -euo pipefail

module load anaconda/3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate researchlab

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/researchlab}"
OUTDIR="${OUTDIR:-$PROJECT_ROOT/data/phase1_athaliana}"
FASTA="${FASTA:-$OUTDIR/raw/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa}"
ALLC_GZ="${ALLC_GZ:-$OUTDIR/raw/GSM2099378_allc_9386.tsv.gz}"

cd "$PROJECT_ROOT"

python scripts/phase1_run_athaliana.py \
  --assemble-only \
  --outdir "$OUTDIR" \
  --fasta "$FASTA" \
  --allc-gz "$ALLC_GZ" \
  --sample-fraction 0.01 \
  --max-rows 200000
