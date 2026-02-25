# researchlab

Phase 1 deliverables for the project “Cross-Species Transferability of Genomic Language Models for DNA Methylation Prediction”:

- Data assembly: build labeled, leakage-safe train/val/test datasets from a reference genome FASTA + methylation call tables.
- Baseline training: train a lightweight, reproducible baseline classifier on sequence windows and report standard metrics.

## Setup (Headnode)

```bash
module load anaconda/3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n researchlab python=3.11 -y
conda activate researchlab
pip install -r requirements.txt
```

## Quickstart (synthetic smoke test)

```bash
python scripts/make_synthetic_example.py --outdir data/example
python scripts/assemble_dataset.py \
  --fasta data/example/reference.fa \
  --calls data/example/methylation_calls.tsv \
  --out data/example/dataset.parquet \
  --window 200 \
  --min-cov 10 \
  --ratio-thresh 0.5 \
  --split-spec data/example/splits.json
python scripts/train_baseline.py \
  --dataset data/example/dataset.parquet \
  --outdir runs/baseline_example
```

## Phase 1 (real public Arabidopsis data download)

This pipeline supports three modes:
- default: download + assemble
- `--download-only`: headnode pre-download and manifest generation
- `--assemble-only`: offline assembly from local pre-downloaded files (SLURM-safe)

```bash
python scripts/phase1_run_athaliana.py --outdir data/phase1_athaliana
python scripts/phase1_run_athaliana.py \
  --download-only \
  --outdir data/phase1_athaliana
python scripts/phase1_run_athaliana.py \
  --assemble-only \
  --outdir data/phase1_athaliana \
  --fasta data/phase1_athaliana/raw/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa \
  --allc-gz data/phase1_athaliana/raw/GSM2099378_allc_9386.tsv.gz
python scripts/train_baseline.py \
  --dataset data/phase1_athaliana/processed/dataset.parquet \
  --outdir runs/baseline_athaliana
```

## Canonical artifacts

Assembly output (`data/phase1_athaliana/processed/`):
- `downloads.summary.json`
- `calls.sampled.tsv`
- `calls.sampled.summary.json`
- `splits.json`
- `dataset.parquet`
- `dataset.summary.json`

Baseline output (`runs/baseline_athaliana/`):
- `model.joblib`
- `metrics.json`
- `run_manifest.json`

## Cluster execution

See:
- `cluster/slurm_phase1_assemble.sh`
- `cluster/slurm_baseline_train.sh`
- `cluster/slurm_transformer_train_gpu.sh`
- `cluster/RUNBOOK.md`

## Transformer model training (GPU-ready)

Train a HuggingFace encoder + binary classifier head on the assembled dataset:

```bash
python scripts/train_transformer.py \
  --dataset data/phase1_athaliana/processed/dataset.parquet \
  --outdir runs/dnabert2_athaliana \
  --model-name zhihan1996/DNABERT-2-117M \
  --train-mode frozen \
  --epochs 3 \
  --batch-size 32 \
  --max-length 256 \
  --device auto
```

Available `--train-mode` values:
- `frozen` (encoder frozen, train only classifier head)
- `top` (unfreeze top N encoder layers + head)
- `full` (full fine-tuning)

Compare runs:

```bash
python scripts/compare_runs.py \
  --run lr:runs/baseline_athaliana \
  --run dnabert2:runs/dnabert2_athaliana \
  --run nt:runs/nucleotide_transformer_athaliana
```

## Expected input formats

`scripts/assemble_dataset.py` supports:

- **Bismark CX report** (tab-separated; optionally gz): `chrom, pos, strand, count_methylated, count_unmethylated, context, trinucleotide`
- **ALLC TSV** (tab-separated; optionally gz): `chrom, pos, strand, mc_class, methylated_bases, total_bases`
- **Generic calls TSV** (tab-separated; optionally gz): must include `chrom`, `pos`, `strand`, `m`, `u` columns (methylated/unmethylated counts).
- **Unified processed CSV** (comma-separated): accepts `(chromosome, position)` or `(chrom, pos)` plus either `(coverage, methylation_ratio)` or `(m, u)`. If `sequence_context` is present it can be used when FASTA isn’t available.

Positions are assumed **1-based** (Bismark-style). The assembler orients windows so the center base is always `C`.
