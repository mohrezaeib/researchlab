# Cluster Runbook (SCIBIOME SLURM)

This runbook uses the repository defaults:
- Phase 1 output: `data/phase1_athaliana`
- Baseline output: `runs/baseline_athaliana`
- Transformer output example: `runs/transformer_athaliana`

## 1. Setup on Headnode

```bash
module load anaconda/3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n researchlab python=3.11 -y
conda activate researchlab
pip install -r requirements.txt
```

## 2. Pre-download data on Headnode (internet required)

Do not do this in SLURM jobs.

```bash
python scripts/phase1_run_athaliana.py \
  --download-only \
  --outdir data/phase1_athaliana
```

Expected metadata:
- `data/phase1_athaliana/processed/downloads.summary.json`

## 3. Submit assembly SLURM job (no internet needed)

```bash
sbatch cluster/slurm_phase1_assemble.sh
```

Expected artifacts:
- `data/phase1_athaliana/processed/calls.sampled.tsv`
- `data/phase1_athaliana/processed/calls.sampled.summary.json`
- `data/phase1_athaliana/processed/splits.json`
- `data/phase1_athaliana/processed/dataset.parquet`
- `data/phase1_athaliana/processed/dataset.summary.json`

## 4. Submit baseline training SLURM job

```bash
sbatch cluster/slurm_baseline_train.sh
```

Expected artifacts:
- `runs/baseline_athaliana/model.joblib`
- `runs/baseline_athaliana/metrics.json`
- `runs/baseline_athaliana/run_manifest.json`

## 5. Debug transformer training on a small subset (recommended first)

On headnode or small interactive test:

```bash
python scripts/train_transformer.py \
  --dataset data/phase1_athaliana/processed/dataset.parquet \
  --outdir runs/transformer_debug \
  --model-name hf-internal-testing/tiny-random-BertModel \
  --train-mode frozen \
  --epochs 1 \
  --batch-size 16 \
  --max-length 256 \
  --debug-max-rows 256
```

Then run a GPU job with a real genomic model (examples):
- `zhihan1996/DNABERT-2-117M`
- `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`

```bash
MODEL_NAME=zhihan1996/DNABERT-2-117M \
OUTDIR=$HOME/researchlab/runs/dnabert2_athaliana \
TRAIN_MODE=frozen \
sbatch cluster/slurm_transformer_train_gpu.sh
```

## 6. Monitor jobs

```bash
squeue -u "$USER"
tail -f slurm-phase1-assemble-<JOBID>.out
tail -f slurm-baseline-train-<JOBID>.out
tail -f slurm-transformer-train-<JOBID>.out
```

## 7. GPU troubleshooting and context-matrix experiments

GPU smoke test:

```bash
sbatch cluster/slurm_gpu_smoketest.sh
```

If this fails, check:
- You are on headnode (`ssh 134.169.129.163`) and submitting with cluster credentials.
- `module load anaconda/3` works inside the SLURM script.
- `conda activate researchlab` points to an env with `torch` installed.

Context window sweep (50/100/200/300/400; upstream/downstream/both):

```bash
OUTROOT=$HOME/researchlab/runs/context_matrix_5pct \
DATASET=$HOME/researchlab/data/phase1_athaliana_5pct_w800/processed/dataset.parquet \
MODEL_NAME=zhihan1996/DNA_bert_6 \
sbatch cluster/slurm_context_matrix_gpu.sh
```

Auto-generated outputs:
- `$OUTROOT/context_matrix_summary.csv`
- `$OUTROOT/context_matrix_summary.md`

## 8. Compare model runs

```bash
python scripts/compare_runs.py \
  --run lr:runs/baseline_athaliana \
  --run dnabert2:runs/dnabert2_athaliana \
  --run nt:runs/nucleotide_transformer_athaliana \
  --out-csv runs/model_comparison.csv
```

## 9. Repeatability check

Run assembly twice with the same seed and parameters, then compare:

```bash
jq .dataset_sha256 data/phase1_athaliana/processed/dataset.summary.json
```

Checksums should match if inputs and parameters are unchanged.
