# researchlab

Phase 1 deliverables for the project “Cross-Species Transferability of Genomic Language Models for DNA Methylation Prediction”:

- Data assembly: build labeled, leakage-safe train/val/test datasets from a reference genome FASTA + methylation call tables.
- Baseline training: train a lightweight, reproducible baseline classifier on sequence windows and report standard metrics.

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

This downloads a TAIR10 reference genome FASTA from Ensembl Plants and a public ALLC methylation table from GEO (GSM2099378), then samples/filters calls and assembles `dataset.parquet` inside this repo.

```bash
python scripts/phase1_run_athaliana.py --outdir data/phase1_athaliana
python scripts/train_baseline.py \
  --dataset data/phase1_athaliana/processed/dataset.parquet \
  --outdir runs/baseline_athaliana
```

## Expected input formats

`scripts/assemble_dataset.py` supports:

- **Bismark CX report** (tab-separated; optionally gz): `chrom, pos, strand, count_methylated, count_unmethylated, context, trinucleotide`
- **ALLC TSV** (tab-separated; optionally gz): `chrom, pos, strand, mc_class, methylated_bases, total_bases`
- **Generic calls TSV** (tab-separated; optionally gz): must include `chrom`, `pos`, `strand`, `m`, `u` columns (methylated/unmethylated counts).
- **Unified processed CSV** (comma-separated): accepts `(chromosome, position)` or `(chrom, pos)` plus either `(coverage, methylation_ratio)` or `(m, u)`. If `sequence_context` is present it can be used when FASTA isn’t available.

Positions are assumed **1-based** (Bismark-style). The assembler orients windows so the center base is always `C`.
