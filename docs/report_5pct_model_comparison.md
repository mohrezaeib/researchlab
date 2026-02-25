# 5% Pilot Model Comparison (Saved Runs + Analysis)

## Objective
- Compare baseline LR vs transformer-family models on the 5% Arabidopsis Phase-1 dataset.
- Save all outputs for supervisor discussion and identify what is blocking full DNABERT2/NT-v2 runs.

## Saved output files
- Full 5% LR run: `runs/baseline_athaliana_5pct_saga/{metrics.json,run_manifest.json,analysis_5pct.json}`
- Matched debug dataset (256 rows per split): `data/phase1_athaliana_5pct_fullscan/processed/dataset.debug256.parquet`
- LR debug run: `runs/lr_5pct_debug256/{metrics.json,run_manifest.json}`
- DNABERT-family debug run: `runs/dnabert6_5pct_debug256/{metrics.json,run_manifest.json,model/,classifier_head.pt}`
- Nucleotide Transformer debug run: `runs/nt500m_5pct_debug256/{metrics.json,run_manifest.json,model/,classifier_head.pt}`
- Comparison CSVs: `runs/model_comparison_5pct_debug256.csv`, `runs/model_comparison_5pct_all.csv`

## Main results
### A) Full 5% baseline (primary reliable benchmark)
- Test rows: 45,019
- AUROC: 0.7069
- AUPRC: 0.0878
- MCC: 0.1111
- F1: 0.1166
- ACC: 0.6908

### B) Matched debug comparison (256 test rows each)
| Model | AUROC | AUPRC | MCC | F1 | ACC |
|---|---:|---:|---:|---:|---:|
| lr_debug | 0.4163 | 0.0280 | 0.0000 | 0.0000 | 0.9688 |
| dnabert_family_debug | 0.5000 | 0.0312 | 0.0000 | 0.0000 | 0.9688 |
| nucleotide_transformer_debug | 0.5852 | 0.0428 | 0.0000 | 0.0606 | 0.0312 |

## Interpretation
- Full 5% LR run is currently the only result at meaningful sample size; it shows clear predictive signal (AUROC ~0.71) but low precision-recall due strong class imbalance.
- Debug transformer runs are not yet decision-grade for model ranking because they were intentionally tiny and 1-epoch quick checks.
- In this tiny setting, all three models are unstable at threshold 0.5 (MCC near 0), which is expected for severe imbalance and minimal tuning.
- Nucleotide-transformer debug has better ranking (AUROC/AUPRC) than debug LR and debug DNABERT-family, but poor threshold calibration (very low accuracy at 0.5).

## Important blocker: exact DNABERT2 + NT-v2 checkpoints
- `zhihan1996/DNABERT-2-117M` failed in this shell due upstream custom-code/config-class incompatibility during `AutoModel.from_pretrained` (remote `BertConfig` mismatch).
- `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species` failed due custom `EsmConfig` not recognized by `AutoModel` under current environment combination.
- Workaround used here for progress: compatible proxies (`zhihan1996/DNA_bert_6` and `InstaDeepAI/nucleotide-transformer-500m-human-ref`).

## Supervisor-ready conclusions
1. Baseline on 5% data is working end-to-end and shows moderate ranking signal; imbalance remains the core challenge.
2. Transformer training infrastructure works and writes reproducible artifacts, but exact target checkpoints need environment pinning on cluster (likely dedicated conda env and known-good transformers/torch versions).
3. Immediate next step is not more local probing; it is reproducible cluster jobs with pinned versions and longer training on full 5% split.

## Recommended next run plan (cluster)
1. Create a dedicated env for genomic transformers (separate from baseline env).
2. Pin versions after compatibility test with target checkpoints.
3. Run frozen-head first, then top-layer fine-tuning, then compare via `scripts/compare_runs.py`.

## Update: Full GPU Context Matrix (15 runs)
- Completed run matrix and saved consolidated analysis:
  - `docs/report_5pct_context_matrix_gpu.md`
  - `runs/context_matrix_5pct_gpu/context_matrix_summary.csv`
  - `runs/context_matrix_5pct_gpu/context_matrix_summary.md`
