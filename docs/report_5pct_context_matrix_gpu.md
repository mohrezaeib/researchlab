# 5% GPU Context-Matrix Report (Transformer)

## Scope
- Dataset: `data/phase1_athaliana_5pct_w800/processed/dataset.parquet`
- Model family: `zhihan1996/DNA_bert_6` (frozen encoder, 1 epoch)
- Matrix: context size `{50,100,200,300,400}` x mode `{upstream,downstream,both}`
- Total runs: 15
- Outputs:
  - `runs/context_matrix_5pct_gpu/context_matrix_summary.csv`
  - `runs/context_matrix_5pct_gpu/context_matrix_summary.md`
  - per-run `metrics.json` + `predictions_val_test.npz`

## GPU / SLURM Status
- GPU path validated on SLURM node `dds8` (A40) with `torch 2.6.0+cu124`.
- Root issue fixed: SLURM scripts were activating cluster Anaconda env instead of `$HOME/miniconda3` env.
- Security compatibility fixed: upgraded from `torch 2.5.1` to `torch >=2.6` (required by current `transformers` safety check).

## Main Results

### Best single runs
- Best test AUROC: `downstream_100` = `0.5598`
- Best test AUPRC: `downstream_300` = `0.0310`
- Best test F1@0.5: `downstream_100` = `0.0529`

### Mean performance by direction mode
- `upstream`: AUROC `0.5298`, AUPRC `0.0268`, F1 `0.0471`
- `downstream`: AUROC `0.5060`, AUPRC `0.0274`, F1 `0.0458`
- `both`: AUROC `0.4950`, AUPRC `0.0244`, F1 `0.0452`

### Mean performance by context size (bp)
- `100bp` is strongest overall (mean AUROC `0.5446`, mean F1 `0.0500`).
- `300bp` is weakest overall (mean AUROC `0.4866`, mean F1 `0.0417`).
- Pattern suggests moderate context helps, wide context degrades signal-to-noise in this frozen-head setup.

## Comparison vs Baseline LR (full 5% run)
- Baseline LR (`runs/baseline_athaliana_5pct_saga/metrics.json`):
  - AUROC `0.7069`, AUPRC `0.0878`, F1 `0.1166`
- Best transformer context run in this matrix:
  - AUROC `0.5598`, AUPRC `0.0310`, F1 `0.0529`
- Conclusion: current frozen transformer setup underperforms the LR baseline by a large margin on all key metrics.

## Why F1 is low (diagnosis)

From saved test predictions across 15 runs:
- True positive rate is only `~2.36%` (high imbalance).
- Predicted positive rate at threshold `0.5` is `~58.33%` on average.
- Mean predicted probability:
  - positives: `~0.5105`
  - negatives: `~0.5092`

Interpretation:
- The model outputs are concentrated near `0.5` with minimal class separation.
- At threshold `0.5`, it predicts far too many positives, causing very low precision (`~2.40%`) and therefore low F1.
- Threshold tuning helps only marginally in most runs, indicating calibration is not the only issue; representation quality and/or trainability is the primary bottleneck.

## Remedies (prioritized)
1. Move from `frozen` to `top` fine-tuning first (`top-layers` 2-4), then test `full` with low LR.
2. Train longer than 1 epoch (at least 3-5) with early stopping on validation AUPRC.
3. Use imbalance-aware loss:
   - `BCEWithLogitsLoss(pos_weight=...)` or focal loss.
4. Tune decision threshold on validation split (report both rank metrics and thresholded metrics).
5. Keep context near `100bp` first; avoid `300-400bp` until fine-tuning is stable.
6. Add context-stratified analysis (CG/CHG/CHH) for transformer runs, matching baseline analysis style.

## Recommendation for next training cycle
- Use `100bp` and `upstream/downstream` (not `both`) as primary candidates.
- Run a staged experiment set:
  1. `frozen` (3 epochs) as control
  2. `top` (3 epochs, top 2-4 layers)
  3. `full` (3 epochs, lower LR)
- Select best by validation AUPRC, then evaluate test metrics and threshold-optimized F1.

