# 5% Data Pilot: Baseline Model Results (Arabidopsis Phase 1)

## Run configuration
- Dataset: `data/phase1_athaliana_5pct_fullscan/processed/dataset.parquet`
- Data assembly: `sample_fraction=0.05`, full-file scan, seed=7
- Model: Logistic Regression baseline (`train_baseline.py`)
- Training args: `solver=saga`, `ngram=3..5`, `max_features=20000`, `max_iter=30`
- Device used: CPU (`norm`); GPU was not available in this local shell
- Note: scikit-learn emitted a convergence warning (`max_iter` reached).

## Dataset summary
- Total rows: **197,710**
- Split counts: train=122,350, val=30,341, test=45,019
- Context counts: CG=15,730, CHG=27,730, CHH=154,250
- Positive rate (all rows): **0.035**
- Positive rate (test split): **0.035**

## Main metrics (test split)
- AUROC: **0.7069**
- AUPRC: **0.0878**
- MCC: **0.1111**
- F1: **0.1166**
- Accuracy: **0.6908**

## Confusion matrix at threshold 0.5 (test)
- TP=919, TN=30,180, FP=13,274, FN=646
- Precision=0.0648, Recall=0.5872

## Context-stratified test performance
| Context | Rows | Positive rate | AUROC | AUPRC | MCC | F1 | ACC |
|---|---:|---:|---:|---:|---:|---:|---:|
| CG | 3,659 | 0.043 | 0.6574 | 0.0985 | 0.0766 | 0.1096 | 0.5469 |
| CHG | 6,365 | 0.116 | 0.6738 | 0.2123 | 0.1515 | 0.2719 | 0.6196 |
| CHH | 34,995 | 0.019 | 0.6979 | 0.0504 | 0.0816 | 0.0691 | 0.7188 |

## Comparison vs earlier 1% run
- Earlier 1% test AUROC=0.5849, AUPRC=0.0546, MCC=0.0458, F1=0.0896, ACC=0.8757
- 5% pilot test AUROC=0.7069, AUPRC=0.0878, MCC=0.1111, F1=0.1166, ACC=0.6908
- Interpretation: 5% pilot gives substantially better ranking and balance-sensitive metrics than the earlier 1% baseline.

## Interpretation for supervisor discussion
- The dataset is strongly imbalanced (low positive rate), so AUPRC and MCC are more informative than accuracy.
- Overall AUROC (~0.707) indicates the model has signal, but AUPRC (~0.088) shows limited precision in positive-site retrieval at default threshold.
- Context-wise behavior differs: CG/CHG/CHH have different prevalences and separability, suggesting context-specific modeling or calibration may improve results.
- Since solver hit max iterations, this pilot is likely under-optimized; performance may improve with more iterations or tuned regularization.

## Recommended immediate next experiments
1. Refit LR with higher `max_iter` and threshold tuning on validation (optimize MCC/F1).
2. Train separate context-specific LR models (CG, CHG, CHH) and compare against unified model.
3. Run DNABERT2 and Nucleotide Transformer frozen-encoder heads on the same 5% dataset for direct comparison.

