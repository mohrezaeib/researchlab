# Next Experiments After Baseline

## Scope for Next Cycle

This cycle stays with the existing baseline pipeline and prepares experiment definitions for the next implementation cycle.

Included:
- Within-species and cross-species evaluation matrix definition
- Context-stratified reporting schema (CG, CHG, CHH)
- Standard result table format for AUROC, AUPRC, MCC, F1, ACC

Not included yet:
- Transformer model implementation and fine-tuning execution

## Experiment Matrix (Placeholders)

| Experiment ID | Train Species | Test Species | Setting | Model | Notes |
|---|---|---|---|---|---|
| E1 | A. thaliana | A. thaliana | within-species | baseline n-gram LR | reference baseline |
| E2 | B. rapa | B. rapa | within-species | baseline n-gram LR | optional once B. rapa dataset exists |
| E3 | A. thaliana | B. rapa | cross-species | baseline n-gram LR | transfer direction 1 |
| E4 | B. rapa | A. thaliana | cross-species | baseline n-gram LR | transfer direction 2 |

## Reporting Schema

### Per-split metrics
- `split`: train | val | test
- `rows`
- `auroc`
- `auprc`
- `mcc`
- `f1`
- `acc`

### Context-stratified metrics
- `context`: CG | CHG | CHH
- `split`
- `rows`
- `auroc`
- `auprc`
- `mcc`
- `f1`
- `acc`

## Evaluation Table Template

| Experiment ID | Split | Context | Rows | AUROC | AUPRC | MCC | F1 | ACC |
|---|---|---|---:|---:|---:|---:|---:|---:|
| E1 | test | all |  |  |  |  |  |  |
| E1 | test | CG |  |  |  |  |  |  |
| E1 | test | CHG |  |  |  |  |  |  |
| E1 | test | CHH |  |  |  |  |  |  |

## Acceptance for Next Cycle Readiness

- Baseline pipeline runs end-to-end with reproducible manifests.
- Results can be inserted into the table template without reformatting.
- Context-stratified metric generation is defined and can be implemented directly.
