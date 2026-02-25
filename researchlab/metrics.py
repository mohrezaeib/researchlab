from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    roc_auc_score,
    f1_score,
    accuracy_score,
)


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, thresh: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= thresh).astype(int)

    metrics: dict[str, float] = {
        "auroc": float("nan"),
        "auprc": float("nan"),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "acc": float(accuracy_score(y_true, y_pred)),
    }

    # AUROC/AUPRC require both classes present
    if len(np.unique(y_true)) == 2:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
        metrics["auprc"] = float(average_precision_score(y_true, y_prob))
    return metrics

