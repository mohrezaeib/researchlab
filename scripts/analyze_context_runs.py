from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from researchlab.metrics import compute_binary_metrics


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate context-window experiment runs.")
    p.add_argument("--root", required=True, help="Directory containing run subfolders named <mode>_<bp>.")
    p.add_argument("--out-csv", required=True)
    p.add_argument("--out-md", required=True)
    return p.parse_args()


def _best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.01, 0.99, 99):
        y_pred = (y_prob >= t).astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


def main() -> None:
    args = _parse_args()
    root = Path(args.root)
    rows: list[dict] = []

    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        metrics_path = d / "metrics.json"
        preds_path = d / "predictions_val_test.npz"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        test = metrics.get("test", {})
        mode, bp = d.name.rsplit("_", 1) if "_" in d.name else (d.name, "nan")
        rec = {
            "run": d.name,
            "mode": mode,
            "bp": int(bp) if bp.isdigit() else np.nan,
            "test_rows": int(test.get("rows", 0)),
            "test_auroc": float(test.get("auroc", np.nan)),
            "test_auprc": float(test.get("auprc", np.nan)),
            "test_mcc": float(test.get("mcc", np.nan)),
            "test_f1_at_0_5": float(test.get("f1", np.nan)),
            "test_acc": float(test.get("acc", np.nan)),
            "val_best_threshold_for_f1": np.nan,
            "val_best_f1": np.nan,
            "test_f1_at_val_best_threshold": np.nan,
            "f1_gain_vs_0_5": np.nan,
        }
        if preds_path.exists():
            arr = np.load(preds_path)
            val_y_true = arr["val_y_true"].astype(int)
            val_y_prob = arr["val_y_prob"].astype(float)
            test_y_true = arr["test_y_true"].astype(int)
            test_y_prob = arr["test_y_prob"].astype(float)
            t, best_val_f1 = _best_f1_threshold(val_y_true, val_y_prob)
            test_tuned = compute_binary_metrics(test_y_true, test_y_prob, thresh=t)
            rec["val_best_threshold_for_f1"] = float(t)
            rec["val_best_f1"] = float(best_val_f1)
            rec["test_f1_at_val_best_threshold"] = float(test_tuned["f1"])
            rec["f1_gain_vs_0_5"] = float(test_tuned["f1"] - rec["test_f1_at_0_5"])
        rows.append(rec)

    if not rows:
        raise SystemExit(f"No runs found under {root}")

    df = pd.DataFrame(rows).sort_values(["mode", "bp"]).reset_index(drop=True)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    best = df.sort_values("test_auprc", ascending=False).iloc[0]
    md_lines = [
        "# Context Matrix Summary",
        "",
        f"- Best run by test AUPRC: `{best['run']}`",
        f"- Best test AUPRC: `{best['test_auprc']:.4f}`",
        f"- Best test F1@0.5: `{best['test_f1_at_0_5']:.4f}`",
        "",
        "## Table",
        "",
        "| Run | Mode | BP | Test AUROC | Test AUPRC | Test F1@0.5 | Test F1@val-opt-thresh | F1 Gain |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        md_lines.append(
            f"| {r['run']} | {r['mode']} | {int(r['bp'])} | {r['test_auroc']:.4f} | {r['test_auprc']:.4f} | "
            f"{r['test_f1_at_0_5']:.4f} | {r['test_f1_at_val_best_threshold']:.4f} | {r['f1_gain_vs_0_5']:.4f} |"
        )

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
