from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from researchlab.features import NgramFeaturizer
from researchlab.metrics import compute_binary_metrics


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"
    return out or "unknown"


def _get_package_versions() -> dict[str, str]:
    pkgs = ["pandas", "numpy", "scikit-learn", "biopython", "pyarrow", "joblib"]
    versions: dict[str, str] = {}
    for pkg in pkgs:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "missing"
    return versions


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a simple baseline methylation classifier.")
    p.add_argument("--dataset", required=True, help="Parquet from scripts/assemble_dataset.py")
    p.add_argument("--outdir", required=True, help="Run output directory")
    p.add_argument("--ngram-min", type=int, default=3)
    p.add_argument("--ngram-max", type=int, default=6)
    p.add_argument("--max-features", type=int, default=200_000)
    p.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength.")
    p.add_argument("--max-iter", type=int, default=200)
    return p.parse_args()


def _fit_and_eval(df: pd.DataFrame, outdir: Path, args: argparse.Namespace) -> dict:
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]

    featurizer = NgramFeaturizer(
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        max_features=args.max_features,
    )
    vectorizer = featurizer.make_vectorizer()

    clf = LogisticRegression(
        C=args.C,
        max_iter=args.max_iter,
        solver="liblinear",
        class_weight="balanced",
    )
    pipe: Pipeline = Pipeline([("vect", vectorizer), ("clf", clf)])

    X_train = train["seq"].tolist()
    y_train = train["label"].to_numpy(dtype=int)
    pipe.fit(X_train, y_train)

    def predict_prob(part: pd.DataFrame) -> np.ndarray:
        if len(part) == 0:
            return np.array([], dtype=float)
        return pipe.predict_proba(part["seq"].tolist())[:, 1]

    result: dict[str, dict] = {"n_rows": int(len(df))}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        y_true = part["label"].to_numpy(dtype=int)
        y_prob = predict_prob(part)
        result[name] = compute_binary_metrics(y_true, y_prob)
        result[name]["rows"] = int(len(part))

    joblib.dump(pipe, outdir / "model.joblib")
    (outdir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    args = _parse_args()
    started_at = _iso_utc_now()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.dataset)
    required = {"seq", "label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"dataset missing columns: {sorted(missing)}")

    metrics = _fit_and_eval(df, outdir, args)
    finished_at = _iso_utc_now()
    run_manifest = {
        "git_commit_sha": _get_git_commit(),
        "command": "python " + " ".join(shlex.quote(x) for x in sys.argv),
        "args": vars(args),
        "dataset_path": str(Path(args.dataset)),
        "outdir": str(outdir),
        "hostname": platform.node(),
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "python_version": sys.version.split()[0],
        "package_versions": _get_package_versions(),
    }
    (outdir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
