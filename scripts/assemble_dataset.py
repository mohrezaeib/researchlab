from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from researchlab.assemble import SplitSpec, assemble_dataset


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assemble methylation dataset from FASTA + calls table.")
    p.add_argument("--fasta", required=True, help="Reference genome FASTA.")
    p.add_argument("--calls", required=True, help="Methylation calls table (tsv or tsv.gz).")
    p.add_argument(
        "--calls-format",
        default="generic_tsv",
        choices=["generic_tsv", "bismark_cx"],
        help="Input calls file format.",
    )
    p.add_argument("--split-spec", required=True, help="JSON with train/val/test chromosome lists.")
    p.add_argument("--out", required=True, help="Output dataset parquet path.")
    p.add_argument("--window", type=int, default=200, help="Even; half window on each side.")
    p.add_argument("--min-cov", type=int, default=10, help="Minimum coverage (m+u).")
    p.add_argument("--ratio-thresh", type=float, default=0.5, help="Label threshold on m/(m+u).")
    p.add_argument(
        "--contexts",
        default="CG,CHG,CHH",
        help="Comma-separated contexts to keep (subset of CG,CHG,CHH).",
    )
    p.add_argument("--max-rows", type=int, default=None, help="Optional cap for quick runs.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    contexts = {c.strip() for c in args.contexts.split(",") if c.strip()}
    spec = SplitSpec.from_json(args.split_spec)

    df = assemble_dataset(
        fasta_path=args.fasta,
        calls_path=args.calls,
        calls_format=args.calls_format,
        split_spec=spec,
        window=args.window,
        min_cov=args.min_cov,
        ratio_thresh=args.ratio_thresh,
        contexts=contexts,
        max_rows=args.max_rows,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    summary = {
        "rows": int(len(df)),
        "splits": df["split"].value_counts(dropna=False).to_dict() if len(df) else {},
        "contexts": df["context"].value_counts(dropna=False).to_dict() if len(df) else {},
    }
    (out.with_suffix(out.suffix + ".summary.json")).write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
