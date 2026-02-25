from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from Bio import SeqIO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create chromosome-level train/val/test split spec JSON.")
    p.add_argument("--fasta", required=True, help="Reference genome FASTA (for contig names).")
    p.add_argument("--out", required=True, help="Output JSON path.")
    p.add_argument(
        "--drop-prefix",
        default="",
        help="Comma-separated prefixes to drop (e.g. 'scaffold,contig').",
    )
    p.add_argument("--val-chroms", default="", help="Comma-separated contig names for validation.")
    p.add_argument("--test-chroms", default="", help="Comma-separated contig names for test.")
    return p.parse_args()


def _csv_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    args = _parse_args()
    fasta = SeqIO.index(str(args.fasta), "fasta")

    drop_prefixes = tuple(_csv_list(args.drop_prefix))
    contigs = [c for c in fasta.keys() if not drop_prefixes or not str(c).startswith(drop_prefixes)]
    contigs = [str(c) for c in contigs]
    if not contigs:
        raise SystemExit("No contigs found after filtering.")

    val = _csv_list(args.val_chroms)
    test = _csv_list(args.test_chroms)
    chosen = set(val) | set(test)
    unknown = [c for c in chosen if c not in contigs]
    if unknown:
        raise SystemExit(f"Unknown contigs in --val-chroms/--test-chroms: {unknown}")

    remaining = [c for c in contigs if c not in chosen]

    # If user didn’t specify, take last contig as test, second last as val.
    if not test:
        if not remaining:
            raise SystemExit("No remaining contigs to assign to test.")
        test = [remaining[-1]]
        remaining = remaining[:-1]
    if not val:
        if not remaining:
            raise SystemExit("No remaining contigs to assign to val.")
        val = [remaining[-1]]
        remaining = remaining[:-1]

    splits = {"train_chroms": remaining, "val_chroms": val, "test_chroms": test}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(splits, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(splits, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

