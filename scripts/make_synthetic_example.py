from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a tiny synthetic FASTA + calls TSV for smoke testing.")
    p.add_argument("--outdir", required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--chroms", type=int, default=3)
    p.add_argument("--chrom-len", type=int, default=50000)
    p.add_argument("--sites", type=int, default=6000)
    return p.parse_args()


def _rand_dna(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def main() -> None:
    args = _parse_args()
    rng = random.Random(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    chrom_names = [f"chr{i+1}" for i in range(args.chroms)]
    chrom_seqs: dict[str, str] = {c: _rand_dna(rng, args.chrom_len) for c in chrom_names}

    # Write FASTA
    fasta_path = outdir / "reference.fa"
    with fasta_path.open("w", encoding="utf-8") as f:
        for chrom, seq in chrom_seqs.items():
            f.write(f">{chrom}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i : i + 80] + "\n")

    # Synthetic methylation calls: bias methylation toward 'CG' centers
    rows = []
    for _ in range(args.sites):
        chrom = rng.choice(chrom_names)
        pos_1based = rng.randint(2, args.chrom_len - 2)
        strand = rng.choice(["+", "-"])
        # generate counts with different probabilities based on local motif
        seq = chrom_seqs[chrom]
        i = pos_1based - 1
        tri = seq[i : i + 3]
        if tri.startswith("CG"):
            p_meth = 0.8
        elif tri[2:3] == "G":
            p_meth = 0.55
        else:
            p_meth = 0.25
        cov = rng.randint(10, 40)
        m = sum(1 for _ in range(cov) if rng.random() < p_meth)
        u = cov - m
        rows.append({"chrom": chrom, "pos": pos_1based, "strand": strand, "m": m, "u": u})

    calls_path = outdir / "methylation_calls.tsv"
    pd.DataFrame(rows).to_csv(calls_path, sep="\t", index=False)

    splits = {
        "train_chroms": chrom_names[: max(1, len(chrom_names) - 2)],
        "val_chroms": [chrom_names[-2]] if len(chrom_names) >= 2 else [],
        "test_chroms": [chrom_names[-1]],
    }
    (outdir / "splits.json").write_text(json.dumps(splits, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote: {fasta_path}")
    print(f"Wrote: {calls_path}")
    print(f"Wrote: {outdir / 'splits.json'}")


if __name__ == "__main__":
    main()

