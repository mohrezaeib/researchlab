from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq

DNA_ALPHABET = set("ACGT")


def _open_maybe_gzip(path: str | Path, mode: str = "rt"):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode)
    return open(path, mode, encoding=None if "b" in mode else "utf-8")


def load_fasta_index(fasta_path: str | Path):
    fasta_path = str(fasta_path)
    return SeqIO.index(fasta_path, "fasta")


def revcomp(seq: str) -> str:
    return str(Seq(seq).reverse_complement())


def normalize_dna(seq: str) -> str:
    seq = seq.upper()
    return "".join(base if base in DNA_ALPHABET else "N" for base in seq)


def classify_context(centered_seq: str) -> Literal["CG", "CHG", "CHH", "NA"]:
    # centered_seq length must be >= 3 and centered at index mid
    if len(centered_seq) < 3:
        return "NA"
    mid = len(centered_seq) // 2
    if centered_seq[mid] != "C":
        return "NA"
    b1 = centered_seq[mid + 1] if mid + 1 < len(centered_seq) else "N"
    b2 = centered_seq[mid + 2] if mid + 2 < len(centered_seq) else "N"
    if b1 == "G":
        return "CG"
    if b2 == "G":
        return "CHG"
    return "CHH"


@dataclass(frozen=True)
class SplitSpec:
    train_chroms: list[str]
    val_chroms: list[str]
    test_chroms: list[str]

    @staticmethod
    def from_json(path: str | Path) -> "SplitSpec":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return SplitSpec(
            train_chroms=list(data["train_chroms"]),
            val_chroms=list(data["val_chroms"]),
            test_chroms=list(data["test_chroms"]),
        )

    def split_for_chrom(self, chrom: str) -> Literal["train", "val", "test", "drop"]:
        if chrom in self.train_chroms:
            return "train"
        if chrom in self.val_chroms:
            return "val"
        if chrom in self.test_chroms:
            return "test"
        return "drop"


def _read_calls_table(path: str | Path, fmt: str) -> pd.DataFrame:
    with _open_maybe_gzip(path, "rt") as f:
        if fmt == "bismark_cx":
            df = pd.read_csv(
                f,
                sep="\t",
                header=None,
                names=[
                    "chrom",
                    "pos",
                    "strand",
                    "m",
                    "u",
                    "context",
                    "trinuc",
                ],
                dtype={"chrom": "string", "pos": "int64", "strand": "string"},
            )
            return df[["chrom", "pos", "strand", "m", "u"]]
        if fmt == "generic_tsv":
            df = pd.read_csv(f, sep="\t")
            required = {"chrom", "pos", "strand", "m", "u"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"calls file missing columns: {sorted(missing)}")
            return df[["chrom", "pos", "strand", "m", "u"]].copy()
        raise ValueError(f"unknown calls format: {fmt!r}")


def assemble_dataset(
    *,
    fasta_path: str | Path,
    calls_path: str | Path,
    calls_format: Literal["bismark_cx", "generic_tsv"],
    split_spec: SplitSpec,
    window: int,
    min_cov: int,
    ratio_thresh: float,
    contexts: set[str] | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    if window % 2 != 0:
        raise ValueError("--window must be even so the cytosine is centered.")
    half = window // 2

    fasta = load_fasta_index(fasta_path)
    calls = _read_calls_table(calls_path, calls_format)

    records: list[dict] = []
    n_kept = 0

    for row in calls.itertuples(index=False):
        chrom = str(row.chrom)
        pos_1based = int(row.pos)
        strand = str(row.strand)
        m = int(row.m)
        u = int(row.u)
        cov = m + u
        if cov < min_cov:
            continue
        ratio = m / cov if cov else 0.0
        label = 1 if ratio >= ratio_thresh else 0

        split = split_spec.split_for_chrom(chrom)
        if split == "drop":
            continue
        if chrom not in fasta:
            continue

        center_0based = pos_1based - 1
        start = center_0based - half
        end = center_0based + half
        if start < 0:
            continue
        chrom_seq = str(fasta[chrom].seq)
        if end >= len(chrom_seq):
            continue

        seq = normalize_dna(chrom_seq[start : end + 1])
        if strand == "-":
            seq = revcomp(seq)

        if seq[half] != "C":
            continue

        context = classify_context(seq)
        if contexts is not None and context not in contexts:
            continue

        records.append(
            {
                "chrom": chrom,
                "pos_1based": pos_1based,
                "strand": strand,
                "split": split,
                "context": context,
                "m": m,
                "u": u,
                "cov": cov,
                "ratio": ratio,
                "label": label,
                "seq": seq,
            }
        )
        n_kept += 1
        if max_rows is not None and n_kept >= max_rows:
            break

    return pd.DataFrame.from_records(records)
