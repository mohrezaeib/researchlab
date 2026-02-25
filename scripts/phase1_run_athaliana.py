from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from researchlab.assemble import SplitSpec, assemble_dataset


ARABIDOPSIS_FASTA_URL = "https://ftp.ebi.ac.uk/ensemblgenomes/pub/plants/release-62/fasta/arabidopsis_thaliana/dna/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz"
GSM2099378_ALLC_URL = "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2099nnn/GSM2099378/suppl/GSM2099378_allc_9386.tsv.gz"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_metadata(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": int(stat.st_size),
        "sha256": _sha256(path),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def download(url: str, dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    existed_before = dest.exists() and dest.stat().st_size > 0
    if existed_before:
        print(f"Download exists, skipping: {dest}")
    else:
        print(f"Downloading: {url}")
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        tmp = dest.with_suffix(dest.suffix + ".partial")
        with urlopen(req, timeout=120) as r, tmp.open("wb") as out:
            shutil.copyfileobj(r, out, length=1024 * 1024)
        tmp.replace(dest)
        print(f"Wrote: {dest} ({dest.stat().st_size:,} bytes)")
    return {
        "url": url,
        "downloaded_at_utc": _iso_utc_now(),
        "existed_before": existed_before,
        **_file_metadata(dest),
    }


def gunzip(src_gz: Path, dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    existed_before = dest.exists() and dest.stat().st_size > 0
    if existed_before:
        print(f"Unzipped exists, skipping: {dest}")
    else:
        tmp = dest.with_suffix(dest.suffix + ".partial")
        with gzip.open(src_gz, "rb") as r, tmp.open("wb") as w:
            shutil.copyfileobj(r, w, length=1024 * 1024)
        tmp.replace(dest)
        print(f"Wrote: {dest} ({dest.stat().st_size:,} bytes)")
    return {
        "source_gz": str(src_gz),
        "unzipped_at_utc": _iso_utc_now(),
        "existed_before": existed_before,
        **_file_metadata(dest),
    }


def _normalize_chrom(chrom: str) -> str:
    c = chrom.strip()
    if c.lower().startswith("chr"):
        c = c[3:]
    return c


def sample_allc_to_calls_tsv(
    *,
    allc_gz: Path,
    out_tsv: Path,
    seed: int,
    sample_fraction: float,
    min_cov: int,
    max_rows: int | None,
) -> dict:
    """
    Stream a GEO ALLC-style gz file and write a much smaller TSV with columns:
    chrom, pos, strand, m, u
    """
    if not (0 < sample_fraction <= 1.0):
        raise ValueError("--sample-fraction must be in (0, 1].")

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    kept = 0
    seen = 0
    cov_filtered = 0

    tmp = out_tsv.with_suffix(out_tsv.suffix + ".partial")
    with gzip.open(allc_gz, "rt") as f, tmp.open("w", encoding="utf-8") as out:
        header = f.readline().strip().split("\t")
        cols = {name: i for i, name in enumerate(header)}
        required = {"chrom", "pos", "strand", "methylated_bases", "total_bases"}
        missing = required - set(cols)
        if missing:
            raise ValueError(f"ALLC missing columns: {sorted(missing)}")

        out.write("chrom\tpos\tstrand\tm\tu\n")
        for line in f:
            seen += 1
            if sample_fraction < 1.0 and rng.random() > sample_fraction:
                continue
            parts = line.rstrip("\n").split("\t")
            chrom = _normalize_chrom(parts[cols["chrom"]])
            pos = parts[cols["pos"]]
            strand = parts[cols["strand"]]
            m = int(parts[cols["methylated_bases"]])
            total = int(parts[cols["total_bases"]])
            cov = total
            if cov < min_cov:
                cov_filtered += 1
                continue
            u = total - m
            out.write(f"{chrom}\t{pos}\t{strand}\t{m}\t{u}\n")
            kept += 1
            if max_rows is not None and kept >= max_rows:
                break

    tmp.replace(out_tsv)
    return {
        "seen_lines": seen,
        "kept_rows": kept,
        "filtered_low_cov": cov_filtered,
        "sample_fraction": sample_fraction,
        "min_cov": min_cov,
        "max_rows": max_rows,
    }


def write_default_splits(out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    splits = {"train_chroms": ["1", "2", "3", "Mt", "Pt"], "val_chroms": ["4"], "test_chroms": ["5"]}
    out_json.write_text(json.dumps(splits, indent=2, sort_keys=True), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1: download Arabidopsis TAIR10 + a public ALLC file, assemble dataset parquet."
    )
    p.add_argument("--outdir", default="data/phase1_athaliana", help="Output directory under the repo.")
    p.add_argument(
        "--download-only",
        action="store_true",
        help="Headnode mode: only download/decompress source files and write downloads.summary.json.",
    )
    p.add_argument(
        "--assemble-only",
        action="store_true",
        help="SLURM mode: only assemble dataset from pre-downloaded local files; never uses network.",
    )
    p.add_argument("--fasta", default="", help="Required with --assemble-only: local uncompressed FASTA path.")
    p.add_argument("--allc-gz", default="", help="Required with --assemble-only: local ALLC .tsv.gz path.")
    p.add_argument("--window", type=int, default=200, help="Even; half window on each side.")
    p.add_argument("--min-cov", type=int, default=10)
    p.add_argument("--ratio-thresh", type=float, default=0.5)
    p.add_argument("--sample-fraction", type=float, default=0.01, help="Subsample ALLC lines to keep runtime low.")
    p.add_argument("--max-rows", type=int, default=200_000, help="Cap sampled rows (after filtering).")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def _write_download_manifest(
    *,
    processed_dir: Path,
    fasta_gz_meta: dict,
    fasta_meta: dict,
    allc_meta: dict,
) -> None:
    manifest = {
        "generated_at_utc": _iso_utc_now(),
        "files": {
            "fasta_gz": fasta_gz_meta,
            "fasta": fasta_meta,
            "allc_gz": allc_meta,
        },
    }
    (processed_dir / "downloads.summary.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    args = _parse_args()
    if args.download_only and args.assemble_only:
        raise SystemExit("Choose only one mode: --download-only or --assemble-only.")
    outdir = Path(args.outdir)
    raw = outdir / "raw"
    processed = outdir / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    fasta_gz = raw / "Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz"
    fasta = raw / "Arabidopsis_thaliana.TAIR10.dna.toplevel.fa"
    allc_gz = raw / "GSM2099378_allc_9386.tsv.gz"
    if args.fasta:
        fasta = Path(args.fasta)
    if args.allc_gz:
        allc_gz = Path(args.allc_gz)

    if args.assemble_only and (not args.fasta or not args.allc_gz):
        raise SystemExit("--assemble-only requires both --fasta and --allc-gz.")

    if not args.assemble_only:
        fasta_gz_meta = download(ARABIDOPSIS_FASTA_URL, fasta_gz)
        fasta_meta = gunzip(fasta_gz, fasta)
        allc_meta = download(GSM2099378_ALLC_URL, allc_gz)
        _write_download_manifest(
            processed_dir=processed,
            fasta_gz_meta=fasta_gz_meta,
            fasta_meta=fasta_meta,
            allc_meta=allc_meta,
        )
        if args.download_only:
            print(f"Wrote: {processed / 'downloads.summary.json'}")
            return

    if not fasta.exists():
        raise SystemExit(f"FASTA not found: {fasta}")
    if not allc_gz.exists():
        raise SystemExit(f"ALLC gz not found: {allc_gz}")

    calls_tsv = processed / "calls.sampled.tsv"
    sampling_summary = sample_allc_to_calls_tsv(
        allc_gz=allc_gz,
        out_tsv=calls_tsv,
        seed=args.seed,
        sample_fraction=args.sample_fraction,
        min_cov=args.min_cov,
        max_rows=args.max_rows,
    )
    (processed / "calls.sampled.summary.json").write_text(
        json.dumps(sampling_summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    splits_json = processed / "splits.json"
    if not splits_json.exists():
        write_default_splits(splits_json)

    df = assemble_dataset(
        fasta_path=str(fasta),
        calls_path=str(calls_tsv),
        calls_format="generic_tsv",
        split_spec=SplitSpec.from_json(splits_json),
        window=args.window,
        min_cov=args.min_cov,
        ratio_thresh=args.ratio_thresh,
        contexts={"CG", "CHG", "CHH"},
        max_rows=None,
    )

    processed.mkdir(parents=True, exist_ok=True)
    dataset_path = processed / "dataset.parquet"
    df.to_parquet(dataset_path, index=False)

    summary = {
        "rows": int(len(df)),
        "splits": df["split"].value_counts().to_dict() if len(df) else {},
        "contexts": df["context"].value_counts().to_dict() if len(df) else {},
        "dataset_sha256": _sha256(dataset_path),
    }
    (processed / "dataset.summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
