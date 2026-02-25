from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare model runs from metrics.json files.")
    p.add_argument(
        "--run",
        action="append",
        required=True,
        help="Format: name:path_to_run_dir_or_metrics_json ; can be repeated.",
    )
    p.add_argument("--out-csv", default="", help="Optional output CSV path.")
    return p.parse_args()


def _read_metrics(path: Path) -> dict:
    if path.is_dir():
        path = path / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(x: float) -> str:
    try:
        return f"{float(x):.4f}"
    except Exception:
        return "nan"


def main() -> None:
    args = _parse_args()
    rows: list[dict] = []
    for item in args.run:
        if ":" not in item:
            raise SystemExit(f"Invalid --run value: {item!r}. Expected name:path.")
        name, p = item.split(":", 1)
        metrics = _read_metrics(Path(p))
        test = metrics.get("test", {})
        rows.append(
            {
                "model": name,
                "rows": int(test.get("rows", 0)),
                "auroc": test.get("auroc", float("nan")),
                "auprc": test.get("auprc", float("nan")),
                "mcc": test.get("mcc", float("nan")),
                "f1": test.get("f1", float("nan")),
                "acc": test.get("acc", float("nan")),
            }
        )

    rows = sorted(rows, key=lambda r: (float("-inf") if str(r["auprc"]) == "nan" else float(r["auprc"])), reverse=True)
    print("| Model | Rows | AUROC | AUPRC | MCC | F1 | ACC |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['model']} | {r['rows']} | {_fmt(r['auroc'])} | {_fmt(r['auprc'])} | {_fmt(r['mcc'])} | {_fmt(r['f1'])} | {_fmt(r['acc'])} |"
        )

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = ["model,rows,auroc,auprc,mcc,f1,acc"]
        lines.extend(
            f"{r['model']},{r['rows']},{r['auroc']},{r['auprc']},{r['mcc']},{r['f1']},{r['acc']}"
            for r in rows
        )
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
