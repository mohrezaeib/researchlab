from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import shlex
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from researchlab.metrics import compute_binary_metrics


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"
    return out or "unknown"


def _package_versions() -> dict[str, str]:
    pkgs = [
        "pandas",
        "numpy",
        "scikit-learn",
        "biopython",
        "pyarrow",
        "joblib",
        "torch",
        "transformers",
        "tqdm",
    ]
    out: dict[str, str] = {}
    for pkg in pkgs:
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = "missing"
    return out


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _select_layers_for_finetune(model: AutoModel, top_layers: int) -> list[nn.Module]:
    candidates = [
        ("encoder.layer", lambda m: getattr(getattr(m, "encoder", None), "layer", None)),
        ("bert.encoder.layer", lambda m: getattr(getattr(getattr(m, "bert", None), "encoder", None), "layer", None)),
        ("roberta.encoder.layer", lambda m: getattr(getattr(getattr(m, "roberta", None), "encoder", None), "layer", None)),
        ("deberta.encoder.layer", lambda m: getattr(getattr(getattr(m, "deberta", None), "encoder", None), "layer", None)),
    ]
    for _, fn in candidates:
        layers = fn(model)
        if layers is not None and len(layers) > 0:
            k = min(top_layers, len(layers))
            return list(layers[-k:])
    return []


class SeqDataset(Dataset):
    def __init__(self, seqs: list[str], labels: list[int]) -> None:
        self.seqs = seqs
        self.labels = labels

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.seqs[idx], int(self.labels[idx])


class HFSequenceClassifier(nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = int(self.encoder.config.hidden_size)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(**inputs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled).squeeze(-1)
        return logits


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train transformer-based methylation classifier.")
    p.add_argument("--dataset", required=True, help="Parquet dataset with seq,label,split columns.")
    p.add_argument("--outdir", required=True)
    p.add_argument(
        "--model-name",
        required=True,
        help="HuggingFace model ID (e.g. zhihan1996/DNABERT-2-117M or InstaDeepAI/nucleotide-transformer-v2-50m-multi-species).",
    )
    p.add_argument("--train-mode", choices=["frozen", "top", "full"], default="frozen")
    p.add_argument("--top-layers", type=int, default=2, help="Used with --train-mode top.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--debug-max-rows", type=int, default=None)
    p.add_argument("--device", default="auto", help="auto|cpu|cuda|cuda:0")
    return p.parse_args()


def _prepare_dataloaders(
    df: pd.DataFrame,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]
    ds_train = SeqDataset(train["seq"].tolist(), train["label"].astype(int).tolist())
    ds_val = SeqDataset(val["seq"].tolist(), val["label"].astype(int).tolist())
    ds_test = SeqDataset(test["seq"].tolist(), test["label"].astype(int).tolist())
    return (
        DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


def _collate(batch, tokenizer: AutoTokenizer, max_length: int, device: torch.device) -> tuple[dict, torch.Tensor]:
    if isinstance(batch, (list, tuple)) and len(batch) == 2 and isinstance(batch[0], (list, tuple)):
        seqs = list(batch[0])
        raw_labels = batch[1]
        if torch.is_tensor(raw_labels):
            labels = raw_labels.to(device=device, dtype=torch.float32)
        else:
            labels = torch.tensor(list(raw_labels), dtype=torch.float32, device=device)
    else:
        seqs = [x[0] for x in batch]
        labels = torch.tensor([x[1] for x in batch], dtype=torch.float32, device=device)
    toks = tokenizer(
        seqs,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    toks = {k: v.to(device) for k, v in toks.items()}
    return toks, labels


@torch.no_grad()
def _predict_probs(
    model: HFSequenceClassifier,
    dl: DataLoader,
    tokenizer: AutoTokenizer,
    max_length: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[int] = []
    y_prob: list[float] = []
    for batch in dl:
        toks, labels = _collate(batch, tokenizer, max_length=max_length, device=device)
        logits = model(**toks)
        probs = torch.sigmoid(logits)
        y_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
        y_prob.extend(probs.detach().cpu().numpy().astype(float).tolist())
    return np.asarray(y_true, dtype=int), np.asarray(y_prob, dtype=float)


def _fit(
    *,
    model: HFSequenceClassifier,
    tokenizer: AutoTokenizer,
    train_dl: DataLoader,
    val_dl: DataLoader,
    test_dl: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    y_train = np.array([int(y) for _, y in train_dl.dataset], dtype=int)
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for p in model.encoder.parameters():
        p.requires_grad = args.train_mode != "frozen"
    if args.train_mode == "top":
        for p in model.encoder.parameters():
            p.requires_grad = False
        for layer in _select_layers_for_finetune(model.encoder, args.top_layers):
            for p in layer.parameters():
                p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_items = 0
        for batch in tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            toks, labels = _collate(batch, tokenizer, max_length=args.max_length, device=device)
            logits = model(**toks)
            loss = criterion(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bs = int(labels.shape[0])
            running_loss += float(loss.item()) * bs
            n_items += bs

        y_val_true, y_val_prob = _predict_probs(model, val_dl, tokenizer, args.max_length, device)
        val_metrics = compute_binary_metrics(y_val_true, y_val_prob)
        val_metrics["rows"] = int(len(y_val_true))
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(n_items, 1),
                "val_auroc": val_metrics["auroc"],
                "val_auprc": val_metrics["auprc"],
            }
        )

    y_train_true, y_train_prob = _predict_probs(model, train_dl, tokenizer, args.max_length, device)
    y_val_true, y_val_prob = _predict_probs(model, val_dl, tokenizer, args.max_length, device)
    y_test_true, y_test_prob = _predict_probs(model, test_dl, tokenizer, args.max_length, device)
    result = {"n_rows": int(len(y_train_true) + len(y_val_true) + len(y_test_true)), "history": history}
    for name, y_t, y_p in [
        ("train", y_train_true, y_train_prob),
        ("val", y_val_true, y_val_prob),
        ("test", y_test_true, y_test_prob),
    ]:
        m = compute_binary_metrics(y_t, y_p)
        m["rows"] = int(len(y_t))
        result[name] = m
    return result


def main() -> None:
    args = _parse_args()
    started_at = _iso_utc_now()
    _set_seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)

    df = pd.read_parquet(args.dataset)
    required = {"seq", "label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"dataset missing columns: {sorted(missing)}")
    if args.debug_max_rows is not None:
        parts = []
        for split in ["train", "val", "test"]:
            parts.append(df[df["split"] == split].head(args.debug_max_rows))
        df = pd.concat(parts, axis=0, ignore_index=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = HFSequenceClassifier(args.model_name).to(device)
    train_dl, val_dl, test_dl = _prepare_dataloaders(df, args.batch_size, args.num_workers)

    metrics = _fit(
        model=model,
        tokenizer=tokenizer,
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        args=args,
        device=device,
    )
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    tokenizer.save_pretrained(outdir / "model")
    model.encoder.save_pretrained(outdir / "model")
    torch.save(model.classifier.state_dict(), outdir / "classifier_head.pt")

    finished_at = _iso_utc_now()
    run_manifest = {
        "git_commit_sha": _get_git_commit(),
        "command": "python " + " ".join(shlex.quote(x) for x in sys.argv),
        "args": vars(args),
        "dataset_path": str(Path(args.dataset)),
        "outdir": str(outdir),
        "hostname": platform.node(),
        "device": str(device),
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "python_version": sys.version.split()[0],
        "package_versions": _package_versions(),
    }
    (outdir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
