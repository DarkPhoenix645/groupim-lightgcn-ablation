"""Identity of one training run, and the shared results ledger.

A run id is the hash of the config fields that define the experiment. The same
config and seed always produce the same id, so a repeat is the same row.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

IDENTITY = (
    "dataset",
    "encoder",
    "k",
    "aggregator",
    "losses",
    "lambda_mi",
    "emb_size",
    "drop_ratio",
    "batch_size",
    "negs_per_group",
    "pretrain_user",
    "pretrain_mi",
    "pretrain_epochs",
    "mi_epochs",
    "epochs",
    "lr",
    "wd",
    "tied_item_embeddings",
    "eval_freq",
    "cuda",
    "seed",
)

LEDGER_FIELDS = (
    "run_id",
    *IDENTITY,
    "n20",
    "n50",
    "r20",
    "r50",
    "n100",
    "seconds",
    "artifact_dir",
    "writes",
)


def canonical(config: dict) -> dict:
    """JSON-stable identity. Floats are rounded so 0.4 and 0.400000 stay one id."""
    out = {}
    for key in IDENTITY:
        value = config[key]
        if key in {"lambda_mi", "drop_ratio", "lr", "wd"}:
            out[key] = round(float(value), 8)
        elif key in {"pretrain_user", "pretrain_mi", "tied_item_embeddings", "cuda"}:
            out[key] = bool(value)
        elif key in {"dataset", "encoder", "aggregator", "losses"}:
            out[key] = str(value)
        else:
            out[key] = int(value)
    return out


def run_id(config: dict) -> str:
    payload = json.dumps(canonical(config), sort_keys=True, separators=(",", ":"))
    import hashlib

    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def upsert_ledger(path: Path, row: dict) -> int:
    """Insert or replace the row with this run id. Returns how many times it has been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+") as lock:
        try:
            import fcntl
        except ImportError:  # Windows has no flock; the GPU box is Linux.
            fcntl = None
        if fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        existing: list[dict[str, str]] = []
        if path.is_file() and path.stat().st_size:
            with path.open(newline="") as handle:
                existing = list(csv.DictReader(handle))
        writes = 1
        replaced = False
        for old in existing:
            if old.get("run_id") == row["run_id"]:
                writes = int(old.get("writes") or "0") + 1
                old.update({key: str(row[key]) for key in LEDGER_FIELDS})
                old["writes"] = str(writes)
                replaced = True
                break
        if not replaced:
            existing.append({key: str(row[key]) for key in LEDGER_FIELDS})
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(LEDGER_FIELDS))
            writer.writeheader()
            writer.writerows(existing)
        if fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return writes


MARKER_START = "<!-- launcher-runs:start -->"
MARKER_END = "<!-- launcher-runs:end -->"


def rebuild_generated_catalog(catalog_path: Path, runs_root: Path) -> None:
    """Rewrite the generated section of the figure catalog from run directories."""
    blocks = ["", "## Launcher runs", ""]
    index = [
        "| Figure | Run | Plotted | Y axis |",
        "| --- | --- | --- | --- |",
    ]
    bodies: list[str] = []
    if runs_root.is_dir():
        for config_path in sorted(runs_root.glob("*/config.json")):
            config = json.loads(config_path.read_text(encoding="utf-8"))
            identity = config_path.parent.name
            metrics_path = config_path.parent / "metrics.csv"
            metrics = {}
            if metrics_path.is_file():
                with metrics_path.open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                    if rows:
                        metrics = rows[0]
            rel = Path("..") / ".." / "results" / "runs" / identity
            for name, plotted, axis in (
                ("val-r20.svg", "Validation Recall@20 every eval epoch", "Recall@20"),
                ("val-r50.svg", "Validation Recall@50 every eval epoch", "Recall@50"),
                ("val-n100.svg", "Validation NDCG@100 every eval epoch", "NDCG@100"),
                ("train-loss.svg", "Mean group-training loss each epoch", "Mean train loss"),
            ):
                index.append(
                    f"| [{name}]({rel / 'figures' / name}) | `{identity}` | {plotted}. Seed {config['seed']}. | {axis} |"
                )
            test = ""
            if metrics:
                test = (
                    f"Test N@20 {float(metrics['n20']):.4f}, N@50 {float(metrics['n50']):.4f}, "
                    f"R@20 {float(metrics['r20']):.4f}, R@50 {float(metrics['r50']):.4f}. "
                    f"Wall clock {metrics['seconds']} s."
                )
            bodies.append(
                "\n".join(
                    [
                        f"## `{identity}`",
                        "",
                        (
                            f"Encoder `{config['encoder']}`, aggregator `{config['aggregator']}`, "
                            f"dataset `{config['dataset']}`, seed {config['seed']}."
                        ),
                        "",
                        "| Field | Value |",
                        "| --- | --- |",
                        f"| Losses | `{config['losses']}` |",
                        f"| λ | {config['lambda_mi']} |",
                        f"| K | {config['k']} |",
                        f"| Embedding dim | {config['emb_size']} |",
                        f"| Epochs | {config['epochs']} group, {config['pretrain_epochs']} user pretrain, {config['mi_epochs']} MI |",
                        f"| Pretrain | user {config['pretrain_user']}, MI {config['pretrain_mi']} |",
                        f"| Tied item embeddings | {config['tied_item_embeddings']} |",
                        f"| CUDA | {config['cuda']} |",
                        f"| Log | [train.log]({rel / 'train.log'}) |",
                        f"| Metrics | [metrics.csv]({rel / 'metrics.csv'}) |",
                        "",
                        test,
                        "",
                    ]
                )
            )
    generated = "\n".join(blocks + index + ["", *bodies]).rstrip() + "\n"
    text = catalog_path.read_text(encoding="utf-8") if catalog_path.is_file() else "# Figure catalog\n"
    if MARKER_START not in text:
        text = text.rstrip() + f"\n\n{MARKER_START}\n{MARKER_END}\n"
    before, rest = text.split(MARKER_START, 1)
    _, after = rest.split(MARKER_END, 1)
    catalog_path.write_text(f"{before}{MARKER_START}\n{generated}{MARKER_END}{after}", encoding="utf-8")
