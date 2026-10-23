"""One config in, one results row out, plus the log and figures that prove the run.

    uv run python -m groupim_lightgcn_ablation.launch --dataset weeplaces --cuda --seed 1111
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from groupim_lightgcn_ablation.plots import write_run_figures
from groupim_lightgcn_ablation.run_record import (
    canonical,
    rebuild_generated_catalog,
    run_id,
    upsert_ledger,
)
from models.models import AGGREGATOR_CHOICES, LOSS_CHOICES

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "results" / "runs"
LEDGER = RUNS.parent / "runs.csv"
CATALOG = REPO / "docs" / "figures" / "catalog.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one GroupIM config and record it")
    parser.add_argument("--dataset", default="weeplaces")
    parser.add_argument("--encoder", choices=["mlp", "lightgcn"], default="mlp")
    parser.add_argument("--k", type=int, default=0, help="LightGCN layers. Ignored for the MLP encoder.")
    parser.add_argument("--aggregator", choices=AGGREGATOR_CHOICES, default="attention")
    parser.add_argument("--losses", choices=LOSS_CHOICES, default="full")
    parser.add_argument("--lambda-mi", type=float, default=1.0)
    parser.add_argument("--emb-size", type=int, default=64)
    parser.add_argument("--drop-ratio", type=float, default=0.4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--negs-per-group", type=int, default=5)
    parser.add_argument("--pretrain-user", action="store_true")
    parser.add_argument("--pretrain-mi", action="store_true")
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--mi-epochs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--tied-item-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-freq", type=int, default=5)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=1111)
    return parser


def config_from_args(args: argparse.Namespace) -> dict:
    return canonical(
        {
            "dataset": args.dataset,
            "encoder": args.encoder,
            "k": args.k,
            "aggregator": args.aggregator,
            "losses": args.losses,
            "lambda_mi": args.lambda_mi,
            "emb_size": args.emb_size,
            "drop_ratio": args.drop_ratio,
            "batch_size": args.batch_size,
            "negs_per_group": args.negs_per_group,
            "pretrain_user": args.pretrain_user,
            "pretrain_mi": args.pretrain_mi,
            "pretrain_epochs": args.pretrain_epochs,
            "mi_epochs": args.mi_epochs,
            "epochs": args.epochs,
            "lr": args.lr,
            "wd": args.wd,
            "tied_item_embeddings": args.tied_item_embeddings,
            "eval_freq": args.eval_freq,
            "cuda": args.cuda,
            "seed": args.seed,
        }
    )


def train_command(config: dict, run_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(REPO / "train.py"),
        "--dataset",
        config["dataset"],
        "--encoder",
        config["encoder"],
        "--k",
        str(config["k"]),
        "--aggregator",
        config["aggregator"],
        "--losses",
        config["losses"],
        "--lambda_mi",
        str(config["lambda_mi"]),
        "--emb_size",
        str(config["emb_size"]),
        "--drop_ratio",
        str(config["drop_ratio"]),
        "--batch_size",
        str(config["batch_size"]),
        "--negs_per_group",
        str(config["negs_per_group"]),
        "--pretrain_epochs",
        str(config["pretrain_epochs"]),
        "--mi-epochs",
        str(config["mi_epochs"]),
        "--epochs",
        str(config["epochs"]),
        "--lr",
        str(config["lr"]),
        "--wd",
        str(config["wd"]),
        "--eval_freq",
        str(config["eval_freq"]),
        "--seed",
        str(config["seed"]),
        "--save",
        str(run_dir / "model_user.pt"),
        "--save_group",
        str(run_dir / "model_group.pt"),
        "--artifact-dir",
        str(run_dir),
    ]
    if config["pretrain_user"]:
        cmd.append("--pretrain_user")
    if config["pretrain_mi"]:
        cmd.append("--pretrain_mi")
    if config["tied_item_embeddings"]:
        cmd.append("--tied-item-embeddings")
    else:
        cmd.append("--no-tied-item-embeddings")
    if config["cuda"]:
        cmd.append("--cuda")
    return cmd


def execute(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n")
        log.flush()
        process = subprocess.Popen(
            cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        assert process.stdout is not None
        for raw in process.stdout:
            sys.stdout.buffer.write(raw)
            sys.stdout.buffer.flush()
            log.write(raw.decode(errors="replace"))
            log.flush()
        return process.wait()


def read_metrics(path: Path) -> dict[str, str]:
    import csv

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"{path} should hold one row, found {len(rows)}")
    return rows[0]


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    identity = run_id(config)
    run_dir = RUNS / identity
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"run_id {identity}", flush=True)
    code = execute(train_command(config, run_dir), run_dir / "train.log")
    if code != 0:
        raise SystemExit(code)
    metrics = read_metrics(run_dir / "metrics.csv")
    figures = write_run_figures(
        (run_dir / "train.log").read_text(encoding="utf-8"),
        run_dir / "figures",
        config["dataset"],
        config["seed"],
    )
    row = {"run_id": identity, **config, "artifact_dir": str(run_dir.relative_to(REPO)), "writes": 1}
    for key in ("n20", "n50", "r20", "r50", "n100", "seconds"):
        row[key] = metrics[key]
    writes = upsert_ledger(LEDGER, row)
    rebuild_generated_catalog(CATALOG, RUNS)
    print(f"recorded {identity} (write {writes})", flush=True)
    for figure in figures:
        print(f"figure {figure.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
