"""Launcher identity, group statistics, plots, and the results ledger."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from groupim_lightgcn_ablation.group_stats import (
    aggregate_diversity,
    mean_pairwise_pearson,
    records_from_batch,
)
from groupim_lightgcn_ablation.launch import build_parser, config_from_args, train_command
from groupim_lightgcn_ablation.plots import parse_group_log, write_run_figures
from groupim_lightgcn_ablation.run_record import rebuild_generated_catalog, run_id, upsert_ledger
from models.models import GroupIM


def _config(**overrides):
    args = build_parser().parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return config_from_args(args)


def test_run_id_is_stable_and_changes_with_seed():
    first = run_id(_config(seed=1111))
    assert run_id(_config(seed=1111)) == first
    assert run_id(_config(seed=1112)) != first


def test_identical_histories_are_perfectly_coherent():
    vectors = np.array([[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]])
    assert mean_pairwise_pearson(vectors) == pytest.approx(1.0)
    assert aggregate_diversity(vectors) == 2


def test_diversity_counts_the_union():
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert aggregate_diversity(vectors) == 2
    assert mean_pairwise_pearson(vectors) < 0


def test_ledger_replaces_the_same_run_id(tmp_path: Path):
    path = tmp_path / "runs.csv"
    fields = (
        "run_id", "dataset", "encoder", "k", "aggregator", "losses", "lambda_mi", "emb_size",
        "drop_ratio", "batch_size", "negs_per_group", "pretrain_user", "pretrain_mi",
        "pretrain_epochs", "mi_epochs", "epochs", "lr", "wd", "tied_item_embeddings", "eval_freq",
        "cuda", "seed", "n20", "n50", "r20", "r50", "n100", "seconds", "artifact_dir", "writes",
    )
    row = {key: "1" for key in fields}
    row["run_id"] = "abc"
    row["seed"] = "1111"
    row["n20"] = "0.1"
    assert upsert_ledger(path, row) == 1
    other = dict(row)
    other["run_id"] = "def"
    other["seed"] = "1112"
    assert upsert_ledger(path, other) == 1
    row["n20"] = "0.2"
    assert upsert_ledger(path, row) == 2
    import csv

    with path.open(newline="") as handle:
        found = list(csv.DictReader(handle))
    assert [item["run_id"] for item in found] == ["abc", "def"]
    assert next(item["n20"] for item in found if item["run_id"] == "abc") == "0.2"
    assert next(item["writes"] for item in found if item["run_id"] == "abc") == "2"


def test_records_skip_padding_and_score_present_members():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GroupIM(64, [4], aggregator_type="attention").to(device)
    group = torch.tensor([[10], [11]], device=device)
    users = torch.tensor([[0, 1, 7], [2, 3, 7]], device=device)
    mask = torch.tensor([[0.0, 0.0, float("-inf")], [0.0, 0.0, float("-inf")]], device=device)
    items = torch.zeros(2, 64, device=device)
    items[0, 1] = 1
    items[1, 2] = 1
    user_items = torch.zeros(2, 3, 64, device=device)
    user_items[0, 0, 0] = 1
    user_items[0, 1, 0] = 1
    user_items[1, 0, 3] = 1
    user_items[1, 1, 4] = 1
    groups, members = records_from_batch(model, group, users, mask, items, user_items)
    assert [row["group_size"] for row in groups] == [2, 2]
    assert groups[0]["coherence"] == pytest.approx(1.0)
    assert groups[0]["aggregate_diversity"] == 1
    assert groups[1]["aggregate_diversity"] == 2
    assert {row["user_id"] for row in members} == {0, 1, 2, 3}


def test_figures_and_catalog_follow_the_log(tmp_path: Path):
    log = """
Train loss: 3.0
| end of epoch   1 | time: 1.00s | n100 (group) 0.2000 | r20 (group) 0.3000 | r50 (group) 0.4000
Train loss: 2.5
| end of epoch   2 | time: 1.00s | n100 (group) 0.2500 | r20 (group) 0.3500 | r50 (group) 0.4500
"""
    val, loss = parse_group_log(log)
    assert [epoch for epoch, *_rest in val] == [1, 2]
    assert [epoch for epoch, _loss in loss] == [1, 2]
    figures = write_run_figures(log, tmp_path / "figures", "weeplaces", 7)
    assert len(figures) == 4
    assert "Recall@20" in figures[0].read_text(encoding="utf-8")
    run_dir = tmp_path / "runs" / "deadbeef"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(_config(seed=7)), encoding="utf-8")
    (run_dir / "metrics.csv").write_text(
        "n20,n50,r20,r50,seconds\n0.1,0.2,0.3,0.4,9\n", encoding="utf-8"
    )
    catalog = tmp_path / "catalog.md"
    catalog.write_text(
        "# Figure catalog\n\n<!-- launcher-runs:start -->\n<!-- launcher-runs:end -->\n",
        encoding="utf-8",
    )
    rebuild_generated_catalog(catalog, tmp_path / "runs")
    text = catalog.read_text(encoding="utf-8")
    assert "`deadbeef`" in text
    assert "seed 7" in text


def test_train_command_points_artifacts_at_the_run_directory(tmp_path: Path):
    command = train_command(_config(seed=3, pretrain_user=True, cuda=True), tmp_path)
    assert "--artifact-dir" in command
    assert str(tmp_path) in command
    assert command[command.index("--seed") + 1] == "3"
    assert "--cuda" in command
    assert "--pretrain_user" in command
