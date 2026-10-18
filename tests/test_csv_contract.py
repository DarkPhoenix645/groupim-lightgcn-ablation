"""Shared author-CSV emitter: synthetic round trip, Weeplaces reference, Table 1."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from groupim_lightgcn_ablation.csv_contract import (
    SPLIT_SEED,
    assert_loaders,
    dataset_statistics,
    emit_author_csvs,
    layout_errors,
    read_author_csvs,
    render_table1_report,
    split_counts,
    write_table1_report,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _synthetic() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    rows: list[tuple[int, int]] = []
    for user in range(40):
        for item in rng.choice(15, size=6, replace=False):
            rows.append((user, int(item)))
    rows.append((0, 100))
    user_item = pd.DataFrame(rows, columns=["user", "item"])

    membership: list[tuple[int, int]] = []
    group_item: list[tuple[int, int]] = []
    for group in range(30):
        first = group % 40
        second = (group * 3 + 1) % 40
        if first == second:
            second = (second + 1) % 40
        membership.extend([(group, first), (group, second)])
        group_item.append((group, int(rng.integers(0, 15))))
    # Same members as group 0 (users 0 and 1), so the emitter must keep them together.
    membership.extend([(30, 0), (30, 1)])
    group_item.append((30, 3))
    return (
        user_item,
        pd.DataFrame(group_item, columns=["group", "item"]),
        pd.DataFrame(membership, columns=["group", "user"]),
    )


def test_split_counts_match_weeplaces_sizes() -> None:
    assert split_counts(8643) == (6050, 864, 1729)
    assert split_counts(22733) == (15913, 2273, 4547)


def test_emit_round_trip_loads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ui, gi, gu = _synthetic()
    out = DATA / "_contract_smoke"
    monkeypatch.chdir(ROOT)
    emit_author_csvs(ui, gi, gu, out, seed=SPLIT_SEED)
    try:
        frames = read_author_csvs(out)
        assert layout_errors(frames) == []
        stats = dataset_statistics(out)
        assert stats["users"] == 40
        assert stats["groups"] == 31
        train_items = set(frames["train_ui.csv"]["item"])
        all_items = set()
        for name in (
            "train_ui.csv",
            "val_ui_tr.csv",
            "val_ui_te.csv",
            "test_ui_tr.csv",
            "test_ui_te.csv",
            "train_gi.csv",
            "val_gi.csv",
            "test_gi.csv",
        ):
            all_items.update(frames[name]["item"])
        assert max(train_items) == max(all_items)
        assert_loaders("_contract_smoke", negs_per_group=5)
    finally:
        for path in out.glob("*.csv"):
            path.unlink()
        out.rmdir()


def test_repeated_member_combination_stays_in_one_split(tmp_path: Path) -> None:
    ui, gi, gu = _synthetic()
    out = tmp_path / "csv"
    emit_author_csvs(ui, gi, gu, out)
    frames = read_author_csvs(out)
    members = frames["group_users.csv"].groupby("group")["user"].apply(lambda s: frozenset(s))
    by_split = {
        "train": set(frames["train_gi.csv"]["group"]),
        "val": set(frames["val_gi.csv"]["group"]),
        "test": set(frames["test_gi.csv"]["group"]),
    }
    homes: dict[frozenset[int], str] = {}
    for split, groups in by_split.items():
        for group in groups:
            combo = members.loc[group]
            previous = homes.setdefault(combo, split)
            assert previous == split


def test_repeated_outings_pack_into_the_training_split(tmp_path: Path) -> None:
    """Thirty copies of one member set exceed the 10% and 20% quotas of 100 groups."""
    rows = [(user, item) for user in range(40) for item in range(6)]
    rows.append((0, 100))
    user_item = pd.DataFrame(rows, columns=["user", "item"])
    membership: list[tuple[int, int]] = []
    group_item: list[tuple[int, int]] = []
    for group in range(30):
        membership.extend([(group, 0), (group, 1)])
        group_item.append((group, group % 6))
    for group in range(30, 100):
        first = 2 + (group % 30)
        second = 3 + (group % 30)
        membership.extend([(group, first), (group, second)])
        group_item.append((group, 0))
    out = tmp_path / "packed"
    emit_author_csvs(
        user_item,
        pd.DataFrame(group_item, columns=["group", "item"]),
        pd.DataFrame(membership, columns=["group", "user"]),
        out,
    )
    frames = read_author_csvs(out)
    train_groups = set(frames["train_gi.csv"]["group"])
    assert set(range(30)).issubset(train_groups)


def test_member_set_larger_than_the_training_quota_is_rejected(tmp_path: Path) -> None:
    rows = [(user, item) for user in range(40) for item in range(6)]
    user_item = pd.DataFrame(rows, columns=["user", "item"])
    membership: list[tuple[int, int]] = []
    group_item: list[tuple[int, int]] = []
    for group in range(80):
        membership.extend([(group, 0), (group, 1)])
        group_item.append((group, 0))
    for group in range(80, 100):
        membership.extend([(group, 2), (group, 3 + (group - 80))])
        group_item.append((group, 1))
    with pytest.raises(ValueError, match="does not fit"):
        emit_author_csvs(
            user_item,
            pd.DataFrame(group_item, columns=["group", "item"]),
            pd.DataFrame(membership, columns=["group", "user"]),
            tmp_path / "oversize",
        )


def test_emit_is_reproducible(tmp_path: Path) -> None:
    ui, gi, gu = _synthetic()
    first = tmp_path / "a"
    second = tmp_path / "b"
    emit_author_csvs(ui, gi, gu, first, seed=7)
    shuffled = ui.sample(frac=1, random_state=1)
    emit_author_csvs(shuffled, gi.iloc[::-1], gu.sample(frac=1, random_state=2), second, seed=7)
    for name in first.iterdir():
        assert (first / name.name).read_bytes() == (second / name.name).read_bytes()


def test_member_without_an_interaction_is_rejected(tmp_path: Path) -> None:
    ui, gi, gu = _synthetic()
    gu = pd.concat([gu, pd.DataFrame({"group": [0], "user": [999]})], ignore_index=True)
    with pytest.raises(ValueError, match="group member"):
        emit_author_csvs(ui, gi, gu, tmp_path / "bad")


def test_weeplaces_blocks_match_the_measured_reference() -> None:
    frames = read_author_csvs(DATA / "weeplaces")
    train_users = set(frames["train_ui.csv"]["user"])
    val_users = set(frames["val_ui_tr.csv"]["user"]) | set(frames["val_ui_te.csv"]["user"])
    test_users = set(frames["test_ui_tr.csv"]["user"]) | set(frames["test_ui_te.csv"]["user"])
    assert (min(train_users), max(train_users), len(train_users)) == (0, 6049, 6050)
    assert (min(val_users), max(val_users), len(val_users)) == (6050, 6913, 864)
    assert (min(test_users), max(test_users), len(test_users)) == (6914, 8642, 1729)
    assert train_users.isdisjoint(val_users) and val_users.isdisjoint(test_users)
    train_pairs = set(
        zip(frames["train_ui.csv"]["user"], frames["train_ui.csv"]["item"], strict=True)
    )
    held_out = pd.concat([frames["val_ui_te.csv"], frames["test_ui_te.csv"]], ignore_index=True)
    held_pairs = set(zip(held_out["user"], held_out["item"], strict=True))
    assert train_pairs.isdisjoint(held_pairs)
    assert set(frames["train_gi.csv"]["group"]) == set(range(15913))
    assert set(frames["val_gi.csv"]["group"]) == set(range(15913, 18186))
    assert set(frames["test_gi.csv"]["group"]) == set(range(18186, 22733))


def test_table1_report_records_weeplaces_and_leaves_the_others(tmp_path: Path) -> None:
    stats = dataset_statistics(DATA / "weeplaces")
    assert stats["users"] == 8643
    assert stats["items"] == 25081
    assert stats["groups"] == 22733
    assert stats["user_item"] == 508485
    assert stats["group_item"] == 67130
    text = render_table1_report(DATA)
    assert "1,358,458" in text
    assert "508,485" in text
    assert "| yelp | status |  | not converted |  |" in text
    for dataset in ("gowalla", "douban"):
        status = f"| {dataset} | status |  | not converted |  |"
        if (DATA / dataset / "train_ui.csv").is_file():
            assert status not in text
        else:
            assert status in text
    report = tmp_path / "table1.md"
    write_table1_report(DATA, report)
    assert report.read_text(encoding="utf-8") == text
