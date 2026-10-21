"""Write and check the nine-file GroupIM CSV contract.

The split rule is the one measured on ``data/weeplaces/``, which is the format
reference. New datasets go through :func:`emit_author_csvs`. They do not get a
second on-disk schema.

Measured layout:

* Users and groups are each partitioned 70/10/20. Ids are one contiguous range
  per entity: training, then validation, then test.
* ``train_ui`` holds every interaction of the training users. A training user
  who is also a member of a validation or test group still keeps those
  interactions here.
* Validation and test users are a disjoint user-level evaluation set. Each of
  those users holds out ``ceil(0.4 * n)`` items (a user with one item holds
  that item out). The rest are fold-in. Held-out pairs are not copied into
  ``train_ui``. On Weeplaces this matches about 93% of evaluation users; the
  rest differ by one item, so the closed rule is the ceiling.
* Group training still reads ``train_ui`` plus the two fold-in files. That is
  how individual interactions of evaluation users enter the user matrix. The
  held-out slice stays a ranking target.
* A validation or test member combination does not appear in training. Groups
  that share a member combination stay in the same split. Larger combinations
  are placed first so a repeated outing still fits one side of the 70/10/20
  cut. The released Weeplaces files leak a few combinations; emitted datasets
  do not.

``SPLIT_SEED`` is 1111. Users, groups, and item holdout draw from three
streams (``seed``, ``seed + 1``, ``seed + 2``) so that one draw does not
reshuffle the others. Within each split, new ids follow sorted raw ids.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_SEED = 1111

USER_ITEM_FILES = (
    "train_ui.csv",
    "val_ui_tr.csv",
    "val_ui_te.csv",
    "test_ui_tr.csv",
    "test_ui_te.csv",
)
GROUP_ITEM_FILES = ("train_gi.csv", "val_gi.csv", "test_gi.csv")
MEMBERSHIP_FILE = "group_users.csv"
AUTHOR_FILES = USER_ITEM_FILES + GROUP_ITEM_FILES + (MEMBERSHIP_FILE,)

# Paper Table 1, in the column order printed in the PDF. Weeplaces user-item
# and group-item totals disagree with the printed averages and with the
# released CSVs; the averages match the CSVs. See docs/table1-comparison.md.
PAPER_TABLE1: dict[str, dict[str, float]] = {
    "yelp": {
        "users": 7037,
        "items": 7671,
        "groups": 10214,
        "user_item": 220091,
        "group_item": 11160,
        "items_per_user": 31.3,
        "items_per_group": 1.09,
        "group_size": 6.8,
    },
    "weeplaces": {
        "users": 8643,
        "items": 25081,
        "groups": 22733,
        "user_item": 1358458,
        "group_item": 180229,
        "items_per_user": 58.83,
        "items_per_group": 2.95,
        "group_size": 2.9,
    },
    "gowalla": {
        "users": 25405,
        "items": 38306,
        "groups": 44565,
        "user_item": 1025500,
        "group_item": 154526,
        "items_per_user": 40.37,
        "items_per_group": 3.47,
        "group_size": 2.8,
    },
    "douban": {
        "users": 23032,
        "items": 10039,
        "groups": 58983,
        "user_item": 1731429,
        "group_item": 93635,
        "items_per_user": 75.17,
        "items_per_group": 1.59,
        "group_size": 4.2,
    },
}

DATASETS = ("weeplaces", "yelp", "gowalla", "douban")

_SPLIT_ORDER = ("train", "val", "test")


def heldout_count(n_items: int) -> int:
    """Items held out of a validation or test user's history."""
    if n_items < 1:
        raise ValueError("a user history must contain at least one item")
    if n_items == 1:
        return 1
    return math.ceil(0.4 * n_items)


def split_counts(n: int) -> tuple[int, int, int]:
    """Return ``(train, val, test)`` sizes for a 70/10/20 partition.

    Half values round half up. This reproduces the author Weeplaces sizes
    (users 6050/864/1729, groups 15913/2273/4547).
    """
    if n < 3:
        raise ValueError(f"need at least 3 entities to fill train/val/test, got {n}")
    n_train = _round_half_up(0.7 * n)
    n_val = _round_half_up(0.1 * n)
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        raise ValueError(
            f"70/10/20 split of {n} leaves an empty split ({n_train}, {n_val}, {n_test})"
        )
    return n_train, n_val, n_test


def emit_author_csvs(
    user_item: pd.DataFrame,
    group_item: pd.DataFrame,
    group_user: pd.DataFrame,
    out_dir: Path,
    *,
    seed: int = SPLIT_SEED,
) -> None:
    """Renumber raw interactions and write the nine author CSVs to ``out_dir``."""
    ui = _require_pairs(user_item, ("user", "item"), "user-item")
    gi = _require_pairs(group_item, ("group", "item"), "group-item")
    gu = _require_pairs(group_user, ("group", "user"), "group-user")

    users = set(ui["user"])
    groups = set(gi["group"])
    if set(gu["group"]) != groups:
        raise ValueError("group-item and group-user must name the same groups")
    missing_members = set(gu["user"]) - users
    if missing_members:
        raise ValueError(
            "every group member needs at least one user-item interaction; "
            f"{len(missing_members)} members have none"
        )

    user_split = _assign_entities(sorted(users, key=_sort_key), seed)
    group_split = _assign_groups(gu, seed + 1)
    user_items = {
        user: sorted(set(items), key=_sort_key) for user, items in ui.groupby("user")["item"]
    }
    heldout, foldin = _holdout_items(user_items, user_split, seed + 2)

    user_ids = _block_ids(user_split)
    group_ids = _block_ids(group_split)
    _pin_max_user_id(user_ids, user_split, foldin)
    item_ids = _item_ids(ui, gi, user_split, heldout)

    frames = _build_frames(
        ui, gi, gu, user_split, group_split, user_ids, group_ids, item_ids, heldout
    )
    errors = layout_errors(frames)
    if errors:
        raise ValueError("emitted CSVs violate the author contract:\n" + "\n".join(errors))
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(out_dir / name, index=False, lineterminator="\n")


def read_author_csvs(directory: Path) -> dict[str, pd.DataFrame]:
    """Read the nine CSVs. Values must be integral. Headers are fixed."""
    frames: dict[str, pd.DataFrame] = {}
    for name in AUTHOR_FILES:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(path)
        columns = (
            ("group", "user")
            if name == MEMBERSHIP_FILE
            else (("group", "item") if name in GROUP_ITEM_FILES else ("user", "item"))
        )
        frame = pd.read_csv(path)
        if tuple(frame.columns) != columns:
            raise ValueError(f"{name} columns {list(frame.columns)} != {list(columns)}")
        frames[name] = _integral_frame(frame, name)
    return frames


def layout_errors(frames: dict[str, pd.DataFrame]) -> list[str]:
    """Return contract violations. An empty list means the frames match."""
    errors: list[str] = []
    ui = {name: frames[name] for name in USER_ITEM_FILES}
    gi = {name: frames[name] for name in GROUP_ITEM_FILES}
    gu = frames[MEMBERSHIP_FILE]

    for name, frame in frames.items():
        if frame.duplicated().any():
            errors.append(f"{name} has duplicate rows")

    train_users = set(ui["train_ui.csv"]["user"])
    val_users = set(ui["val_ui_tr.csv"]["user"]) | set(ui["val_ui_te.csv"]["user"])
    test_users = set(ui["test_ui_tr.csv"]["user"]) | set(ui["test_ui_te.csv"]["user"])
    blocks = {"train": train_users, "val": val_users, "test": test_users}
    if train_users & val_users or train_users & test_users or val_users & test_users:
        errors.append("user splits overlap")
    _check_contiguous_blocks(blocks, "user", errors)

    try:
        expected = split_counts(len(train_users | val_users | test_users))
    except ValueError as exc:
        errors.append(str(exc))
    else:
        actual = (len(train_users), len(val_users), len(test_users))
        if actual != expected:
            errors.append(f"user split {actual} != 70/10/20 {expected}")

    _check_holdout(ui["val_ui_tr.csv"], ui["val_ui_te.csv"], "val", errors)
    _check_holdout(ui["test_ui_tr.csv"], ui["test_ui_te.csv"], "test", errors)
    train_pairs = _pairs(ui["train_ui.csv"])
    for name in ("val_ui_tr.csv", "val_ui_te.csv", "test_ui_tr.csv", "test_ui_te.csv"):
        overlap = len(train_pairs & _pairs(ui[name]))
        if overlap:
            errors.append(f"{overlap} {name} pairs are also in train_ui")

    group_blocks = {
        "train": set(gi["train_gi.csv"]["group"]),
        "val": set(gi["val_gi.csv"]["group"]),
        "test": set(gi["test_gi.csv"]["group"]),
    }
    if group_blocks["train"] & group_blocks["val"] or group_blocks["train"] & group_blocks["test"]:
        errors.append("group splits overlap")
    if group_blocks["val"] & group_blocks["test"]:
        errors.append("validation and test groups overlap")
    _check_contiguous_blocks(group_blocks, "group", errors)
    all_groups = set().union(*group_blocks.values())
    try:
        expected_groups = split_counts(len(all_groups))
    except ValueError as exc:
        errors.append(str(exc))
    else:
        actual_groups = tuple(len(group_blocks[name]) for name in _SPLIT_ORDER)
        if actual_groups != expected_groups:
            errors.append(f"group split {actual_groups} != 70/10/20 {expected_groups}")
    if set(gu["group"]) != all_groups:
        errors.append("group_users groups differ from group-item groups")

    all_users = train_users | val_users | test_users
    if set(gu["user"]) - all_users:
        errors.append("group_users names a user with no user-item rows")

    member_sets = {split: _member_sets(gu, groups) for split, groups in group_blocks.items()}
    leaked_val = member_sets["val"] & member_sets["train"]
    leaked_test = member_sets["test"] & member_sets["train"]
    if leaked_val or leaked_test:
        errors.append(
            f"ephemeral split leaked {len(leaked_val)} val and {len(leaked_test)} test "
            "member combinations into training"
        )

    all_items: set[int] = set()
    for frame in list(ui.values()) + list(gi.values()):
        all_items.update(int(item) for item in frame["item"])
    if all_items:
        train_item_max = int(ui["train_ui.csv"]["item"].max()) if len(ui["train_ui.csv"]) else -1
        if train_item_max != max(all_items):
            errors.append(f"train_ui max item {train_item_max} != global max item {max(all_items)}")
        if min(all_items) != 0 or max(all_items) + 1 != len(all_items):
            errors.append("item ids are not a contiguous range starting at 0")

    present_users: set[int] = set()
    for name in ("train_ui.csv", "val_ui_tr.csv", "test_ui_tr.csv"):
        present_users.update(int(user) for user in ui[name]["user"])
    if all_users and max(all_users) not in present_users:
        errors.append("the highest user id is absent from train_ui and the fold-in files")
    return errors


def assert_loaders(dataset: str, *, negs_per_group: int = 5) -> None:
    """Construct the author loaders and read one batch from each split.

    The dataset directory must be ``data/<dataset>/`` relative to the current
    working directory. That is the path the loaders hard-code. ``train.py`` puts
    that directory on the import path; this check does the same.
    """
    root = str(Path.cwd())
    if root not in sys.path:
        sys.path.insert(0, root)
    from utils.group_utils import EvalGroupDataset, TrainGroupDataset
    from utils.user_utils import EvalUserDataset, TrainUserDataset

    train_users = TrainUserDataset(dataset)
    n_items = train_users.n_items
    _ = train_users[0]
    train_groups = TrainGroupDataset(dataset, n_items, negs_per_group)
    _ = train_groups[0]
    padding = train_groups.padding_idx
    for datatype in ("val", "test"):
        eval_users = EvalUserDataset(dataset, n_items, datatype)
        _ = eval_users[0]
        eval_groups = EvalGroupDataset(dataset, n_items, padding, datatype)
        _ = eval_groups[0]


def dataset_statistics(directory: Path) -> dict[str, float]:
    """Full-dataset counts, across all nine files, for a Table 1 comparison."""
    frames = read_author_csvs(directory)
    ui = pd.concat([frames[name] for name in USER_ITEM_FILES], ignore_index=True)
    gi = pd.concat([frames[name] for name in GROUP_ITEM_FILES], ignore_index=True)
    gu = frames[MEMBERSHIP_FILE]
    n_users = int(ui["user"].nunique())
    n_items = int(pd.concat([ui["item"], gi["item"]], ignore_index=True).nunique())
    n_groups = int(gi["group"].nunique())
    n_ui = len(ui)
    n_gi = len(gi)
    return {
        "users": n_users,
        "items": n_items,
        "groups": n_groups,
        "user_item": n_ui,
        "group_item": n_gi,
        "items_per_user": n_ui / n_users,
        "items_per_group": n_gi / n_groups,
        "group_size": len(gu) / n_groups,
    }


def render_table1_report(data_root: Path, notes_dir: Path | None = None) -> str:
    """Markdown Table 1 comparison. Missing datasets stay as paper targets only."""
    lines = [
        "# Table 1 comparison",
        "",
        "Paper counts are GroupIM Table 1. Computed counts read the nine author CSVs:",
        "distinct users, distinct items, distinct groups, interaction rows, and the three averages.",
        "A gap is recorded here. A dataset is not dropped because a count misses the paper (ADR-0005).",
        "",
        "Weeplaces is the format reference. Its user, item, and group counts match Table 1.",
        "The printed Weeplaces user-item total (1,358,458) and group-item total (180,229) do not",
        "match the printed averages or the released CSVs. The averages do: about 58.83 items per",
        "user and 2.95 items per group. Weeplaces is not rebuilt.",
        "",
        "A dataset row is filled from data/<name>/ when its nine CSVs exist.",
        "A dataset without those files stays not converted.",
        "Per-dataset notes live in docs/dataset-notes/.",
        "",
        "| Dataset | Field | Paper | Computed | Delta |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for dataset in DATASETS:
        paper = PAPER_TABLE1[dataset]
        directory = data_root / dataset
        if not (directory / "train_ui.csv").is_file():
            lines.append(f"| {dataset} | status |  | not converted |  |")
            continue
        stats = dataset_statistics(directory)
        for field, paper_value in paper.items():
            computed = stats[field]
            delta = computed - paper_value
            lines.append(
                f"| {dataset} | {field} | {_format_stat(paper_value)} | "
                f"{_format_stat(computed)} | {_format_stat(delta)} |"
            )
    if notes_dir is not None and notes_dir.is_dir():
        for note in sorted(notes_dir.glob("*.md")):
            lines.extend(["", note.read_text(encoding="utf-8").rstrip()])
    lines.append("")
    return "\n".join(lines)


def write_table1_report(data_root: Path, out_path: Path, notes_dir: Path | None = None) -> None:
    out_path.write_text(render_table1_report(data_root, notes_dir), encoding="utf-8")


def _round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


def _sort_key(value: object) -> tuple:
    if isinstance(value, (int, np.integer)):
        return (0, int(value))
    return (1, str(value))


def _require_pairs(frame: pd.DataFrame, columns: tuple[str, str], label: str) -> pd.DataFrame:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} frame is missing columns {missing}")
    out = frame.loc[:, list(columns)].copy()
    if out.empty:
        raise ValueError(f"{label} frame is empty")
    if out.duplicated().any():
        raise ValueError(f"{label} frame has duplicate pairs")
    if out.isna().any().any():
        raise ValueError(f"{label} frame has missing values")
    return out


def _assign_entities(entities: list, seed: int) -> dict[str, list]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(entities))
    shuffled = [entities[int(index)] for index in order]
    n_train, n_val, _ = split_counts(len(shuffled))
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def _assign_groups(group_user: pd.DataFrame, seed: int) -> dict[str, list]:
    clusters: dict[frozenset, list] = {}
    for group, members in group_user.groupby("group")["user"]:
        clusters.setdefault(frozenset(members), []).append(group)
    keys = list(clusters)
    rng = np.random.default_rng(seed)
    n_train, n_val, n_test = split_counts(sum(len(groups) for groups in clusters.values()))
    capacity = {"train": n_train, "val": n_val, "test": n_test}
    rank = {"train": 0, "val": 1, "test": 2}
    assigned: dict[str, list] = {split: [] for split in _SPLIT_ORDER}
    order = [int(index) for index in rng.permutation(len(keys))]
    order.sort(key=lambda index: len(clusters[keys[index]]), reverse=True)
    for index in order:
        key = keys[index]
        groups = clusters[key]
        fits = [split for split, room in capacity.items() if room >= len(groups)]
        if not fits:
            raise ValueError(
                "a member combination covers "
                f"{len(groups)} groups and does not fit a 70/10/20 split"
            )
        choice = max(fits, key=lambda split: (capacity[split], -rank[split]))
        capacity[choice] -= len(groups)
        assigned[choice].extend(groups)
    return assigned


def _holdout_items(user_items: dict, user_split: dict[str, list], seed: int) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    heldout: dict = {}
    foldin: dict = {}
    for split in ("val", "test"):
        for user in sorted(user_split[split], key=_sort_key):
            items = user_items[user]
            order = rng.permutation(len(items))
            shuffled = [items[int(index)] for index in order]
            cut = heldout_count(len(shuffled))
            heldout[user] = shuffled[:cut]
            foldin[user] = shuffled[cut:]
    return heldout, foldin


def _block_ids(split_members: dict[str, list]) -> dict:
    mapping: dict = {}
    next_id = 0
    for split in _SPLIT_ORDER:
        for raw in sorted(split_members[split], key=_sort_key):
            mapping[raw] = next_id
            next_id += 1
    return mapping


def _pin_max_user_id(user_ids: dict, user_split: dict[str, list], foldin: dict) -> None:
    """Keep the highest user id inside the test block and present in a fold-in file.

    The group loader sizes its user matrix from the maximum id in ``train_ui``
    plus the two fold-in files. The test block owns that maximum id.
    """
    for split in ("val", "test"):
        if not any(foldin.get(user) for user in user_split[split]):
            raise ValueError(
                f"need at least one {split} user with 2 or more items so {split}_ui_tr is non-empty"
            )
    present_test = [user for user in user_split["test"] if foldin.get(user)]
    max_raw = max(user_ids, key=lambda raw: user_ids[raw])
    if foldin.get(max_raw):
        return
    donor = max(present_test, key=lambda user: user_ids[user])
    user_ids[max_raw], user_ids[donor] = user_ids[donor], user_ids[max_raw]


def _item_ids(
    user_item: pd.DataFrame,
    group_item: pd.DataFrame,
    user_split: dict[str, list],
    heldout: dict,
) -> dict:
    """Map raw items onto ``0 .. n-1`` with the highest id present in ``train_ui``.

    ``TrainUserDataset`` sets ``n_items`` from ``train_ui.item.max() + 1``. Every
    other file indexes into that width, so the maximum id has to occur there.
    """
    raw_items = set(user_item["item"]) | set(group_item["item"])
    mapping = {raw: index for index, raw in enumerate(sorted(raw_items, key=_sort_key))}
    train_users = set(user_split["train"])
    heldout_pairs = {(user, item) for user, items in heldout.items() for item in items}
    train_raw = {
        item
        for user, item in zip(user_item["user"], user_item["item"], strict=True)
        if user in train_users and (user, item) not in heldout_pairs
    }
    if not train_raw:
        raise ValueError("train_ui would contain no items")
    max_raw = max(mapping, key=mapping.get)
    if max_raw in train_raw:
        return mapping
    donor = min(train_raw, key=mapping.get)
    mapping[max_raw], mapping[donor] = mapping[donor], mapping[max_raw]
    return mapping


def _build_frames(
    user_item: pd.DataFrame,
    group_item: pd.DataFrame,
    group_user: pd.DataFrame,
    user_split: dict[str, list],
    group_split: dict[str, list],
    user_ids: dict,
    group_ids: dict,
    item_ids: dict,
    heldout: dict,
) -> dict[str, pd.DataFrame]:
    split_of_user = {user: split for split, users in user_split.items() for user in users}
    split_of_group = {group: split for split, groups in group_split.items() for group in groups}
    rows: dict[str, list[tuple[int, int]]] = {name: [] for name in USER_ITEM_FILES}
    file_of = {
        ("train", "all"): "train_ui.csv",
        ("val", "fold"): "val_ui_tr.csv",
        ("val", "hold"): "val_ui_te.csv",
        ("test", "fold"): "test_ui_tr.csv",
        ("test", "hold"): "test_ui_te.csv",
    }
    heldout_sets = {user: set(items) for user, items in heldout.items()}
    for user, item in zip(user_item["user"], user_item["item"], strict=True):
        split = split_of_user[user]
        if split == "train":
            kind = "all"
        elif item in heldout_sets.get(user, ()):
            kind = "hold"
        else:
            kind = "fold"
        rows[file_of[(split, kind)]].append((user_ids[user], item_ids[item]))

    gi_rows: dict[str, list[tuple[int, int]]] = {
        "train_gi.csv": [],
        "val_gi.csv": [],
        "test_gi.csv": [],
    }
    gi_file = {"train": "train_gi.csv", "val": "val_gi.csv", "test": "test_gi.csv"}
    for group, item in zip(group_item["group"], group_item["item"], strict=True):
        gi_rows[gi_file[split_of_group[group]]].append((group_ids[group], item_ids[item]))

    membership = [
        (group_ids[group], user_ids[user])
        for group, user in zip(group_user["group"], group_user["user"], strict=True)
    ]
    frames = {name: _pair_frame(pairs, ("user", "item")) for name, pairs in rows.items()}
    frames.update({name: _pair_frame(pairs, ("group", "item")) for name, pairs in gi_rows.items()})
    frames[MEMBERSHIP_FILE] = _pair_frame(membership, ("group", "user"))
    return frames


def _pair_frame(pairs: list[tuple[int, int]], columns: tuple[str, str]) -> pd.DataFrame:
    frame = pd.DataFrame(pairs, columns=list(columns), dtype="int64")
    return frame.drop_duplicates().sort_values(list(columns)).reset_index(drop=True)


def _integral_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    converted = frame.copy()
    for column in converted.columns:
        numeric = pd.to_numeric(converted[column], errors="coerce")
        if numeric.isna().any() or not np.all(numeric.to_numpy() == np.floor(numeric.to_numpy())):
            raise ValueError(f"{name}.{column} has a non-integral value")
        converted[column] = numeric.astype("int64")
    return converted


def _check_contiguous_blocks(blocks: dict[str, set[int]], label: str, errors: list[str]) -> None:
    cursor = 0
    for split in _SPLIT_ORDER:
        ids = blocks[split]
        if not ids:
            errors.append(f"{label} {split} block is empty")
            return
        expected = set(range(cursor, cursor + len(ids)))
        if ids != expected:
            errors.append(f"{label} {split} ids are not the contiguous block starting at {cursor}")
        cursor += len(ids)


def _check_holdout(
    fold_in: pd.DataFrame, held_out: pd.DataFrame, split: str, errors: list[str]
) -> None:
    if _pairs(fold_in) & _pairs(held_out):
        errors.append(f"{split} fold-in and held-out share pairs")
    fold_counts = fold_in.groupby("user").size()
    hold_counts = held_out.groupby("user").size()
    both = pd.concat([fold_counts.rename("fold"), hold_counts.rename("hold")], axis=1).fillna(0)
    both = both.astype(int)
    for user, row in both.iterrows():
        total = int(row.fold + row.hold)
        expected = heldout_count(total)
        if int(row.hold) != expected:
            errors.append(
                f"{split} user {user} holds out {int(row.hold)} items, expected {expected}"
            )


def _pairs(frame: pd.DataFrame) -> set[tuple[int, int]]:
    return set(zip(frame.iloc[:, 0].astype(int), frame.iloc[:, 1].astype(int), strict=True))


def _member_sets(group_user: pd.DataFrame, groups: set[int]) -> set[frozenset[int]]:
    subset = group_user[group_user["group"].isin(groups)]
    return {
        frozenset(int(user) for user in members) for _, members in subset.groupby("group")["user"]
    }


def _format_stat(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.4f}"
