"""Douban event attendance to the author CSV contract.

Paper §5.1: items are event venues, a group is the people who attended an
event together, and users and venues with fewer than 10 interactions are
removed.

The public file GroupIM cites is the Douban-Event benchmark released with
Yin et al., ICDE 2019 (Google Drive folder from Hongzhi Yin's dataset page).
That release stores event ids, not venue ids, and the groups are already the
co-attendance sets. This converter applies the 10+ filter to those events and
keeps each released group whose surviving members still share a surviving event.

Raw files stay in ``data/raw/douban/`` (gitignored):

- ``user_events.dat``
- ``groupid_users.dat``
- ``groupid_events.dat``

    uv run python -m groupim_lightgcn_ablation.douban
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from groupim_lightgcn_ablation.csv_contract import (
    PAPER_TABLE1,
    _format_stat,
    dataset_statistics,
    emit_author_csvs,
    write_table1_report,
)
from groupim_lightgcn_ablation.yelp import Visit, filter_visits

USER_EVENTS = "user_events.dat"
GROUP_USERS = "groupid_users.dat"
GROUP_EVENTS = "groupid_events.dat"
RAW_FILES = (USER_EVENTS, GROUP_USERS, GROUP_EVENTS)
MIN_COUNT = 10
_STAMP = datetime(1970, 1, 1)


@dataclass(frozen=True)
class DoubanSummary:
    users: int
    items: int
    groups: int
    visits: int
    source_users: int
    source_events: int
    source_groups: int
    source_group_rows: int
    min_count: int


def convert_douban(
    raw_dir: Path,
    out_dir: Path,
    *,
    min_count: int = MIN_COUNT,
) -> DoubanSummary:
    """Read the Douban-Event benchmark and write the nine author CSVs."""
    missing = [name for name in RAW_FILES if not (raw_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing {missing} under {raw_dir}")

    user_events, _user_rows = _read_lists(raw_dir / USER_EVENTS)
    group_users, group_rows = _read_lists(raw_dir / GROUP_USERS)
    group_events, _event_rows = _read_lists(raw_dir / GROUP_EVENTS)
    visits = [
        Visit(user, event, _STAMP) for user, events in user_events.items() for event in events
    ]
    kept = filter_visits(visits, min_count)
    if not kept:
        raise ValueError(f"no attendance left after the {min_count}+ filter")
    by_user: dict[str, set[str]] = {}
    for visit in kept:
        by_user.setdefault(visit.user, set()).add(visit.item)

    group_item_rows: list[tuple[str, str]] = []
    group_user_rows: list[tuple[str, str]] = []
    for group, members in group_users.items():
        people = [member for member in members if member in by_user]
        if len(people) < 2:
            continue
        attended: set[str] = set()
        for member in people:
            attended.update(by_user[member])
        events = [event for event in group_events.get(group, []) if event in attended]
        if not events:
            continue
        for event in events:
            group_item_rows.append((group, event))
        for member in people:
            group_user_rows.append((group, member))
    if not group_item_rows:
        raise ValueError("no co-attendance groups survived the filter")

    user_item = pd.DataFrame([(visit.user, visit.item) for visit in kept], columns=["user", "item"])
    emit_author_csvs(
        user_item,
        pd.DataFrame(group_item_rows, columns=["group", "item"]),
        pd.DataFrame(group_user_rows, columns=["group", "user"]),
        out_dir,
    )
    return DoubanSummary(
        users=int(user_item["user"].nunique()),
        items=int(user_item["item"].nunique()),
        groups=len({group for group, _ in group_item_rows}),
        visits=len(user_item),
        source_users=len(user_events),
        source_events=len({event for events in user_events.values() for event in events}),
        source_groups=len(group_users),
        source_group_rows=group_rows,
        min_count=min_count,
    )


def douban_deviation_note(summary: DoubanSummary, stats: dict[str, float]) -> str:
    """Text appended to the Table 1 comparison after a Douban conversion."""
    paper = PAPER_TABLE1["douban"]
    repeated = ""
    if summary.source_group_rows != summary.source_groups:
        repeated = f" ({summary.source_group_rows:,} lines; repeated ids have the same members)"
    lines = [
        "## Douban",
        "",
        "Source: Douban-Event benchmark, Yin et al. ICDE 2019, from Hongzhi Yin's dataset page.",
        "The release stores event ids. It has no venue id, so the item is the event.",
        "A group is a co-attendance set from that benchmark, kept when at least two",
        f"members and one shared event survive the {summary.min_count}+ filter.",
        (
            f"Benchmark before the filter: {summary.source_users:,} users, "
            f"{summary.source_events:,} events, {summary.source_groups:,} groups{repeated}."
        ),
        f"Distinct attendance after the filter: {summary.visits:,}.",
        "",
        "| Field | Paper | Computed | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for field, paper_value in paper.items():
        computed = stats[field]
        lines.append(
            f"| {field} | {_format_stat(paper_value)} | {_format_stat(computed)} | "
            f"{_format_stat(computed - paper_value)} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert Douban events to author CSVs")
    parser.add_argument("--raw", type=Path, default=Path("data/raw/douban/douban_sh"))
    parser.add_argument("--out", type=Path, default=Path("data/douban"))
    parser.add_argument("--notes", type=Path, default=Path("docs/dataset-notes/douban.md"))
    parser.add_argument("--table1", type=Path, default=Path("docs/table1-comparison.md"))
    args = parser.parse_args(argv)
    try:
        summary = convert_douban(args.raw, args.out)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}. Download Datasets.zip from Hongzhi Yin's Douban-Event and Yelp "
            "folder (https://drive.google.com/drive/folders/1YXhBaKh75k8JBwDmYEvl2S_YICqoZjPW) "
            f"and place douban_sh/ under {args.raw.parent}."
        ) from exc
    stats = dataset_statistics(args.out)
    args.notes.parent.mkdir(parents=True, exist_ok=True)
    args.notes.write_text(douban_deviation_note(summary, stats), encoding="utf-8")
    write_table1_report(args.out.parent, args.table1, notes_dir=args.notes.parent)
    print(
        f"wrote {args.out} ({summary.users:,} users, {summary.items:,} items, "
        f"{summary.groups:,} groups)",
        flush=True,
    )


def _read_lists(path: Path) -> tuple[dict[str, list[str]], int]:
    """Map the left-hand id to the comma-separated ids on the right.

    Repeated ids keep one entry. Membership lists that differ only by order
    collapse to the same set.
    """
    rows: dict[str, list[str]] = {}
    n_rows = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            n_rows += 1
            left, _, right = text.partition("\t")
            values = rows.setdefault(left, [])
            for part in right.split(","):
                part = part.strip()
                if part and part not in values:
                    values.append(part)
    if not rows:
        raise ValueError(f"{path} has no rows")
    return rows, n_rows


if __name__ == "__main__":
    main()
