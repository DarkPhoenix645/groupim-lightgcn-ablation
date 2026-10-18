"""Yelp Los Angeles restaurants to the author CSV contract.

Paper §5.1: keep restaurant check-ins in Los Angeles, build a group from a
user and the friends who visited that restaurant in the same short window, and
keep users and restaurants with five or more visits.

The public Yelp check-in file has no user id, so visits come from reviews.
``date`` values with a clock time use a 15-minute window. Date-only values, and
timestamps that are all midnight, use the calendar day and the Table 1 note
says the 15-minute cut could not be applied.

Each anchor user in an outing produces one group: that user plus friends who
are also in the outing. The same member set is not repeated for every anchor.
A later outing of the same people is a new group. Every surviving visit stays
in the user-item table.

Raw JSON stays in ``data/raw/yelp/`` (gitignored). The nine CSVs go to ``data/yelp/``.

    uv run python -m groupim_lightgcn_ablation.yelp
"""

from __future__ import annotations

import argparse
import bisect
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from groupim_lightgcn_ablation.csv_contract import (
    PAPER_TABLE1,
    _format_stat,
    dataset_statistics,
    emit_author_csvs,
    write_table1_report,
)

BUSINESS_FILE = "yelp_academic_dataset_business.json"
REVIEW_FILE = "yelp_academic_dataset_review.json"
USER_FILE = "yelp_academic_dataset_user.json"
RAW_FILES = (BUSINESS_FILE, REVIEW_FILE, USER_FILE)
BENCHMARK_FILES = ("user_events.dat", "groupid_users.dat", "groupid_events.dat")
BENCHMARK_DIR = Path("data/raw/douban/yelp_la")

WINDOW_15_MIN = "15 minutes"
WINDOW_SAME_DAY = "same calendar day"
MIN_COUNT = 5
_FIFTEEN_MINUTES = timedelta(minutes=15)


@dataclass(frozen=True)
class Visit:
    user: str
    item: str
    at: datetime


@dataclass(frozen=True)
class YelpSummary:
    window: str
    restaurants: int
    reviews: int
    visits: int
    users: int
    items: int
    groups: int
    min_count: int


def convert_yelp(
    raw_dir: Path,
    out_dir: Path,
    *,
    min_count: int = MIN_COUNT,
) -> YelpSummary:
    """Read a Yelp JSON dump and write the nine author CSVs to ``out_dir``."""
    missing = [name for name in RAW_FILES if not (raw_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing {missing} under {raw_dir}")

    restaurants = la_restaurant_ids(raw_dir / BUSINESS_FILE)
    if not restaurants:
        raise ValueError(f"no Los Angeles restaurants in {raw_dir / BUSINESS_FILE}")
    visits, timed = read_visits(raw_dir / REVIEW_FILE, restaurants)
    review_count = len(visits)
    visits = filter_visits(visits, min_count)
    if not visits:
        raise ValueError(f"no visits left after the {min_count}-visit filter")
    window = WINDOW_15_MIN if timed else WINDOW_SAME_DAY
    users = {visit.user for visit in visits}
    friends = friend_graph(raw_dir / USER_FILE, users)
    group_item, group_user = group_interactions(visits, friends, window)
    user_item = pd.DataFrame(
        [(visit.user, visit.item) for visit in visits], columns=["user", "item"]
    )
    emit_author_csvs(user_item, group_item, group_user, out_dir)
    return YelpSummary(
        window=window,
        restaurants=len(restaurants),
        reviews=review_count,
        visits=len(user_item),
        users=int(user_item["user"].nunique()),
        items=int(user_item["item"].nunique()),
        groups=int(group_item["group"].nunique()),
        min_count=min_count,
    )


def la_restaurant_ids(path: Path) -> set[str]:
    """Business ids whose city is Los Angeles, state is CA, and category is Restaurants."""
    found: set[str] = set()
    for business in _iter_objects(path):
        if _is_la_restaurant(business):
            found.add(str(business["business_id"]))
    return found


def read_visits(path: Path, restaurants: set[str]) -> tuple[list[Visit], bool]:
    """Reviews of ``restaurants``. The bool is true when any date has a clock time."""
    visits: list[Visit] = []
    timed = False
    scanned = 0
    for review in _iter_objects(path):
        scanned += 1
        if scanned % 1_000_000 == 0:
            print(f"reviews scanned: {scanned:,}", flush=True)
        business_id = str(review.get("business_id", ""))
        if business_id not in restaurants:
            continue
        user_id = review.get("user_id")
        if not user_id or "date" not in review:
            continue
        at, has_clock = _parse_date(str(review["date"]))
        timed = timed or has_clock
        visits.append(Visit(str(user_id), business_id, at))
    print(f"reviews scanned: {scanned:,}; kept {len(visits):,}", flush=True)
    return visits, timed


def filter_visits(visits: list[Visit], min_count: int) -> list[Visit]:
    """Keep users and restaurants with at least ``min_count`` distinct partners.

    The cut repeats until it is stable. Several reviews of one restaurant count
    once, at the earliest timestamp.
    """
    if min_count < 1:
        raise ValueError("min_count must be at least 1")
    kept = _collapse_visits(visits)
    while True:
        user_items: dict[str, set[str]] = defaultdict(set)
        item_users: dict[str, set[str]] = defaultdict(set)
        for visit in kept:
            user_items[visit.user].add(visit.item)
            item_users[visit.item].add(visit.user)
        users = {user for user, items in user_items.items() if len(items) >= min_count}
        items = {item for item, people in item_users.items() if len(people) >= min_count}
        filtered = [visit for visit in kept if visit.user in users and visit.item in items]
        if len(filtered) == len(kept):
            return filtered
        kept = filtered


def friend_graph(path: Path, users: set[str]) -> dict[str, set[str]]:
    """Undirected friends among ``users``. A one-sided Yelp edge is kept."""
    graph: dict[str, set[str]] = {user: set() for user in users}
    for record in _iter_objects(path):
        user = str(record.get("user_id", ""))
        if user not in users:
            continue
        for friend in _friend_ids(record.get("friends", "")):
            if friend in users and friend != user:
                graph[user].add(friend)
                graph[friend].add(user)
    return graph


def group_interactions(
    visits: list[Visit],
    friends: dict[str, set[str]],
    window: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One group per anchor member-set inside an outing, and one item: the restaurant.

    A 15-minute group is the anchor plus friends with a visit in that window.
    Visits are not chained: a friend 40 minutes later stays in a later outing.
    The same member set in the same window is written once. A later outing of
    those people is another group.
    """
    if window not in (WINDOW_15_MIN, WINDOW_SAME_DAY):
        raise ValueError(f"unknown window {window}")
    by_item: dict[str, list[Visit]] = defaultdict(list)
    for visit in visits:
        by_item[visit.item].append(visit)

    group_item_rows: list[tuple[int, str]] = []
    group_user_rows: list[tuple[int, str]] = []
    next_group = 0
    for item, rows in by_item.items():
        rows.sort(key=lambda visit: (visit.at, visit.user))
        seen: set[tuple[frozenset[str], datetime]] = set()
        for present, in_window in _windows(rows, window):
            for user in sorted(present):
                members = frozenset((friends.get(user, set()) & present) | {user})
                if len(members) < 2:
                    continue
                stamp = min(visit.at for visit in in_window if visit.user in members)
                if (members, stamp) in seen:
                    continue
                seen.add((members, stamp))
                group_item_rows.append((next_group, item))
                for member in sorted(members):
                    group_user_rows.append((next_group, member))
                next_group += 1
    if not group_item_rows:
        raise ValueError("the friend window produced no groups")
    return (
        pd.DataFrame(group_item_rows, columns=["group", "item"]),
        pd.DataFrame(group_user_rows, columns=["group", "user"]),
    )


@dataclass(frozen=True)
class YelpBenchmarkSummary:
    users: int
    items: int
    groups: int
    visits: int
    source_users: int
    source_events: int
    source_groups: int
    source_group_rows: int
    min_count: int


def convert_yelp_benchmark(
    raw_dir: Path,
    out_dir: Path,
    *,
    min_count: int = MIN_COUNT,
) -> YelpBenchmarkSummary:
    """Read Yin et al.'s Yelp-LA release and write the nine author CSVs.

    Groups are already built and the files have no timestamps, so the 15-minute
    window is not applied again. Users and businesses with fewer than
    ``min_count`` distinct partners are dropped.
    """
    user_item, group_item, group_user, meta = released_group_frames(raw_dir, min_count)
    emit_author_csvs(user_item, group_item, group_user, out_dir)
    return YelpBenchmarkSummary(
        users=int(user_item["user"].nunique()),
        items=int(user_item["item"].nunique()),
        groups=int(group_item["group"].nunique()),
        visits=len(user_item),
        source_users=meta["source_users"],
        source_events=meta["source_events"],
        source_groups=meta["source_groups"],
        source_group_rows=meta["source_group_rows"],
        min_count=min_count,
    )


def released_group_frames(
    raw_dir: Path,
    min_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """User-item, group-item, and group-user tables from a Yin ``.dat`` release."""
    missing = [name for name in BENCHMARK_FILES if not (raw_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing {missing} under {raw_dir}")
    user_events, _user_rows = read_id_lists(raw_dir / BENCHMARK_FILES[0])
    group_users, group_rows = read_id_lists(raw_dir / BENCHMARK_FILES[1])
    group_events, _event_rows = read_id_lists(raw_dir / BENCHMARK_FILES[2])
    visits = [
        Visit(user, event, datetime(1970, 1, 1))
        for user, events in user_events.items()
        for event in events
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
        raise ValueError("no released groups survived the filter")
    user_item = pd.DataFrame([(visit.user, visit.item) for visit in kept], columns=["user", "item"])
    return (
        user_item,
        pd.DataFrame(group_item_rows, columns=["group", "item"]),
        pd.DataFrame(group_user_rows, columns=["group", "user"]),
        {
            "source_users": len(user_events),
            "source_events": len({event for events in user_events.values() for event in events}),
            "source_groups": len(group_users),
            "source_group_rows": group_rows,
        },
    )


def read_id_lists(path: Path) -> tuple[dict[str, list[str]], int]:
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


def yelp_benchmark_note(summary: YelpBenchmarkSummary, stats: dict[str, float]) -> str:
    """Table 1 note for the Yin Yelp-LA release."""
    paper = PAPER_TABLE1["yelp"]
    repeated = ""
    if summary.source_group_rows != summary.source_groups:
        repeated = f" ({summary.source_group_rows:,} lines; repeated ids have the same members)"
    lines = [
        "## Yelp",
        "",
        "Source: Yelp-LA benchmark, Yin et al. ICDE 2019, from Hongzhi Yin's dataset page.",
        "The Yelp Open Dataset has no Los Angeles restaurants, so it is not the source.",
        "Groups in this release are already built, and the files have no timestamps,",
        "so the 15-minute window was not applied again.",
        f"Users and businesses with fewer than {summary.min_count} distinct partners are dropped.",
        (
            f"Benchmark before the filter: {summary.source_users:,} users, "
            f"{summary.source_events:,} businesses, {summary.source_groups:,} groups{repeated}."
        ),
        f"Distinct visits after the filter: {summary.visits:,}.",
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


def yelp_deviation_note(summary: YelpSummary, stats: dict[str, float]) -> str:
    """Text appended to the Table 1 comparison after a Yelp JSON conversion."""
    paper = PAPER_TABLE1["yelp"]
    lines = [
        "## Yelp",
        "",
        f"Window: {summary.window}.",
        f"Los Angeles restaurants in the business file: {summary.restaurants:,}.",
        f"Visits after the {summary.min_count}+ filter: {summary.visits:,}.",
        (
            "Every surviving visit is an individual interaction. "
            "A group is one outing of a user and the friends who were there."
        ),
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
    if summary.window == WINDOW_SAME_DAY:
        lines.extend(
            [
                "",
                (
                    "Review dates have no clock time, so the 15-minute cut could not be applied. "
                    "Friends who reviewed the same restaurant on the same calendar day form the group."
                ),
            ]
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert Yelp Los Angeles restaurants to author CSVs"
    )
    parser.add_argument("--raw", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("data/yelp"))
    parser.add_argument("--notes", type=Path, default=Path("docs/dataset-notes/yelp.md"))
    parser.add_argument("--table1", type=Path, default=Path("docs/table1-comparison.md"))
    args = parser.parse_args(argv)
    raw = args.raw if args.raw is not None else _default_raw()
    try:
        if _is_benchmark(raw):
            summary = convert_yelp_benchmark(raw, args.out)
        else:
            summary = convert_yelp(raw, args.out)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}. Place Yin Yelp-LA .dat files in {BENCHMARK_DIR}, or download the "
            f"Yelp Open Dataset and place {', '.join(RAW_FILES)} in data/raw/yelp."
        ) from exc
    stats = dataset_statistics(args.out)
    note = (
        yelp_benchmark_note(summary, stats)
        if isinstance(summary, YelpBenchmarkSummary)
        else yelp_deviation_note(summary, stats)
    )
    args.notes.parent.mkdir(parents=True, exist_ok=True)
    args.notes.write_text(note, encoding="utf-8")
    write_table1_report(args.out.parent, args.table1, notes_dir=args.notes.parent)
    print(
        f"wrote {args.out} ({summary.users:,} users, {summary.items:,} items, "
        f"{summary.groups:,} groups)",
        flush=True,
    )


def _default_raw() -> Path:
    if _is_benchmark(BENCHMARK_DIR):
        return BENCHMARK_DIR
    return Path("data/raw/yelp")


def _is_benchmark(raw: Path) -> bool:
    return all((raw / name).is_file() for name in BENCHMARK_FILES)


def _is_la_restaurant(business: dict) -> bool:
    if str(business.get("state", "")).strip() != "CA":
        return False
    if str(business.get("city", "")).strip() != "Los Angeles":
        return False
    categories = business.get("categories")
    if categories is None:
        return False
    if isinstance(categories, list):
        tokens = {str(part).strip() for part in categories}
    else:
        tokens = {part.strip() for part in str(categories).split(",")}
    return "Restaurants" in tokens


def _parse_date(value: str) -> tuple[datetime, bool]:
    text = value.strip()
    if len(text) == 10:
        return datetime.fromisoformat(text), False
    parsed = datetime.fromisoformat(text.replace("Z", ""))
    has_clock = not (
        parsed.hour == 0 and parsed.minute == 0 and parsed.second == 0 and parsed.microsecond == 0
    )
    return parsed, has_clock


def _collapse_visits(visits: list[Visit]) -> list[Visit]:
    best: dict[tuple[str, str], Visit] = {}
    for visit in visits:
        key = (visit.user, visit.item)
        current = best.get(key)
        if current is None or visit.at < current.at:
            best[key] = visit
    return list(best.values())


def _friend_ids(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text == "None":
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _windows(rows: list[Visit], window: str) -> list[tuple[set[str], list[Visit]]]:
    """Users who share a window, and the visits inside that window."""
    if window == WINDOW_SAME_DAY:
        by_day: dict[datetime, list[Visit]] = defaultdict(list)
        for visit in rows:
            by_day[datetime(visit.at.year, visit.at.month, visit.at.day)].append(visit)
        return [({visit.user for visit in day_rows}, day_rows) for day_rows in by_day.values()]

    times = [visit.at for visit in rows]
    windows: list[tuple[set[str], list[Visit]]] = []
    for visit in rows:
        start = bisect.bisect_left(times, visit.at - _FIFTEEN_MINUTES)
        stop = bisect.bisect_right(times, visit.at + _FIFTEEN_MINUTES)
        chosen = rows[start:stop]
        windows.append(({row.user for row in chosen}, chosen))
    return windows


def _iter_objects(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not a JSON object") from exc


if __name__ == "__main__":
    main()
