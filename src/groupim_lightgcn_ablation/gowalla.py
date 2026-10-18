"""Gowalla US restaurant check-ins to the author CSV contract.

Paper §5.1: restaurant check-ins across US cities through June 2011, plus the
friend graph. A group is a user and the friends who checked in at the same
place within 15 minutes. Users and places with fewer than five distinct
partners are dropped.

The public file that matches that cutoff and has place categories is the
Gowalla dump redistributed at https://doi.org/10.6084/m9.figshare.22126586
(collected via the Gowalla API, CC0). SNAP's Gowalla file ends in October 2010
and has no categories, so it cannot apply the restaurant rule.

Raw files stay in ``data/raw/gowalla/`` (gitignored):

- ``gowalla_checkins.csv``
- ``gowalla_spots_subset1.csv``
- ``gowalla_friendship.csv``
- ``gowalla_category_structure.json``

    uv run python -m groupim_lightgcn_ablation.gowalla
"""

from __future__ import annotations

import argparse
import csv
import json
import re
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
from groupim_lightgcn_ablation.yelp import (
    WINDOW_15_MIN,
    Visit,
    filter_visits,
    group_interactions,
)

CHECKINS_FILE = "gowalla_checkins.csv"
SPOTS_FILE = "gowalla_spots_subset1.csv"
FRIENDS_FILE = "gowalla_friendship.csv"
CATEGORIES_FILE = "gowalla_category_structure.json"
RAW_FILES = (CHECKINS_FILE, SPOTS_FILE, FRIENDS_FILE, CATEGORIES_FILE)

# "Up to June 2011": keep timestamps in June, drop July.
CUTOFF = datetime(2011, 7, 1)
MIN_COUNT = 5
_CATEGORY_ID = re.compile(r"/categories/(\d+)")


@dataclass(frozen=True)
class GowallaSummary:
    restaurants: int
    checkins: int
    visits: int
    users: int
    items: int
    groups: int
    member_sets: int
    items_per_member_set: float
    member_set_size: float
    min_count: int


def convert_gowalla(
    raw_dir: Path,
    out_dir: Path,
    *,
    min_count: int = MIN_COUNT,
) -> GowallaSummary:
    """Read the Gowalla CSVs and write the nine author CSVs to ``out_dir``."""
    missing = [name for name in RAW_FILES if not (raw_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing {missing} under {raw_dir}")

    food_ids = food_category_ids(raw_dir / CATEGORIES_FILE)
    restaurants = us_restaurant_ids(raw_dir / SPOTS_FILE, food_ids)
    if not restaurants:
        raise ValueError("no US restaurants in the spots file")
    visits = read_checkins(raw_dir / CHECKINS_FILE, restaurants)
    checkin_count = len(visits)
    kept = filter_visits(visits, min_count)
    if not kept:
        raise ValueError(f"no visits left after the {min_count}-visit filter")
    keep_pairs = {(visit.user, visit.item) for visit in kept}
    timed = [visit for visit in visits if (visit.user, visit.item) in keep_pairs]
    users = {visit.user for visit in kept}
    friends = friend_graph(raw_dir / FRIENDS_FILE, users)
    group_item, group_user = group_interactions(timed, friends, WINDOW_15_MIN)
    n_groups = int(group_item["group"].nunique())
    n_sets, items_per_set, set_size = member_set_stats(group_item, group_user)
    print(f"groups built: {n_groups:,} ({n_sets:,} member sets)", flush=True)
    user_item = pd.DataFrame([(visit.user, visit.item) for visit in kept], columns=["user", "item"])
    emit_author_csvs(user_item, group_item, group_user, out_dir)
    return GowallaSummary(
        restaurants=len(restaurants),
        checkins=checkin_count,
        visits=len(user_item),
        users=int(user_item["user"].nunique()),
        items=int(user_item["item"].nunique()),
        groups=n_groups,
        member_sets=n_sets,
        items_per_member_set=items_per_set,
        member_set_size=set_size,
        min_count=min_count,
    )


def food_category_ids(path: Path) -> set[str]:
    """Category ids under the Food root, including Food itself."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    found: set[str] = set()

    def walk(node: dict, under_food: bool) -> None:
        is_food = under_food or node.get("name") == "Food"
        if is_food:
            match = _CATEGORY_ID.search(str(node.get("url", "")))
            if match:
                found.add(match.group(1))
        for child in node.get("spot_categories") or []:
            walk(child, is_food)

    for root in payload["spot_categories"]:
        walk(root, False)
    if not found:
        raise ValueError(f"{path} has no Food categories")
    return found


def us_restaurant_ids(path: Path, food_ids: set[str]) -> set[str]:
    """Spot ids in the United States whose category is under Food."""
    found: set[str] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            categories = row.get("spot_categories") or ""
            if food_ids.isdisjoint(_CATEGORY_ID.findall(categories)):
                continue
            try:
                lat = float(row["lat"])
                lng = float(row["lng"])
            except (KeyError, TypeError, ValueError):
                continue
            if in_united_states(lat, lng):
                found.add(str(row["id"]))
    return found


def read_checkins(path: Path, restaurants: set[str]) -> list[Visit]:
    """Check-ins at ``restaurants`` with a timestamp before July 2011."""
    visits: list[Visit] = []
    scanned = 0
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            scanned += 1
            if scanned % 5_000_000 == 0:
                print(f"check-ins scanned: {scanned:,}", flush=True)
            place = str(row.get("placeid", ""))
            if place not in restaurants:
                continue
            user = row.get("userid")
            if not user or "datetime" not in row:
                continue
            at = datetime.fromisoformat(row["datetime"].strip()).replace(tzinfo=None)
            if at >= CUTOFF:
                continue
            visits.append(Visit(str(user), place, at))
    print(f"check-ins scanned: {scanned:,}; kept {len(visits):,}", flush=True)
    return visits


def friend_graph(path: Path, users: set[str]) -> dict[str, set[str]]:
    """Undirected friendships among ``users``."""
    graph: dict[str, set[str]] = {user: set() for user in users}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            left = str(row.get("userid1", ""))
            right = str(row.get("userid2", ""))
            if left in users and right in users and left != right:
                graph[left].add(right)
                graph[right].add(left)
    return graph


def in_united_states(lat: float, lng: float) -> bool:
    """Continental United States, Alaska, and Hawaii. Territories are left out."""
    if 24.5 <= lat <= 49.4 and -125.0 <= lng <= -66.9:
        return True
    if 51.0 <= lat <= 71.5 and -170.0 <= lng <= -129.0:
        return True
    return 18.5 <= lat <= 22.5 and -160.5 <= lng <= -154.5


def member_set_stats(
    group_item: pd.DataFrame, group_user: pd.DataFrame
) -> tuple[int, float, float]:
    """Distinct member sets, restaurants per set, and mean set size.

    Each outing is its own group. The same people at another restaurant are
    another group, so items per group stays 1. Table 1's items-per-group is
    the collapsed figure.
    """
    members = group_user.groupby("group")["user"].agg(lambda values: frozenset(values))
    places: dict[frozenset, set] = {}
    for group, item in zip(group_item["group"], group_item["item"], strict=True):
        places.setdefault(members.loc[group], set()).add(item)
    n_sets = len(places)
    mean_items = sum(len(items) for items in places.values()) / n_sets
    mean_size = sum(len(combo) for combo in places) / n_sets
    return n_sets, mean_items, mean_size


def gowalla_deviation_note(summary: GowallaSummary, stats: dict[str, float]) -> str:
    paper = PAPER_TABLE1["gowalla"]
    lines = [
        "## Gowalla",
        "",
        "Source: figshare 10.6084/m9.figshare.22126586 (Gowalla API dump).",
        "Restaurants are spots whose category is under Food.",
        "Places are limited to a latitude/longitude box for the continental US, Alaska, and Hawaii.",
        "Check-ins at or after 2011-07-01 are dropped.",
        "Window: 15 minutes.",
        f"US Food spots: {summary.restaurants:,}.",
        f"Check-ins before the {summary.min_count}+ filter: {summary.checkins:,}.",
        f"Distinct visits after the filter: {summary.visits:,}.",
        (
            "Each group is one 15-minute outing at one restaurant, so items per group is 1. "
            f"The same people on another outing are another group. Those {summary.groups:,} "
            f"groups are {summary.member_sets:,} member sets, with "
            f"{summary.items_per_member_set:.2f} restaurants per set and mean size "
            f"{summary.member_set_size:.2f}. Table 1 has 44,565 groups, 3.47 items per group, "
            "and group size 2.8."
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
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert Gowalla US restaurants to author CSVs")
    parser.add_argument("--raw", type=Path, default=Path("data/raw/gowalla"))
    parser.add_argument("--out", type=Path, default=Path("data/gowalla"))
    parser.add_argument("--notes", type=Path, default=Path("docs/dataset-notes/gowalla.md"))
    parser.add_argument("--table1", type=Path, default=Path("docs/table1-comparison.md"))
    args = parser.parse_args(argv)
    try:
        summary = convert_gowalla(args.raw, args.out)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}. Download gowalla_checkins.csv, gowalla_spots_subset1.csv, "
            "gowalla_friendship.csv, and gowalla_category_structure.json from "
            f"https://doi.org/10.6084/m9.figshare.22126586 into {args.raw}."
        ) from exc
    stats = dataset_statistics(args.out)
    args.notes.parent.mkdir(parents=True, exist_ok=True)
    args.notes.write_text(gowalla_deviation_note(summary, stats), encoding="utf-8")
    write_table1_report(args.out.parent, args.table1, notes_dir=args.notes.parent)
    print(
        f"wrote {args.out} ({summary.users:,} users, {summary.items:,} items, "
        f"{summary.groups:,} groups)",
        flush=True,
    )


if __name__ == "__main__":
    main()
