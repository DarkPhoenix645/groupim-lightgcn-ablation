"""Yelp Los Angeles conversion: filters, friend windows, and the author CSV emit."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from groupim_lightgcn_ablation.csv_contract import assert_loaders, layout_errors, read_author_csvs
from groupim_lightgcn_ablation.yelp import (
    WINDOW_15_MIN,
    WINDOW_SAME_DAY,
    Visit,
    convert_yelp,
    filter_visits,
    friend_graph,
    group_interactions,
    la_restaurant_ids,
    main,
    read_visits,
)

ROOT = Path(__file__).resolve().parents[1]


def _dump(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _business(business_id: str, city: str, categories: str, state: str = "CA") -> dict:
    return {
        "business_id": business_id,
        "city": city,
        "state": state,
        "categories": categories,
    }


def test_la_restaurant_filter_keeps_category_tokens(tmp_path: Path) -> None:
    path = tmp_path / "business.json"
    _dump(
        path,
        [
            _business("la", "Los Angeles", "Restaurants, Mexican"),
            _business("padded", " Los Angeles ", "Restaurants"),
            _business("phoenix", "Phoenix", "Restaurants", state="AZ"),
            _business("supplies", "Los Angeles", "Restaurant Supplies"),
            _business("bar", "Los Angeles", "Bars, Nightlife"),
        ],
    )
    assert la_restaurant_ids(path) == {"la", "padded"}


def test_five_visit_filter_is_stable_and_collapses_repeat_reviews() -> None:
    visits = []
    for user in range(5):
        for item in range(5):
            visits.append(Visit(f"u{user}", f"b{item}", datetime(2020, 1, 1, 12, user, item)))
    visits.append(Visit("u0", "b0", datetime(2020, 1, 2)))
    visits.append(Visit("sparse", "b0", datetime(2020, 1, 1)))
    visits.append(Visit("u0", "rare", datetime(2020, 1, 1)))
    kept = filter_visits(visits, 5)
    pairs = {(visit.user, visit.item) for visit in kept}
    assert ("sparse", "b0") not in pairs
    assert ("u0", "rare") not in pairs
    assert ("u0", "b0") in pairs
    assert len([visit for visit in kept if visit.user == "u0" and visit.item == "b0"]) == 1
    assert next(
        visit.at for visit in kept if visit.user == "u0" and visit.item == "b0"
    ) == datetime(2020, 1, 1, 12, 0, 0)


def test_filter_cascades_when_dropping_an_item_drops_a_user() -> None:
    visits = [Visit("u", f"b{i}", datetime(2020, 1, i + 1)) for i in range(5)]
    for other in range(4):
        for item in range(4):
            visits.append(Visit(f"o{other}", f"b{item}", datetime(2020, 1, 1)))
    visits.append(Visit("lonely", "b4", datetime(2020, 1, 1)))
    kept = filter_visits(visits, 5)
    assert kept == []


def test_same_day_window_groups_friends_and_dedupes_anchors(tmp_path: Path) -> None:
    visits = [
        Visit("a", "r", datetime(2020, 1, 1, 9, 0)),
        Visit("b", "r", datetime(2020, 1, 1, 18, 0)),
        Visit("c", "r", datetime(2020, 1, 2, 9, 0)),
        Visit("d", "r", datetime(2020, 1, 1, 10, 0)),
    ]
    friends = {"a": {"b"}, "b": {"a"}, "c": set(), "d": set()}
    group_item, group_user = group_interactions(visits, friends, WINDOW_SAME_DAY)
    assert list(group_item["item"]) == ["r"]
    assert set(group_user["user"]) == {"a", "b"}


def test_fifteen_minute_window_splits_later_friends() -> None:
    visits = [
        Visit("a", "r", datetime(2020, 1, 1, 12, 0)),
        Visit("d", "r", datetime(2020, 1, 1, 12, 5)),
        Visit("b", "r", datetime(2020, 1, 1, 12, 10)),
        Visit("c", "r", datetime(2020, 1, 1, 12, 40)),
    ]
    friends = {
        "a": {"b", "c"},
        "b": {"a"},
        "c": {"a"},
        "d": set(),
    }
    group_item, group_user = group_interactions(visits, friends, WINDOW_15_MIN)
    members = {
        frozenset(group_user.loc[group_user["group"] == group, "user"])
        for group in group_item["group"]
    }
    assert members == {frozenset({"a", "b"})}


def test_anchor_keeps_each_friend_subset() -> None:
    visits = [
        Visit("a", "r", datetime(2020, 1, 1, 12, 0)),
        Visit("b", "r", datetime(2020, 1, 1, 12, 5)),
        Visit("c", "r", datetime(2020, 1, 1, 12, 6)),
    ]
    friends = {"a": {"b", "c"}, "b": {"a"}, "c": {"a"}}
    _, group_user = group_interactions(visits, friends, WINDOW_15_MIN)
    members = {
        frozenset(group_user.loc[group_user["group"] == group, "user"])
        for group in group_user["group"].unique()
    }
    assert members == {
        frozenset({"a", "b", "c"}),
        frozenset({"a", "b"}),
        frozenset({"a", "c"}),
    }


def test_one_sided_friend_edge_is_symmetrized(tmp_path: Path) -> None:
    path = tmp_path / "user.json"
    _dump(
        path,
        [
            {"user_id": "a", "friends": "b, ignored"},
            {"user_id": "b", "friends": "None"},
            {"user_id": "other", "friends": "a, b"},
        ],
    )
    graph = friend_graph(path, {"a", "b"})
    assert graph["a"] == {"b"}
    assert graph["b"] == {"a"}


def test_convert_writes_author_csvs_and_records_a_date_only_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    businesses = [_business(f"b{i:02d}", "Los Angeles", "Restaurants") for i in range(15)]
    businesses.append(_business("phx", "Phoenix", "Restaurants", state="AZ"))
    _dump(raw / "yelp_academic_dataset_business.json", businesses)
    reviews = []
    users = []
    for index in range(40):
        user = f"u{index:02d}"
        partner = f"u{index ^ 1:02d}"
        users.append({"user_id": user, "friends": partner})
        for item in range(15):
            reviews.append({"user_id": user, "business_id": f"b{item:02d}", "date": "2019-06-01"})
    reviews.append({"user_id": "sparse", "business_id": "b00", "date": "2019-06-01"})
    reviews.append({"user_id": "u00", "business_id": "phx", "date": "2019-06-01 19:30:00"})
    _dump(raw / "yelp_academic_dataset_review.json", reviews)
    _dump(raw / "yelp_academic_dataset_user.json", users)

    out = ROOT / "data" / "_yelp_smoke"
    notes = tmp_path / "notes"
    table1 = tmp_path / "table1.md"
    monkeypatch.chdir(ROOT)
    try:
        summary = convert_yelp(raw, out, min_count=5)
        assert summary.window == WINDOW_SAME_DAY
        assert summary.users == 40
        assert summary.items == 15
        assert summary.reviews == 40 * 15 + 1
        assert layout_errors(read_author_csvs(out)) == []
        assert_loaders("_yelp_smoke", negs_per_group=5)
        main(
            [
                "--raw",
                str(raw),
                "--out",
                str(out),
                "--notes",
                str(notes / "yelp.md"),
                "--table1",
                str(table1),
            ]
        )
        note = (notes / "yelp.md").read_text(encoding="utf-8")
        assert "same calendar day" in note
        assert "15-minute cut could not be applied" in note
        assert note in table1.read_text(encoding="utf-8")
    finally:
        if out.exists():
            for path in out.glob("*"):
                path.unlink()
            out.rmdir()


def test_clock_time_selects_the_fifteen_minute_window(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _dump(
        raw / "yelp_academic_dataset_business.json", [_business("r", "Los Angeles", "Restaurants")]
    )
    _dump(
        raw / "yelp_academic_dataset_review.json",
        [
            {"user_id": "a", "business_id": "r", "date": "2020-01-01 12:00:00"},
            {"user_id": "b", "business_id": "r", "date": "2020-01-01 12:10:00"},
            {"user_id": "c", "business_id": "r", "date": "2020-01-01 13:00:00"},
        ],
    )
    visits, timed = read_visits(raw / "yelp_academic_dataset_review.json", {"r"})
    assert timed
    friends = {"a": {"b", "c"}, "b": {"a"}, "c": {"a"}}
    group_item, group_user = group_interactions(
        visits, friends, WINDOW_15_MIN if timed else WINDOW_SAME_DAY
    )
    members = {
        frozenset(group_user.loc[group_user["group"] == group, "user"])
        for group in group_item["group"]
    }
    assert members == {frozenset({"a", "b"})}


def test_main_asks_for_the_download_when_raw_files_are_missing(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="Yelp Open Dataset"):
        main(["--raw", str(tmp_path / "missing"), "--out", str(tmp_path / "out")])


def test_benchmark_keeps_groups_after_the_five_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "yelp_la"
    raw.mkdir()
    users = [str(index) for index in range(24)]
    events = [str(index) for index in range(6)]
    (raw / "user_events.dat").write_text(
        "\n".join(f"{user}\t{','.join(events)}" for user in users)
        + "\nsparse\t0,1\n0\t0,1,rare\n1\t0,1,rare\n",
        encoding="utf-8",
    )
    (raw / "groupid_users.dat").write_text(
        "\n".join(f"{index}\t{index % 24},{(index + 1) % 24}" for index in range(24))
        + "\nlonely\tsparse,0\nrare-only\t0,1\n",
        encoding="utf-8",
    )
    (raw / "groupid_events.dat").write_text(
        "\n".join(f"{index}\t0" for index in range(24)) + "\nlonely\t0\nrare-only\trare\n",
        encoding="utf-8",
    )
    out = ROOT / "data" / "_yelp_benchmark_smoke"
    notes = tmp_path / "notes"
    table1 = tmp_path / "table1.md"
    monkeypatch.chdir(ROOT)
    try:
        main(
            [
                "--raw",
                str(raw),
                "--out",
                str(out),
                "--notes",
                str(notes / "yelp.md"),
                "--table1",
                str(table1),
            ]
        )
        frames = read_author_csvs(out)
        assert layout_errors(frames) == []
        users = set()
        for name in (
            "train_ui.csv",
            "val_ui_tr.csv",
            "val_ui_te.csv",
            "test_ui_tr.csv",
            "test_ui_te.csv",
        ):
            users.update(frames[name]["user"])
        assert len(users) == 24
        note = (notes / "yelp.md").read_text(encoding="utf-8")
        assert "15-minute window was not applied again" in note
        assert "Yin" in note
        assert "These metrics are on the Yin-LA variant, not the paper's Yelp benchmark." in note
        assert_loaders("_yelp_benchmark_smoke", negs_per_group=5)
    finally:
        if out.exists():
            for path in out.glob("*"):
                path.unlink()
            out.rmdir()
