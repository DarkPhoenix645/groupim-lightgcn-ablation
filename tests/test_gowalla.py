"""Gowalla restaurant conversion: Food spots, US box, June 2011, 15-minute groups."""

import csv
import json
from pathlib import Path

import pytest

from groupim_lightgcn_ablation.csv_contract import assert_loaders, layout_errors, read_author_csvs
from groupim_lightgcn_ablation.gowalla import (
    convert_gowalla,
    food_category_ids,
    in_united_states,
    main,
)

ROOT = Path(__file__).resolve().parents[1]


def _write_categories(path: Path) -> None:
    payload = {
        "spot_categories": [
            {
                "name": "Food",
                "url": "/categories/1",
                "spot_categories": [
                    {"name": "BBQ", "url": "/categories/17", "spot_categories": []}
                ],
            },
            {
                "name": "Nightlife",
                "url": "/categories/2",
                "spot_categories": [
                    {"name": "Bar", "url": "/categories/99", "spot_categories": []}
                ],
            },
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_food_ids_include_children_and_skip_nightlife(tmp_path: Path) -> None:
    path = tmp_path / "categories.json"
    _write_categories(path)
    assert food_category_ids(path) == {"1", "17"}


def test_united_states_box() -> None:
    assert in_united_states(34.05, -118.25)
    assert in_united_states(64.8, -147.7)
    assert in_united_states(21.3, -157.8)
    assert not in_united_states(51.5, -0.1)
    assert not in_united_states(19.4, -99.1)


def test_convert_keeps_us_food_checkins_through_june(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_categories(raw / "gowalla_category_structure.json")
    spots = [
        {
            "id": str(place),
            "lat": "30.27",
            "lng": "-97.74",
            "spot_categories": "[{'url': '/categories/17', 'name': 'BBQ'}]",
        }
        for place in range(5)
    ]
    spots.append(
        {
            "id": "bar",
            "lat": "30.27",
            "lng": "-97.74",
            "spot_categories": "[{'url': '/categories/99', 'name': 'Bar'}]",
        }
    )
    spots.append(
        {
            "id": "london",
            "lat": "51.50",
            "lng": "-0.12",
            "spot_categories": "[{'url': '/categories/17', 'name': 'BBQ'}]",
        }
    )
    _write_csv(raw / "gowalla_spots_subset1.csv", spots)

    checkins = []
    friends = []
    for index in range(20):
        user = str(index)
        friends.append({"userid1": user, "userid2": str((index + 1) % 20)})
        checkins.append({"userid": user, "placeid": "0", "datetime": "2011-05-01 12:00:00"})
        for place in range(1, 5):
            checkins.append(
                {
                    "userid": user,
                    "placeid": str(place),
                    "datetime": f"2011-05-{2 + index:02d} 12:00:00",
                }
            )
    checkins.append({"userid": "0", "placeid": "0", "datetime": "2011-07-01 00:00:00"})
    checkins.append({"userid": "0", "placeid": "bar", "datetime": "2011-05-01 12:00:00"})
    checkins.append({"userid": "0", "placeid": "london", "datetime": "2011-05-01 12:00:00"})
    _write_csv(raw / "gowalla_checkins.csv", checkins)
    _write_csv(raw / "gowalla_friendship.csv", friends)

    out = ROOT / "data" / "_gowalla_smoke"
    notes = tmp_path / "notes"
    table1 = tmp_path / "table1.md"
    monkeypatch.chdir(ROOT)
    try:
        summary = convert_gowalla(raw, out)
        assert summary.users == 20
        assert summary.items == 5
        assert summary.restaurants == 5
        assert summary.checkins == 100
        assert layout_errors(read_author_csvs(out)) == []
        assert_loaders("_gowalla_smoke", negs_per_group=5)
        main(
            [
                "--raw",
                str(raw),
                "--out",
                str(out),
                "--notes",
                str(notes / "gowalla.md"),
                "--table1",
                str(table1),
            ]
        )
        note = (notes / "gowalla.md").read_text(encoding="utf-8")
        assert "15 minutes" in note
        assert "2011-07-01" in note
        assert note in table1.read_text(encoding="utf-8")
    finally:
        if out.exists():
            for path in out.glob("*"):
                path.unlink()
            out.rmdir()


def test_main_names_the_download_when_files_are_missing(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="figshare"):
        main(["--raw", str(tmp_path / "missing"), "--out", str(tmp_path / "out")])
