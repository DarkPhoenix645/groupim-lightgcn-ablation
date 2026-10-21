"""Douban event conversion: 10+ filter and co-attendance groups."""

from pathlib import Path

import pytest

from groupim_lightgcn_ablation.csv_contract import assert_loaders, layout_errors, read_author_csvs
from groupim_lightgcn_ablation.douban import convert_douban, main

ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_convert_keeps_shared_events_after_the_ten_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    users = [str(index) for index in range(24)]
    events = [str(index) for index in range(12)]
    _write(
        raw / "user_events.dat",
        [f"{user}\t{','.join(events)}" for user in users]
        + ["sparse\t0,1,2", "0\t0,1,2,rare", "1\t0,1,2,rare"],
    )
    _write(
        raw / "groupid_users.dat",
        [f"{index}\t{index % 24},{(index + 1) % 24}" for index in range(24)]
        + ["lonely\tsparse,0", "rare-only\t0,1"],
    )
    _write(
        raw / "groupid_events.dat",
        [f"{index}\t0" for index in range(24)] + ["lonely\t0", "rare-only\trare"],
    )

    out = ROOT / "data" / "_douban_smoke"
    notes = tmp_path / "notes"
    table1 = tmp_path / "table1.md"
    monkeypatch.chdir(ROOT)
    try:
        summary = convert_douban(raw, out)
        assert summary.users == 24
        assert summary.items == 12
        assert summary.groups == 24
        frames = read_author_csvs(out)
        assert layout_errors(frames) == []
        assert_loaders("_douban_smoke", negs_per_group=5)
        main(
            [
                "--raw",
                str(raw),
                "--out",
                str(out),
                "--notes",
                str(notes / "douban.md"),
                "--table1",
                str(table1),
            ]
        )
        note = (notes / "douban.md").read_text(encoding="utf-8")
        assert "10+" in note
        assert "event" in note
        assert note in table1.read_text(encoding="utf-8")
    finally:
        if out.exists():
            for path in out.glob("*"):
                path.unlink()
            out.rmdir()


def test_main_names_the_download_when_files_are_missing(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="Douban-Event"):
        main(["--raw", str(tmp_path / "missing"), "--out", str(tmp_path / "out")])
