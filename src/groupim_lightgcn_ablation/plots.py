"""SVG charts in the same layout as the Weeplaces baseline figures."""

from __future__ import annotations

import math
import re
from pathlib import Path

PAPER_TEST = {
    "weeplaces": {"r20": 0.716, "r50": 0.813},
}


def write_run_figures(log_text: str, figure_dir: Path, dataset: str, seed: int) -> list[Path]:
    """Write val-r20, val-r50, val-n100, and train-loss SVGs. Returns the paths."""
    val, loss = parse_group_log(log_text)
    figure_dir.mkdir(parents=True, exist_ok=True)
    paper = PAPER_TEST.get(dataset, {})
    r20_ref = [(paper["r20"], "paper test R@20 0.716")] if "r20" in paper else []
    r50_ref = [(paper["r50"], "paper test R@50 0.813")] if "r50" in paper else []
    specs = [
        (
            "val-r20.svg",
            f"{dataset} validation Recall@20, seed {seed}",
            "Recall@20",
            [(e, r) for e, _n, r, _r50 in val],
            r20_ref,
        ),
        (
            "val-r50.svg",
            f"{dataset} validation Recall@50, seed {seed}",
            "Recall@50",
            [(e, r) for e, _n, _r, r in val],
            r50_ref,
        ),
        (
            "val-n100.svg",
            f"{dataset} validation NDCG@100, seed {seed}",
            "NDCG@100",
            [(e, n) for e, n, _r, _r50 in val],
            [],
        ),
        (
            "train-loss.svg",
            f"{dataset} group train loss, seed {seed}",
            "Mean train loss",
            loss,
            [],
        ),
    ]
    written = []
    for name, title, ylabel, points, refs in specs:
        path = figure_dir / name
        path.write_text(line_chart(title, ylabel, points, refs))
        written.append(path)
    return written


def parse_group_log(text: str) -> tuple[list[tuple[int, float, float, float]], list[tuple[int, float]]]:
    """Validation (epoch, n100, r20, r50) and per-epoch train loss."""
    epoch_re = re.compile(
        r"end of epoch\s+(\d+)\s+\| time:\s+[\d.]+s \| n100 \(group\)\s+([\d.]+) \| "
        r"r20 \(group\)\s+([\d.]+) \| r50 \(group\)\s+([\d.]+)"
    )
    loss_re = re.compile(r"^Train loss:\s+([\d.]+)\s*$", re.MULTILINE)
    val = [
        (int(epoch), float(n100), float(r20), float(r50))
        for epoch, n100, r20, r50 in epoch_re.findall(text)
    ]
    losses = [float(value) for value in loss_re.findall(text)]
    if not val or not losses:
        return val, []
    head = text.split("end of epoch", 1)[0]
    before = len(loss_re.findall(head))
    start = val[0][0] - before + 1
    return val, [(start + index, loss) for index, loss in enumerate(losses)]


def line_chart(title: str, ylabel: str, points: list[tuple[int, float]], refs: list[tuple[float, str]]) -> str:
    width, height = 860, 420
    left, right, top, bottom = 64, 24, 36, 48
    plot_w, plot_h = width - left - right, height - top - bottom
    xs = [epoch for epoch, _value in points] or [1]
    ys = [value for _epoch, value in points]
    x0, x1 = 1, max(200, max(xs))
    finite = [value for value in ys if math.isfinite(value)]
    for value, _label in refs:
        finite.append(value)
    if not finite:
        finite = [0.0, 1.0]
    low, high = min(finite), max(finite)
    pad = (high - low) * 0.08 or 0.05
    y_min, y_max = low - pad, high + pad

    def x_of(epoch: int) -> float:
        return left + (epoch - x0) / (x1 - x0) * plot_w

    def y_of(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="22" font-family="sans-serif" font-size="16" fill="#111827">{title}</text>',
    ]
    for tick in range(0, x1 + 1, 40):
        shown = 1 if tick == 0 else tick
        x = x_of(shown)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#e5e7eb"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{height - 18}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#6b7280">{shown}</text>'
        )
    for step in range(6):
        value = y_min + (y_max - y_min) * step / 5
        y = y_of(value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#6b7280">{value:.2f}</text>'
        )
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>')
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>')
    parts.append(
        f'<text x="{left + plot_w / 2}" y="{height - 4}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#374151">Group-training epoch</text>'
    )
    parts.append(
        f'<text x="16" y="{top + plot_h / 2}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#374151" transform="rotate(-90 16 {top + plot_h / 2})">{ylabel}</text>'
    )
    for value, label in refs:
        y = y_of(value)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#9ca3af" stroke-dasharray="4 4"/>'
        )
        parts.append(
            f'<text x="{left + plot_w - 4}" y="{y - 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#6b7280">{label}</text>'
        )
    if len(points) >= 2:
        drawn = " ".join(f"{x_of(epoch):.1f},{y_of(value):.2f}" for epoch, value in points)
        parts.append(f'<polyline fill="none" stroke="#1d4e89" stroke-width="2" points="{drawn}"/>')
    parts.append("</svg>\n")
    return "\n".join(parts)
