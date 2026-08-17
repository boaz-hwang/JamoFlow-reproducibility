#!/usr/bin/env python3
"""Generate the paper's result figures from tracked aggregate evidence only."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "paper" / "figures"

COMPACT_QUALITY = ROOT / "results/phase3-inference-final-v2/summary.json"
COMPACT_ACTUAL = ROOT / "results/phase3-inference-actual-v5r3/summary.json"
INITIAL_SCALE = ROOT / "results/scale-schedule-preflight-v1/summary.json"
INITIAL_SCALE_PLAN = ROOT / "data/manifests/scale-schedule-preflight-v1.json"
EXTENDED_SCALE = ROOT / "results/scale-schedule-extrapolation-v1/summary.json"
EXTENDED_SCALE_PLAN = ROOT / "data/manifests/scale-schedule-extrapolation-v1.json"
TRAINED_W72 = ROOT / "results/balanced-200m-trained-screen-v1/training-summary.json"
TRAINED_W80 = ROOT / "results/balanced-200m-w80-rescue-v1/training-summary.json"
TRAINED_W80_ACTUAL = ROOT / "results/balanced-200m-w80-rescue-v1/actual-summary.json"

FONT = "-apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif"
INK = "#18212f"
MUTED = "#5f6b7a"
GRID = "#d8dee8"
BLUE = "#2563eb"
TEAL = "#0f9d8a"
ORANGE = "#e97918"
RED = "#c43d4b"
LIGHT_BLUE = "#dbeafe"
LIGHT_ORANGE = "#ffedd5"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


class SVG:
    def __init__(self, width: int, height: int, *, title: str, sources: list[Path]) -> None:
        self.width = width
        self.height = height
        source_text = "; ".join(
            f"{path.relative_to(ROOT)}={sha256(path)}" for path in sources
        )
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f"<title id=\"title\">{esc(title)}</title>",
            f"<desc id=\"desc\">Generated from tracked aggregate evidence. {esc(source_text)}</desc>",
            '<rect width="100%" height="100%" fill="#ffffff"/>',
        ]

    def line(self, x1: float, y1: float, x2: float, y2: float, *, color: str = INK,
             width: float = 1.0, dash: str | None = None) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="{width:.2f}"{dash_attr}/>'
        )

    def rect(self, x: float, y: float, width: float, height: float, *, fill: str,
             stroke: str | None = None, radius: float = 0.0) -> None:
        stroke_attr = f' stroke="{stroke}"' if stroke else ""
        self.parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
            f'rx="{radius:.2f}" fill="{fill}"{stroke_attr}/>'
        )

    def circle(self, x: float, y: float, radius: float, *, fill: str,
               stroke: str | None = None, stroke_width: float = 1.0) -> None:
        stroke_attr = ""
        if stroke:
            stroke_attr = f' stroke="{stroke}" stroke-width="{stroke_width:.2f}"'
        self.parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{fill}"{stroke_attr}/>'
        )

    def text(self, x: float, y: float, value: object, *, size: float = 14.0,
             weight: int = 400, color: str = INK, anchor: str = "start") -> None:
        self.parts.append(
            f'<text x="{x:.2f}" y="{y:.2f}" font-family="{FONT}" font-size="{size:.2f}" '
            f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{esc(value)}</text>'
        )

    def finish(self, path: Path) -> None:
        self.parts.append("</svg>")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.parts) + "\n", encoding="utf-8")


def linear(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if not lo < hi:
        raise ValueError("invalid linear scale")
    return out_lo + (value - lo) * (out_hi - out_lo) / (hi - lo)


def log_x(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    return linear(math.log10(value), math.log10(lo), math.log10(hi), out_lo, out_hi)


def error_bar(svg: SVG, x: float, low_y: float, high_y: float, *, color: str) -> None:
    svg.line(x, low_y, x, high_y, color=color, width=2)
    svg.line(x - 6, low_y, x + 6, low_y, color=color, width=2)
    svg.line(x - 6, high_y, x + 6, high_y, color=color, width=2)


def generate_trained_scale_figure(output_dir: Path) -> None:
    compact_quality = load_json(COMPACT_QUALITY)
    compact_actual = load_json(COMPACT_ACTUAL)
    trained_w72 = load_json(TRAINED_W72)
    trained_w80 = load_json(TRAINED_W80)
    trained_actual = load_json(TRAINED_W80_ACTUAL)

    cq = compact_quality["final_quality_gate"]["candidate_vs_matched_efficiency_baseline"]
    compact_latency = compact_actual["latency"]
    w72_quality = trained_w72["quality"]
    w80_quality = trained_w80["quality"]
    w80_latency = trained_actual["actual_timing"]["by_mode"]

    quality_rows = [
        ("19.6M W72", float(cq["mean_difference_bpb"]),
         float(cq["paired_seed_t_95_lower_bpb"]), float(cq["paired_seed_t_95_upper_bpb"]), BLUE),
        ("188.6M W72", float(w72_quality["w72_minus_c86_bpb"]), None, None, RED),
        ("188.6M W80", float(w80_quality["w80_minus_c86_bpb"]),
         float(w80_quality["block_bootstrap"]["lower"]),
         float(w80_quality["block_bootstrap"]["upper"]), ORANGE),
    ]
    actual_rows = [
        ("19.6M", "controlled", compact_latency["controlled_replay"]["end_to_end_ms"], BLUE),
        ("19.6M", "free", compact_latency["free_running_utf8_greedy"]["end_to_end_ms"], TEAL),
        ("188.6M", "controlled", w80_latency["controlled_replay"], ORANGE),
        ("188.6M", "free", w80_latency["free_running_utf8_greedy"], "#9b6bdb"),
    ]

    svg = SVG(
        1120,
        610,
        title="Quality retention and actual inference at two trained scales",
        sources=[COMPACT_QUALITY, COMPACT_ACTUAL, TRAINED_W72, TRAINED_W80, TRAINED_W80_ACTUAL],
    )
    svg.text(40, 42, "Quality-qualified boundary schedules at two trained scales", size=24, weight=700)
    svg.text(40, 68, "Error bars are the protocol-specific 95% intervals; lower latency reduction is not inferred from patch count.", size=13, color=MUTED)

    # Panel A: quality deltas.
    left_x0, left_x1 = 75.0, 520.0
    top, bottom = 125.0, 520.0
    svg.text(left_x0, 102, "A  Calibration BPB difference from C86", size=17, weight=700)
    svg.text(left_x0, 122, "Lower is better; dashed line is the +0.010 noninferiority margin", size=12, color=MUTED)
    y_quality = lambda value: linear(value, 0.0, 0.026, bottom, top + 35)
    for tick in (0.0, 0.005, 0.010, 0.015, 0.020, 0.025):
        y = y_quality(tick)
        svg.line(left_x0, y, left_x1, y, color=GRID, width=1)
        svg.text(left_x0 - 10, y + 5, f"{tick:.3f}", size=11, color=MUTED, anchor="end")
    margin_y = y_quality(0.010)
    svg.line(left_x0, margin_y, left_x1, margin_y, color=RED, width=1.5, dash="6 5")
    x_positions = (160.0, 300.0, 440.0)
    for x, (label, point, low, high, color) in zip(x_positions, quality_rows, strict=True):
        if low is not None and high is not None:
            error_bar(svg, x, y_quality(high), y_quality(low), color=color)
        svg.circle(x, y_quality(point), 7, fill=color, stroke="#ffffff", stroke_width=2)
        svg.text(x, bottom + 28, label, size=13, weight=600, anchor="middle")
        svg.text(x, y_quality(point) - 13, f"+{point:.4f}", size=12, weight=600, color=color, anchor="middle")
    svg.text(300, bottom + 54, "W72 fails at 188.6M; the single W80 rescue restores quality.", size=12, color=MUTED, anchor="middle")

    # Panel B: actual latency reductions.
    right_x0, right_x1 = 625.0, 1080.0
    svg.text(right_x0, 102, "B  End-to-end latency reduction", size=17, weight=700)
    svg.text(right_x0, 122, "Five fresh-process sessions at each scale; positive means faster", size=12, color=MUTED)
    y_actual = lambda value: linear(value, 0.0, 0.04, bottom, top + 35)
    for tick in (0.0, 0.01, 0.02, 0.03, 0.04):
        y = y_actual(tick)
        svg.line(right_x0, y, right_x1, y, color=GRID, width=1)
        svg.text(right_x0 - 10, y + 5, f"{tick * 100:.0f}%", size=11, color=MUTED, anchor="end")
    actual_x = (690.0, 790.0, 915.0, 1015.0)
    for x, (scale, mode, row, color) in zip(actual_x, actual_rows, strict=True):
        if "crossed_median_latency_reduction" in row:
            point = float(row["crossed_median_latency_reduction"])
            low = float(row["bootstrap_percentile_95_lower"])
            high = float(row["bootstrap_percentile_95_upper"])
        else:
            point = float(row["end_to_end_reduction"])
            interval = row["crossed_bootstrap_95_interval"]
            low = float(interval["lower"])
            high = float(interval["upper"])
        error_bar(svg, x, y_actual(high), y_actual(low), color=color)
        svg.circle(x, y_actual(point), 7, fill=color, stroke="#ffffff", stroke_width=2)
        svg.text(x, y_actual(point) - 13, f"{point * 100:.3f}%", size=12, weight=600, color=color, anchor="middle")
        svg.text(x, bottom + 24, scale, size=12, weight=600, anchor="middle")
        svg.text(x, bottom + 43, mode, size=11, color=MUTED, anchor="middle")
    svg.text(852, bottom + 70, "The larger trained screen replicates the small effect; it does not amplify it.", size=12, color=MUTED, anchor="middle")
    svg.finish(output_dir / "trained-scale-evidence.svg")


def generate_scale_headroom_figure(output_dir: Path) -> None:
    compact_actual = load_json(COMPACT_ACTUAL)
    initial = load_json(INITIAL_SCALE)
    initial_plan = load_json(INITIAL_SCALE_PLAN)
    extended = load_json(EXTENDED_SCALE)
    extended_plan = load_json(EXTENDED_SCALE_PLAN)
    trained_actual = load_json(TRAINED_W80_ACTUAL)

    random_points: list[tuple[int, float, float, float, str]] = []
    for key in ("50", "75", "100"):
        row = initial["aggregate"]["rows"][key]
        params = int(initial_plan["models"][key]["expected_parameter_count"])
        interval = row["prompt_bootstrap_95_interval"]
        random_points.append((params, float(row["median_reduction"]), float(interval["lower"]), float(interval["upper"]), "initial"))
    for key in ("200", "400", "800", "1600"):
        row = extended["aggregate"]["rows"][key]
        params = int(extended_plan["models"][key]["expected_parameter_count"])
        interval = row["prompt_bootstrap_95_interval"]
        random_points.append((params, float(row["median_reduction"]), float(interval["lower"]), float(interval["upper"]), "extension"))

    compact_row = compact_actual["latency"]["controlled_replay"]["end_to_end_ms"]
    compact_params = int(compact_actual["timing_pair"]["roles"]["candidate"]["total_parameter_count"])
    trained_row = trained_actual["actual_timing"]["by_mode"]["controlled_replay"]
    trained_points = [
        (compact_params, float(compact_row["crossed_median_latency_reduction"]),
         float(compact_row["bootstrap_percentile_95_lower"]), float(compact_row["bootstrap_percentile_95_upper"]), "19.6M W72"),
        (188_639_808, float(trained_row["end_to_end_reduction"]),
         float(trained_row["crossed_bootstrap_95_interval"]["lower"]),
         float(trained_row["crossed_bootstrap_95_interval"]["upper"]), "188.6M W80"),
    ]

    svg = SVG(
        560,
        410,
        title="Random-weight systems headroom versus trained quality-qualified reductions",
        sources=[COMPACT_ACTUAL, INITIAL_SCALE, INITIAL_SCALE_PLAN, EXTENDED_SCALE, EXTENDED_SCALE_PLAN, TRAINED_W80_ACTUAL],
    )
    x0, x1, y0, y1 = 52.0, 542.0, 310.0, 28.0
    x_scale = lambda value: log_x(value, 15_000_000, 2_000_000_000, x0, x1)
    y_scale = lambda value: linear(value, 0.0, 0.12, y0, y1)
    for tick in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12):
        y = y_scale(tick)
        svg.line(x0, y, x1, y, color=GRID, width=1)
        svg.text(x0 - 7, y + 5, f"{tick * 100:.0f}%", size=17, color=MUTED, anchor="end")
    for value, label in ((20_000_000, "20M"), (50_000_000, "50M"), (100_000_000, "100M"),
                         (200_000_000, "200M"), (400_000_000, "400M"),
                         (800_000_000, "800M"), (1_600_000_000, "1.6B")):
        x = x_scale(value)
        svg.line(x, y0, x, y0 + 6, color=INK, width=1)
        svg.text(x, y0 + 23, label, size=16, color=MUTED, anchor="middle")
    svg.line(x0, y_scale(0.10), x1, y_scale(0.10), color=RED, width=1.5, dash="7 5")
    svg.text(x1 - 3, y_scale(0.10) - 7, "10% threshold", size=15, color=RED, anchor="end")

    polyline = " ".join(f"{x_scale(p):.2f},{y_scale(point):.2f}" for p, point, _, _, _ in random_points)
    svg.parts.append(f'<polyline points="{polyline}" fill="none" stroke="{BLUE}" stroke-width="3"/>')
    for params, point, low, high, stage in random_points:
        x = x_scale(params)
        error_bar(svg, x, y_scale(high), y_scale(low), color=BLUE)
        svg.circle(x, y_scale(point), 6, fill=BLUE if stage == "extension" else LIGHT_BLUE, stroke=BLUE, stroke_width=2)
    for params, point, low, high, label in trained_points:
        x = x_scale(params)
        error_bar(svg, x, y_scale(high), y_scale(low), color=ORANGE)
        svg.rect(x - 6, y_scale(point) - 6, 12, 12, fill=ORANGE, stroke="#ffffff", radius=1)
        offset = -13 if params < 50_000_000 else 20
        svg.text(x + 6, y_scale(point) + offset, f"{label}: {point * 100:.3f}%", size=16, weight=600, color=ORANGE)

    svg.line(62, 354, 94, 354, color=BLUE, width=3)
    svg.circle(78, 354, 5, fill=LIGHT_BLUE, stroke=BLUE, stroke_width=2)
    svg.text(104, 360, "same-weight random W72 vs C86", size=17, color=MUTED)
    svg.rect(72, 374, 12, 12, fill=ORANGE, stroke="#ffffff", radius=1)
    svg.text(104, 386, "trained quality-qualified schedule vs C86", size=17, color=MUTED)
    svg.text(297, 407, "Model parameters (log scale)", size=18, weight=600, anchor="middle")
    svg.finish(output_dir / "scale-headroom-versus-trained.svg")


def generate(output_dir: Path) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_trained_scale_figure(output_dir)
    generate_scale_headroom_figure(output_dir)
    magick = shutil.which("magick")
    if magick is None:
        raise RuntimeError("ImageMagick `magick` is required to create deterministic PNG fallbacks")
    outputs: list[Path] = []
    for name in ("scale-headroom-versus-trained", "trained-scale-evidence"):
        svg_path = output_dir / f"{name}.svg"
        png_path = svg_path.with_suffix(".png")
        subprocess.run(
            [
                magick,
                "-background",
                "white",
                str(svg_path),
                "-alpha",
                "remove",
                "-alpha",
                "off",
                "-depth",
                "8",
                "-type",
                "TrueColor",
                "-strip",
                str(png_path),
            ],
            check=True,
        )
        outputs.extend((svg_path, png_path))
    return tuple(outputs)


def verify() -> None:
    with tempfile.TemporaryDirectory(prefix="jamoflow-paper-figures-") as temp_dir:
        generated = generate(Path(temp_dir))
        for generated_path in generated:
            tracked_path = FIGURE_DIR / generated_path.name
            if not tracked_path.is_file():
                raise ValueError(f"missing tracked figure: {tracked_path.relative_to(ROOT)}")
            if generated_path.read_bytes() != tracked_path.read_bytes():
                raise ValueError(f"tracked figure is stale: {tracked_path.relative_to(ROOT)}")
    print("paper_figures=verified")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURE_DIR,
        help="output directory (default: paper/figures)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="regenerate in a temporary directory and compare with tracked figures",
    )
    args = parser.parse_args()
    if args.verify:
        verify()
        return
    output_dir = args.output_dir.resolve()
    outputs = generate(output_dir)
    print("paper_figures=generated")
    for path in sorted(outputs):
        label = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        print(f"{label} sha256={sha256(path)}")


if __name__ == "__main__":
    main()
