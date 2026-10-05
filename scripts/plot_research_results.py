"""Create dependency-free SVG plots from the frozen replay and Kronos reports."""

from __future__ import annotations

import html
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from ws.labels import regular_hours
from ws.store.event_store import BarStore

COLORS = {
    "Capped JEV + Kronos": "#1769aa",
    "Capped JEV only": "#e4572e",
    "Uncapped JEV + Kronos": "#2a9d8f",
    "Uncapped JEV only": "#8e5ea2",
    "SPY buy-and-hold": "#444444",
}


def text(x: float, y: float, value: str, size: int = 13, anchor: str = "start", fill: str = "#222") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{fill}" font-family="Arial, sans-serif">{html.escape(value)}</text>'
    )


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = "#999", width: float = 1) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}" />'


def scale(value: float, ymin: float, ymax: float, top: float, bottom: float) -> float:
    return bottom - (value - ymin) / (ymax - ymin) * (bottom - top)


def get_series(report_path: str, strategy: str) -> dict[date, float]:
    report = json.loads(Path(report_path).read_text())
    return {
        date.fromisoformat(day): float(value) for day, value in report["strategies"][strategy]["daily_equity"].items()
    }


def replay_svg(out: Path) -> None:
    reports = {
        "Capped JEV + Kronos": ("docs/broad-replay-report.json", "jev_kronos"),
        "Capped JEV only": ("docs/broad-replay-report.json", "text_only"),
        "Uncapped JEV + Kronos": ("docs/broad-replay-no-sector-cap-report.json", "jev_kronos"),
        "Uncapped JEV only": ("docs/broad-replay-no-sector-cap-report.json", "text_only"),
    }
    series = {label: get_series(path, key) for label, (path, key) in reports.items()}
    days = sorted(set.intersection(*(set(values) for values in series.values())))
    bars = regular_hours(BarStore(Path("data"), "15Min").read(["SPY"])).sort("ts")
    spy_daily = (
        bars.with_columns(session=pl.col("ts").dt.convert_time_zone("America/New_York").dt.date())
        .group_by("session", maintain_order=True)
        .agg(pl.col("open").first(), pl.col("close").last())
        .sort("session")
        .filter(pl.col("session").is_in(days))
    )
    capital, cost = 500.0, 15.0 / 10000
    qty = capital / (float(spy_daily["open"][0]) * (1 + cost))
    spy_vals = [float(close) * qty for close in spy_daily["close"]]
    spy_vals[-1] *= 1 - cost
    series["SPY buy-and-hold"] = {day: value for day, value in zip(days, spy_vals, strict=True)}
    days = sorted(set.intersection(*(set(values) for values in series.values())))

    width, height = 1200, 820
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    ]
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(text(600, 36, "INVALIDATED legacy replay: equity and drawdown", 23, "middle", "#a00"))
    parts.append(
        text(
            600,
            58,
            "400-day Kronos + P(up) gate; sample filtered on invalid outputs; diagnostic only",
            13,
            "middle",
            "#555",
        )
    )
    left, right = 95, 1145
    for _panel, y0, y1, title_text, ymin, ymax, ystep, is_dd in [
        (0, 105, 390, "Daily account value ($)", 440, 585, 25, False),
        (1, 485, 750, "Drawdown from prior peak (%)", -0.20, 0.02, 0.05, True),
    ]:
        parts.append(text(left, y0 - 15, title_text, 17))
        plot_top, plot_bottom = y0, y1

        tick = ymin
        while tick <= ymax + 1e-9:
            y = scale(tick, ymin, ymax, plot_top, plot_bottom)
            parts.append(line(left, y, right, y, "#e7e7e7"))
            label = f"{tick:.0%}" if is_dd else f"${tick:.0f}"
            parts.append(text(left - 10, y + 4, label, 11, "end", "#666"))
            tick += ystep
        parts.append(line(left, plot_bottom, right, plot_bottom, "#555"))
        for label, values in series.items():
            observed = [values[d] for d in days]
            if is_dd:
                peak = 500.0
                transformed = []
                for val in observed:
                    peak = max(peak, val)
                    transformed.append(val / peak - 1)
                observed = transformed
            pts = []
            for i, val in enumerate(observed):
                x = left + i / max(1, len(days) - 1) * (right - left)
                pts.append(f"{x:.1f},{scale(val, ymin, ymax, plot_top, plot_bottom):.1f}")
            parts.append(
                f'<polyline points="{" ".join(pts)}" fill="none" stroke="{COLORS[label]}" stroke-width="2.2"/>'
            )
        for i in sorted({0, len(days) // 2, len(days) - 1}):
            x = left + i / max(1, len(days) - 1) * (right - left)
            parts.append(line(x, plot_bottom, x, plot_bottom + 5, "#555"))
            parts.append(text(x, plot_bottom + 20, days[i].strftime("%b %d"), 11, "middle", "#666"))
    legend_y = 790
    x = 95
    for label, color in COLORS.items():
        parts.append(line(x, legend_y - 5, x + 25, legend_y - 5, color, 3))
        parts.append(text(x + 31, legend_y, label, 12))
        x += 215
    parts.append("</svg>")
    out.write_text("\n".join(parts) + "\n")


def kronos_svg(out: Path) -> None:
    report = json.loads(Path("docs/kronos-feature-report.json").read_text())
    cfg = report["settings"]
    direction = [
        ("L40 / T1", cfg["L40_T1"]["metrics"]["forecast_excess"]),
        ("L90 / T0.6", cfg["L90_T0.6"]["metrics"]["forecast_excess"]),
        ("L90 / T1", cfg["L90_T1"]["metrics"]["forecast_excess"]),
        ("L400 / T1", cfg["L400_T1"]["metrics"]["forecast_excess"]),
        ("20d reversal", cfg["L40_T1"]["metrics"]["reversal20"]),
        ("12–1 momentum", cfg["L40_T1"]["metrics"]["momentum12_1"]),
    ]
    volatility = [
        ("L40 dispersion", cfg["L40_T1"]["metrics"]["forecast_dispersion"]),
        ("L90 / T0.6", cfg["L90_T0.6"]["metrics"]["forecast_dispersion"]),
        ("L90 / T1", cfg["L90_T1"]["metrics"]["forecast_dispersion"]),
        ("L400 / T1", cfg["L400_T1"]["metrics"]["forecast_dispersion"]),
        ("ATR(14)", cfg["L40_T1"]["metrics"]["atr14"]),
    ]
    width, height = 1200, 710
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    ]
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(text(600, 35, "Kronos daily feature test", 23, "middle"))
    parts.append(
        text(
            600,
            57,
            "Mean cross-sectional rank correlation by session; error bars are 5-session block-bootstrap 95% intervals",
            13,
            "middle",
            "#555",
        )
    )

    def draw_panel(
        rows: list[tuple[str, dict[str, Any]]], top: int, bottom: int, title_text: str, lo: float, hi: float
    ) -> None:
        left, right = 105, 1155
        parts.append(text(left, top - 20, title_text, 17))

        def ymap(v: float) -> float:
            return bottom - (v - lo) / (hi - lo) * (bottom - top)

        for v in np_ticks(lo, hi, 0.1):
            y = ymap(v)
            parts.append(line(left, y, right, y, "#e5e5e5"))
            parts.append(text(left - 10, y + 4, f"{v:+.1f}", 11, "end", "#666"))
        zero = ymap(0)
        parts.append(line(left, zero, right, zero, "#555", 1.5))
        slot = (right - left) / len(rows)
        for i, (label, metric) in enumerate(rows):
            x = left + slot * (i + 0.5)
            value = float(metric["mean_daily_rank_ic"])
            ci = metric["block5_bootstrap_95pct_ci"]
            low_ci = float(ci[0]) if ci[0] is not None else value
            high_ci = float(ci[1]) if ci[1] is not None else value
            yv = ymap(value)
            color = "#247ba0" if "L" in label else "#f18f01"
            parts.append(
                f'<rect x="{x - 22:.1f}" y="{min(yv, zero):.1f}" width="44" '
                f'height="{max(1, abs(zero - yv)):.1f}" fill="{color}" opacity="0.85"/>'
            )
            parts.append(line(x, ymap(low_ci), x, ymap(high_ci), "#222", 1.5))
            parts.append(line(x - 6, ymap(low_ci), x + 6, ymap(low_ci), "#222", 1.5))
            parts.append(line(x - 6, ymap(high_ci), x + 6, ymap(high_ci), "#222", 1.5))
            parts.append(text(x, bottom + 20, label, 11, "middle"))
            parts.append(text(x, yv - 7 if value >= 0 else yv + 18, f"{value:+.3f}", 11, "middle", "#222"))

    draw_panel(
        direction, 115, 310, "Direction: Kronos forecast excess vs. realized stock-minus-SPY return", -0.25, 0.25
    )
    draw_panel(volatility, 445, 625, "Volatility: forecast path dispersion vs. realized volatility", 0, 0.9)
    parts.append(
        text(
            600,
            682,
            "ATR(14) was stronger than Kronos dispersion; the 40-bar signal is not an untouched holdout.",
            12,
            "middle",
            "#555",
        )
    )
    parts.append("</svg>")
    out.write_text("\n".join(parts) + "\n")


def intraday_svg(out: Path) -> None:
    reports = [
        ("09:45 / 15m", json.loads(Path("docs/kronos-intraday-15m-report.json").read_text())),
        ("Afternoon / 30m", json.loads(Path("docs/kronos-intraday-30m-report.json").read_text())),
    ]
    width, height = 1000, 540
    left, right, top, bottom = 110, 940, 110, 415
    lo, hi = -0.3, 0.4

    def ymap(v: float) -> float:
        return bottom - (v - lo) / (hi - lo) * (bottom - top)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    ]
    parts.extend(
        ['<rect width="100%" height="100%" fill="white"/>', text(500, 38, "Kronos at the decision time", 23, "middle")]
    )
    parts.append(
        text(
            500,
            63,
            "Mean cross-sectional rank correlation with five-session excess return; "
            "5-session block-bootstrap intervals",
            13,
            "middle",
            "#555",
        )
    )
    for tick in np_ticks(lo, hi, 0.1):
        y = ymap(tick)
        parts.append(line(left, y, right, y, "#e5e5e5"))
        parts.append(text(left - 10, y + 4, f"{tick:+.1f}", 11, "end", "#666"))
    zero = ymap(0)
    parts.append(line(left, zero, right, zero, "#555", 1.5))
    colors = ["#247ba0", "#f18f01", "#8e5ea2"]
    group_width = (right - left) / len(reports)
    bar_width = 60
    for j, (cycle_label, report) in enumerate(reports):
        center = left + group_width * (j + 0.5)
        parts.append(text(center, bottom + 33, cycle_label, 14, "middle"))
        parts.append(
            text(center, bottom + 52, f"n={report['samples']}; {report['decision_cycles']} dates", 11, "middle", "#555")
        )
        for i, feature in enumerate(["forecast_excess", "reversal20", "forecast_excess_after_price_controls"]):
            metric = report["metrics"][feature]
            value = float(metric["mean_cycle_rank_ic"])
            ci = metric["block5_bootstrap_95pct_ci"]
            low_ci, high_ci = float(ci[0]), float(ci[1])
            x = center + (i - 1) * 100
            yv = ymap(value)
            parts.append(
                f'<rect x="{x - bar_width / 2:.1f}" y="{min(yv, zero):.1f}" '
                f'width="{bar_width}" height="{max(1, abs(zero - yv)):.1f}" '
                f'fill="{colors[i]}" opacity="0.86"/>'
            )
            parts.append(line(x, ymap(low_ci), x, ymap(high_ci), "#222", 1.5))
            parts.append(line(x - 7, ymap(low_ci), x + 7, ymap(low_ci), "#222", 1.5))
            parts.append(line(x - 7, ymap(high_ci), x + 7, ymap(high_ci), "#222", 1.5))
            parts.append(text(x, yv - 8 if value >= 0 else yv + 18, f"{value:+.3f}", 12, "middle"))
    parts.append(
        text(
            500,
            510,
            "Exploratory, overlapping five-session returns; selected 23-stock watchlist; not a trading backtest.",
            12,
            "middle",
            "#555",
        )
    )
    parts.append(
        text(
            500,
            485,
            "Blue: Kronos · orange: 20-day reversal · purple: Kronos after price controls; "
            "bars show bootstrap intervals.",
            12,
            "middle",
            "#555",
        )
    )
    parts.append("</svg>")
    out.write_text("\n".join(parts) + "\n")


def kronos_pit_svg(out: Path) -> None:
    reports = [
        ("Morning / 15m", json.loads(Path("docs/kronos-pit-2024-2026-15m-report.json").read_text())),
        ("Afternoon / 30m", json.loads(Path("docs/kronos-pit-2024-2026-30m-report.json").read_text())),
    ]
    width, height = 1080, 560
    left, right, top, bottom = 105, 1010, 105, 420
    features = [
        ("forecast_excess", "Kronos, SPY-relative", "#247ba0"),
        ("reversal20", "20-day reversal", "#f18f01"),
        ("forecast_excess_after_price_controls", "Kronos after controls", "#8e5ea2"),
    ]
    ci_extents = [
        abs(float(bound))
        for _, report in reports
        for feature, _, _ in features
        for bound in report["metrics"][feature]["block5_bootstrap_95pct_ci"]
        if bound is not None
    ]
    max_ci = max(ci_extents, default=0.08)
    tick_step = 0.02 if max_ci <= 0.1 else 0.05 if max_ci <= 0.3 else 0.1
    limit = round(math.ceil((max_ci + tick_step * 1.25) / tick_step) * tick_step, 6)
    lo, hi = -limit, limit
    tick_format = ".2f" if tick_step < 0.1 else ".1f"

    def ymap(v: float) -> float:
        return bottom - (v - lo) / (hi - lo) * (bottom - top)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(width / 2, 36, "Kronos across the July 2024–June 2026 PIT sample", 22, "middle"),
        text(
            width / 2,
            62,
            "Mean per-cycle rank IC for five-session returns; error bars are 5-session block-bootstrap 95% intervals",
            13,
            "middle",
            "#555",
        ),
    ]
    for tick in np_ticks(lo, hi, tick_step):
        y = ymap(tick)
        parts.append(line(left, y, right, y, "#e5e5e5"))
        parts.append(text(left - 10, y + 4, format(tick, f"+{tick_format}"), 11, "end", "#666"))
    zero = ymap(0)
    parts.append(line(left, zero, right, zero, "#555", 1.5))

    group_width = (right - left) / len(reports)
    slot = group_width / len(features)
    bar_width = 54
    for j, (label, report) in enumerate(reports):
        center = left + group_width * (j + 0.5)
        parts.append(text(center, bottom + 32, label, 14, "middle"))
        parts.append(
            text(
                center,
                bottom + 51,
                f"n={report['samples']:,}; {report['decision_cycles']:,} cycles",
                11,
                "middle",
                "#555",
            )
        )
        for i, (feature, _, color) in enumerate(features):
            metric = report["metrics"][feature]
            value = float(metric["mean_cycle_rank_ic"])
            ci = metric["block5_bootstrap_95pct_ci"]
            x = center + (i - 1) * slot
            yv = ymap(value)
            parts.append(
                f'<rect x="{x - bar_width / 2:.1f}" y="{min(yv, zero):.1f}" '
                f'width="{bar_width}" height="{max(1, abs(zero - yv)):.1f}" '
                f'fill="{color}" opacity="0.86"/>'
            )
            if ci[0] is not None and ci[1] is not None:
                low_ci, high_ci = ymap(float(ci[0])), ymap(float(ci[1]))
                parts.extend(
                    [
                        line(x, low_ci, x, high_ci, "#222", 1.5),
                        line(x - 7, low_ci, x + 7, low_ci, "#222", 1.5),
                        line(x - 7, high_ci, x + 7, high_ci, "#222", 1.5),
                    ]
                )
            parts.append(text(x, yv - 8 if value >= 0 else yv + 18, f"{value:+.3f}", 11, "middle"))

    legend_x = 155
    for i, (_, label, color) in enumerate(features):
        x = legend_x + i * 285
        parts.append(f'<rect x="{x}" y="492" width="16" height="12" fill="{color}"/>')
        parts.append(text(x + 23, 503, label, 12))
    parts.append(
        text(
            width / 2,
            535,
            "Exploratory feature test, not a portfolio backtest; current asset snapshot leaves survivorship bias.",
            12,
            "middle",
            "#555",
        )
    )
    parts.append("</svg>")
    out.write_text("\n".join(parts) + "\n")


def np_ticks(lo: float, hi: float, step: float) -> list[float]:
    count = round((hi - lo) / step)
    return [lo + i * step for i in range(count + 1)]


def main() -> None:
    replay_svg(Path("docs/broad-replay-visual.svg"))
    kronos_svg(Path("docs/kronos-feature-visual.svg"))
    intraday_svg(Path("docs/kronos-intraday-visual.svg"))
    kronos_pit_svg(Path("docs/kronos-pit-2024-2026-visual.svg"))
    print("Wrote replay and Kronos daily/intraday SVG charts")


if __name__ == "__main__":
    main()
