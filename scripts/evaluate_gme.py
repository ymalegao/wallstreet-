"""Retrospective, leakage-aware JEV classification study for GameStop's January 2021 episode."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from ws.models.text import TextModel
from ws.store.atomic import atomic_text
from ws.store.event_store import EventStore

ET = ZoneInfo("America/New_York")
REVISION = "b63f651ce8ed64481d3f5e73ecdb05f740042f01"


def ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        rank = (i + 1 + j) / 2
        for k in range(i, j):
            result[ordered[k][0]] = rank
        i = j
    return result


def correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    x, y = ranks(left), ranks(right)
    mx, my = sum(x) / len(x), sum(y) / len(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    return cov / math.sqrt(vx * vy) if vx and vy else None


def run(model: TextModel, rows: list[dict[str, Any]], output: Path, limit: int) -> list[dict[str, Any]]:
    raw = output.read_text() if output.exists() else ""
    current = json.loads(raw) if raw else {}
    by_id = {row["event_id"]: row for row in current.get("articles", [])}
    selected = rows[:limit] if limit else rows

    def score(row: dict[str, Any]) -> dict[str, Any]:
        text = f"{row['headline']}\n{row['body']}"[:7000]
        prediction = model.score_event(text, "GME")
        return {
            "event_id": row["event_id"],
            "published_at": row["published_ts"].isoformat(),
            "available_at": row["first_seen_ts"].isoformat(),
            "was_revised": bool(row["updated_ts"] and row["updated_ts"] > row["published_ts"]),
            "headline": row["headline"],
            **prediction,
        }

    todo = [row for row in selected if row["event_id"] not in by_id]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for i, prediction in enumerate(pool.map(score, todo), start=1):
            by_id[prediction["event_id"]] = prediction
            if i % 25 == 0:
                atomic_text(
                    output,
                    json.dumps(
                        {
                            "model": "JEV-9B decision head",
                            "revision": REVISION,
                            "complete": False,
                            "mode": "zero-shot retrospective demonstration; no fine-tuning",
                            "articles": sorted(by_id.values(), key=lambda row: (row["available_at"], row["event_id"])),
                        },
                        indent=2,
                    )
                    + "\n",
                )
                print(f"GME articles classified this run: {i}/{len(todo)}", flush=True)
    result = sorted(by_id.values(), key=lambda row: (row["available_at"], row["event_id"]))
    atomic_text(
        output,
        json.dumps(
            {
                "model": "JEV-9B decision head",
                "revision": REVISION,
                "complete": len(result) == len(selected),
                "mode": "zero-shot retrospective demonstration; no fine-tuning",
                "limitations": [
                    "JEV postdates January 2021 and may know this famous event; this is not a valid "
                    "out-of-sample trading backtest.",
                    "No human sentiment/materiality labels exist for these articles, so this run tests "
                    "inference and signal behavior, not classification accuracy.",
                    "Revised articles are delayed to updated_at; vendor history does not reconstruct "
                    "the original article versions.",
                    "Historical bars have no bid/ask quotes, slippage or market-impact measurements.",
                ],
                "articles": result,
            },
            indent=2,
        )
        + "\n",
    )
    return result


def price_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text())
    if isinstance(rows, dict):
        rows = rows["bars"]
    if rows and "t" in rows[0]:
        rows = [
            {"ts": row["t"], "open": row["o"], "high": row["h"], "low": row["l"], "close": row["c"], "volume": row["v"]}
            for row in rows
        ]
    frame = pl.DataFrame(rows, strict=False).with_columns(pl.col("ts").str.to_datetime(time_zone="UTC"))
    return frame.sort("ts").to_dicts()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8001")
    parser.add_argument("--limit", type=int, default=0, help="limit scored articles; zero scores all")
    parser.add_argument("--output", type=Path, default=Path("data/gme-2021-jev.json"))
    args = parser.parse_args()
    events = EventStore(Path("data")).read().filter(
        (pl.col("source") == "alpaca_news") & pl.col("tickers").list.contains("GME")
    )
    start, end = date(2021, 1, 1), date(2021, 2, 16)
    rows = [
        row
        for row in events.iter_rows(named=True)
        if start <= row["published_ts"].date() < end and row["headline"]
    ]
    model = TextModel(
        args.endpoint,
        "jev-decision",
        Path("data/inference-cache"),
        "jev",
        Path("data/models/JEV-9B"),
        REVISION,
    )
    articles = run(model, rows, args.output, args.limit)
    minute = price_rows(Path("data/universe/gme_2021_1Min_full.json"))
    rth = []
    for minute_row in minute:
        ts = minute_row["ts"].astimezone(ET)
        if (ts.hour, ts.minute) >= (9, 30) and (ts.hour, ts.minute) < (16, 0):
            rth.append({**minute_row, "local_ts": ts})
    daily = price_rows(Path("data/universe/gme_2021_1Day_probe.json"))
    spy = price_rows(Path("data/universe/spy_2021_1Day_probe.json"))
    daily_by_date = {row["ts"].astimezone(ET).date(): row for row in daily}
    spy_by_date = {row["ts"].astimezone(ET).date(): row for row in spy}
    sessions = sorted(daily_by_date)
    labels: list[dict[str, Any]] = []
    for article in articles:
        available = datetime.fromisoformat(article["available_at"]) + timedelta(minutes=5)
        entry = next((bar for bar in rth if bar["ts"] >= available), None)
        if entry is None:
            continue
        entry_day = entry["local_ts"].date()
        try:
            index = sessions.index(entry_day)
        except ValueError:
            continue
        same_day_close = daily_by_date[entry_day]["close"]
        label_row: dict[str, Any] = {
            "event_id": article["event_id"],
            "available_at": article["available_at"],
            "ticker": "GME",
            "signal": article["signal"],
            "sentiment": article["sentiment"],
            "material": article["material"],
            "entry_open": entry["open"],
            "entry_at": entry["ts"].isoformat(),
            "same_day_close_return": same_day_close / entry["open"] - 1,
        }
        for horizon in (1, 5):
            target = index + horizon
            label_row[f"close_return_{horizon}d"] = (
                daily_by_date[sessions[target]]["close"] / entry["open"] - 1 if target < len(sessions) else None
            )
        labels.append(label_row)
    valid = [r for r in labels if r["close_return_1d"] is not None]
    by_entry_day: dict[date, list[dict[str, Any]]] = {}
    for row in valid:
        day = datetime.fromisoformat(row["entry_at"]).astimezone(ET).date()
        by_entry_day.setdefault(day, []).append(row)
    daily_signals = []
    for day, group in sorted(by_entry_day.items()):
        idx = sessions.index(day)
        if idx + 1 >= len(sessions):
            continue
        signal = sum(r["signal"] for r in group) / len(group)
        start_close = daily_by_date[day]["close"]
        next_close = daily_by_date[sessions[idx + 1]]["close"]
        daily_signals.append(
            {
                "session": str(day),
                "n": len(group),
                "mean_signal": signal,
                "next_close_return": next_close / start_close - 1,
            }
        )
    gme_start, gme_end = date(2021, 1, 22), date(2021, 2, 4)
    gme_hold = daily_by_date[gme_end]["close"] / daily_by_date[gme_start]["close"] - 1
    spy_hold = spy_by_date[gme_end]["close"] / spy_by_date[gme_start]["close"] - 1
    peak = max(
        (bar for bar in rth if gme_start <= bar["local_ts"].date() <= gme_end),
        key=lambda bar: bar["high"],
    )
    peak_return = peak["high"] / daily_by_date[gme_start]["close"] - 1
    finite_1d = [r for r in valid if r["close_return_1d"] is not None]
    finite_5d = [r for r in labels if r["close_return_5d"] is not None]
    output = {
        "status": "RETROSPECTIVE CASE STUDY ONLY; not out-of-sample evidence",
        "article_count": len(articles),
        "article_revised_count": sum(r["was_revised"] for r in articles),
        "articles_with_executable_minute_entry": len(labels),
        "signal_distribution": {
            label: sum(r["sentiment"] == label for r in articles) for label in ["negative", "neutral", "positive"]
        },
        "material_rate": sum(r["material"] for r in articles) / max(1, len(articles)),
        "same_day_signal_vs_close_return_spearman": correlation(
            [r["signal"] for r in finite_1d], [r["same_day_close_return"] for r in finite_1d]
        ),
        "five_session_signal_vs_close_return_spearman": correlation(
            [r["signal"] for r in finite_5d], [r["close_return_5d"] for r in finite_5d]
        ),
        "day_aggregated_signal_vs_next_close_spearman": correlation(
            [r["mean_signal"] for r in daily_signals], [r["next_close_return"] for r in daily_signals]
        ),
        "buy_and_hold_2021_01_22_to_2021_02_04": {"GME": gme_hold, "SPY": spy_hold},
        "intraday_peak_2021_01_22_to_2021_02_04": {
            "timestamp": peak["ts"].isoformat(),
            "high": peak["high"],
            "return_from_2021_01_22_close": peak_return,
        },
        "event_labels": labels,
        "daily_signal_diagnostics": daily_signals,
        "limitations": [
            "JEV was released years after January 2021 and may have memorized the squeeze; all outputs "
            "are post-hoc.",
            "Historical story text has no manual ground-truth sentiment or materiality labels, so "
            "predictive correlation does not measure classifier accuracy.",
            "This event window has very few independent daily observations and no executable "
            "quote/spread or market-impact data.",
            "The five-minute availability delay is unverified; revised text is delayed to the vendor "
            "update time.",
            "This result is one famous episode selected after its outcome; it cannot support an alpha claim.",
        ],
    }
    report_path = Path("docs/gme-2021-report.json")
    atomic_text(report_path, json.dumps(output, indent=2, allow_nan=False) + "\n")
    lines = [
        "# GME January 2021 JEV case study",
        "",
        f"**{output['status']}**",
        "",
        f"JEV scored {len(articles)} GME-tickered articles; {output['article_revised_count']} were later revised. "
        f"A timestamp-safe minute entry was available for {len(labels)} articles.",
        "",
        f"Buy and hold, Jan 22 close to Feb 4 close: GME {gme_hold:.1%}; SPY {spy_hold:.1%}.",
        f"The best intraday high after Jan 22 was {peak['high']:.2f} on {peak['local_ts']:%b %-d}; "
        f"that was {peak_return:.1%} above the Jan 22 close, before the subsequent reversal.",
        "",
        f"Spearman(signal, same-day close return) = {output['same_day_signal_vs_close_return_spearman']}; "
        f"Spearman(signal, 5-session return) = {output['five_session_signal_vs_close_return_spearman']}.",
        "",
        "This is an inference demonstration, not a clean backtest: JEV postdates the event, "
        "there are no manual labels, and the historical window is tiny.",
        "",
        *[f"- {item}" for item in output["limitations"]],
    ]
    atomic_text(Path("docs/gme-2021-report.md"), "\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
