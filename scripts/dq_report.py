"""Data-quality report for the M1 exit gate. Writes docs/data-quality-report.md.

    uv run python scripts/dq_report.py

Hard failures (exit code 1):
  * any event with first_seen_ts > ingested_at, or with first_seen_ts in the future
  * any event without a usable ticker list among eligible names below the coverage target
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from ws import sources_config, universe
from ws.config import load_settings
from ws.labels import regular_hours
from ws.schema import Event, TsOrigin
from ws.store.dedupe import near_duplicate_clusters
from ws.store.event_store import BarStore, EventStore

COVERAGE_TARGET = 0.95


def main() -> int:
    st = load_settings()
    ev = EventStore(st.data_dir).read()
    out = ["# Data-quality report", "", f"Generated {datetime.now(UTC).isoformat()}", ""]
    failures: list[str] = []

    if ev.is_empty():
        print("no events; run backfill first")
        return 1

    out += [
        "## Volume",
        "",
        "| source | year | events | per day (median) | revised % | no tickers % |",
        "|---|---|---|---|---|---|",
    ]
    per = (
        ev.with_columns(year=pl.col("first_seen_ts").dt.year(), day=pl.col("first_seen_ts").dt.date())
        .group_by("source", "year")
        .agg(
            pl.len().alias("n"),
            pl.col("day").value_counts().struct.field("count").median().alias("per_day"),
            (pl.col("updated_ts") > pl.col("published_ts")).mean().alias("revised"),
            (pl.col("tickers").list.len() == 0).mean().alias("no_tickers"),
        )
        .sort("source", "year")
    )
    for r in per.iter_rows(named=True):
        out.append(
            f"| {r['source']} | {r['year']} | {r['n']} | {r['per_day']:.0f} "
            f"| {100 * (r['revised'] or 0):.1f} | {100 * r['no_tickers']:.1f} |"
        )

    out += ["", "## Timestamp audit", ""]
    acausal = ev.filter(pl.col("first_seen_ts") > pl.col("ingested_at")).height
    future = ev.filter(pl.col("first_seen_ts") > datetime.now(UTC) + timedelta(minutes=5)).height
    out.append(f"- first_seen_ts > ingested_at: **{acausal}**")
    out.append(f"- first_seen_ts in the future: **{future}**")
    if acausal or future:
        failures.append("timestamp audit")
    by_origin = ev.group_by("ts_origin").len()
    out.append(f"- ts_origin counts: {dict(by_origin.iter_rows())}")
    et_min = ev.select(
        (
            pl.col("first_seen_ts").dt.convert_time_zone("America/New_York").dt.hour() * 60
            + pl.col("first_seen_ts").dt.convert_time_zone("America/New_York").dt.minute()
        ).alias("m")
    )["m"]
    out.append(f"- share of events outside 09:30-16:00 ET: {((et_min >= 16 * 60) | (et_min < 9 * 60 + 30)).mean():.1%}")

    daily = BarStore(st.data_dir, "1Day_raw").read()
    if daily.is_empty():
        out += ["", "## Coverage", "", "- not computed: no daily bars (run `backfill.py daily-bars`)"]
        failures.append("coverage not computed")
    else:
        elig = universe.eligibility(daily).filter(pl.col("eligible"))
        elig_months = elig.with_columns(month=pl.col("session").dt.strftime("%Y-%m")).select("symbol", "month").unique()
        ev_months = (
            ev.explode("tickers", empty_as_null=True)
            .select(pl.col("tickers").alias("symbol"), pl.col("first_seen_ts").dt.strftime("%Y-%m").alias("month"))
            .unique()
        )
        covered = elig_months.join(ev_months, on=["symbol", "month"], how="semi").height
        cov = covered / max(elig_months.height, 1)
        out += [
            "",
            "## Coverage",
            "",
            f"- eligible ticker-months with >= 1 event: **{cov:.1%}** (target {COVERAGE_TARGET:.0%})",
        ]
        if cov < COVERAGE_TARGET:
            failures.append("coverage")

    intraday = BarStore(st.data_dir, "15Min").read(["SPY"])
    if not intraday.is_empty():
        rth = regular_hours(intraday)
        per_session = rth.group_by(pl.col("ts").dt.convert_time_zone("America/New_York").dt.date()).len()
        out += [
            "",
            "## Bars (SPY 15Min)",
            "",
            f"- bars outside regular hours: {intraday.height - rth.height}",
            f"- sessions: {per_session.height}, bars/session "
            f"min={per_session['len'].min()} median={per_session['len'].median()}",
        ]

    try:
        thr = float(sources_config.verified(sources_config.load(), "dedupe.jaccard_threshold"))
        recent = ev.filter(pl.col("first_seen_ts") > datetime.now(UTC) - timedelta(days=30))
        events = [
            Event(**{**r, "ts_origin": TsOrigin(r["ts_origin"]), "meta": {}}) for r in recent.iter_rows(named=True)
        ]
        clusters = near_duplicate_clusters(events, thr)
        dup = sum(1 for k, v in clusters.items() if k != v)
        out += [
            "",
            "## Near-duplicates (last 30 days)",
            "",
            f"- {dup}/{len(events)} items are copies of an earlier item (J >= {thr})",
        ]
    except sources_config.UnverifiedSetting as exc:
        out += ["", "## Near-duplicates", "", f"- skipped: {exc}"]

    out += ["", f"**Result:** {'FAIL: ' + ', '.join(failures) if failures else 'PASS'}"]
    Path("docs/data-quality-report.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
