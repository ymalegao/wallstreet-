"""Fail-closed data checks. Exploratory reports explicitly retain unverified assumptions."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from ws import sources_config, universe
from ws.labels import regular_hours
from ws.store.event_store import BarStore, EventStore


def audit(
    root: Path, cfg: dict[str, Any], exploratory: bool = False, manifest_path: Path | None = None
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, Any] = {}
    selected_manifest = manifest_path or root / "research-manifest.json"
    manifest = json.loads(selected_manifest.read_text()) if selected_manifest.exists() else None
    ev = EventStore(root).read()
    if ev.is_empty():
        failures.append("no events")
    else:
        metrics["events"] = ev.height
        metrics["sources"] = dict(ev.group_by("source").len().iter_rows())
        metrics["duplicate_ids"] = ev.height - ev["event_id"].n_unique()
        metrics["acausal_timestamps"] = ev.filter(
            (pl.col("first_seen_ts") > pl.col("ingested_at")) | (pl.col("first_seen_ts") > datetime.now(UTC))
        ).height
        metrics["revised_articles"] = ev.filter(pl.col("updated_ts") > pl.col("published_ts")).height
        metrics["revisions_used_early"] = ev.filter(
            (pl.col("ts_origin") == "vendor") & (pl.col("updated_ts") > pl.col("first_seen_ts"))
        ).height
        for key in ["duplicate_ids", "acausal_timestamps", "revisions_used_early"]:
            if metrics[key]:
                failures.append(key)
        local = pl.col("first_seen_ts").dt.convert_time_zone("America/New_York")
        minutes = local.dt.hour().cast(pl.Int32) * 60 + local.dt.minute().cast(pl.Int32)
        metrics["outside_clock_hours_fraction"] = ev.select(((minutes < 570) | (minutes >= 960)).mean()).item()
    daily = BarStore(root, "1Day_raw").read()
    intraday = BarStore(root, "15Min").read()
    if daily.is_empty():
        failures.append("no daily bars")
    else:
        elig = universe.eligibility(daily).filter(pl.col("eligible"))
        eligible_months = (
            elig.with_columns(month=pl.col("session").dt.strftime("%Y-%m")).select("symbol", "month").unique()
        )
        metrics["eligible_ticker_months"] = eligible_months.height
        if not ev.is_empty():
            # Warm-up history is not part of the requested news coverage window.
            if manifest is not None:
                eligible_months = eligible_months.filter(
                    (pl.col("month") >= manifest["start"][:7]) & (pl.col("month") <= manifest["end"][:7])
                )
                if manifest.get("symbols"):
                    eligible_months = eligible_months.filter(pl.col("symbol").is_in(manifest["symbols"]))
                metrics["eligible_ticker_months_in_manifest"] = eligible_months.height
            event_months = (
                ev.explode("tickers", empty_as_null=True)
                .select(pl.col("tickers").alias("symbol"), pl.col("first_seen_ts").dt.strftime("%Y-%m").alias("month"))
                .unique()
            )
            if manifest is not None:
                event_months = event_months.filter(
                    (pl.col("month") >= manifest["start"][:7]) & (pl.col("month") <= manifest["end"][:7])
                )
                if manifest.get("symbols"):
                    event_months = event_months.filter(pl.col("symbol").is_in(manifest["symbols"]))
            coverage = eligible_months.join(event_months, on=["symbol", "month"], how="semi").height / max(
                1, eligible_months.height
            )
            metrics["news_ticker_month_coverage"] = coverage
            if coverage < 0.95:
                failures.append("news coverage below 95%")
        if not eligible_months.height:
            failures.append("no eligible ticker-months")
    if intraday.is_empty():
        failures.append("no intraday bars")
    else:
        metrics["intraday_bars"] = intraday.height
        if "SPY" not in intraday["symbol"].unique().to_list():
            failures.append("no SPY intraday benchmark")
        if intraday.select(pl.struct("symbol", "ts").n_unique()).item() != intraday.height:
            failures.append("duplicate intraday bars")
        rth = regular_hours(intraday)
        metrics["regular_hours_bars"] = rth.height
        if not rth.height:
            failures.append("no regular-hours bars")
        if intraday.filter((pl.col("close") <= 0) | (~pl.col("close").is_finite())).height:
            failures.append("invalid bar prices")
    reject_path = root / "state" / "rejects.jsonl"
    metrics["normalization_rejects"] = (
        sum(bool(line.strip()) for line in reject_path.read_text().splitlines()) if reject_path.exists() else 0
    )
    if metrics["normalization_rejects"]:
        failures.append("normalization rejects")
    for setting in ["edgar.acceptance_tz", "bars.feed", "dedupe.jaccard_threshold", "latency.news_minutes"]:
        try:
            sources_config.verified(cfg, setting)
        except sources_config.UnverifiedSetting:
            (warnings if exploratory else failures).append(f"unverified {setting}")
    warnings.append("Symbol coverage is limited to the declared sample, not a survivorship-free market universe")
    return {
        "status": "FAIL" if failures else ("EXPLORATORY" if exploratory else "PASS"),
        "failures": failures,
        "warnings": warnings,
        "metrics": metrics,
    }
