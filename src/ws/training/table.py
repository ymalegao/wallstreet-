"""Point-in-time features and forward labels for px-v1 model training."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import polars as pl

from ws.features.px_v1 import PX_V1_FEATURES, compute_px_v1
from ws.store.history import BarHistory
from ws.store.pit_bars import PITBarStore

LABEL_HORIZON = 21
FEATURE_SET_VERSION = "px-v1"
_XNYS = xcals.get_calendar("XNYS")


def build_training_table(
    store_path: Path,
    universe_path: Path,
    output_path: Path,
    *,
    horizon: int = LABEL_HORIZON,
) -> dict[str, Any]:
    """Build only pre-holdout rows and labels from an adjusted Alpaca snapshot.

    The holdout boundary comes from the immutable ingestion manifest. The bar query is
    capped at the final session before that boundary, so this code never loads holdout
    outcomes.
    """
    if horizon != LABEL_HORIZON:
        raise ValueError(f"px-v1 label horizon is fixed at {LABEL_HORIZON} sessions")
    metadata_output_path = output_path.with_suffix(output_path.suffix + ".json")
    if output_path.exists() or metadata_output_path.exists():
        raise FileExistsError(
            f"training table snapshots are immutable and this output already exists: {output_path}"
        )
    universe_data = json.loads(universe_path.read_text(encoding="utf-8"))
    universe = tuple(sorted({str(symbol).strip().upper() for symbol in universe_data["symbols"]}))
    if not universe or any(not symbol for symbol in universe):
        raise ValueError("universe must contain at least one non-empty symbol")
    manifest_path = store_path / "snapshot.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing immutable Alpaca snapshot manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feed = str(manifest.get("feed", "")).lower()
    if feed not in {"iex", "sip"} or manifest.get("adjustment") != "all":
        raise ValueError("training requires an adjusted Alpaca IEX or SIP snapshot")
    missing_symbols = sorted((set(universe) | {"SPY"}) - set(manifest.get("symbols", [])))
    if missing_symbols:
        raise ValueError(f"snapshot manifest does not include universe symbols: {missing_symbols}")
    earliest_values = [date.fromisoformat(value) for value in manifest.get("actual_earliest_by_symbol", {}).values()]
    latest_values = [date.fromisoformat(value) for value in manifest.get("actual_latest_by_symbol", {}).values()]
    if not earliest_values or not latest_values:
        raise ValueError("snapshot manifest contains no returned bars")
    session_start = _snapshot_session_start(manifest)
    latest_session = max(latest_values)
    labeled_end = _shift_session(latest_session, -(horizon + 1))
    holdout_start = _one_year_before(labeled_end)
    sessions = _session_dates(session_start, labeled_end)
    eligible = [session for session in sessions if _shift_session(session, horizon + 1) < holdout_start]
    if not eligible:
        raise ValueError("snapshot does not have enough pre-holdout history to build labeled rows")
    prior_sessions = _session_dates(eligible[0], holdout_start - timedelta(days=1))
    if not prior_sessions:
        raise ValueError("no pre-holdout bar sessions are available")
    through = _decision_time(prior_sessions[-1])
    symbols = (*universe, "SPY")
    history = BarHistory.load(PITBarStore(store_path), symbols, through=through)
    visible = history.visible(through, symbols=symbols)
    frame = _build_rows(
        visible,
        universe,
        sessions,
        holdout_start=holdout_start,
        first_eligible_session=eligible[0],
        horizon=horizon,
    )
    if frame.is_empty():
        raise ValueError("no training rows were generated from the visible Alpaca snapshot")

    snapshot_id = str(manifest["snapshot_id"])
    metadata = {
        "snapshot_id": snapshot_id,
        "market_data_feed": feed,
        "feature_set": FEATURE_SET_VERSION,
        "feature_names": list(PX_V1_FEATURES),
        "label_horizon": horizon,
        "session_start": session_start.isoformat(),
        "holdout_start": holdout_start.isoformat(),
        "labeled_end": labeled_end.isoformat(),
        "universe_id": str(universe_data.get("index", "universe")),
        "universe_sha256": hashlib.sha256(universe_path.read_bytes()).hexdigest(),
        "survivorship_biased": True,
        "source_store_path": str(store_path.resolve()),
        "universe_path": str(universe_path.resolve()),
    }
    _write_table(frame, output_path, metadata)
    return {**metadata, "row_count": frame.height, "date_count": frame.get_column("date").n_unique()}


def write_data_coverage(manifest: dict[str, Any], output_path: Path) -> None:
    """Render actual vendor coverage from a completed bar-download manifest."""
    symbols = sorted(set(manifest.get("symbols", [])))
    earliest = manifest.get("actual_earliest_by_symbol", {})
    latest = manifest.get("actual_latest_by_symbol", {})
    lines = [
        "# Data coverage",
        "",
        f"- Snapshot: `{manifest.get('snapshot_id', 'unknown')}`",
        f"- Feed: `{manifest.get('feed', 'unknown')}`",
        f"- Adjustment: `{manifest.get('adjustment', 'unknown')}`",
        f"- Requested range: {manifest.get('requested_start', '?')} through {manifest.get('requested_end', '?')}",
        "- Universe: current S&P 100 ETF holdings proxy applied historically; survivorship-biased.",
        (
            "- Data quality: IEX is a single-venue feed and is for development and testing only; "
            "do not use this snapshot for live trading decisions."
            if str(manifest.get("feed", "")).lower() == "iex"
            else "- Data quality: consolidated SIP feed."
        ),
        "",
        "| Symbol | Earliest returned session | Latest returned session |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| {symbol} | {earliest.get(symbol, 'no bars returned')} | {latest.get(symbol, 'no bars returned')} |"
        for symbol in symbols
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_pre_holdout_table(table_path: Path, *, holdout_start: date) -> pl.DataFrame:
    """Read only labeled rows whose decision and outcome dates precede the holdout."""
    lazy_frame = pl.scan_parquet(table_path)
    required = {"date", "label_exit_date", "label", "raw_forward_return"}
    missing = required - set(lazy_frame.collect_schema().names())
    if missing:
        raise ValueError(f"training table is missing required columns: {sorted(missing)}")
    return lazy_frame.filter(
        (pl.col("date") < holdout_start) & (pl.col("label_exit_date") < holdout_start)
    ).collect()


def _build_rows(
    visible_bars: pl.DataFrame,
    universe: tuple[str, ...],
    sessions: list[date],
    *,
    holdout_start: date,
    first_eligible_session: date,
    horizon: int,
    allow_holdout: bool = False,
) -> pl.DataFrame:
    """Build features and labels from bars already bounded before holdout."""
    if visible_bars.is_empty():
        return pl.DataFrame()
    local_dates = visible_bars.get_column("event_time").dt.convert_time_zone("America/New_York").dt.date()
    visible_bars = visible_bars.with_columns(local_dates.alias("session_date"))
    opens: dict[tuple[str, date], float] = {}
    for bar_row in visible_bars.select("symbol", "session_date", "open").iter_rows(named=True):
        opens[(str(bar_row["symbol"]).upper(), bar_row["session_date"])] = float(bar_row["open"])

    rows: list[dict[str, object]] = []
    started = False
    for index, session_date in enumerate(sessions):
        if session_date < first_eligible_session or (allow_holdout and session_date < holdout_start):
            continue
        exit_index = index + horizon + 1
        if exit_index >= len(sessions) or (not allow_holdout and sessions[exit_index] >= holdout_start):
            continue
        tick_at = _decision_time(session_date)
        date_bars = visible_bars.filter(
            (pl.col("session_date") <= session_date) & (pl.col("available_at") <= tick_at)
        )
        recent_sessions = sorted(date_bars.get_column("session_date").unique().to_list())[-300:]
        date_bars = date_bars.filter(pl.col("session_date").is_in(recent_sessions)).drop("session_date")
        feature_bars = date_bars.filter(pl.col("symbol").is_in(list(universe)))
        benchmark_bars = date_bars.filter(pl.col("symbol") == "SPY")
        features = compute_px_v1(feature_bars, benchmark_bars)
        feature_rows = {row["symbol"]: row for row in features.iter_rows(named=True)}
        valid_momentum = sum(
            feature_rows.get(symbol, {}).get("mom_12_1") is not None for symbol in universe
        )
        if not started:
            if valid_momentum / len(universe) < 0.8:
                continue
            started = True
        raw_returns: dict[str, float] = {}
        entry_date = sessions[index + 1]
        exit_date = sessions[exit_index]
        for symbol in universe:
            entry = opens.get((symbol, entry_date))
            exit_price = opens.get((symbol, exit_date))
            if entry is not None and exit_price is not None and entry > 0:
                raw_returns[symbol] = exit_price / entry - 1.0
        labels = _rank_percentiles(raw_returns)
        for symbol in universe:
            feature = feature_rows.get(symbol, {name: None for name in (*PX_V1_FEATURES, "adv_dollars_20d_raw")})
            record: dict[str, object] = {
                "date": session_date,
                "symbol": symbol,
                **{name: feature.get(name) for name in PX_V1_FEATURES},
                "adv_dollars_20d_raw": feature.get("adv_dollars_20d_raw"),
                "label": labels.get(symbol),
                "raw_forward_return": raw_returns.get(symbol),
                "label_exit_date": exit_date,
            }
            rows.append(record)
    if not rows:
        return pl.DataFrame()
    result = pl.DataFrame(rows).sort(["date", "symbol"])
    if not allow_holdout and result.filter(pl.col("label_exit_date") >= holdout_start).height:
        raise AssertionError("training label reaches the holdout boundary")
    return result


def _rank_percentiles(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    result: dict[str, float] = {}
    count = len(ordered)
    index = 0
    while index < count:
        end = index + 1
        while end < count and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        percentile = (average_rank - 1.0) / (count - 1) if count > 1 else 0.0
        for symbol, _ in ordered[index:end]:
            result[symbol] = percentile
        index = end
    return result


def _session_dates(start: date, end: date) -> list[date]:
    if end < start:
        return []
    return [session.date() for session in _XNYS.sessions_in_range(start.isoformat(), end.isoformat())]


def _snapshot_session_start(manifest: dict[str, Any]) -> date:
    """Start calendar iteration at the first available bar and supported XNYS session."""
    requested = date.fromisoformat(manifest["requested_start"])
    actual = [date.fromisoformat(value) for value in manifest.get("actual_earliest_by_symbol", {}).values()]
    if not actual:
        raise ValueError("snapshot manifest contains no earliest-bar dates")
    calendar_start = _XNYS.first_session.date()
    return max(requested, min(actual), calendar_start)


def _shift_session(value: date, offset: int) -> date:
    margin = max(60, abs(offset) * 3)
    start, end = value - timedelta(days=margin), value + timedelta(days=margin)
    sessions = _session_dates(start, end)
    try:
        index = sessions.index(value)
    except ValueError as error:
        raise ValueError(f"{value} is not an XNYS session") from error
    shifted = index + offset
    if shifted < 0 or shifted >= len(sessions):
        raise ValueError(f"cannot shift {value} by {offset} XNYS sessions")
    return sessions[shifted]


def _one_year_before(value: date) -> date:
    try:
        candidate = value.replace(year=value.year - 1)
    except ValueError:
        candidate = value.replace(year=value.year - 1, day=28)
    next_sessions = _session_dates(candidate, candidate + timedelta(days=10))
    if not next_sessions:
        raise ValueError(f"no XNYS session found near holdout start {candidate}")
    return next_sessions[0]


def _decision_time(session_date: date) -> datetime:
    session = _XNYS.date_to_session(session_date.isoformat(), direction="none")
    close = _XNYS.session_close(session).to_pydatetime()
    return close + timedelta(minutes=30)


def _write_table(frame: pl.DataFrame, output_path: Path, metadata: dict[str, Any]) -> None:
    from ws.store.atomic import atomic_path

    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    if output_path.exists() or metadata_path.exists():
        raise FileExistsError(f"training table snapshots are immutable: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    string_metadata = {key: json.dumps(value, sort_keys=True) for key, value in metadata.items()}
    with atomic_path(output_path) as temporary:
        frame.write_parquet(temporary, metadata=string_metadata)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
