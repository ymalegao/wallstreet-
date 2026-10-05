from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from ws.metrics import classification
from ws.replay import Governor, Position, Rules, run_replay


def test_settlement_and_caps_are_enforced():
    g = Governor(Rules())
    ts = datetime(2025, 3, 10, 14, tzinfo=UTC)
    g.positions["AAPL"] = Position("AAPL", 1, ts, 100, ts + timedelta(days=5), 90, 1.0)
    g.cash = 0
    g.sell("AAPL", ts, 110, "test")
    assert g.cash == 0
    assert g.allocation("MSFT", 110)[0] == 0
    g.settle(date(2025, 3, 10))
    assert g.cash == 0
    g.settle(date(2025, 3, 11))
    assert g.cash > 0
    for ticker in ["AAPL", "MSFT"]:
        g.positions[ticker] = Position(ticker, 1, ts, 100, ts, 90, 1.0)
    assert g.allocation("NVDA", 500)[1] == "sector cap"


def test_unclassified_symbols_do_not_share_a_fake_sector_cap():
    g = Governor(Rules(max_positions=10))
    ts = datetime(2025, 3, 10, 14, tzinfo=UTC)
    for ticker in ["UNMAPPED1", "UNMAPPED2", "UNMAPPED3"]:
        g.positions[ticker] = Position(ticker, 1, ts, 100, ts, 90, 1.0)
    assert g.allocation("UNMAPPED4", 500)[0] > 0


def test_entries_are_after_decisions_and_stops_gap_at_open():
    t = datetime(2025, 3, 10, 14, tzinfo=UTC)  # 10:00 ET
    rows = [
        {
            "symbol": "AAPL",
            "ts": t + timedelta(minutes=15 * i),
            "open": p,
            "high": p + 1,
            "low": p - 1,
            "close": p,
            "volume": 100,
        }
        for i, p in enumerate([100.0, 80.0, 81.0])
    ]
    bars = pl.DataFrame(rows)
    c = {
        "ticker": "AAPL",
        "cycle_ts": t - timedelta(minutes=15),
        "feature_ts": t - timedelta(minutes=15),
        "entry_ts": t,
        "exit_ts": t + timedelta(minutes=30),
        "atr_fraction": 0.05,
        "signal": 1.0,
    }
    out = run_replay([c], bars, Rules(cost_bps_per_side=0))
    assert out["trades"][0]["exit_px"] == 80  # gap through the $90 stop
    assert out["trades"][0]["exit_reason"] == "ATR stop"
    assert out["trades"][0]["pnl"] == -20
    with pytest.raises(ValueError):
        run_replay([{**c, "entry_ts": c["cycle_ts"]}], bars)
    with pytest.raises(ValueError):
        run_replay([{**c, "feature_ts": t}], bars)


def test_invalid_predictions_count_as_misses():
    m = classification(["positive", "positive"], ["positive", "ERROR"], ["negative", "neutral", "positive"])
    assert m["accuracy"] == 0.5
    assert m["per_class"]["positive"]["recall"] == 0.5


def test_daily_model_timestamps_use_exchange_date_across_dst():
    from ws.market_time import daily_model_timestamp, session_model_timestamp

    # Alpaca daily bars are stamped at 04:00 UTC in summer and 05:00 UTC in winter.
    summer = datetime(2026, 7, 1, 4, tzinfo=UTC)
    winter = datetime(2025, 1, 2, 5, tzinfo=UTC)
    assert daily_model_timestamp(summer) == session_model_timestamp(date(2026, 7, 1))
    assert daily_model_timestamp(winter) == session_model_timestamp(date(2025, 1, 2))
    with pytest.raises(ValueError):
        daily_model_timestamp(datetime(2025, 1, 2))
