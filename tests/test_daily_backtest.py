from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from ws.backtest import simulate_daily
from ws.broker.costs import CostModel


def _panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "event_time": [
                datetime(2026, 1, 5, 21, tzinfo=UTC),
                datetime(2026, 1, 6, 21, tzinfo=UTC),
            ],
            "open": [100.0, 111.0],
            "close": [110.0, 122.1],
            "target_weight": [0.5, 0.5],
            "signal_available_at": [
                datetime(2026, 1, 2, 21, 15, tzinfo=UTC),
                datetime(2026, 1, 5, 21, 15, tzinfo=UTC),
            ],
            "adv_dollars": [10_000_000.0, 10_000_000.0],
            "annual_volatility": [0.2, 0.2],
        }
    )


def test_daily_simulator_executes_prior_close_signal_and_tracks_cash() -> None:
    result = simulate_daily(
        _panel(),
        initial_equity=1_000.0,
        cost_model=CostModel(commission_bps=0, half_spread_bps=0, impact_coefficient=0),
    )
    assert result.get_column("equity").to_list() == pytest.approx([1_050.0, 1_107.75])
    assert result.get_column("net_return").to_list() == pytest.approx([0.05, 0.055])
    assert result.get_column("transaction_costs").to_list() == [0.0, 0.0]


def test_daily_simulator_rejects_signal_that_is_not_available_at_open() -> None:
    panel = _panel().head(1).with_columns(
        pl.lit(datetime(2026, 1, 5, 21, 15, tzinfo=UTC)).alias("signal_available_at")
    )
    with pytest.raises(ValueError, match="unavailable at session open"):
        simulate_daily(panel)

