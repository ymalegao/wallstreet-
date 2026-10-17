from __future__ import annotations

import pytest

from ws.broker.costs import CostModel
from ws.portfolio import PortfolioConstructor, RiskManager


def test_constructor_is_deterministic_and_ignores_nonpositive_scores() -> None:
    targets = PortfolioConstructor(holdings=2).construct({"b": 0.5, "A": 0.5, "c": 0.0})
    assert targets.ranked_symbols == ("A", "B")
    assert targets.weights == {"A": 0.5, "B": 0.5}


def test_risk_manager_applies_name_sector_volatility_and_drawdown_limits() -> None:
    decision = RiskManager().apply(
        {"A": 0.10, "B": 0.10, "C": 0.10},
        sectors={"A": "Tech", "B": "Tech", "C": "Health"},
        annual_volatility=0.30,
        drawdown=0.16,
        current_weights={"A": 0.02, "B": 0.00, "C": 0.01},
    )
    assert decision.weights == {"A": pytest.approx(0.01), "C": pytest.approx(0.005)}
    assert "max_name_weight" in decision.constraints
    assert "drawdown_no_new_buys" in decision.constraints
    assert "volatility_target" in decision.constraints


def test_liquidity_cap_is_reported() -> None:
    decision = RiskManager().apply(
        {"A": 0.05, "B": 0.05},
        sectors={"A": "Tech", "B": "Tech"},
        equity=1000,
        adv_dollars={"A": 1000, "B": 1000},
    )
    assert decision.weights == {"A": pytest.approx(0.01), "B": pytest.approx(0.01)}
    assert "liquidity_cap" in decision.constraints


def test_sector_cap_scales_names_within_the_sector() -> None:
    requested = {f"TICKER{i}": 0.05 for i in range(10)}
    sectors = {symbol: "Tech" for symbol in requested}
    decision = RiskManager().apply(requested, sectors=sectors)
    assert decision.gross_exposure == pytest.approx(0.30)
    assert all(weight == pytest.approx(0.03) for weight in decision.weights.values())
    assert "max_sector_weight" in decision.constraints


def test_cost_model_increases_with_trade_size_and_volatility() -> None:
    costs = CostModel()
    small = costs.rate(order_notional=100, adv_dollars=1_000_000, annual_volatility=0.2)
    large = costs.rate(order_notional=10_000, adv_dollars=1_000_000, annual_volatility=0.4)
    assert large > small > 0
    assert costs.dollars(order_notional=100, adv_dollars=1_000_000, annual_volatility=0.2) == pytest.approx(
        100 * small
    )
