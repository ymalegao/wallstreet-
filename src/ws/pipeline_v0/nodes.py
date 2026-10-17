"""Required source, feature, alpha, portfolio, and risk nodes for pipeline v0."""

from __future__ import annotations

from math import isfinite

from ws.broker.base import Broker
from ws.core.node import Node, NodeContext
from ws.pipeline_v0.messages import (
    Features,
    Insights,
    MarketSnapshot,
    Order,
    Orders,
    PortfolioState,
    PortfolioTargets,
    RiskAdjustedTargets,
)
from ws.pipeline_v0.metrics import adv_dollars_20d, momentum_12_1, portfolio_vol_21d, realized_vol_21d
from ws.portfolio.constructor import PortfolioConstructor as ScorePortfolioConstructor
from ws.portfolio.risk import RiskConfig, RiskManager


class FeatureBuilder(Node):
    """Compute fixed-window momentum, realized volatility, and dollar ADV features."""

    name = "features"
    reads = (MarketSnapshot,)
    writes = Features

    async def run(self, ctx: NodeContext) -> Features:
        """Calculate cross-sectional features from bars visible in the snapshot."""
        bars = ctx.inputs[MarketSnapshot].bars
        momentum = momentum_12_1(bars)
        volatility = realized_vol_21d(bars)
        liquidity = adv_dollars_20d(bars)
        frame = momentum.join(volatility, on="symbol", how="left").join(liquidity, on="symbol", how="left")
        return Features(tick=ctx.tick, producer=self.name, frame=frame.sort("symbol"), feature_set="mom-v0")


class Alpha(Node):
    """Emit the configured 12-1 momentum score for every eligible symbol."""

    name = "alpha"
    reads = (Features,)
    writes = Insights

    def __init__(self, model_id: str = "momentum-12-1") -> None:
        super().__init__()
        if model_id != "momentum-12-1":
            raise ValueError("pipeline v0 supports only model_id='momentum-12-1'")
        self.model_id = model_id

    async def run(self, ctx: NodeContext) -> Insights:
        """Turn non-null finite 12-1 momentum features into symbol scores."""
        frame = ctx.inputs[Features].frame
        scores: dict[str, float] = {}
        for row in frame.iter_rows(named=True):
            score = row["mom_12_1"]
            if score is not None and isfinite(float(score)):
                scores[row["symbol"]] = float(score)
        return Insights(tick=ctx.tick, producer=self.name, scores=scores, model_id=self.model_id)


class PortfolioConstruction(Node):
    """Convert positive momentum scores into equal-weight long-only targets."""

    name = "portfolio_construction"
    reads = (Insights,)
    writes = PortfolioTargets
    kind = "stateful"

    def __init__(self, holdings: int = 25) -> None:
        super().__init__()
        self.constructor = ScorePortfolioConstructor(holdings=holdings)

    async def run(self, ctx: NodeContext) -> PortfolioTargets:
        """Build equal-weight targets from the top positive scores."""
        result = self.constructor.construct(ctx.inputs[Insights].scores)
        return PortfolioTargets(
            tick=ctx.tick,
            producer=self.name,
            weights=dict(result.weights),
            ranked_symbols=result.ranked_symbols,
        )


class RiskManagement(Node):
    """Apply deterministic long-only, liquidity, volatility, and drawdown constraints."""

    name = "risk_management"
    reads = (PortfolioTargets, PortfolioState, MarketSnapshot)
    writes = RiskAdjustedTargets
    kind = "stateful"

    def __init__(
        self,
        max_name_weight: float = 0.05,
        max_gross_weight: float = 1.0,
        max_sector_weight: float = 1.0,
        target_annual_volatility: float = 0.15,
        drawdown_halt: float = 0.15,
        max_adv_fraction: float = 0.01,
    ) -> None:
        super().__init__()
        self.manager = RiskManager(
            RiskConfig(
                max_name_weight=max_name_weight,
                max_gross_weight=max_gross_weight,
                max_sector_weight=max_sector_weight,
                target_annual_volatility=target_annual_volatility,
                drawdown_halt=drawdown_halt,
                max_adv_fraction=max_adv_fraction,
            )
        )
        self._sector_cap_disabled_runs: set[str] = set()

    async def run(self, ctx: NodeContext) -> RiskAdjustedTargets:
        """Return risk-approved weights and record the disabled sector cap once per run."""
        targets = ctx.inputs[PortfolioTargets]
        state = ctx.inputs[PortfolioState]
        snapshot = ctx.inputs[MarketSnapshot]
        config = self.manager.config
        if config.max_sector_weight >= 1.0 and ctx.run_id not in self._sector_cap_disabled_runs:
            ctx.log.emit("sector_cap_disabled", run_id=ctx.run_id, tick=ctx.tick.id)
            self._sector_cap_disabled_runs.add(ctx.run_id)

        current_weights = {
            symbol: value / state.equity for symbol, value in state.market_values.items() if state.equity > 0
        }
        drawdown = 1.0 - state.equity / state.peak_equity if state.peak_equity > 0 else 0.0
        liquidity_frame = adv_dollars_20d(snapshot.bars)
        adv = {
            row["symbol"]: float(row["adv_dollars_20d"])
            for row in liquidity_frame.iter_rows(named=True)
            if row["adv_dollars_20d"] is not None
        }
        requested_volatility = portfolio_vol_21d(snapshot.bars, targets.weights)
        decision = self.manager.apply(
            targets.weights,
            sectors={},
            annual_volatility=requested_volatility if requested_volatility > 0 else None,
            drawdown=drawdown,
            current_weights=current_weights,
            equity=state.equity,
            adv_dollars=adv,
        )
        return RiskAdjustedTargets(
            tick=ctx.tick,
            producer=self.name,
            weights=dict(decision.weights),
            gross_exposure=decision.gross_exposure,
            constraints=decision.constraints,
        )


class Execution(Node):
    """Turn risk-approved weights into close-referenced orders for the broker."""

    name = "execution"
    reads = (RiskAdjustedTargets, PortfolioState, MarketSnapshot)
    writes = Orders
    kind = "stateful"

    def __init__(self, broker: Broker, min_order_dollars: float = 1.0) -> None:
        super().__init__()
        if not isfinite(min_order_dollars) or min_order_dollars < 0:
            raise ValueError("min_order_dollars must be finite and non-negative")
        self.broker = broker
        self.min_order_dollars = min_order_dollars

    async def run(self, ctx: NodeContext) -> Orders:
        """Submit sells then buys sized from approved weights and current marked values."""
        approved = ctx.inputs[RiskAdjustedTargets]
        state = ctx.inputs[PortfolioState]
        snapshot = ctx.inputs[MarketSnapshot]
        latest_closes: dict[str, float] = {}
        for row in snapshot.bars.sort(["event_time", "symbol"]).iter_rows(named=True):
            latest_closes[row["symbol"]] = float(row["close"])

        orders: list[Order] = []
        skipped_small = 0
        symbols = set(approved.weights) | set(state.positions)
        sells: list[Order] = []
        buys: list[Order] = []
        for symbol in sorted(symbols):
            target_value = float(approved.weights.get(symbol, 0.0)) * state.equity
            current_value = state.market_values.get(symbol)
            if current_value is None:
                current_value = state.positions.get(symbol, 0.0) * latest_closes.get(symbol, 0.0)
            delta = target_value - current_value
            if abs(delta) < self.min_order_dollars:
                skipped_small += 1
                continue
            reference_price = latest_closes.get(symbol)
            if reference_price is None and state.positions.get(symbol, 0.0) > 0:
                shares = state.positions[symbol]
                reference_price = state.market_values.get(symbol, 0.0) / shares if shares > 0 else None
            if reference_price is None or reference_price <= 0:
                raise ValueError(f"no visible reference price for order in {symbol}")
            order = Order(
                symbol=symbol,
                side="buy" if delta > 0 else "sell",
                notional=abs(delta),
                reference_price=reference_price,
            )
            (buys if delta > 0 else sells).append(order)

        orders.extend(sorted(sells, key=lambda order: order.symbol))
        orders.extend(sorted(buys, key=lambda order: order.symbol))
        self.broker.submit(orders, placed_at=ctx.clock.now())
        return Orders(
            tick=ctx.tick,
            producer=self.name,
            orders=tuple(orders),
            submitted=len(orders),
            skipped_small=skipped_small,
        )
