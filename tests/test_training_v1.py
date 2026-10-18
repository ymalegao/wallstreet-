from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Sequence
from datetime import UTC, date, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

import exchange_calendars as xcals
import lightgbm as lgb
import polars as pl
import pytest
from fixtures.pipeline_v0 import build_fixture

from ws.broker import BrokerState
from ws.core import Pipeline
from ws.core.messages import message_digest
from ws.features.px_v1 import PX_V1_FEATURES, compute_px_v1
from ws.pipeline_v0.messages import (
    Features,
    Insights,
    MarketSnapshot,
    Order,
    PortfolioState,
    RiskAdjustedTargets,
)
from ws.pipeline_v0.nodes import Alpha, Execution, FeatureBuilder, PortfolioConstruction, RiskManagement
from ws.pipeline_v0.snapshot import SnapshotProvider
from ws.schedule import daily_ticks
from ws.store.history import BarHistory
from ws.store.pit_bars import PITBarStore
from ws.training.evaluation import _baseline_metrics
from ws.training.final_eval import final_eval
from ws.training.final_fit import fit_final
from ws.training.metrics import is_month_end, rank_ic_by_date
from ws.training.placebo import permute_fold_labels
from ws.training.registry import append_registry, count_trials, reserve_holdout
from ws.training.splits import WalkForwardFold, build_folds
from ws.training.table import LABEL_HORIZON, build_training_table
from ws.training.trainer import PLACEBO_SEED, _fit_one

_XNYS = xcals.get_calendar("XNYS")


@pytest.fixture(scope="module")
def training_data(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[PITBarStore, Path, Path, pl.DataFrame, dict[str, Any]]:
    root = tmp_path_factory.mktemp("training-v1")
    store, universe_path, symbols = build_fixture(root)
    dates = [session.date().isoformat() for session in _XNYS.sessions_in_range("2021-01-04", "2023-12-29")]
    manifest = {
        "snapshot_id": "fixture-iex-snapshot-001",
        "feed": "iex",
        "adjustment": "all",
        "symbols": [*symbols, "SPY"],
        "requested_start": "2000-01-01",
        "requested_end": "2023-12-29",
        "actual_earliest_by_symbol": {symbol: dates[0] for symbol in (*symbols, "SPY")},
        "actual_latest_by_symbol": {symbol: dates[-1] for symbol in (*symbols, "SPY")},
    }
    (store.dir / "snapshot.json").write_text(json.dumps(manifest), encoding="utf-8")
    table_path = root / "px-v1.parquet"
    metadata = build_training_table(store.dir, universe_path, table_path)
    return store, universe_path, table_path, pl.read_parquet(table_path), metadata


def test_t1_label_uses_next_open_and_exit_open(
    training_data: tuple[PITBarStore, Path, Path, pl.DataFrame, dict[str, Any]],
) -> None:
    store, _, table_path, table, metadata = training_data
    assert table.height > 0
    assert table.filter(pl.col("label_exit_date") >= date.fromisoformat(metadata["holdout_start"])).is_empty()
    chosen = table.filter(pl.col("label").is_not_null()).row(0, named=True)
    holdout_start = date.fromisoformat(metadata["holdout_start"])
    eligible_last_decision = _XNYS.session_offset(
        _XNYS.date_to_session(holdout_start.isoformat(), direction="none"),
        -(LABEL_HORIZON + 2),
    ).date()
    assert table["date"].max() == eligible_last_decision
    all_bars = store.read(_clock(date.fromisoformat(metadata["labeled_end"])), symbols=[chosen["symbol"]])
    prices = {
        row["event_time"].astimezone(UTC).date(): float(row["open"])
        for row in all_bars.iter_rows(named=True)
    }
    entry_session = _XNYS.next_session(_XNYS.date_to_session(chosen["date"].isoformat(), direction="none"))
    entry_date = entry_session.date()
    exit_session = _XNYS.session_offset(entry_session, LABEL_HORIZON)
    exit_date = exit_session.date()
    expected = prices[exit_date] / prices[entry_date] - 1.0
    assert chosen["raw_forward_return"] == pytest.approx(expected)
    assert chosen["label_exit_date"] == exit_date
    parquet_meta = pl.read_parquet_metadata(table_path)
    encoded = str(parquet_meta)
    assert metadata["snapshot_id"] in encoded
    assert metadata["feature_set"] in encoded
    assert metadata["market_data_feed"] == "iex"


def test_training_table_snapshot_output_cannot_be_overwritten(
    training_data: tuple[PITBarStore, Path, Path, pl.DataFrame, dict[str, Any]],
) -> None:
    store, universe_path, table_path, _, _ = training_data
    with pytest.raises(FileExistsError, match="immutable"):
        build_training_table(store.dir, universe_path, table_path)


def test_t2_px_v1_is_unchanged_by_future_rows(
    training_data: tuple[PITBarStore, Path, Path, pl.DataFrame, dict[str, Any]], tmp_path: Path
) -> None:
    store, _, _, _, _ = training_data
    tick = daily_ticks(date(2023, 12, 20), date(2023, 12, 20))[0]
    symbols = json.loads((store.dir.parent / "universe.json").read_text())["symbols"]
    history = BarHistory.load(store, (*symbols, "SPY"), through=tick.at)
    visible = history.visible(tick.at)
    baseline = compute_px_v1(visible.filter(pl.col("symbol") != "SPY"), visible.filter(pl.col("symbol") == "SPY"))
    later = BarHistory.load(
        store,
        (*symbols, "SPY"),
        through=datetime_with_offset(tick.at, timedelta(days=5)),
    )
    with_future = later.visible(tick.at)
    compared = compute_px_v1(
        with_future.filter(pl.col("symbol") != "SPY"),
        with_future.filter(pl.col("symbol") == "SPY"),
    )
    assert baseline.equals(compared)


def test_t3_training_table_and_feature_node_have_exact_parity(
    training_data: tuple[PITBarStore, Path, Path, pl.DataFrame, dict[str, Any]], tmp_path: Path
) -> None:
    store, universe_path, _, table, _ = training_data
    universe = tuple(json.loads(universe_path.read_text(encoding="utf-8"))["symbols"])
    available_dates = table.filter(pl.col("date") >= date(2022, 3, 1))["date"].unique().to_list()
    chosen_dates = sorted(random.Random(3).sample(available_dates, 5))
    for chosen_date in chosen_dates:
        tick = daily_ticks(chosen_date, chosen_date)[0]
        history = BarHistory.load(store, (*universe, "SPY"), through=tick.at)
        snapshot = SnapshotProvider(history, universe_path).snapshot(tick)
        pipeline = Pipeline.build(FeatureBuilder(feature_set="px-v1"), inputs=(MarketSnapshot,))
        outputs = asyncio.run(
            pipeline.run(
                tick,
                run_id=f"px-v1-parity-{chosen_date.isoformat()}",
                log_path=tmp_path / f"parity-{chosen_date.isoformat()}.jsonl",
                inputs={MarketSnapshot: snapshot},
            )
        )
        pipeline_frame = outputs[Features].frame
        table_frame = table.filter(pl.col("date") == chosen_date).sort("symbol")
        for symbol in universe:
            table_row = table_frame.filter(pl.col("symbol") == symbol).row(0, named=True)
            pipeline_row = pipeline_frame.filter(pl.col("symbol") == symbol).row(0, named=True)
            for feature in (*PX_V1_FEATURES, "adv_dollars_20d_raw"):
                left, right = table_row[feature], pipeline_row[feature]
                if left is None:
                    assert right is None
                else:
                    assert right == left


def test_t4_folds_purge_label_windows_and_embargo_test_boundary() -> None:
    sessions = [session.date() for session in _XNYS.sessions_in_range("2014-01-02", "2023-12-29")]
    rows = []
    for index, session_date in enumerate(sessions[:-LABEL_HORIZON - 1]):
        exit_date = sessions[index + LABEL_HORIZON + 1]
        for symbol in ("AAA", "BBB", "CCC"):
            rows.append(
                {
                    "date": session_date,
                    "symbol": symbol,
                    "label_exit_date": exit_date,
                    "label": 0.5,
                    "raw_forward_return": 0.01,
                }
            )
    folds = build_folds(pl.DataFrame(rows), horizon=LABEL_HORIZON)
    assert folds
    for fold in folds:
        test_start = date(int(fold.fold), 1, 1)
        assert fold.train.filter(pl.col("label_exit_date") >= test_start).is_empty()
        prior_sessions = [session.date() for session in _XNYS.sessions_in_range("2010-01-01", test_start.isoformat())]
        embargo = set(prior_sessions[-LABEL_HORIZON - 1 : -1])
        assert not set(fold.train["date"].to_list()).intersection(embargo)


def test_t5_holdout_reservation_is_single_use(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    append_registry(
        registry,
        [
            {"model_id": "fixture", "record_type": "final_model", "config_sha256": "abc", "seed": seed}
            for seed in range(5)
        ],
    )
    reserve_holdout(registry, "fixture", "abc")
    with pytest.raises(ValueError, match="already been reserved"):
        reserve_holdout(registry, "fixture", "abc")


def test_holdout_reservation_requires_registered_final_models(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    append_registry(registry, [{"model_id": "fixture", "record_type": "trial", "seed": 0}])
    with pytest.raises(ValueError, match="all five registered final seed models"):
        reserve_holdout(registry, "fixture", "abc")


def test_trial_count_includes_registered_fits_across_snapshots(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    append_registry(
        registry,
        [
            {"record_type": "trial", "model_id": "snapshot-a", "seed": 0},
            {"record_type": "placebo_trial", "model_id": "snapshot-a", "seed": 0},
            {"record_type": "trial", "model_id": "snapshot-b", "seed": 1},
            {"record_type": "final_model", "model_id": "snapshot-b", "seed": 0},
            {"record_type": "holdout_started", "model_id": "snapshot-b"},
        ],
    )
    assert count_trials(registry) == 4


def test_final_fit_reports_unspecified_lambda_tie(
    training_data: tuple[PITBarStore, Path, Path, pl.DataFrame, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, table_path, _, metadata = training_data
    model_id = f"pxv1-{metadata['snapshot_id'][:12]}"
    registry_path = tmp_path / "models" / "registry.jsonl"
    monkeypatch.setattr("ws.training.final_fit._ROOT", tmp_path)
    monkeypatch.setattr("ws.training.final_fit._REGISTRY", registry_path)
    append_registry(
        registry_path,
        [
            {
                "record_type": "trial",
                "model_id": model_id,
                "fold": fold,
                "params": {
                    "num_leaves": 15,
                    "min_data_in_leaf": 50,
                    "lambda_l2": lambda_l2,
                },
                "selected": True,
            }
            for fold, lambda_l2 in (("2021", 0), ("2022", 10))
        ],
    )
    with pytest.raises(ValueError, match="remains tied on lambda_l2"):
        fit_final(table_path, iteration_policy="median-fold", num_threads=1)


def test_final_eval_reads_holdout_once_and_renders_report(
    training_data: tuple[PITBarStore, Path, Path, pl.DataFrame, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _store, _universe_path, _table_path, table, metadata = training_data
    model_id = f"pxv1-{metadata['snapshot_id'][:12]}"
    holdout_start = date.fromisoformat(metadata["holdout_start"])
    holdout_row = table.tail(1).with_columns(
        pl.lit(holdout_start).cast(pl.Date).alias("date"),
        pl.lit(holdout_start + timedelta(days=21)).cast(pl.Date).alias("label_exit_date"),
        pl.lit(1.0).alias("label"),
        pl.lit(0.5).alias("raw_forward_return"),
    )
    contaminated_table_path = tmp_path / "table-with-holdout-row.parquet"
    pl.concat([table, holdout_row], how="vertical").write_parquet(contaminated_table_path)
    contaminated_table_path.with_suffix(".parquet.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    model_root = tmp_path / "models"
    registry_path = model_root / "registry.jsonl"
    monkeypatch.setattr("ws.training.final_fit._ROOT", tmp_path)
    monkeypatch.setattr("ws.training.final_fit._REGISTRY", registry_path)
    monkeypatch.setattr("ws.training.final_eval._ROOT", tmp_path)
    monkeypatch.setattr("ws.training.final_eval._REGISTRY", registry_path)
    append_registry(
        registry_path,
        [
            {
                "record_type": "trial",
                "model_id": model_id,
                "fold": "2022",
                "params": {"num_leaves": 15, "min_data_in_leaf": 50, "lambda_l2": 0},
                "selected": True,
                "best_iteration": 5,
            }
        ],
    )
    fit_final(contaminated_table_path, iteration_policy="median-fold", num_threads=1)
    final_models = [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("record_type") == "final_model"
    ]
    assert final_models
    assert all(record["train_range"][1] < holdout_start.isoformat() for record in final_models)

    portfolio_metrics = {
        "total_return": 0.0,
        "annualized_return": 0.0,
        "annualized_volatility": 0.0,
        "sharpe": None,
        "max_drawdown": 0.0,
        "turnover": 0.0,
        "total_costs": 0.0,
    }
    portfolio_scenario = {
        "lightgbm": portfolio_metrics,
        "momentum_12_1": portfolio_metrics,
        "equal_weight_buy_hold": portfolio_metrics,
        "spy_buy_hold": portfolio_metrics,
        "monthly": [],
        "concentration": {"top_five_stocks": [], "top_five_months": []},
        "artifacts": {},
    }
    monkeypatch.setattr(
        "ws.training.final_eval.run_portfolio_scenarios",
        lambda *args, **kwargs: {"1x": portfolio_scenario, "2x": portfolio_scenario},
    )

    manifest_path = model_root / model_id / "manifest.json"
    original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed_manifest = {**original_manifest, "holdout_start": (holdout_start + timedelta(days=1)).isoformat()}
    manifest_path.write_text(json.dumps(changed_manifest), encoding="utf-8")

    def fail_on_unverified_model_read(*args, **kwargs):
        raise AssertionError("a changed holdout boundary must be rejected before reading bars")

    with monkeypatch.context() as context:
        context.setattr("ws.training.final_eval.BarHistory.load", fail_on_unverified_model_read)
        with pytest.raises(ValueError, match="model config or model files changed"):
            final_eval(model_id)
    manifest_path.write_text(json.dumps(original_manifest), encoding="utf-8")

    report = final_eval(model_id)
    assert report["rows"] > 0
    assert report["holdout_range"][0] == metadata["holdout_start"]
    report_path = tmp_path / "runs" / "training" / model_id / "report.md"
    assert report_path.is_file()
    rendered_report = report_path.read_text(encoding="utf-8")
    assert "Locked holdout" in rendered_report
    assert "Positive IC fraction" in rendered_report
    assert "Leakage checks" in rendered_report

    def fail_on_second_read(*args, **kwargs):
        raise AssertionError("a repeated holdout run must stop before loading bars")

    monkeypatch.setattr("ws.training.final_eval.BarHistory.load", fail_on_second_read)
    with pytest.raises(ValueError, match="already been reserved or run"):
        final_eval(model_id)


def test_monthly_rebalance_holds_between_dates_and_executes_risk_sales(tmp_path: Path) -> None:
    class CaptureBroker:
        def __init__(self) -> None:
            self.batches: list[tuple[Sequence[Order], Any]] = []

        def state(self, as_of):
            return BrokerState(cash=1000.0, positions={}, market_values={}, equity=1000.0, peak_equity=1000.0)

        def submit(self, orders, placed_at):
            self.batches.append((tuple(orders), placed_at))

    broker = CaptureBroker()
    pipeline = Pipeline.build(
        Execution(broker, rebalance_rule="monthly"),
        inputs=(RiskAdjustedTargets, PortfolioState, MarketSnapshot),
    )
    symbols = ("AAA", "BBB")

    def run_day(
        session_date: date,
        *,
        risk_reductions: dict[str, float],
        rebalance: dict[str, float],
        has_position: bool,
    ):
        tick = daily_ticks(session_date, session_date)[0]
        bars = pl.DataFrame(
            {
                "symbol": ["AAA", "BBB"],
                "event_time": [tick.at, tick.at],
                "close": [100.0, 100.0],
            }
        )
        snapshot = MarketSnapshot(
            tick=tick,
            producer="test",
            universe=symbols,
            universe_id="test",
            bars=bars,
            last_session=session_date,
        )
        positions = {"BBB": 0.5} if has_position else {}
        market_values = {"BBB": 50.0} if has_position else {}
        state = PortfolioState(
            tick=tick,
            producer="test",
            cash=950.0 if has_position else 1000.0,
            positions=positions,
            market_values=market_values,
            equity=1000.0,
            peak_equity=1000.0,
        )
        targets = RiskAdjustedTargets(
            tick=tick,
            producer="test",
            weights=rebalance,
            gross_exposure=sum(rebalance.values()),
            constraints=(),
            risk_reductions=risk_reductions,
        )
        return asyncio.run(
            pipeline.run(
                tick,
                run_id=f"monthly-{session_date.isoformat()}",
                log_path=tmp_path / f"{session_date.isoformat()}.jsonl",
                inputs={RiskAdjustedTargets: targets, PortfolioState: state, MarketSnapshot: snapshot},
            )
        )

    run_day(date(2023, 11, 15), risk_reductions={}, rebalance={"BBB": 0.05}, has_position=False)
    assert [order.side for order in broker.batches[-1][0]] == ["buy"]
    run_day(date(2023, 11, 16), risk_reductions={}, rebalance={"AAA": 0.05}, has_position=True)
    assert broker.batches[-1][0] == ()
    run_day(date(2023, 11, 17), risk_reductions={"BBB": 0.02}, rebalance={"AAA": 0.05}, has_position=True)
    assert [(order.symbol, order.side) for order in broker.batches[-1][0]] == [("BBB", "sell")]
    run_day(date(2023, 11, 30), risk_reductions={}, rebalance={"AAA": 0.05}, has_position=True)
    assert {(order.symbol, order.side) for order in broker.batches[-1][0]} == {
        ("AAA", "buy"),
        ("BBB", "sell"),
    }


def test_baseline_turnover_is_traded_notional_over_mean_equity(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baselines.csv"
    baseline_path.write_text(
        "date,strategy,equity,net_return,turnover,transaction_costs\n"
        "2023-11-15,equal_weight_buy_hold,100,0,0,0\n"
        "2023-11-16,equal_weight_buy_hold,110,0.1,0.9,1\n",
        encoding="utf-8",
    )
    metrics = _baseline_metrics(baseline_path, "equal_weight_buy_hold")
    assert metrics["turnover"] == pytest.approx(90 / 105)


def test_t6_planted_feature_has_positive_test_rank_ic() -> None:
    rng = random.Random(2501)
    sessions = [session.date() for session in _XNYS.sessions_in_range("2020-01-02", "2021-12-31")]
    feature_rows = []
    split_train, split_valid = 260, 340
    for session_date in sessions:
        for index in range(80):
            signal = rng.uniform(-1, 1)
            outcome = signal + rng.gauss(0, 0.2)
            features = {name: rng.uniform(-1, 1) for name in PX_V1_FEATURES}
            features[PX_V1_FEATURES[0]] = signal
            feature_rows.append(
                {
                    "date": session_date,
                    "symbol": f"S{index:03d}",
                    **features,
                    "label": (outcome + 1) / 2,
                    "raw_forward_return": outcome,
                    "label_exit_date": session_date + timedelta(days=1),
                }
            )
    frame = pl.DataFrame(feature_rows)
    train = frame.filter(pl.col("date") < sessions[split_train])
    valid = frame.filter((pl.col("date") >= sessions[split_train]) & (pl.col("date") < sessions[split_valid]))
    test = frame.filter(pl.col("date") >= sessions[split_valid])
    fold = WalkForwardFold(
        fold="planted",
        train=train,
        valid=valid,
        test=test,
        train_range=(min(train["date"].to_list()), max(train["date"].to_list())),
        valid_range=(min(valid["date"].to_list()), max(valid["date"].to_list())),
        test_range=(min(test["date"].to_list()), max(test["date"].to_list())),
    )
    model, best_iteration = _fit_one(
        fold,
        {"num_leaves": 15, "min_data_in_leaf": 50, "lambda_l2": 0},
        0,
        num_threads=1,
    )
    predictions = model.predict(test.select(PX_V1_FEATURES).to_numpy(), num_iteration=best_iteration)
    summary = rank_ic_by_date(test, predictions)
    assert summary
    assert mean(float(item["rank_ic"]) for item in summary) > 0.1


def test_t7_training_label_placebo_has_no_test_signal() -> None:
    signal_rng = random.Random(701)
    feature_rng = random.Random(98765)
    all_dates = [
        session.date()
        for session in _XNYS.sessions_in_range("2014-01-01", "2020-12-31")
        if is_month_end(session.date())
    ]
    train_dates = [session_date for session_date in all_dates if session_date.year <= 2018]
    valid_dates = [session_date for session_date in all_dates if session_date.year == 2019]
    test_dates = [session_date for session_date in all_dates if session_date.year == 2020]
    rows = []
    for session_date in (*train_dates, *valid_dates, *test_dates):
        values = []
        for symbol_index in range(1000):
            signal = signal_rng.uniform(-1, 1)
            outcome = signal + signal_rng.gauss(0, 0.15)
            values.append((symbol_index, signal, outcome))
        ordered = sorted(values, key=lambda item: item[2])
        ranked = {symbol_index: rank / (len(ordered) - 1) for rank, (symbol_index, _, _) in enumerate(ordered)}
        for symbol_index, signal, outcome in values:
            features = {name: feature_rng.uniform(-1, 1) for name in PX_V1_FEATURES}
            features[PX_V1_FEATURES[0]] = signal
            rows.append(
                {
                    "date": session_date,
                    "symbol": f"S{symbol_index:04d}",
                    **features,
                    "label": ranked[symbol_index],
                    "raw_forward_return": outcome,
                    "label_exit_date": session_date + timedelta(days=21),
                }
            )
    frame = pl.DataFrame(rows)
    train = frame.filter(pl.col("date").is_in(train_dates))
    valid = frame.filter(pl.col("date").is_in(valid_dates))
    test = frame.filter(pl.col("date").is_in(test_dates))
    fold = WalkForwardFold(
        fold="placebo",
        train=train,
        valid=valid,
        test=test,
        train_range=(min(train["date"].to_list()), max(train["date"].to_list())),
        valid_range=(min(valid["date"].to_list()), max(valid["date"].to_list())),
        test_range=(min(test["date"].to_list()), max(test["date"].to_list())),
    )
    placebo_fold = permute_fold_labels(fold, seed=PLACEBO_SEED, scope="train-only")
    seed_predictions = []
    for seed in range(5):
        model, best_iteration = _fit_one(
            placebo_fold,
            {"num_leaves": 15, "min_data_in_leaf": 200, "lambda_l2": 10},
            seed,
            num_threads=1,
        )
        seed_predictions.append(
            model.predict(placebo_fold.test.select(PX_V1_FEATURES).to_numpy(), num_iteration=best_iteration)
        )
    predictions = [
        sum(seed_values[index] for seed_values in seed_predictions) / len(seed_predictions)
        for index in range(test.height)
    ]
    placebo_ic = rank_ic_by_date(test, predictions)
    assert placebo_ic
    placebo_mean = mean(float(item["rank_ic"]) for item in placebo_ic)
    monthly_ic = [(item["date"], round(float(item["rank_ic"]), 3)) for item in placebo_ic]
    assert abs(placebo_mean) <= 0.02, f"mean={placebo_mean:.4f}; monthly IC={monthly_ic}"


def test_t8_same_lightgbm_seed_writes_identical_model() -> None:
    rows = []
    rng = random.Random(801)
    dates = [session.date() for session in _XNYS.sessions_in_range("2021-01-04", "2021-08-31")]
    for session_date in dates:
        for index in range(80):
            signal = rng.uniform(-1, 1)
            label = signal + rng.gauss(0, 0.2)
            rows.append(
                {
                    "date": session_date,
                    "symbol": f"S{index:03d}",
                    **{name: signal if name == PX_V1_FEATURES[0] else rng.uniform(-1, 1) for name in PX_V1_FEATURES},
                    "label": (label + 1) / 2,
                    "raw_forward_return": label,
                    "label_exit_date": session_date + timedelta(days=21),
                }
            )
    frame = pl.DataFrame(rows)
    cutoff, valid_cutoff = dates[100], dates[130]
    train = frame.filter(pl.col("date") < cutoff)
    valid = frame.filter((pl.col("date") >= cutoff) & (pl.col("date") < valid_cutoff))
    test = frame.filter(pl.col("date") >= valid_cutoff)
    fold = WalkForwardFold(
        fold="determinism",
        train=train,
        valid=valid,
        test=test,
        train_range=(min(train["date"].to_list()), max(train["date"].to_list())),
        valid_range=(min(valid["date"].to_list()), max(valid["date"].to_list())),
        test_range=(min(test["date"].to_list()), max(test["date"].to_list())),
    )
    config = {"num_leaves": 15, "min_data_in_leaf": 50, "lambda_l2": 0}
    first, first_iteration = _fit_one(fold, config, 3, num_threads=1)
    second, second_iteration = _fit_one(fold, config, 3, num_threads=1)
    assert first_iteration == second_iteration
    assert first.model_to_string(num_iteration=first_iteration) == second.model_to_string(
        num_iteration=second_iteration
    )


def test_t9_lightgbm_pipeline_replays_with_identical_digests(
    training_data: tuple[PITBarStore, Path, Path, pl.DataFrame, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, universe_path, _, _, _ = training_data
    model_root = tmp_path / "models"
    model_dir = model_root / "fixture"
    model_dir.mkdir(parents=True)
    rng = random.Random(9)
    feature_frame = pl.DataFrame(
        [{name: rng.uniform(-1, 1) for name in PX_V1_FEATURES} for _ in range(200)]
    )
    labels = [rng.random() for _ in range(feature_frame.height)]
    dataset = lgb.Dataset(feature_frame.to_numpy(), label=labels, feature_name=list(PX_V1_FEATURES))
    booster = lgb.train(
        {
            "objective": "regression",
            "metric": "l2",
            "num_leaves": 15,
            "min_data_in_leaf": 5,
            "verbosity": -1,
            "num_threads": 1,
        },
        dataset,
        num_boost_round=10,
    )
    for seed in range(5):
        booster.save_model(str(model_dir / f"seed_{seed}.txt"))
    (model_dir / "manifest.json").write_text(
        json.dumps({"feature_set": "px-v1", "feature_names": list(PX_V1_FEATURES), "seeds": list(range(5))}),
        encoding="utf-8",
    )
    monkeypatch.setattr("ws.pipeline_v0.nodes._MODEL_ROOT", model_root)

    class CaptureBroker:
        def __init__(self) -> None:
            self.orders: list[tuple[Sequence[Order], Any]] = []

        def state(self, as_of):
            return BrokerState(cash=1000.0, positions={}, market_values={}, equity=1000.0, peak_equity=1000.0)

        def submit(self, orders, placed_at):
            self.orders.append((tuple(orders), placed_at))

    tick = daily_ticks(date(2023, 12, 29), date(2023, 12, 29))[0]
    history = BarHistory.load(store, (*json.loads(universe_path.read_text())["symbols"], "SPY"), through=tick.at)
    provider = SnapshotProvider(history, universe_path)
    state = PortfolioState(
        tick=tick,
        producer="test-coordinator",
        cash=1000.0,
        positions={},
        market_values={},
        equity=1000.0,
        peak_equity=1000.0,
    )
    inputs = {MarketSnapshot: provider.snapshot(tick), PortfolioState: state}

    def build(broker: CaptureBroker) -> Pipeline:
        return Pipeline.build(
            FeatureBuilder(feature_set="px-v1"),
            Alpha(model_id="lgbm@fixture"),
            PortfolioConstruction(holdings=20),
            RiskManagement(),
            Execution(broker),
            inputs=(MarketSnapshot, PortfolioState),
        )

    first_broker = CaptureBroker()
    first_pipeline = build(first_broker)
    recorded_log = tmp_path / "recorded.jsonl"
    first_outputs = asyncio.run(
        first_pipeline.run(tick, run_id="model-replay", log_path=recorded_log, inputs=inputs)
    )
    assert Insights in first_outputs
    assert first_broker.orders

    second_pipeline = build(CaptureBroker())
    replayed = asyncio.run(
        second_pipeline.replay(
            [tick],
            recorded_log=recorded_log,
            recorded_run_id="model-replay",
            replay_log=tmp_path / "replayed.jsonl",
            inputs_for_tick=lambda _: inputs,
        )
    )
    assert len(replayed) == 1
    assert {key: message_digest(value) for key, value in first_outputs.items()} == {
        key: message_digest(value) for key, value in replayed[0].items()
    }


def _clock(value: date):
    from ws.core.clock import Tick

    return _VisibleClock(Tick(id=value.isoformat(), at=daily_ticks(value, value)[0].at))


class _VisibleClock:
    def __init__(self, tick: Any) -> None:
        self._tick = tick

    def now(self):
        return self._tick.at

    def tick(self):
        return self._tick


def datetime_with_offset(value, offset):
    return value + offset
