from __future__ import annotations

import json
import shutil
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from ws.fundamentals.query import annual_value, fundamentals_are_stale
from ws.fundamentals.store import _filing_available_at, build_facts_table, visible_facts

_FIXTURES = Path(__file__).parent / "fixtures" / "fundamentals"


def test_filing_date_is_available_after_next_session_close_plus_fifteen_minutes() -> None:
    available_at = _filing_available_at(datetime(2024, 1, 2).date())

    assert available_at == datetime(2024, 1, 3, 21, 15, tzinfo=UTC)
    facts = pl.DataFrame({"available_at": [available_at], "value": [1.0]})
    assert visible_facts(facts, datetime(2024, 1, 2, 23, 0, tzinfo=UTC)).is_empty()
    assert visible_facts(facts, datetime(2024, 1, 3, 21, 16, tzinfo=UTC)).height == 1


def test_build_facts_table_extracts_requested_ciks_and_writes_snapshot_hash(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    shutil.copyfile(_FIXTURES / "company_tickers.json", raw_dir / "company_tickers.json")
    for archive_name, fixture_name in (
        ("companyfacts.zip", "companyfacts.CIK0000000001.json"),
        ("submissions.zip", "submissions.CIK0000000001.json"),
    ):
        with zipfile.ZipFile(raw_dir / archive_name, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(_FIXTURES / fixture_name, arcname="CIK0000000001.json")

    result = build_facts_table(raw_dir, ("TST", "MISSING"), tmp_path / "facts")
    facts_path = Path(result["parquet_path"])
    frame = pl.read_parquet(facts_path)

    assert result["missing_tickers"] == ["MISSING"]
    assert set(frame["cik"].to_list()) == {"0000000001"}
    assert "dei:EntityCommonStockSharesOutstanding" in frame["tag"].to_list()
    sidecar = json.loads(facts_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["snapshot_id"] == facts_path.stem
    assert sidecar["input_sha256"]["companyfacts.zip"]
    assert sidecar["sic_by_cik"] == {"0000000001": 3571}


def test_annual_fact_selection_respects_visibility_restatements_tag_order_and_staleness() -> None:
    rows = [
        {
            "cik": "0000000001",
            "tag": tag,
            "unit": "USD",
            "value": value,
            "period_start": start,
            "period_end": end,
            "form": form,
            "fiscal_year": 2023,
            "fiscal_period": "FY",
            "accession": accession,
            "filed": filed,
            "available_at": available_at,
        }
        for tag, value, start, end, form, accession, filed, available_at in (
            (
                "NetIncomeLoss",
                100.0,
                date(2023, 1, 1),
                date(2023, 12, 31),
                "10-K",
                "first",
                date(2024, 2, 1),
                datetime(2024, 2, 2, 21, 15, tzinfo=UTC),
            ),
            (
                "NetIncomeLoss",
                125.0,
                date(2023, 1, 1),
                date(2023, 12, 31),
                "10-K/A",
                "restated",
                date(2024, 3, 1),
                datetime(2024, 3, 4, 21, 15, tzinfo=UTC),
            ),
            (
                "ProfitLoss",
                200.0,
                date(2023, 1, 1),
                date(2023, 12, 31),
                "10-K",
                "alternate",
                date(2024, 2, 1),
                datetime(2024, 2, 2, 21, 15, tzinfo=UTC),
            ),
            (
                "NetIncomeLoss",
                50.0,
                date(2023, 1, 1),
                date(2023, 9, 30),
                "10-Q",
                "quarterly",
                date(2023, 11, 1),
                datetime(2023, 11, 2, 21, 15, tzinfo=UTC),
            ),
            (
                "Assets",
                900.0,
                None,
                date(2022, 12, 31),
                "10-K",
                "old-year",
                date(2023, 2, 1),
                datetime(2023, 2, 2, 21, 15, tzinfo=UTC),
            ),
        )
    ]
    facts = pl.DataFrame(rows)

    visible_before_restatement = visible_facts(facts, datetime(2024, 2, 15, tzinfo=UTC))
    assert annual_value(
        visible_before_restatement,
        cik="0000000001",
        tags=("NetIncomeLoss", "ProfitLoss"),
        kind="flow",
    ) == 100.0
    visible_after_restatement = visible_facts(facts, datetime(2024, 3, 5, tzinfo=UTC))
    assert annual_value(
        visible_after_restatement,
        cik="0000000001",
        tags=("NetIncomeLoss", "ProfitLoss"),
        kind="flow",
    ) == 125.0
    assert fundamentals_are_stale(visible_after_restatement, "0000000001", date(2025, 7, 1))
