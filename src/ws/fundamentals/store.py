"""Build immutable, point-in-time SEC company-fact snapshots for one universe."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import polars as pl

from ws.fundamentals.query import FundamentalsIndex
from ws.store.atomic import atomic_path, atomic_text
from ws.fundamentals.industry import parse_ff12_sic_zip

FACTS_SCHEMA: dict[str, pl.DataType] = {
    "cik": pl.String,
    "tag": pl.String,
    "unit": pl.String,
    "value": pl.Float64,
    "period_start": pl.Date,
    "period_end": pl.Date,
    "form": pl.String,
    "fiscal_year": pl.Int64,
    "fiscal_period": pl.String,
    "accession": pl.String,
    "filed": pl.Date,
    "available_at": pl.Datetime("us", "UTC"),
}
_FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A"}
_UNITS = {"USD", "shares"}
_XNYS = xcals.get_calendar("XNYS")


class FundamentalsStore:
    """Load one immutable facts snapshot and expose facts visible at a decision time."""

    def __init__(self, path: Path) -> None:
        if not path.is_file() or path.suffix != ".parquet":
            raise FileNotFoundError(f"SEC facts snapshot parquet does not exist: {path}")
        metadata_path = path.with_suffix(".json")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"SEC facts snapshot metadata does not exist: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("snapshot_id") != path.stem:
            raise ValueError("SEC facts snapshot metadata id does not match its filename")
        self.path = path
        self.snapshot_id = path.stem
        self.raw_dir = Path(metadata["raw_dir"])
        self.industry_source_path = (
            Path(metadata["industry_source_path"]) if metadata.get("industry_source_path") else None
        )
        self.ticker_to_cik = {
            str(symbol).strip().upper(): str(cik).zfill(10)
            for symbol, cik in metadata.get("ticker_to_cik", {}).items()
        }
        self.sic_by_cik = {
            str(cik).zfill(10): int(sic)
            for cik, sic in metadata.get("sic_by_cik", {}).items()
        }
        self.industry_by_cik = {
            str(cik).zfill(10): str(industry)
            for cik, industry in metadata.get("industry_by_cik", {}).items()
        }
        self.missing_tickers = tuple(str(value) for value in metadata.get("missing_tickers", ()))
        self._facts = pl.read_parquet(path)
        if set(self._facts.columns) != set(FACTS_SCHEMA):
            raise ValueError("SEC facts snapshot has an incompatible schema")
        self.index = FundamentalsIndex(self._facts)

    @property
    def facts(self) -> pl.DataFrame:
        """Return the immutable source facts for coverage reports and diagnostics."""
        return self._facts

    def visible_at(self, as_of: datetime) -> pl.DataFrame:
        """Return the point-in-time view of this snapshot."""
        return visible_facts(self._facts, as_of)


def build_facts_table(
    raw_dir: Path,
    universe_symbols: tuple[str, ...],
    output_dir: Path,
    *,
    industry_zip: Path | None = None,
) -> dict[str, Any]:
    """Extract only requested CIKs from SEC bulk archives and write one immutable parquet.

    The directory layout of raw inputs is deliberately supplied by the caller. The
    derived snapshot is content-addressed from those files and the ticker universe.
    """
    required = ("company_tickers.json", "companyfacts.zip", "submissions.zip")
    raw_files = {name: raw_dir / name for name in required}
    missing_files = [str(path) for path in raw_files.values() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"SEC raw snapshot is missing required files: {missing_files}")
    if not universe_symbols:
        raise ValueError("universe_symbols must contain at least one symbol")

    normalized_symbols = tuple(sorted({symbol.strip().upper() for symbol in universe_symbols if symbol.strip()}))
    if not normalized_symbols:
        raise ValueError("universe_symbols must contain at least one non-empty symbol")
    ticker_to_cik = _ticker_to_cik(raw_files["company_tickers.json"], normalized_symbols)
    missing_tickers = sorted(set(normalized_symbols) - set(ticker_to_cik))
    selected_ciks = set(ticker_to_cik.values())

    found_facts_ciks: set[str] = set()
    rows: list[dict[str, Any]] = []
    for cik, payload in _read_selected_zip_json(raw_files["companyfacts.zip"], selected_ciks):
        found_facts_ciks.add(cik)
        rows.extend(_company_fact_rows(cik, payload))
    sic_by_cik: dict[str, int] = {}
    found_submissions_ciks: set[str] = set()
    for cik, payload in _read_selected_zip_json(raw_files["submissions.zip"], selected_ciks):
        found_submissions_ciks.add(cik)
        sic = _as_int(payload.get("sic"))
        if sic is not None:
            sic_by_cik[cik] = sic

    sic_to_industry = parse_ff12_sic_zip(industry_zip) if industry_zip is not None else {}

    rows.sort(
        key=lambda row: (
            row["cik"],
            row["tag"],
            row["period_end"],
            row["filed"],
            row["accession"] or "",
            row["unit"],
        )
    )
    frame = pl.DataFrame(rows, schema=FACTS_SCHEMA) if rows else pl.DataFrame(schema=FACTS_SCHEMA)
    input_hashes = {name: _sha256(path) for name, path in sorted(raw_files.items())}
    if industry_zip is not None:
        input_hashes[industry_zip.name] = _sha256(industry_zip)
    snapshot_payload = {
        "input_sha256": input_hashes,
        "universe_symbols": normalized_symbols,
        "ticker_to_cik": ticker_to_cik,
    }
    snapshot_id = hashlib.sha256(
        json.dumps(snapshot_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{snapshot_id}.parquet"
    metadata_path = output_dir / f"{snapshot_id}.json"
    if parquet_path.exists() or metadata_path.exists():
        raise FileExistsError(f"SEC fact snapshots are immutable: {parquet_path}")
    with atomic_path(parquet_path) as pending:
        frame.write_parquet(pending)
    metadata = {
        "snapshot_id": snapshot_id,
        "row_count": frame.height,
        "raw_dir": str(raw_dir.resolve()),
        "input_sha256": input_hashes,
        "universe_symbols": list(normalized_symbols),
        "ticker_to_cik": ticker_to_cik,
        "missing_tickers": missing_tickers,
        "missing_companyfacts_ciks": sorted(selected_ciks - found_facts_ciks),
        "missing_submissions_ciks": sorted(selected_ciks - found_submissions_ciks),
        "sic_by_cik": dict(sorted(sic_by_cik.items())),
        "industry_by_cik": {
            cik: sic_to_industry.get(sic_by_cik.get(cik), "Other")
            for cik in sorted(selected_ciks)
        },
        "industry_source_sha256": _sha256(industry_zip) if industry_zip is not None else None,
        "industry_source_path": str(industry_zip.resolve()) if industry_zip is not None else None,
    }
    atomic_text(metadata_path, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return {**metadata, "parquet_path": str(parquet_path)}


def visible_facts(facts: pl.DataFrame, as_of: datetime) -> pl.DataFrame:
    """Return facts available by a timezone-aware point-in-time decision timestamp."""
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    if "available_at" not in facts.columns:
        raise ValueError("facts frame is missing available_at")
    return facts.filter(pl.col("available_at") <= as_of.astimezone(UTC))


def _company_fact_rows(cik: str, payload: dict[str, Any]):
    """Convert one selected SEC JSON member before releasing it from memory."""
    facts = payload.get("facts", {})
    for namespace, tag_data in facts.items():
        if not isinstance(tag_data, dict):
            continue
        for raw_tag, definition in tag_data.items():
            tag = f"{namespace}:{raw_tag}" if namespace != "us-gaap" else raw_tag
            units = definition.get("units", {}) if isinstance(definition, dict) else {}
            if not isinstance(units, dict):
                continue
            for unit, entries in units.items():
                if unit not in _UNITS or not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict) or entry.get("form") not in _FORMS:
                        continue
                    filed = _parse_date(entry.get("filed"))
                    period_end = _parse_date(entry.get("end"))
                    if filed is None or period_end is None:
                        continue
                    value = entry.get("val")
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        continue
                    yield {
                        "cik": cik,
                        "tag": tag,
                        "unit": unit,
                        "value": float(value),
                        "period_start": _parse_date(entry.get("start")),
                        "period_end": period_end,
                        "form": str(entry["form"]),
                        "fiscal_year": _as_int(entry.get("fy")),
                        "fiscal_period": str(entry["fp"]) if entry.get("fp") is not None else None,
                        "accession": str(entry["accn"]) if entry.get("accn") is not None else None,
                        "filed": filed,
                        "available_at": _filing_available_at(filed),
                    }


def _ticker_to_cik(path: Path, symbols: tuple[str, ...]) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        records = payload.values()
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError("company_tickers.json must contain an object or list")
    wanted = set(symbols)
    result: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        ticker = str(record.get("ticker", "")).strip().upper()
        cik_value = record.get("cik_str", record.get("cik"))
        if ticker not in wanted or cik_value is None:
            continue
        cik = str(cik_value).strip().zfill(10)
        if ticker in result and result[ticker] != cik:
            raise ValueError(f"company_tickers.json maps {ticker} to multiple CIKs")
        result[ticker] = cik
    return dict(sorted(result.items()))


def _read_selected_zip_json(path: Path, ciks: set[str]):
    """Yield only selected top-level company JSON members, one file at a time."""
    with zipfile.ZipFile(path) as archive:
        by_cik: dict[str, zipfile.ZipInfo] = {}
        for info in archive.infolist():
            name = Path(info.filename).name
            if not name.startswith("CIK") or not name.endswith(".json"):
                continue
            cik = name[3:-5]
            if cik in ciks and "/old/" not in info.filename:
                by_cik[cik] = info
        for cik, info in sorted(by_cik.items()):
            with archive.open(info) as stream:
                payload = json.load(stream)
            if isinstance(payload, dict):
                yield cik, payload


@lru_cache(maxsize=None)
def _filing_available_at(filed: date) -> datetime:
    """Use the close plus 15 minutes of the first XNYS session strictly after filed."""
    start = filed + timedelta(days=1)
    sessions = _XNYS.sessions_in_range(start.isoformat(), (start + timedelta(days=14)).isoformat())
    if len(sessions) == 0:
        raise ValueError(f"no XNYS session found after filing date {filed}")
    close = _XNYS.session_close(sessions[0]).to_pydatetime().astimezone(UTC)
    return close + timedelta(minutes=15)


def _parse_date(value: Any) -> date | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
