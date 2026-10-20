"""Build one universe-scoped SEC fundamentals snapshot and coverage report."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from ws.fundamentals.query import (
    ASSETS_TAGS,
    BOOK_EQUITY_TAGS,
    COST_OF_REVENUE_TAGS,
    FLOW_TAG_GROUPS,
    GROSS_PROFIT_TAGS,
    OPERATING_CASH_FLOW_TAGS,
    REVENUE_TAGS,
    SHARES_TAGS,
)
from ws.fundamentals.store import FundamentalsStore, build_facts_table

_ROOT = Path(__file__).parents[3]
_UNIVERSE = _ROOT / "configs" / "universe" / "sp100-current.json"
_COVERAGE = _ROOT / "docs" / "fund-v1-coverage.md"
_ITEM_TAGS = {
    "Book equity": BOOK_EQUITY_TAGS,
    "Net income": FLOW_TAG_GROUPS["net_income"],
    "Operating cash flow": OPERATING_CASH_FLOW_TAGS,
    "Assets": ASSETS_TAGS,
    "Revenue": REVENUE_TAGS,
    "Cost of revenue": COST_OF_REVENUE_TAGS,
    "Gross profit": GROSS_PROFIT_TAGS,
    "Shares outstanding": SHARES_TAGS,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ws.fundamentals.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="extract universe CIKs from SEC bulk archives")
    build.add_argument("--raw", type=Path, required=True)
    build.add_argument("--universe", type=Path, default=_UNIVERSE)
    build.add_argument("--out", type=Path, default=_ROOT / "data" / "sec" / "facts")
    build.add_argument("--industry-zip", type=Path)
    build.add_argument("--coverage", type=Path, default=_COVERAGE)
    args = parser.parse_args(argv)

    universe_data = json.loads(args.universe.read_text(encoding="utf-8"))
    universe = tuple(str(symbol).strip().upper() for symbol in universe_data["symbols"])
    industry_zip = args.industry_zip or (
        args.raw.parent.parent.parent
        / "ken-french"
        / "raw"
        / args.raw.name
        / "Siccodes12.zip"
    )
    if not industry_zip.is_file():
        raise FileNotFoundError(f"Ken French FF12 SIC archive is required: {industry_zip}")
    result = build_facts_table(args.raw, universe, args.out, industry_zip=industry_zip)
    store = FundamentalsStore(Path(str(result["parquet_path"])))
    write_coverage_report(store, args.universe, args.coverage)
    print(
        json.dumps(
            {
                "snapshot_id": result["snapshot_id"],
                "row_count": result["row_count"],
                "missing_tickers": result["missing_tickers"],
                "missing_companyfacts_ciks": result["missing_companyfacts_ciks"],
                "parquet_path": result["parquet_path"],
                "coverage_report": str(args.coverage),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def write_coverage_report(store: FundamentalsStore, universe_path: Path, output_path: Path) -> None:
    universe_data = json.loads(universe_path.read_text(encoding="utf-8"))
    symbols = tuple(sorted({str(value).strip().upper() for value in universe_data["symbols"]}))
    tags_by_cik: dict[str, set[str]] = defaultdict(set)
    for cik, tag in store.facts.select("cik", "tag").iter_rows():
        tags_by_cik[str(cik)].add(str(tag))
    tickers_by_cik: dict[str, list[str]] = defaultdict(list)
    for ticker, cik in store.ticker_to_cik.items():
        tickers_by_cik[cik].append(ticker)

    lines = [
        "# Fundamentals v1 coverage",
        "",
        f"- Fundamentals snapshot: `{store.snapshot_id}`",
        f"- Universe: `{universe_data.get('index', 'unknown')}` ({len(symbols)} current constituents; survivorship-biased historically).",
        f"- SEC facts rows: {store.facts.height:,}",
        f"- Tickers missing from `company_tickers.json`: {', '.join(store.missing_tickers) or 'none'}",
        "- Dual share-class CIK mappings share issuer fundamentals; market equity uses each ticker's close with issuer-level shares, as specified.",
        "",
        "## Per-ticker coverage",
        "",
        "| Ticker | CIK | SIC | FF12 industry | SEC facts | Book equity | Net income | Operating cash flow | Assets | Revenue | Cost of revenue | Gross profit | Shares outstanding | Shared CIK tickers |",
        "|---|---:|---:|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for symbol in symbols:
        cik = store.ticker_to_cik.get(symbol)
        if cik is None:
            lines.append(f"| {symbol} | — | — | Other | missing ticker mapping | — | — | — | — | — | — | — | — | — |")
            continue
        tags = tags_by_cik.get(cik, set())
        facts_status = "present" if tags else "missing companyfacts member"
        availability = ["yes" if set(priorities) & tags else "no" for priorities in _ITEM_TAGS.values()]
        sic = store.sic_by_cik.get(cik)
        industry = store.industry_by_cik.get(cik, "Other")
        related_tickers = sorted(tickers_by_cik.get(cik, ()))
        shared = ", ".join(related_tickers) if len(related_tickers) > 1 else "—"
        lines.append(
            f"| {symbol} | {cik} | {sic if sic is not None else '—'} | {industry} | {facts_status} | "
            + " | ".join(availability)
            + f" | {shared} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Item coverage means the source contains at least one eligible USD or share-count tag for that CIK; it does not guarantee a non-null feature on every decision date.",
            "- Features are null before their source filing becomes available and after the annual facts become more than 18 months stale.",
            "- SIC is from the SEC submissions snapshot and FF12 categories are from the dated Ken French `Siccodes12.zip` file; unmapped or missing SIC values are `Other`.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
