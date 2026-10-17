"""Save a dated S&P 100 proxy from iShares OEF U.S. equity holdings."""

from __future__ import annotations

import argparse
import csv
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

URL = "https://www.ishares.com/us/products/239723/ishares-s-p-100-etf/latest-holdings.csv"


def parse_holdings(csv_text: str) -> tuple[str, list[str]]:
    lines = csv_text.splitlines()
    as_of_line = next((line for line in lines if line.startswith('Fund Holdings as of,"')), None)
    if as_of_line is None:
        raise ValueError("iShares holdings CSV has no holdings as-of date")
    as_of = next(csv.reader([as_of_line]))[1]
    header_index = next((i for i, line in enumerate(lines) if line.startswith("Ticker,Name,")), None)
    if header_index is None:
        raise ValueError("iShares holdings CSV header not found")
    rows = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
    symbols = sorted(
        {
            row["Ticker"].strip().replace(" ", ".").upper()
            for row in rows
            if row["Asset Class"] == "Equity"
            and row["Location"] == "United States"
            and row["Exchange"] in {"NASDAQ", "NYSE"}
            and row["Ticker"].strip()
        }
    )
    if not symbols:
        raise ValueError("no U.S. equity constituents found in iShares holdings CSV")
    return as_of, symbols


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("configs/universe/sp100-current.json"))
    args = parser.parse_args()
    response = httpx.get(URL, timeout=30, follow_redirects=True)
    response.raise_for_status()
    as_of, symbols = parse_holdings(response.text)
    snapshot = {
        "index": "S&P 100",
        "proxy": "U.S. equity holdings of iShares OEF",
        "as_of": datetime.strptime(as_of, "%b %d, %Y").date().isoformat(),
        "retrieved_at": datetime.now(UTC).isoformat(),
        "source_url": URL,
        "source_note": (
            "ETF holdings are a dated proxy; dual share classes can make ticker count differ from company count."
        ),
        "symbols": symbols,
        "symbol_count": len(symbols),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, indent=2) + "\n")
    print(f"saved {len(symbols)} tickers as of {snapshot['as_of']} to {args.output}")


if __name__ == "__main__":
    main()
