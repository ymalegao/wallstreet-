"""Compare current-asset and delisting-inclusive PIT universes and quantify bar coverage."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ws.store.atomic import atomic_text
from ws.store.bar_loader import read_bar_store


def summarize(
    prior: dict[str, Any],
    revised: dict[str, Any],
    candidate_records: list[dict[str, Any]],
    cached_bar_symbols: set[str],
    daily_stores: list[str] | None = None,
) -> dict[str, Any]:
    by_symbol = {row["symbol"]: row for row in candidate_records}
    prior_by_session = prior["selection_by_session"]
    revised_by_session = revised["selection_by_session"]
    sessions = sorted(set(prior_by_session) & set(revised_by_session))
    changes: Counter[str] = Counter()
    for session in sessions:
        changes.update(set(revised_by_session[session]) - set(prior_by_session[session]))
    added_symbols = sorted(changes)
    selected_delisted = {
        symbol: {
            "selected_ticker_sessions": changes[symbol],
            "delisted_utc": by_symbol.get(symbol, {}).get("delisted_utc"),
            "status": by_symbol.get(symbol, {}).get("status", "unknown"),
        }
        for symbol in added_symbols
        if by_symbol.get(symbol, {}).get("status") == "inactive"
    }
    selected_symbols = set(revised.get("selected_symbols", []))
    pool_symbols = set(by_symbol)
    inactive = {symbol for symbol in pool_symbols if by_symbol.get(symbol, {}).get("status") == "inactive"}
    known = {}
    for symbol in ["SIVB", "FRC", "BBBY"]:
        known[symbol] = {
            "in_candidate_pool": symbol in by_symbol,
            "daily_bars_cached": symbol in cached_bar_symbols,
            "selected_in_revised_window": symbol in selected_symbols,
            "delisted_utc": by_symbol.get(symbol, {}).get("delisted_utc"),
        }
    return {
        "status": "EXPLORATORY_SURVIVORSHIP_COVERAGE_AUDIT",
        "period": revised["decision_window"],
        "candidate_pool_source": revised["candidate_pool"]["source"],
        "candidate_as_of": revised["candidate_pool"]["as_of"],
        "daily_bar_stores": daily_stores or ["1Day_raw"],
        "sessions_compared": len(sessions),
        "sessions_with_changed_top25": sum(
            prior_by_session[session] != revised_by_session[session] for session in sessions
        ),
        "prior_candidate_count": prior["candidate_pool"]["candidate_count"],
        "revised_candidate_count": revised["candidate_pool"]["candidate_count"],
        "revised_candidate_symbols_with_cached_daily_bars": len(pool_symbols & cached_bar_symbols),
        "revised_candidate_symbols_without_cached_daily_bars": len(pool_symbols - cached_bar_symbols),
        "inactive_candidate_symbols_in_revised_pool": len(inactive),
        "inactive_candidate_symbols_with_cached_daily_bars": len(inactive & cached_bar_symbols),
        "newly_selected_symbols": added_symbols,
        "newly_selected_symbol_ticker_sessions": dict(changes),
        "newly_selected_currently_inactive_symbols": selected_delisted,
        "selected_inactive_ticker_sessions": sum(
            1
            for _session, tickers in revised_by_session.items()
            for symbol in tickers
            if by_symbol.get(symbol, {}).get("status") == "inactive"
        ),
        "known_delisted_audit": known,
        "limitations": [
            "The reference snapshot includes inactive securities but is not a complete historical "
            "exchange-membership file.",
            "Only symbols with provider bars can be ranked; absent bars can still hide formerly liquid "
            "delisted stocks.",
            "Inactive is measured as of the current reference snapshot, not the security's status on each "
            "session.",
            "This audit corrects candidate coverage where data exists; it does not certify zero survivorship "
            "bias.",
        ],
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    known = report["known_delisted_audit"]
    lines = [
        "# Survivorship coverage audit",
        "",
        "This audit compares the Alpaca current/inactive-asset candidate set with the Massive active/inactive "
        "reference set, then reruns the same lagged price/liquidity top-25 selector over cached daily bars.",
        "",
        f"Study window: {report['period']['start_inclusive']} to {report['period']['end_exclusive']} (exclusive).",
        f"Daily-bar sources: {', '.join(report['daily_bar_stores'])}.",
        "",
        "| Measure | Result |",
        "|---|---:|",
        f"| Candidate symbols, prior | {report['prior_candidate_count']:,} |",
        f"| Candidate symbols, revised | {report['revised_candidate_count']:,} |",
        f"| Revised candidates with cached daily bars | "
        f"{report['revised_candidate_symbols_with_cached_daily_bars']:,} |",
        f"| Revised candidates missing cached daily bars | "
        f"{report['revised_candidate_symbols_without_cached_daily_bars']:,} |",
        f"| Inactive candidate symbols in revised pool | {report['inactive_candidate_symbols_in_revised_pool']:,} |",
        f"| Inactive candidates with cached daily bars | "
        f"{report['inactive_candidate_symbols_with_cached_daily_bars']:,} |",
        f"| Sessions with a changed top 25 | {report['sessions_with_changed_top25']:,}/"
        f"{report['sessions_compared']:,} |",
        f"| Selected inactive ticker-sessions | {report['selected_inactive_ticker_sessions']:,} |",
        "",
        "## Added symbols selected by the revised universe",
        "",
    ]
    if report["newly_selected_symbols"]:
        lines.extend(
            ["| Symbol | Ticker-sessions added | Current status | Delisted date |", "|---|---:|---|---|"]
        )
        for symbol in report["newly_selected_symbols"]:
            details = report["newly_selected_currently_inactive_symbols"].get(symbol)
            status = details["status"] if details else "active"
            delisted = details["delisted_utc"] if details else None
            count = report["newly_selected_symbol_ticker_sessions"][symbol]
            lines.append(f"| {symbol} | {count} | {status} | {delisted or '—'} |")
    else:
        lines.append("No newly selected symbols had cached bars and displaced the prior top 25 in this window.")
    lines.extend(
        [
            "",
            "## Known delisted examples",
            "",
            "| Symbol | In candidate pool | Daily bars cached | Selected | Delisted date |",
            "|---|:---:|:---:|:---:|---|",
        ]
    )
    for symbol, row in known.items():
        lines.append(
            f"| {symbol} | {'yes' if row['in_candidate_pool'] else 'no'} "
            f"| {'yes' if row['daily_bars_cached'] else 'no'} "
            f"| {'yes' if row['selected_in_revised_window'] else 'no'} | {row['delisted_utc'] or '—'} |"
        )
    lines.extend(["", "## Limits", ""])
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    lines.append("")
    atomic_text(path, "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, default=Path("data/kronos-pit-2024-2026/universe.json"))
    parser.add_argument(
        "--revised", type=Path, default=Path("data/kronos-pit-2024-2026-survivorship/universe.json")
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("data/universe/massive-reference-2026-10-05/stock-candidates.json"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--additional-daily-store", action="append", default=[])
    parser.add_argument("--json", type=Path, default=Path("docs/survivorship-coverage.json"))
    parser.add_argument("--markdown", type=Path, default=Path("docs/survivorship-coverage.md"))
    args = parser.parse_args()
    prior = json.loads(args.prior.read_text())
    revised = json.loads(args.revised.read_text())
    candidates = json.loads(args.candidates.read_text())
    daily_stores = ["1Day_raw", *args.additional_daily_store]
    bar_symbols: set[str] = set()
    for store in daily_stores:
        bars = read_bar_store(args.data_dir, store)
        if not bars.is_empty():
            bar_symbols.update(bars["symbol"].unique().to_list())
    report = summarize(prior, revised, candidates["records"], bar_symbols, daily_stores)
    atomic_text(args.json, json.dumps(report, indent=2) + "\n")
    write_markdown(args.markdown, report)
    print(json.dumps({key: value for key, value in report.items() if key != "limitations"}, indent=2))


if __name__ == "__main__":
    main()
