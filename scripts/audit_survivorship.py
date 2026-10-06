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
    candidate_records: list[dict[str, Any]] | None,
    cached_bar_symbols: set[str],
    daily_stores: list[str] | None = None,
    candidate_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    by_symbol = {row["symbol"]: row for row in (candidate_records or [])}
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
    pool_symbols = set(by_symbol) if candidate_records is not None else cached_bar_symbols
    inactive = {symbol for symbol in pool_symbols if by_symbol.get(symbol, {}).get("status") == "inactive"}
    known = {}
    for symbol in ["SIVB", "FRC", "BBBY"]:
        known[symbol] = {
            "in_candidate_pool": symbol in pool_symbols,
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
        "inactive_candidate_symbols_in_revised_pool": len(inactive) if candidate_records is not None else None,
        "inactive_candidate_symbols_with_cached_daily_bars": (
            len(inactive & cached_bar_symbols) if candidate_records is not None else None
        ),
        "newly_selected_symbols": added_symbols,
        "newly_selected_symbol_ticker_sessions": dict(changes),
        "newly_selected_currently_inactive_symbols": selected_delisted,
        "selected_inactive_ticker_sessions": (
            sum(
                1
                for _session, tickers in revised_by_session.items()
                for symbol in tickers
                if by_symbol.get(symbol, {}).get("status") == "inactive"
            )
            if candidate_records is not None
            else None
        ),
        "known_delisted_audit": known,
        "candidate_reference_coverage": {
            key: candidate_metadata.get(key)
            for key in (
                "massive_cached_inactive_xnas_records",
                "massive_cached_pages",
                "massive_inactive_crawl_complete",
                "massive_cached_ticker_range",
                "alpaca_candidate_count",
                "massive_inactive_candidate_count",
            )
            if candidate_metadata is not None and key in candidate_metadata
        },
        "limitations": [
            "Point-in-time grouped bars can include securities no longer active, but do not identify historical "
            "exchange membership or an inactive/delisting status by themselves.",
            "Only symbols with provider bars can be ranked; absent bars can still hide formerly liquid "
            "delisted stocks.",
            "Inactive is measured as of the current reference snapshot, not the security's status on each "
            "session.",
            "This audit corrects candidate coverage where data exists; it does not certify zero survivorship "
            "bias.",
            *(candidate_metadata.get("limitations", []) if candidate_metadata is not None else []),
        ],
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    known = report["known_delisted_audit"]
    inactive_pool = report["inactive_candidate_symbols_in_revised_pool"]
    inactive_with_bars = report["inactive_candidate_symbols_with_cached_daily_bars"]
    selected_inactive = report["selected_inactive_ticker_sessions"]
    lines = [
        "# Survivorship coverage audit",
        "",
        "This audit compares the prior current-asset top-25 with a revised point-in-time selection over "
        "historical daily bars. The grouped-bar pool includes any ticker with a reported bar on a session, "
        "including securities no longer active today.",
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
        f"| Inactive candidate symbols in revised pool | {inactive_pool if inactive_pool is not None else 'n/a'} |",
        f"| Inactive candidates with cached daily bars | "
        f"{inactive_with_bars if inactive_with_bars is not None else 'n/a'} |",
        f"| Sessions with a changed top 25 | {report['sessions_with_changed_top25']:,}/"
        f"{report['sessions_compared']:,} |",
        f"| Selected inactive ticker-sessions | {selected_inactive if selected_inactive is not None else 'n/a'} |",
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
            status = details["status"] if details else "unknown"
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
    coverage = report.get("candidate_reference_coverage", {})
    if coverage:
        lines.extend(
            [
                "Inactive-reference crawl coverage: "
                f"{coverage.get('massive_cached_inactive_xnas_records', 'unknown'):,} cached XNAS rows across "
                f"{coverage.get('massive_cached_pages', 'unknown'):,} pages; ticker range "
                f"{coverage.get('massive_cached_ticker_range', 'unknown')}; complete="
                f"{coverage.get('massive_inactive_crawl_complete', 'unknown')}",
                "",
            ]
        )
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    lines.append("")
    atomic_text(path, "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, default=Path("data/kronos-pit-2024-2026/universe.json"))
    parser.add_argument(
        "--revised", type=Path, default=Path("data/kronos-pit-2024-2026-survivorship/universe.json")
    )
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--daily-store", type=str, default="1Day_raw")
    parser.add_argument("--additional-daily-store", action="append", default=[])
    parser.add_argument("--json", type=Path, default=Path("docs/survivorship-coverage.json"))
    parser.add_argument("--markdown", type=Path, default=Path("docs/survivorship-coverage.md"))
    args = parser.parse_args()
    prior = json.loads(args.prior.read_text())
    revised = json.loads(args.revised.read_text())
    candidates = json.loads(args.candidates.read_text()) if args.candidates else None
    daily_stores = [args.daily_store, *args.additional_daily_store]
    bar_symbols: set[str] = set()
    for store in daily_stores:
        bars = read_bar_store(args.data_dir, store)
        if not bars.is_empty():
            bar_symbols.update(bars["symbol"].unique().to_list())
    report = summarize(
        prior,
        revised,
        candidates["records"] if candidates else None,
        bar_symbols,
        daily_stores,
        candidates,
    )
    atomic_text(args.json, json.dumps(report, indent=2) + "\n")
    write_markdown(args.markdown, report)
    print(json.dumps({key: value for key, value in report.items() if key != "limitations"}, indent=2))


if __name__ == "__main__":
    main()
