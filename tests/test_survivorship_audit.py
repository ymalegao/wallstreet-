from __future__ import annotations

from scripts.audit_survivorship import summarize


def test_survivorship_audit_counts_added_inactive_candidates_with_bars() -> None:
    prior = {
        "selection_by_session": {"2025-01-02": ["A", "B"]},
        "candidate_pool": {"candidate_count": 3},
    }
    revised = {
        "selection_by_session": {"2025-01-02": ["B", "C"]},
        "selected_symbols": ["B", "C"],
        "decision_window": {"start_inclusive": "2025-01-01", "end_exclusive": "2025-02-01"},
        "candidate_pool": {"candidate_count": 4, "source": "Massive", "as_of": "2026-10-05"},
    }
    records = [
        {"symbol": "A", "status": "active"},
        {"symbol": "B", "status": "active"},
        {"symbol": "C", "status": "inactive", "delisted_utc": "2025-01-03T00:00:00Z"},
        {"symbol": "D", "status": "inactive", "delisted_utc": "2024-01-03T00:00:00Z"},
    ]

    report = summarize(prior, revised, records, {"A", "B", "C"})

    assert report["sessions_with_changed_top25"] == 1
    assert report["newly_selected_symbols"] == ["C"]
    assert report["newly_selected_symbol_ticker_sessions"] == {"C": 1}
    assert report["inactive_candidate_symbols_in_revised_pool"] == 2
    assert report["inactive_candidate_symbols_with_cached_daily_bars"] == 1
    assert report["selected_inactive_ticker_sessions"] == 1
