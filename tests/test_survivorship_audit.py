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


def test_point_in_time_bar_audit_marks_security_status_as_unknown_without_reference_file() -> None:
    prior = {
        "selection_by_session": {"2025-01-02": ["A", "B"]},
        "candidate_pool": {"candidate_count": 3},
    }
    revised = {
        "selection_by_session": {"2025-01-02": ["B", "BBBY"]},
        "selected_symbols": ["B", "BBBY"],
        "decision_window": {"start_inclusive": "2025-01-01", "end_exclusive": "2025-02-01"},
        "candidate_pool": {"candidate_count": 100, "source": "Massive grouped bars", "as_of": "point-in-time"},
    }

    report = summarize(prior, revised, None, {"A", "B", "BBBY"}, ["grouped:massive_1Day_raw"])

    assert report["newly_selected_symbols"] == ["BBBY"]
    assert report["inactive_candidate_symbols_in_revised_pool"] is None
    assert report["selected_inactive_ticker_sessions"] is None
    assert report["known_delisted_audit"]["BBBY"]["in_candidate_pool"] is True


def test_survivorship_audit_carries_partial_reference_coverage_and_limitations() -> None:
    prior = {
        "selection_by_session": {"2025-01-02": ["A"]},
        "candidate_pool": {"candidate_count": 1},
    }
    revised = {
        "selection_by_session": {"2025-01-02": ["A"]},
        "selected_symbols": ["A"],
        "decision_window": {"start_inclusive": "2025-01-01", "end_exclusive": "2025-02-01"},
        "candidate_pool": {"candidate_count": 2, "source": "Massive", "as_of": "2026-10-05"},
    }
    metadata = {
        "massive_cached_inactive_xnas_records": 129700,
        "massive_cached_pages": 1288,
        "massive_inactive_crawl_complete": False,
        "massive_cached_ticker_range": ["A", "EQFN"],
        "limitations": ["Inactive-reference crawl is incomplete."],
    }

    report = summarize(prior, revised, [{"symbol": "A", "status": "active"}], {"A"}, candidate_metadata=metadata)

    assert report["candidate_reference_coverage"]["massive_inactive_crawl_complete"] is False
    assert report["candidate_reference_coverage"]["massive_cached_ticker_range"] == ["A", "EQFN"]
    assert "Inactive-reference crawl is incomplete." in report["limitations"]
