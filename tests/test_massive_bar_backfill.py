from __future__ import annotations

import json
from datetime import date

from scripts.backfill_massive_bars import recently_delisted_candidates, without_api_key


def test_massive_backfill_limits_to_recently_delisted_symbols(tmp_path) -> None:
    path = tmp_path / "candidates.json"
    path.write_text(
        json.dumps(
            {
                "records": [
                    {"symbol": "RECENT", "status": "inactive", "delisted_utc": "2025-02-01T00:00:00Z"},
                    {"symbol": "OLD", "status": "inactive", "delisted_utc": "2022-02-01T00:00:00Z"},
                    {"symbol": "ACTIVE", "status": "active", "delisted_utc": "2025-02-01T00:00:00Z"},
                    {"symbol": "NO_DATE", "status": "inactive"},
                ]
            }
        )
    )

    assert recently_delisted_candidates(path, start=date(2024, 7, 1)) == ["RECENT"]


def test_massive_next_page_url_does_not_retain_credentials() -> None:
    clean = without_api_key("https://api.massive.com/v2/aggs?cursor=opaque&apiKey=secret")

    assert clean == "https://api.massive.com/v2/aggs?cursor=opaque"
    assert "secret" not in clean
