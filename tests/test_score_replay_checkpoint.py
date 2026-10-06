from __future__ import annotations

import hashlib
import json

import pytest

from scripts.score_replay import read_checkpoint


def test_score_checkpoint_resumes_only_for_matching_input_prefix(tmp_path) -> None:
    rows = [{"ticker": "AAPL", "cycle": "2025-01-02T14:45:00+00:00"}]
    digest = hashlib.sha256(b"frozen").hexdigest()
    score = {"ticker": rows[0]["ticker"], "cycle": rows[0]["cycle"], "signal": 1}
    path = tmp_path / "signals.json"
    path.write_text(
        json.dumps({"model": "jev-decision", "revision": "rev1", "input_sha256": digest, "signals": [score]})
    )

    assert read_checkpoint(path, digest, "jev-decision", "rev1", rows) == [score]
    with pytest.raises(ValueError, match="different frozen inputs"):
        read_checkpoint(path, "other", "jev-decision", "rev1", rows)
    with pytest.raises(ValueError, match="different model revision"):
        read_checkpoint(path, digest, "jev-decision", "rev2", rows)
