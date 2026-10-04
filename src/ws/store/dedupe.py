"""Near-duplicate detection (syndicated copies of the same article).

This handles copies of the *same text*. Different articles about the same story are not merged
here: the evaluation unit is the (ticker, decision cycle) pair (see labels.py), which already
collapses every item about a ticker that arrives before one cycle into one sample.

The Jaccard threshold is NOT fixed in code. ``scripts/probe_apis.py`` measures the similarity
distribution on real cross-source pairs and prints example pairs per band; the chosen value goes in
``configs/sources.yaml``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import timedelta

from datasketch import MinHash, MinHashLSH

from ws.schema import Event

_WORD = re.compile(r"[a-z0-9]+")
SHINGLE = 5
NUM_PERM = 128


def shingles(text: str, k: int = SHINGLE) -> set[str]:
    words = _WORD.findall(text.lower())
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def event_text(e: Event) -> str:
    return f"{e.headline} {e.body}"


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _minhash(sh: set[str]) -> MinHash:
    m = MinHash(num_perm=NUM_PERM)
    for s in sh:
        m.update(s.encode())
    return m


def near_duplicate_clusters(
    events: Sequence[Event], threshold: float, window: timedelta = timedelta(hours=48)
) -> dict[str, str]:
    """Map every event_id to its cluster's canonical event_id (the earliest ``first_seen_ts``).

    Candidates come from MinHash LSH; each candidate pair is then confirmed with exact Jaccard,
    a shared ticker (when both have tickers) and the time window, so LSH false positives never merge.
    """
    ordered = sorted(events, key=lambda e: (e.first_seen_ts, e.event_id))
    sh = {e.event_id: shingles(event_text(e)) for e in ordered}
    by_id = {e.event_id: e for e in ordered}
    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    canonical: dict[str, str] = {}
    for e in ordered:
        if not sh[e.event_id]:
            canonical[e.event_id] = e.event_id
            continue
        mh = _minhash(sh[e.event_id])
        match = None
        for cand_id in lsh.query(mh):
            c = by_id[cand_id]
            if e.first_seen_ts - c.first_seen_ts > window:
                continue
            if e.tickers and c.tickers and not set(e.tickers) & set(c.tickers):
                continue
            if jaccard(sh[e.event_id], sh[cand_id]) >= threshold:
                root = canonical[cand_id]
                if match is None or by_id[root].first_seen_ts < by_id[match].first_seen_ts:
                    match = root
        canonical[e.event_id] = match or e.event_id
        lsh.insert(e.event_id, mh)
    return canonical
