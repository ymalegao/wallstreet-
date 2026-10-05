"""Score the frozen ticker-cycle input set; no outcome data enters model prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ws.models.text import TextModel
from ws.store.atomic import atomic_text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="jev-decision")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8001")
    ap.add_argument("--kind", choices=["chat", "jev"], default="jev")
    ap.add_argument("--model-dir", type=Path, default=Path("data/models/JEV-9B"))
    ap.add_argument("--revision", default="b63f651ce8ed64481d3f5e73ecdb05f740042f01")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--inputs", type=Path, default=Path("data/replay-inputs.json"))
    ap.add_argument("--output", type=Path, default=Path("data/jev-signals.json"))
    a = ap.parse_args()
    raw = a.inputs.read_text()
    rows = json.loads(raw)
    if a.limit:
        rows = rows[: a.limit]
    model = TextModel(a.endpoint, a.model, Path("data/inference-cache"), a.kind, a.model_dir, a.revision)

    def score(row):
        scores = [{"event_id": e["id"], **model.score_event(e["text"], row["ticker"])} for e in row["events"]]
        strongest = max(scores, key=lambda s: abs(s["signal"]))
        return {
            "ticker": row["ticker"],
            "cycle": row["cycle"],
            "session": row["session"],
            "signal": strongest["signal"],
            "events": scores,
        }

    results = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i, result in enumerate(pool.map(score, rows)):
            results.append(result)
            if (i + 1) % 25 == 0:
                print(f"Scored {i + 1}/{len(rows)} cycles", flush=True)
                atomic_text(
                    a.output,
                    json.dumps(
                        {
                            "model": a.model,
                            "revision": a.revision,
                            "complete": False,
                            "input_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                            "signals": results,
                        }
                    )
                    + "\n",
                )
    atomic_text(
        a.output,
        json.dumps(
            {
                "model": a.model,
                "revision": a.revision,
                "complete": len(rows) == len(json.loads(raw)),
                "input_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "signals": results,
            }
        )
        + "\n",
    )


if __name__ == "__main__":
    main()
