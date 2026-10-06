"""Score the frozen ticker-cycle input set; no outcome data enters model prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ws.models.text import TextModel
from ws.store.atomic import atomic_text


def read_checkpoint(
    path: Path,
    input_sha256: str,
    model: str,
    revision: str,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Resume a partial scoring run only when it is an exact prefix of the frozen inputs."""
    if not path.exists():
        return []
    checkpoint = json.loads(path.read_text())
    if checkpoint.get("input_sha256") != input_sha256:
        raise ValueError("Existing score checkpoint belongs to different frozen inputs")
    if checkpoint.get("model") != model or checkpoint.get("revision") != revision:
        raise ValueError("Existing score checkpoint belongs to a different model revision")
    results = checkpoint.get("signals")
    if not isinstance(results, list) or len(results) > len(rows):
        raise ValueError("Existing score checkpoint has an invalid signal list")
    for source, result in zip(rows, results, strict=False):
        if (source["ticker"], source["cycle"]) != (result.get("ticker"), result.get("cycle")):
            raise ValueError("Existing score checkpoint is not a prefix of the frozen inputs")
    if checkpoint.get("complete") and len(results) != len(rows):
        raise ValueError("Existing score checkpoint is marked complete but is truncated")
    return results


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
    all_rows = json.loads(raw)
    rows = all_rows
    if a.limit:
        rows = rows[: a.limit]
    input_sha256 = hashlib.sha256(raw.encode()).hexdigest()
    results = read_checkpoint(a.output, input_sha256, a.model, a.revision, rows)
    if len(results) == len(rows):
        atomic_text(
            a.output,
            json.dumps(
                {
                    "model": a.model,
                    "revision": a.revision,
                    "complete": len(rows) == len(all_rows),
                    "input_sha256": input_sha256,
                    "signals": results,
                }
            )
            + "\n",
        )
        print(f"Score checkpoint already covers all {len(rows)} requested cycles", flush=True)
        return
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

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i, result in enumerate(pool.map(score, rows[len(results) :]), start=len(results)):
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
                            "input_sha256": input_sha256,
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
                "complete": len(rows) == len(all_rows),
                "input_sha256": input_sha256,
                "signals": results,
            }
        )
        + "\n",
    )


if __name__ == "__main__":
    main()
