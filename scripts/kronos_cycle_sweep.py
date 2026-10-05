"""Generate fixed-path Kronos forecasts from bar history available at a decision cycle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ws.store.atomic import atomic_text

sys.path.insert(0, str(Path("data/vendor/Kronos").resolve()))
from model import Kronos, KronosPredictor, KronosTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--paths", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.paths < 2:
        raise ValueError("At least two paths required")

    torch.set_num_threads(4)
    tokenizer = KronosTokenizer.from_pretrained("data/models/Kronos-Tokenizer-base", local_files_only=True).eval()
    model = Kronos.from_pretrained("data/models/Kronos-small", local_files_only=True).eval()
    predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)
    jobs: list[dict[str, Any]] = json.loads(a.inputs.read_text())
    a.cache_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for i, job in enumerate(jobs):
        identity = {
            "job": job,
            "paths": a.paths,
            "temperature": a.temperature,
            "seed": a.seed,
            "upstream": "67b630e67f6a18c9c9be918d9b4337c960db1e9a",
            "model": "901c26c1332695a2a8f243eb2f37243a37bea320",
            "tokenizer": "0e0117387f39004a9016484a186a908917e22426",
        }
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        target = a.cache_dir / f"{job['ticker']}-{job['cycle'].replace(':', '-')}.json"
        if target.exists():
            cached = json.loads(target.read_text())
            if cached.get("fingerprint") == fingerprint:
                results.append(cached)
                continue

        frame = pd.DataFrame(job["history"])[["open", "high", "low", "close", "volume", "amount"]]
        xs = pd.Series(pd.to_datetime([row["ts"] for row in job["history"]]))
        ys = pd.Series(pd.to_datetime(job["future_timestamps"]))
        torch.manual_seed(a.seed)
        started = time.monotonic()
        paths = predictor.predict_batch(
            [frame.copy() for _ in range(a.paths)],
            [xs] * a.paths,
            [ys] * a.paths,
            pred_len=len(ys),
            sample_count=1,
            T=a.temperature,
            top_p=0.9,
            verbose=False,
        )
        closes = np.array([path["close"].to_numpy() for path in paths])
        result: dict[str, Any] = {
            "ticker": job["ticker"],
            "session": job["session"],
            "cycle": job["cycle"],
            "interval_minutes": job["interval_minutes"],
            "target_session": job["target_session"],
            "feature_end": job["history"][-1]["ts"],
            "fingerprint": fingerprint,
            "paths": a.paths,
            "temperature": a.temperature,
            "reference_close": float(job["history"][-1]["close"]),
            "seconds": time.monotonic() - started,
        }
        if not np.isfinite(closes).all() or (closes <= 0).any():
            result["status"] = "invalid_forecast_prices"
            result["nonfinite_values"] = int(np.size(closes) - np.isfinite(closes).sum())
            result["nonpositive_values"] = int(np.sum(np.isfinite(closes) & (closes <= 0)))
        else:
            result["status"] = "ok"
            terminal = closes[:, -1]
            result["terminal_closes"] = terminal.tolist()
            result["mean_return"] = float(np.mean(terminal / result["reference_close"] - 1))
            result["return_std"] = float(np.std(terminal / result["reference_close"] - 1, ddof=1))
            result["p_up"] = float(np.mean(terminal > result["reference_close"]))
        atomic_text(target, json.dumps(result) + "\n")
        results.append(result)
        if (i + 1) % 25 == 0 or i + 1 == len(jobs):
            atomic_text(a.output, json.dumps({"complete": False, "forecasts": results}) + "\n")
            print(
                f"Kronos {job['interval_minutes']}m {i + 1}/{len(jobs)} ({len(job['future_timestamps'])} steps)",
                flush=True,
            )

    atomic_text(a.output, json.dumps({"complete": True, "forecasts": results}) + "\n")


if __name__ == "__main__":
    main()
