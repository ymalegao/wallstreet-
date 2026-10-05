"""Generate resumable daily Kronos path features for a fixed input universe."""

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

from ws.market_time import session_model_timestamp
from ws.store.atomic import atomic_text

sys.path.insert(0, str(Path("data/vendor/Kronos").resolve()))
from model import Kronos, KronosPredictor, KronosTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--lookback", type=int, required=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--paths", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.lookback < 2 or a.paths < 2 or not 0 < a.temperature <= 2:
        raise ValueError("Require lookback >= 2, paths >= 2, and 0 < temperature <= 2")

    torch.set_num_threads(4)
    tokenizer = KronosTokenizer.from_pretrained("data/models/Kronos-Tokenizer-base", local_files_only=True).eval()
    model = Kronos.from_pretrained("data/models/Kronos-small", local_files_only=True).eval()
    predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)
    jobs: list[dict[str, Any]] = json.loads(a.inputs.read_text())
    results: list[dict[str, Any]] = []
    a.cache_dir.mkdir(parents=True, exist_ok=True)

    for i, job in enumerate(jobs):
        history = job["history"][-a.lookback :]
        identity = {
            "job": job,
            "lookback": a.lookback,
            "paths": a.paths,
            "seed": a.seed,
            "temperature": a.temperature,
            "top_p": 0.9,
            "upstream": "67b630e67f6a18c9c9be918d9b4337c960db1e9a",
            "model": "901c26c1332695a2a8f243eb2f37243a37bea320",
            "tokenizer": "0e0117387f39004a9016484a186a908917e22426",
        }
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        target = a.cache_dir / f"{job['ticker']}-{job['session']}.json"
        if target.exists():
            cached = json.loads(target.read_text())
            if cached.get("fingerprint") == fingerprint:
                results.append(cached)
                continue

        frame = pd.DataFrame(history)
        xs = pd.Series(pd.to_datetime([row["ts"][:10] for row in history]))
        ys = pd.Series(
            pd.to_datetime([session_model_timestamp(pd.Timestamp(x).date()) for x in job["future_sessions"]])
        )
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
            "fingerprint": fingerprint,
            "history_end": history[-1]["ts"],
            "lookback": a.lookback,
            "paths": a.paths,
            "temperature": a.temperature,
            "reference_close": float(history[-1]["close"]),
            "seconds": time.monotonic() - started,
        }
        if not np.isfinite(closes).all() or (closes <= 0).any():
            result["status"] = "invalid_forecast_prices"
        else:
            result["status"] = "ok"
            result["terminal_closes"] = closes[:, -1].tolist()
            result["p_up_from_previous_close"] = float((closes[:, -1] > result["reference_close"]).mean())
            result["mean_return"] = float(np.mean(closes[:, -1] / result["reference_close"] - 1))
            result["return_std"] = float(np.std(closes[:, -1] / result["reference_close"] - 1, ddof=1))
        atomic_text(target, json.dumps(result) + "\n")
        results.append(result)
        if (i + 1) % 25 == 0 or i + 1 == len(jobs):
            atomic_text(
                a.output,
                json.dumps({"complete": False, "inputs": str(a.inputs), "features": results}) + "\n",
            )
            print(f"Kronos L{a.lookback} T{a.temperature}: {i + 1}/{len(jobs)}", flush=True)

    atomic_text(a.output, json.dumps({"complete": True, "inputs": str(a.inputs), "features": results}) + "\n")


if __name__ == "__main__":
    main()
