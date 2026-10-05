"""Run inside Dockerfile.research. Individual paths retained; no averaged-path P(up)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ws.market_time import session_model_timestamp

# Official upstream source is pinned in the experiment manifest.
sys.path.insert(0, str(Path("data/vendor/Kronos").resolve()))
from model import Kronos, KronosPredictor, KronosTokenizer

from ws.store.atomic import atomic_text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--inputs", type=Path, default=Path("data/kronos-inputs.json"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/kronos-cache"))
    ap.add_argument("--keep-alive", action="store_true", help="Keep model weights loaded after forecasting")
    a = ap.parse_args()
    if a.paths < 2:
        raise ValueError("At least two paths required")
    torch.set_num_threads(4)
    tokenizer = KronosTokenizer.from_pretrained("data/models/Kronos-Tokenizer-base", local_files_only=True)
    model = Kronos.from_pretrained("data/models/Kronos-small", local_files_only=True)
    # Both upstream model classes start in training mode unless the caller switches them.
    model.eval()
    tokenizer.eval()
    predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)
    jobs = json.loads(a.inputs.read_text())
    if a.limit:
        jobs = jobs[: a.limit]
    for i, job in enumerate(jobs):
        identity = {
            "job": job,
            "paths": a.paths,
            "seed": 42,
            "inference_mode": "eval",
            "timestamp_mode": "session-date-midnight",
            "upstream": "67b630e67f6a18c9e9be918d9b4337c960db1e9a",
            "model": "901c26c1332695a2a8f243eb2f37243a37bea320",
            "tokenizer": "0e0117387f39004a9016484a186a908917e22426",
            "temperature": 1.0,
            "top_p": 0.9,
        }
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        target = a.cache_dir / f"{job['ticker']}-{job['session']}.json"
        if target.exists() and json.loads(target.read_text()).get("fingerprint") == fingerprint:
            continue
        torch.manual_seed(42)
        frame = pd.DataFrame(job["history"])
        # Daily bars are dated by exchange session, not by their vendor UTC publication clock.
        xs = pd.Series(pd.to_datetime([x[:10] for x in frame["ts"]]))
        ys = pd.Series([session_model_timestamp(pd.Timestamp(x).date()) for x in job["future_sessions"]])
        t = time.monotonic()
        # Repeated series + sample_count=1 preserves all stochastic draws separately.
        paths = predictor.predict_batch(
            [frame.copy() for _ in range(a.paths)],
            [xs] * a.paths,
            [ys] * a.paths,
            pred_len=len(ys),
            sample_count=1,
            T=1.0,
            top_p=0.9,
            verbose=False,
        )
        closes = np.array([p["close"].to_numpy() for p in paths])
        if not np.isfinite(closes).all() or (closes <= 0).any():
            atomic_text(
                target,
                json.dumps(
                    {
                        "ticker": job["ticker"],
                        "session": job["session"],
                        "fingerprint": fingerprint,
                        "history_end": job["history"][-1]["ts"],
                        "paths": a.paths,
                        "status": "invalid_forecast_prices",
                    }
                )
                + "\n",
            )
            print(f"Kronos {i + 1}/{len(jobs)}: invalid forecast prices; recorded and skipped", flush=True)
            continue
        reference = float(frame["close"].iloc[-1])
        terminal = closes[:, -1]
        result = {
            "ticker": job["ticker"],
            "session": job["session"],
            "fingerprint": fingerprint,
            "history_end": job["history"][-1]["ts"],
            "paths": a.paths,
            "reference_close": reference,
            "terminal_closes": terminal.tolist(),
            "close_paths": closes.tolist(),
            "p_up_from_previous_close": float((terminal > reference).mean()),
            "return_std": float(np.std(terminal / reference - 1, ddof=1)),
            "seconds": time.monotonic() - t,
            "provenance": {k: v for k, v in identity.items() if k != "job"},
        }
        atomic_text(target, json.dumps(result) + "\n")
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Kronos {i + 1}/{len(jobs)}: {result['seconds']:.2f}s", flush=True)
    if a.keep_alive:
        print("Kronos forecasts complete; model remains loaded. Send SIGTERM to stop.", flush=True)
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    main()
