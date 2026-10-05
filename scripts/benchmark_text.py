"""Zero-shot sentiment checks on pinned public datasets. No fitting or prompt search."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import polars as pl

from ws.metrics import classification
from ws.models.text import LABELS, TextModel
from ws.store.atomic import atomic_text

DATASETS = {
    "phrasebank": (
        "takala/financial_phrasebank",
        "8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964",
        "data/FinancialPhraseBank-v1.0.zip",
    ),
    "fiqa": (
        "TheFinAI/fiqa-sentiment-classification",
        "b669064ceb406dfc33a86a8ddce203b7395a0fa1",
        "data/test-00000-of-00001-0fb9f3a47c7d0fce.parquet",
    ),
}


def dataset(name: str, root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repo, revision, file = DATASETS[name]
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{file}"
    dest = root / name / Path(file).name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = httpx.get(url, follow_redirects=True, timeout=60)
        r.raise_for_status()
        dest.write_bytes(r.content)
    if name == "phrasebank":
        archive = zipfile.ZipFile(io.BytesIO(dest.read_bytes()))
        member = next(x for x in archive.namelist() if x.endswith("Sentences_AllAgree.txt"))
        rows = []
        for i, line in enumerate(archive.read(member).decode("latin-1").splitlines()):
            text, label = line.rsplit("@", 1)
            rows.append({"id": str(i), "text": text, "gold": label.strip(), "target": ""})
        split = "allagree; original dataset has no held-out split"
    else:
        rows = [
            {
                "id": r["_id"],
                "text": r["sentence"],
                "target": r["target"],
                "gold": "positive" if r["score"] > 0.1 else "negative" if r["score"] < -0.1 else "neutral",
            }
            for r in pl.read_parquet(dest).iter_rows(named=True)
        ]
        split = "test; continuous scores discretized with fixed +/-0.1 thresholds"
    return rows, {
        "repo": repo,
        "revision": revision,
        "file": file,
        "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        "split": split,
        "total_rows": len(rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000")
    ap.add_argument("--kind", choices=["chat", "jev"], default="chat")
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--revision", default="local-unpinned")
    ap.add_argument("--limit", type=int, default=256)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--output", type=Path, default=Path("docs/qwen-benchmark.json"))
    a = ap.parse_args()
    model = TextModel(a.endpoint, a.model, Path("data/inference-cache"), a.kind, a.model_dir, a.revision)
    report: dict[str, Any] = {
        "model": a.model,
        "revision": a.revision,
        "mode": "zero-shot, no fine-tuning",
        "seed": 42,
        "limitations": [
            "Public datasets may have been in model training; these are capability checks, not clean generalization.",
            "Sentiment accuracy is not a trading profitability measurement.",
            "PhraseBank is CC-BY-NC-SA-3.0; research use only. FiQA mirror declares MIT.",
        ],
        "datasets": {},
    }
    for name in DATASETS:
        rows, provenance = dataset(name, Path("data/benchmarks"))
        random.Random(42).shuffle(rows)
        if a.limit:
            rows = rows[: a.limit]

        def predict(row):
            try:
                pred = model.sentiment(row["text"], row["target"])
                return {"id": row["id"], "gold": row["gold"], **pred}
            except Exception as exc:
                return {"id": row["id"], "gold": row["gold"], "label": "ERROR", "error": type(exc).__name__}

        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            predictions = []
            for i, pred in enumerate(pool.map(predict, rows)):
                predictions.append(pred)
                if (i + 1) % 50 == 0:
                    print(f"{name}: {i + 1}/{len(rows)}", flush=True)
        metrics = classification([p["gold"] for p in predictions], [p["label"] for p in predictions], LABELS)
        metrics["errors"] = sum(p["label"] == "ERROR" for p in predictions)
        report["datasets"][name] = {"provenance": provenance, "metrics": metrics, "predictions": predictions}
        atomic_text(a.output, json.dumps(report, indent=2) + "\n")
        print(name, json.dumps(metrics), flush=True)
    lines = [
        "# Zero-shot financial-text benchmark",
        "",
        f"Model: `{a.model}`. Revision: `{a.revision}`.",
        "",
        "| Dataset | N | Accuracy | Macro F1 | Majority accuracy | Errors |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in report["datasets"].items():
        m = result["metrics"]
        lines.append(
            f"| {name} | {m['n']} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} "
            f"| {m['majority_class_accuracy']:.3f} | {m['errors']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        *report["limitations"],
        "",
        "FiQA uses its test split with fixed ±0.1 score thresholds. PhraseBank uses the all-agree set.",
        "Subset selection uses seed 42. No threshold/prompt was tuned on these evaluation labels.",
    ]
    atomic_text(a.output.with_suffix(".md"), "\n".join(lines) + "\n")
    if any(d["metrics"]["errors"] for d in report["datasets"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
