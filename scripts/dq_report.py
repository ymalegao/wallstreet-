"""Audit collected data. --exploratory never claims the full M1 gate passed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ws import sources_config
from ws.config import load_settings
from ws.quality import audit
from ws.store.atomic import atomic_text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exploratory", action="store_true")
    ap.add_argument("--manifest", type=Path, help="research manifest used to bound symbol/date coverage checks")
    ap.add_argument("--output", type=Path, default=Path("docs/data-quality-report.md"))
    args = ap.parse_args()
    result = audit(load_settings().data_dir, sources_config.load(), args.exploratory, args.manifest)
    lines = [
        "# Data quality report",
        "",
        f"**Result: {result['status']}**",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(result["metrics"], indent=2),
        "```",
        "",
        "## Failures",
        "",
    ]
    lines.extend(f"- {x}" for x in result["failures"])
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {x}" for x in result["warnings"])
    atomic_text(args.output, "\n".join(lines) + "\n")
    atomic_text(args.output.with_suffix(".json"), json.dumps(result, indent=2) + "\n")
    print("\n".join(lines))
    return int(bool(result["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
