"""Validate manual intent examples and create train/validation/test JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.coli import load_config
from src.data.intent import prepare


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/intent.yaml"))
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    result = prepare(ROOT / config.get("raw_dir", "data/raw/intent"), ROOT / config.get("processed_dir", "data/processed/intent"), config)
    report = ROOT / "reports/data_exploration/intent_data.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Report: {report}")
    return 0 if result["status"] != "invalid_annotations" else 1


if __name__ == "__main__":
    raise SystemExit(main())
