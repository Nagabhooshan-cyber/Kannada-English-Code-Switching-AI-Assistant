"""Prepare CoLI-Kanglish into internal JSONL and write exploration statistics."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.coli import load_config, prepare


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = ROOT
    config = load_config(args.config)
    paths = config.get("paths", {})
    raw_dir = root / paths.get("raw_dir", "data/raw/coli_kanglish")
    output_dir = root / paths.get("processed_dir", "data/processed/language_id")
    report_dir = root / paths.get("exploration_dir", "reports/data_exploration")
    summary = prepare(raw_dir, output_dir, config)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "coli_kanglish_summary.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Exploration report: {report_path}")
    if summary.get("status") == "dataset_missing":
        print("No training examples were written because source data is not available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
