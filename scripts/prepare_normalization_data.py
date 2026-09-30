"""Validate manual normalization pairs and generate task-specific splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.coli import load_config
from src.data.normalization import prepare


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/normalization.yaml"))
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    raw_dir = ROOT / config.get("raw_dir", "data/raw/normalization")
    output_dir = ROOT / config.get("processed_dir", "data/processed/normalization")
    report_path = ROOT / config.get("report_path", "reports/data_exploration/normalization_data.json")
    summary = prepare(raw_dir, output_dir, config.get("split", {}))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {report_path}")
    return 0 if summary["status"] != "invalid_annotations" else 1


if __name__ == "__main__":
    raise SystemExit(main())
