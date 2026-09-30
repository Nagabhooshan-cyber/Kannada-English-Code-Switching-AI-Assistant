"""Create an empty CSV template for manually annotated normalization pairs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["source", "target_kannada", "target_transliteration", "source_id", "notes"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/normalization/manual_pairs_template.csv"))
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        print(f"Template already exists; left unchanged: {output}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
    print(f"Created empty annotation template: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
