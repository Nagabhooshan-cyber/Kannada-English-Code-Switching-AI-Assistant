"""Extract supported date, time, activity, destination, and topic entities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.entity_extractor import extract_entities


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Romanized Kannada/English or Kannada-script text")
    args = parser.parse_args()
    print(json.dumps({"text": args.text, "entities": extract_entities(args.text)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
