"""Convert Romanized Kannada words to Kannada script while preserving English words."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.normalizer import LexiconNormalizer
from src.models.roman_kannada import RomanKannadaConverter
from src.inference.pipeline import AssistantPipeline, TOKEN_RE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Romanized Kannada/English text")
    parser.add_argument("--mode", choices=("normalization", "transliteration"), default="normalization")
    parser.add_argument("--pairs-dir", type=Path, default=Path("data/raw/normalization"))
    args = parser.parse_args()
    pairs_dir = args.pairs_dir if args.pairs_dir.is_absolute() else ROOT / args.pairs_dir
    lexicon = LexiconNormalizer.from_directory(pairs_dir, args.mode)
    pipeline = AssistantPipeline(ROOT)
    matches = list(TOKEN_RE.finditer(args.text))
    tokens = [
        {"text": match.group(), "language": label, "start": match.start(), "end": match.end()}
        for match, label in zip(matches, pipeline._language_labels([match.group() for match in matches]))
    ]
    result = RomanKannadaConverter(
        lexicon, ROOT / "data/processed/language_id/train.jsonl",
    ).convert(args.text, tokens)
    result.update({"mode": args.mode, "note": "Unique one-edit typos are matched against Kannada-labeled training words; ambiguous inputs are preserved."})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
