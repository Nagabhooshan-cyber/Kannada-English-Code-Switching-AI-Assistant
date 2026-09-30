"""Translate English/Kannada text with optional pretrained IndicTrans2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.coli import load_config
from src.models.translator import IndicTransTranslator, TranslationUnavailableError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Input sentence")
    parser.add_argument("--source-language", choices=("kn", "en"), required=True)
    parser.add_argument("--target-language", choices=("kn", "en"), required=True)
    args = parser.parse_args()
    config = load_config(ROOT / "configs/translation.yaml")
    translator = IndicTransTranslator(ROOT, config)
    try:
        result = {
            "status": "translated", "source_language": args.source_language,
            "target_language": args.target_language,
            "text": translator.translate(args.text, args.source_language, args.target_language),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except TranslationUnavailableError as error:
        print(json.dumps({"status": "unavailable", "message": str(error)}, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
