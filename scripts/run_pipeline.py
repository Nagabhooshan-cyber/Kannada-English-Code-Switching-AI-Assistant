"""Run the currently available end-to-end text pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.pipeline import AssistantPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Romanized Kannada/English or Kannada-script text")
    parser.add_argument("--language-backend", choices=("auto", "baseline", "transformer"), default="auto")
    parser.add_argument("--translate", action="store_true", help="Request optional pretrained Kannada/English translation")
    parser.add_argument("--source-language", choices=("kn", "en"), default="kn")
    parser.add_argument("--target-language", choices=("kn", "en"), default="en")
    args = parser.parse_args()
    pipeline = AssistantPipeline(ROOT, language_backend=args.language_backend)
    print(json.dumps(pipeline.process(
        args.text, request_translation=args.translate,
        source_language=args.source_language, target_language=args.target_language,
    ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
