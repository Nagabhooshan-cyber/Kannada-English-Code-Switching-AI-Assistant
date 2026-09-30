"""Fine-tune and evaluate the pretrained transformer language-ID model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.coli import load_config
from src.models.language_id_transformer import train_transformer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/language_id_transformer.yaml"))
    args = parser.parse_args()
    metrics = train_transformer(load_config(ROOT / args.config), ROOT)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
