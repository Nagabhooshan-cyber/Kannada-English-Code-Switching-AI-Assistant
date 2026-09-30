"""Load, validate, split, and write manually annotated normalization pairs."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def discover_pair_files(raw_dir: Path) -> list[Path]:
    """Return supported pair files below the configured raw directory."""
    if not raw_dir.exists():
        return []
    return sorted(path for path in raw_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".csv", ".jsonl", ".ndjson"})


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or [])
            if "source" not in fields or not ({"target_kannada", "target_transliteration"} & fields):
                raise ValueError("CSV needs source and at least one target column")
            return list(reader)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def load_pairs(raw_dir: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read human-entered pairs; reject conflicting targets for the same source."""
    pairs: list[dict[str, str]] = []
    issues: list[str] = []
    seen: dict[str, tuple[str, str]] = {}
    for path in discover_pair_files(raw_dir):
        try:
            rows = _read_rows(path)
        except (OSError, json.JSONDecodeError, csv.Error, ValueError) as error:
            issues.append(f"{path.name}: {error}")
            continue
        for line_number, row in enumerate(rows, start=2):
            source = _clean(row.get("source"))
            target_kannada = _clean(row.get("target_kannada"))
            target_transliteration = _clean(row.get("target_transliteration"))
            if not any((source, target_kannada, target_transliteration)):
                continue
            if not source or not (target_kannada or target_transliteration):
                issues.append(f"{path.name}:{line_number}: source and at least one target are required")
                continue
            key = source.casefold()
            targets = (target_kannada, target_transliteration)
            if key in seen:
                if seen[key] != targets:
                    issues.append(f"{path.name}:{line_number}: conflicting target for source {source!r}")
                continue
            seen[key] = targets
            pairs.append({
                "source": source,
                "target_kannada": target_kannada,
                "target_transliteration": target_transliteration,
                "source_id": _clean(row.get("source_id")) or path.name,
                "source_file": path.name,
            })
    return pairs, issues


def split_pairs(pairs: list[dict[str, str]], ratios: dict[str, float], seed: int) -> dict[str, list[dict[str, str]]]:
    """Split by source_id when feasible; fall back to deterministic row splits."""
    names = ("train", "validation", "test")
    if any(float(ratios[name]) <= 0 for name in names) or abs(sum(float(ratios[name]) for name in names) - 1.0) > 1e-6:
        raise ValueError("Split ratios must be positive and sum to 1.0")
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for pair in pairs:
        groups[pair["source_id"]].append(pair)
    rng = random.Random(seed)
    group_names = list(groups)
    rng.shuffle(group_names)
    partitions: dict[str, list[dict[str, str]]] = {name: [] for name in names}
    desired = {name: len(pairs) * float(ratios[name]) for name in names}
    for group_name in sorted(group_names, key=lambda key: len(groups[key]), reverse=True):
        name = min(names, key=lambda item: len(partitions[item]) / max(desired[item], 1))
        partitions[name].extend(groups[group_name])
    if len(pairs) >= 3 and any(not partitions[name] for name in names):
        shuffled = pairs[:]
        rng.shuffle(shuffled)
        train_end = max(1, round(len(shuffled) * float(ratios["train"])))
        validation_end = max(train_end + 1, round(len(shuffled) * (float(ratios["train"]) + float(ratios["validation"]))))
        validation_end = min(validation_end, len(shuffled) - 1)
        partitions = {
            "train": shuffled[:train_end],
            "validation": shuffled[train_end:validation_end],
            "test": shuffled[validation_end:],
        }
    return partitions


def write_jsonl(pairs: list[dict[str, str]], path: Path, target_field: str) -> None:
    """Write only examples that include the requested target field."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for pair in pairs:
            if pair[target_field]:
                item = {"source": pair["source"], "target": pair[target_field], "source_id": pair["source_id"]}
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")


def prepare(raw_dir: Path, output_dir: Path, split_config: dict[str, Any]) -> dict[str, Any]:
    """Prepare independent transliteration and normalization pair splits."""
    pairs, issues = load_pairs(raw_dir)
    if issues:
        return {"status": "invalid_annotations", "issues": issues, "pair_count": len(pairs)}
    if not pairs:
        return {
            "status": "awaiting_manual_annotations", "pair_count": 0,
            "message": f"Fill the CSV template or add annotated JSONL pairs under {raw_dir}. No synthetic examples are generated.",
        }
    ratios = {"train": float(split_config.get("train", 0.8)), "validation": float(split_config.get("validation", 0.1)), "test": float(split_config.get("test", 0.1))}
    seed = int(split_config.get("seed", 42))
    results: dict[str, Any] = {"status": "prepared", "pair_count": len(pairs), "splits": {}}
    for task, target_field in (("normalization", "target_kannada"), ("transliteration", "target_transliteration")):
        task_pairs = [pair for pair in pairs if pair[target_field]]
        if len(task_pairs) < 3:
            results["splits"][task] = {"pair_count": len(task_pairs), "status": "needs_at_least_3_pairs"}
            continue
        partitions = split_pairs(task_pairs, ratios, seed)
        for split_name, records in partitions.items():
            write_jsonl(records, output_dir / task / f"{split_name}.jsonl", target_field)
        results["splits"][task] = {"pair_count": len(task_pairs), "counts": {name: len(value) for name, value in partitions.items()}}
    return results
