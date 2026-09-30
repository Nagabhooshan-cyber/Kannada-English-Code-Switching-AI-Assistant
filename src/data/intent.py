"""Load, validate, deduplicate, and split manually labeled intent utterances."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"text", "intent"}.issubset(set(reader.fieldnames or [])):
                raise ValueError("CSV requires text and intent columns")
            return list(reader)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_intents(raw_dir: Path, allowed_intents: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """Load supported files and reject missing, unknown, or conflicting labels."""
    paths = sorted(path for path in raw_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".csv", ".jsonl", ".ndjson"}) if raw_dir.exists() else []
    records: list[dict[str, Any]] = []
    issues: list[str] = []
    seen: dict[str, str] = {}
    allowed = set(allowed_intents)
    for path in paths:
        try:
            rows = _read_rows(path)
        except (OSError, ValueError, json.JSONDecodeError, csv.Error) as error:
            issues.append(f"{path.name}: {error}")
            continue
        for row_number, row in enumerate(rows, start=2):
            text = "" if row.get("text") is None else str(row.get("text")).strip()
            intent = "" if row.get("intent") is None else str(row.get("intent")).strip()
            if not text and not intent:
                continue
            if not text or not intent:
                issues.append(f"{path.name}:{row_number}: text and intent are required")
                continue
            if intent not in allowed:
                issues.append(f"{path.name}:{row_number}: unsupported intent {intent!r}")
                continue
            key = " ".join(text.casefold().split())
            if key in seen:
                if seen[key] != intent:
                    issues.append(f"{path.name}:{row_number}: conflicting intent for duplicate text {text!r}")
                continue
            seen[key] = intent
            entities: Any = row.get("entities", {})
            if entities is None or (isinstance(entities, str) and not entities.strip()):
                entities = {}
            if isinstance(entities, str) and entities.strip():
                try:
                    entities = json.loads(entities)
                except json.JSONDecodeError:
                    issues.append(f"{path.name}:{row_number}: entities must be valid JSON")
                    continue
            if not isinstance(entities, dict):
                issues.append(f"{path.name}:{row_number}: entities must be a JSON object")
                continue
            records.append({
                "text": text,
                "intent": intent,
                "entities": entities,
                "source_id": str(row.get("source_id") or path.name),
                "source_file": path.name,
            })
    return records, issues


def split_intents(records: list[dict[str, Any]], ratios: dict[str, float], seed: int) -> dict[str, list[dict[str, Any]]]:
    """Split by source when possible, otherwise use deterministic record splitting."""
    names = ("train", "validation", "test")
    if any(float(ratios[name]) <= 0 for name in names) or abs(sum(float(ratios[name]) for name in names) - 1.0) > 1e-6:
        raise ValueError("Split ratios must be positive and sum to 1.0")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["source_id"])].append(record)
    rng = random.Random(seed)
    group_keys = list(groups)
    rng.shuffle(group_keys)
    partitions = {name: [] for name in names}
    desired = {name: len(records) * float(ratios[name]) for name in names}
    for group_key in sorted(group_keys, key=lambda key: len(groups[key]), reverse=True):
        name = min(names, key=lambda item: len(partitions[item]) / max(desired[item], 1))
        partitions[name].extend(groups[group_key])
    if len(records) >= 3 and any(not partitions[name] for name in names):
        shuffled = records[:]
        rng.shuffle(shuffled)
        train_end = max(1, round(len(shuffled) * float(ratios["train"])))
        validation_end = max(train_end + 1, round(len(shuffled) * (float(ratios["train"]) + float(ratios["validation"]))))
        validation_end = min(validation_end, len(shuffled) - 1)
        partitions = {"train": shuffled[:train_end], "validation": shuffled[train_end:validation_end], "test": shuffled[validation_end:]}
    return partitions


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write records as UTF-8 JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare(raw_dir: Path, output_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Validate intent annotations and prepare reproducible splits."""
    records, issues = load_intents(raw_dir, list(config.get("allowed_intents", [])))
    if issues:
        return {"status": "invalid_annotations", "issues": issues, "valid_record_count": len(records)}
    if not records:
        return {"status": "awaiting_manual_annotations", "record_count": 0,
                "message": f"Add human-labeled utterances under {raw_dir}; no synthetic examples are generated."}
    split_config = config.get("split", {})
    ratios = {"train": float(split_config.get("train", .8)), "validation": float(split_config.get("validation", .1)), "test": float(split_config.get("test", .1))}
    partitions = split_intents(records, ratios, int(split_config.get("seed", 42)))
    for name, part in partitions.items():
        write_jsonl(part, output_dir / f"{name}.jsonl")
    return {
        "status": "prepared", "record_count": len(records),
        "class_distribution": {intent: sum(record["intent"] == intent for record in records) for intent in config["allowed_intents"]},
        "split_counts": {name: len(part) for name, part in partitions.items()},
        "split_strategy": "Deduplicate exact normalized text, then group by source_id where feasible; seeded record split is the fallback.",
    }
