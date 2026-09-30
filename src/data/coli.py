"""Read and prepare CoLI-Kanglish word-level language-ID data."""

from __future__ import annotations

import csv
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)
SUPPORTED_SUFFIXES = {".csv", ".tsv", ".json", ".jsonl", ".ndjson"}
TOKEN_COLUMNS = ("tokens", "token", "words", "word")
LABEL_COLUMNS = ("labels", "label", "tags", "tag", "language", "lang")
TEXT_COLUMNS = ("text", "sentence", "comment", "utterance")
SOURCE_COLUMNS = ("source_id", "source", "thread_id", "document_id", "speaker_id")
LABEL_ALIASES = {
    "en-kn": "mixed", "kn-en": "mixed", "kannada-english": "mixed",
    "english-kannada": "mixed", "mixed-language": "mixed",
}


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML configuration and resolve no paths (paths are relative to project root)."""
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def discover_files(raw_dir: Path) -> list[Path]:
    """Find supported source files recursively, leaving raw data untouched."""
    if not raw_dir.exists():
        return []
    return sorted(p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)


def _read_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if suffix == ".json":
        try:
            frame = pd.read_json(path, lines=True, dtype=False)
        except ValueError:
            frame = pd.read_json(path, dtype=False)
        if isinstance(frame, pd.Series):
            frame = frame.to_frame().T
        return frame
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(records)


def _select_column(columns: list[str], configured: str | None, candidates: tuple[str, ...]) -> str | None:
    if configured:
        if configured not in columns:
            raise ValueError(f"Configured column {configured!r} not found; available: {columns}")
        return configured
    by_lower = {column.lower(): column for column in columns}
    return next((by_lower[name] for name in candidates if name in by_lower), None)


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    # Common serialized list formats are accepted without evaluating input.
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed]
        except json.JSONDecodeError:
            pass
    return text.split()


def normalize_frame(frame: pd.DataFrame, source_file: Path, column_config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Convert supported sentence-level or token-row formats to internal records."""
    config = column_config or {}
    columns = [str(column) for column in frame.columns]
    token_col = _select_column(columns, config.get("tokens"), TOKEN_COLUMNS)
    label_col = _select_column(columns, config.get("labels"), LABEL_COLUMNS)
    text_col = _select_column(columns, config.get("text"), TEXT_COLUMNS)
    source_col = _select_column(columns, config.get("source_id"), SOURCE_COLUMNS)
    records: list[dict[str, Any]] = []

    if token_col and label_col:
        # A row may contain aligned token and label lists, or one token/label pair.
        for row_number, row in frame.iterrows():
            tokens = _as_list(row[token_col])
            labels = _as_list(row[label_col])
            labels = [LABEL_ALIASES.get(label.casefold(), label.casefold()) for label in labels]
            if not tokens and not labels:
                continue
            if len(tokens) != len(labels):
                raise ValueError(f"{source_file}:{row_number + 2}: {len(tokens)} tokens but {len(labels)} labels")
            if not tokens:
                continue
            records.append({"tokens": tokens, "labels": labels,
                            "source_id": str(row[source_col]) if source_col and row[source_col] else source_file.name,
                            "source_file": source_file.name})
    elif text_col and label_col:
        # Sentence text plus whitespace-aligned labels.
        for row_number, row in frame.iterrows():
            tokens = str(row[text_col]).split()
            labels = _as_list(row[label_col])
            labels = [LABEL_ALIASES.get(label.casefold(), label.casefold()) for label in labels]
            if not tokens and not labels:
                continue
            if len(tokens) != len(labels):
                raise ValueError(f"{source_file}:{row_number + 2}: text has {len(tokens)} tokens but {len(labels)} labels")
            records.append({"tokens": tokens, "labels": labels,
                            "source_id": str(row[source_col]) if source_col and row[source_col] else source_file.name,
                            "source_file": source_file.name})
    elif token_col and label_col and len(frame.columns) >= 2:
        # Reserved for clarity if source schema support grows.
        raise ValueError(f"Could not interpret token/label columns in {source_file}")
    else:
        raise ValueError(
            f"Unsupported columns in {source_file}: {columns}. Expected token and label columns "
            "(e.g. tokens/labels) or sentence text with aligned labels. Set configs/data.yaml column names."
        )

    return [record for record in records if record["tokens"]]


def load_records(raw_dir: Path, column_config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Discover all supported files and normalize them into sentence records."""
    files = discover_files(raw_dir)
    if not files:
        return []
    records: list[dict[str, Any]] = []
    for path in files:
        records.extend(normalize_frame(_read_file(path), path, column_config))
    return records


def load_file(path: Path, column_config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Load and normalize one explicitly selected labeled data file."""
    return normalize_frame(_read_file(path), path, column_config)


def validate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate aligned word-level sequences and summarize data quality."""
    issues: list[str] = []
    for index, record in enumerate(records):
        tokens, labels = record.get("tokens"), record.get("labels")
        if not isinstance(tokens, list) or not isinstance(labels, list) or len(tokens) != len(labels):
            issues.append(f"record {index}: tokens and labels must be equal-length lists")
        elif any(not str(token).strip() for token in tokens) or any(not str(label).strip() for label in labels):
            issues.append(f"record {index}: blank token or label")
    return {"valid": not issues, "record_count": len(records), "issues": issues}


def split_records(records: list[dict[str, Any]], ratios: dict[str, float], seed: int) -> dict[str, list[dict[str, Any]]]:
    """Deduplicate exact token/label sequences, then split by source when feasible."""
    total = sum(float(ratios[name]) for name in ("train", "validation", "test"))
    if any(float(ratios[name]) <= 0 for name in ("train", "validation", "test")) or abs(total - 1.0) > 1e-6:
        raise ValueError("Split ratios train/validation/test must be positive and sum to 1.0")
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    for record in records:
        key = (tuple(t.casefold() for t in record["tokens"]), tuple(l.casefold() for l in record["labels"]))
        if key not in seen:
            seen.add(key)
            deduped.append(record)

    groups: dict[str, list[dict[str, Any]]] = {}
    for record in deduped:
        groups.setdefault(str(record.get("source_id") or record.get("source_file") or "unknown"), []).append(record)
    rng = random.Random(seed)
    group_keys = list(groups)
    rng.shuffle(group_keys)
    counts = {key: 0 for key in ("train", "validation", "test")}
    partitions = {key: [] for key in counts}
    desired = {key: len(deduped) * float(ratios[key]) for key in counts}
    for source in sorted(group_keys, key=lambda key: len(groups[key]), reverse=True):
        target = min(counts, key=lambda key: counts[key] / max(desired[key], 1))
        partitions[target].extend(groups[source])
        counts[target] += len(groups[source])
    # If one source owns all examples, source-level isolation would empty splits.
    if len(deduped) >= 3 and any(not partitions[name] for name in partitions):
        LOGGER.warning("Source grouping cannot populate every split; falling back to seeded record split.")
        shuffled = deduped[:]
        rng.shuffle(shuffled)
        train_end = max(1, round(len(shuffled) * float(ratios["train"])))
        valid_end = max(train_end + 1, round(len(shuffled) * (float(ratios["train"]) + float(ratios["validation"]))))
        valid_end = min(valid_end, len(shuffled) - 1)
        partitions = {"train": shuffled[:train_end], "validation": shuffled[train_end:valid_end], "test": shuffled[valid_end:]}
    return partitions


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write records as UTF-8 JSON Lines."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def explore_records(records: list[dict[str, Any]], similarity_threshold: float = 0.90) -> dict[str, Any]:
    """Calculate dataset and token-label statistics without training a model."""
    from difflib import SequenceMatcher

    label_counts: Counter[str] = Counter()
    token_counts: Counter[str] = Counter()
    punctuation = Counter()
    sentence_keys: list[str] = []
    token_label_counts: dict[str, Counter[str]] = {}
    for record in records:
        tokens = record["tokens"]
        sentence_keys.append(" ".join(token.casefold() for token in tokens))
        for token, label in zip(tokens, record["labels"]):
            label = label.casefold()
            token_counts[token.casefold()] += 1
            label_counts[label] += 1
            token_label_counts.setdefault(label, Counter())[token.casefold()] += 1
            if any(char in "!?.,;:()[]{}\"'" for char in token):
                punctuation["tokens_with_punctuation"] += 1
            for char in token:
                if char in "!?.,;:()[]{}\"'":
                    punctuation[char] += 1
    duplicates = len(sentence_keys) - len(set(sentence_keys))
    # Compare only records with matching first-token buckets to avoid all-pairs work on large corpora.
    buckets: dict[str, list[str]] = {}
    for sentence in set(sentence_keys):
        first = sentence.split(maxsplit=1)[0][:3] if sentence else ""
        buckets.setdefault(first, []).append(sentence)
    near_pairs = 0
    for bucket in buckets.values():
        if len(bucket) > 300:
            bucket = sorted(bucket)[:300]
        for i, left in enumerate(bucket):
            for right in bucket[i + 1:]:
                if SequenceMatcher(None, left, right).ratio() >= similarity_threshold:
                    near_pairs += 1
    return {
        "record_count": len(records), "token_count": sum(map(len, (r["tokens"] for r in records))),
        "class_distribution": dict(label_counts.most_common()),
        "token_distribution": dict(token_counts.most_common(50)),
        "common_tokens_by_label": {label: dict(counts.most_common(30)) for label, counts in token_label_counts.items()},
        "mixed_token_frequency": label_counts.get("mixed", 0),
        "punctuation_statistics": dict(punctuation),
        "exact_duplicate_count": duplicates, "near_duplicate_pairs_sampled": near_pairs,
        "near_duplicate_similarity_threshold": similarity_threshold,
        "spelling_variants_sampled": _spelling_variants(token_counts),
        "note": "Labels are reported as found in source data; no sentiment/offensive labels are remapped.",
    }


def _spelling_variants(counts: Counter[str]) -> dict[str, list[str]]:
    """Report likely Romanized variants for a small set of common forms, if present."""
    targets = ("nanu", "naanu", "naan", "hogbeku", "hogbek", "hogbekaa")
    present = [word for word in targets if word in counts]
    return {"observed_forms": present, "counts": {word: counts[word] for word in present}}


def prepare(raw_dir: Path, output_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Load, validate, deduplicate, split, write processed files, and return exploration."""
    columns = config.get("columns", {})
    records = load_records(raw_dir, columns)
    report = validate_records(records)
    if not report["valid"]:
        raise ValueError("Invalid dataset: " + "; ".join(report["issues"][:10]))
    if not records:
        return {"status": "dataset_missing", "raw_dir": str(raw_dir), "discovered_files": [],
                "message": f"Place licensed CoLI-Kanglish files in {raw_dir} to prepare data. No dataset was fabricated."}
    split_config = config.get("split", {})
    partitions = split_records(records, split_config or {"train": .8, "validation": .1, "test": .1}, int(split_config.get("seed", 42)))
    for name, partition in partitions.items():
        write_jsonl(partition, output_dir / f"{name}.jsonl")
    deduplicated_count = sum(map(len, partitions.values()))
    exploration = explore_records(records, float(config.get("near_duplicate_similarity", .90)))
    exploration.update({"status": "prepared", "raw_record_count": len(records), "deduplicated_record_count": deduplicated_count,
                        "split_record_counts": {name: len(part) for name, part in partitions.items()},
                        "split_strategy": "Exact sentence/label deduplication followed by source_id/source_file grouping; falls back to seeded record split when grouping cannot populate each split."})
    return exploration
