"""Character n-gram TF-IDF and logistic regression baseline for word language ID."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.data.coli import load_file


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read UTF-8 JSONL records from a prepared split."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing prepared split: {path}. Run scripts/prepare_data.py first.")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"Prepared split is empty: {path}")
    return records


def flatten_records(records: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Flatten aligned word/tag sequences into word-level training examples."""
    words: list[str] = []
    labels: list[str] = []
    for index, record in enumerate(records):
        tokens = record.get("tokens")
        tags = record.get("labels")
        if not isinstance(tokens, list) or not isinstance(tags, list) or len(tokens) != len(tags):
            raise ValueError(f"Record {index} must have equal-length token and label lists")
        for token, tag in zip(tokens, tags):
            word, label = str(token).strip(), str(tag).strip()
            if word and label:
                words.append(word)
                labels.append(label)
    if not words:
        raise ValueError("No nonempty labeled tokens found")
    return words, labels


def build_model(config: dict[str, Any]) -> Pipeline:
    """Create the configurable character n-gram baseline pipeline."""
    vectorizer_config = config.get("vectorizer", {})
    classifier_config = config.get("classifier", {})
    vectorizer = TfidfVectorizer(
        analyzer=vectorizer_config.get("analyzer", "char"),
        ngram_range=(int(vectorizer_config.get("ngram_min", 2)), int(vectorizer_config.get("ngram_max", 5))),
        sublinear_tf=bool(vectorizer_config.get("sublinear_tf", True)),
        lowercase=True,
        strip_accents=None,
    )
    classifier = LogisticRegression(
        max_iter=int(classifier_config.get("max_iter", 2000)),
        class_weight=classifier_config.get("class_weight"),
        random_state=int(config.get("seed", 42)),
    )
    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])


def evaluate(model: Pipeline, split_path: Path) -> dict[str, Any]:
    """Evaluate a trained model and return standard class and aggregate metrics."""
    words, labels = flatten_records(load_jsonl(split_path))
    return evaluate_examples(model, words, labels)


def evaluate_records(model: Pipeline, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate a model on normalized records loaded from an external labeled source."""
    words, labels = flatten_records(records)
    return evaluate_examples(model, words, labels)


def evaluate_examples(model: Pipeline, words: list[str], labels: list[str]) -> dict[str, Any]:
    """Calculate standard aggregate and per-class metrics for word examples."""
    predictions = model.predict(words).tolist()
    classes = sorted(set(labels) | set(predictions))
    return {
        "examples": len(words),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "classes": classes,
        "classification_report": classification_report(labels, predictions, labels=classes, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=classes).tolist(),
    }


def train_and_evaluate(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Fit on train only, evaluate validation and derived test, and persist artifacts."""
    data_dir = project_root / config.get("data_dir", "data/processed/language_id")
    train_words, train_labels = flatten_records(load_jsonl(data_dir / "train.jsonl"))
    model = build_model(config)
    model.fit(train_words, train_labels)

    model_path = project_root / config.get("model_path", "models/language_id_baseline.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics = {
        "model": "character n-gram TF-IDF (2-5) + logistic regression",
        "training_examples": len(train_words),
        "training_classes": sorted(set(train_labels)),
        "validation": evaluate(model, data_dir / "validation.jsonl"),
        "held_out_test": evaluate(model, data_dir / "test.jsonl"),
        "test_data_note": (
            "held_out_test is the reproducible 10% record-level split from Train_Kanglish.csv. "
            "official_test, when present, is evaluated separately and never used for training."
        ),
        "model_path": str(model_path),
    }
    official_test_path = config.get("official_test_path")
    if official_test_path:
        external_path = project_root / official_test_path
        if external_path.is_file():
            metrics["official_test"] = evaluate_records(model, load_file(external_path, config.get("columns")))
        else:
            metrics["official_test_note"] = f"Configured official test file not found: {external_path}"
    metrics_path = project_root / config.get("metrics_path", "reports/evaluation/language_id_baseline.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics["metrics_path"] = str(metrics_path)
    return metrics
