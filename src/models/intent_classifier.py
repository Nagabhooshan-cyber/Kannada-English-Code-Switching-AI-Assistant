"""Word/character TF-IDF logistic-regression intent baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline


def load_split(path: Path) -> tuple[list[str], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing intent split {path}; run scripts/prepare_intent_data.py first.")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Intent split is empty: {path}")
    return [str(row["text"]) for row in rows], [str(row["intent"]) for row in rows]


def train(config: dict[str, Any], root: Path) -> dict[str, Any]:
    """Train on manually labeled data only, then evaluate on validation/test splits."""
    data_dir = root / config.get("processed_dir", "data/processed/intent")
    train_texts, train_labels = load_split(data_dir / "train.jsonl")
    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1, sublinear_tf=True)),
    ])
    model = Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=int(config.get("split", {}).get("seed", 42)))),
    ])
    model.fit(train_texts, train_labels)

    def evaluate(split_name: str) -> dict[str, Any]:
        texts, labels = load_split(data_dir / f"{split_name}.jsonl")
        predictions = model.predict(texts)
        return {
            "examples": len(labels),
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
            "classification_report": classification_report(labels, predictions, output_dict=True, zero_division=0),
        }

    model_path = root / config.get("model_path", "models/intent_baseline.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics = {
        "model": "word and character TF-IDF + class-balanced logistic regression",
        "training_examples": len(train_texts),
        "validation": evaluate("validation"),
        "test": evaluate("test"),
        "model_path": str(model_path),
    }
    metrics_path = root / config.get("metrics_path", "reports/evaluation/intent_baseline.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics["metrics_path"] = str(metrics_path)
    return metrics
