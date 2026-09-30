"""Fine-tune a pretrained multilingual token classifier for word-level language ID."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.data.coli import load_file
from src.models.language_id_baseline import flatten_records, load_jsonl


class WordTagDataset(Dataset):
    """Tokenize word/tag sequences while assigning labels to first subwords."""

    def __init__(self, records: list[dict[str, Any]], tokenizer: Any, label_to_id: dict[str, int], max_length: int):
        self.features: list[dict[str, list[int]]] = []
        for index, record in enumerate(records):
            words = record.get("tokens")
            labels = record.get("labels")
            if not isinstance(words, list) or not isinstance(labels, list) or len(words) != len(labels):
                raise ValueError(f"Record {index} must contain equal-length tokens and labels")
            if not words:
                continue
            unknown = sorted(set(str(label) for label in labels) - set(label_to_id))
            if unknown:
                raise ValueError(f"Labels in split absent from training set: {unknown}")
            encoded = tokenizer(words, is_split_into_words=True, truncation=True, max_length=max_length)
            word_ids = encoded.word_ids()
            aligned: list[int] = []
            previous_word_id = None
            for word_id in word_ids:
                if word_id is None or word_id == previous_word_id:
                    aligned.append(-100)
                else:
                    aligned.append(label_to_id[str(labels[word_id])])
                previous_word_id = word_id
            self.features.append({**encoded, "labels": aligned})

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.features[index]


def make_collator(tokenizer: Any):
    """Create dynamic padding collator, using -100 for padded target positions."""
    def collate(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        labels = [item["labels"] for item in batch]
        model_inputs = [{key: value for key, value in item.items() if key != "labels"} for item in batch]
        padded = tokenizer.pad(model_inputs, padding=True, return_tensors="pt")
        target_length = padded["input_ids"].shape[1]
        padded_labels = torch.full((len(batch), target_length), -100, dtype=torch.long)
        for index, sequence in enumerate(labels):
            start = target_length - len(sequence) if tokenizer.padding_side == "left" else 0
            padded_labels[index, start:start + len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        padded["labels"] = padded_labels
        return padded
    return collate


def _batch_words(logits: torch.Tensor, labels: torch.Tensor, id_to_label: dict[int, str]) -> tuple[list[str], list[str]]:
    predictions: list[str] = []
    references: list[str] = []
    for prediction_row, label_row in zip(logits.argmax(dim=-1), labels):
        for prediction, reference in zip(prediction_row.tolist(), label_row.tolist()):
            if reference != -100:
                predictions.append(id_to_label[prediction])
                references.append(id_to_label[reference])
    return predictions, references


def _metrics(references: list[str], predictions: list[str]) -> dict[str, Any]:
    classes = sorted(set(references) | set(predictions))
    return {
        "examples": len(references),
        "accuracy": float(accuracy_score(references, predictions)),
        "macro_precision": float(precision_score(references, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(references, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(references, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(references, predictions, average="weighted", zero_division=0)),
        "classes": classes,
        "classification_report": classification_report(references, predictions, labels=classes, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(references, predictions, labels=classes).tolist(),
    }


def _evaluate(model: Any, loader: DataLoader, device: torch.device, id_to_label: dict[int, str]) -> dict[str, Any]:
    model.eval()
    predictions: list[str] = []
    references: list[str] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**{key: value for key, value in batch.items() if key != "labels"})
            batch_predictions, batch_references = _batch_words(output.logits.cpu(), batch["labels"].cpu(), id_to_label)
            predictions.extend(batch_predictions)
            references.extend(batch_references)
    return _metrics(references, predictions)


def train_transformer(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Fine-tune configured checkpoint and evaluate it on prepared and official data."""
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_dir = project_root / config.get("data_dir", "data/processed/language_id")
    train_records = load_jsonl(data_dir / "train.jsonl")
    train_words, train_labels = flatten_records(train_records)
    label_list = sorted(set(train_labels))
    label_to_id = {label: index for index, label in enumerate(label_list)}
    id_to_label = {index: label for label, index in label_to_id.items()}

    checkpoint = str(config.get("model_name", "google/muril-base-cased"))
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError(f"Checkpoint {checkpoint!r} requires a fast tokenizer for word alignment")
    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint, num_labels=len(label_list), id2label=id_to_label, label2id=label_to_id,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    collate = make_collator(tokenizer)
    max_length = int(config.get("max_length", 64))
    batch_size = int(config.get("batch_size", 16))
    train_dataset = WordTagDataset(train_records, tokenizer, label_to_id, max_length)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, generator=generator)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.get("learning_rate", 2e-5)),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )

    epoch_losses: list[float] = []
    for _epoch in range(int(config.get("epochs", 2))):
        model.train()
        loss_total = 0.0
        batches = 0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1.0)))
            optimizer.step()
            loss_total += float(output.loss.detach().cpu())
            batches += 1
        epoch_losses.append(loss_total / max(batches, 1))

    def make_loader(records: list[dict[str, Any]]) -> DataLoader:
        dataset = WordTagDataset(records, tokenizer, label_to_id, max_length)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    model.eval()
    metrics: dict[str, Any] = {
        "model": checkpoint,
        "device": str(device),
        "epochs": int(config.get("epochs", 2)),
        "epoch_mean_training_loss": epoch_losses,
        "training_examples": len(train_words),
        "labels": label_list,
        "validation": _evaluate(model, make_loader(load_jsonl(data_dir / "validation.jsonl")), device, id_to_label),
        "held_out_test": _evaluate(model, make_loader(load_jsonl(data_dir / "test.jsonl")), device, id_to_label),
        "test_data_note": "held_out_test is the derived record-level split; official test remains a separate evaluation only.",
    }
    external_test_path = config.get("official_test_path")
    if external_test_path:
        external_path = project_root / external_test_path
        if external_path.is_file():
            external_records = load_file(external_path)
            metrics["official_test"] = _evaluate(model, make_loader(external_records), device, id_to_label)
        else:
            metrics["official_test_note"] = f"Configured official test file not found: {external_path}"

    output_dir = project_root / config.get("output_dir", "models/language_id_transformer")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metrics["model_path"] = str(output_dir)
    metrics_path = project_root / config.get("metrics_path", "reports/evaluation/language_id_transformer.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics["metrics_path"] = str(metrics_path)
    return metrics
