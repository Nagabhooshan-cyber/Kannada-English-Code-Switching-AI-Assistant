"""Compose available text components into a structured assistant response."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import regex as unicode_regex

from src.inference.entity_extractor import extract_entities
from src.data.coli import load_config
from src.models.normalizer import LexiconNormalizer
from src.models.roman_kannada import RomanKannadaConverter
from src.models.translator import IndicTransTranslator, TranslationUnavailableError
from src.support.ai_support import GeminiSupport
from src.support.decision import decide_support

TOKEN_RE = unicode_regex.compile(
    r"[\w\p{M}]+(?:[-'][\w\p{M}]+)*|[^\w\p{M}\s]",
    flags=unicode_regex.UNICODE,
)
KANNADA_RE = unicode_regex.compile(r"[\u0c80-\u0cff]")
KANNADA_ROMAN_SUFFIXES = ("ige", "alli", "nalli", "inda", "ninda", "annu", "jothe", "mele", "ge", "na")
KNOWN_ENGLISH_BASES = {
    "assignment", "appointment", "class", "exam", "homework", "interview", "meeting",
    "office", "presentation", "project", "report", "school", "shop", "shopping", "task", "work",
}
COMMON_ENGLISH_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can",
    "could", "did", "do", "does", "for", "from", "had", "has", "have", "he", "her",
    "here", "him", "his", "how", "i", "if", "in", "is", "it", "its", "me", "my",
    "no", "not", "of", "on", "or", "our", "please", "she", "should", "so", "that",
    "the", "their", "them", "then", "there", "these", "they", "this", "those", "to",
    "was", "we", "were", "what", "when", "where", "which", "who", "will", "with",
    "would", "you", "your", "remind", "tomorrow", "today", "issue",
}


def _split_kannada_suffix(token: str, start: int) -> list[tuple[str, int, int, bool]]:
    """Separate recognized Romanized Kannada case markers from attached words."""
    folded = token.casefold()
    for suffix in KANNADA_ROMAN_SUFFIXES:
        hyphenated = f"-{suffix}"
        if folded.endswith(hyphenated) and len(token) > len(hyphenated):
            cut = len(token) - len(hyphenated)
            return [(token[:cut], start, start + cut, False),
                    (token[cut + 1:], start + cut + 1, start + len(token), True)]
        if folded.endswith(suffix) and len(token) > len(suffix):
            cut = len(token) - len(suffix)
            if folded[:cut] in KNOWN_ENGLISH_BASES:
                return [(token[:cut], start, start + cut, False),
                        (token[cut:], start + cut, start + len(token), True)]
    return [(token, start, start + len(token), False)]


class AssistantPipeline:
    """Load and run model components once per instance, keeping unavailable parts explicit."""

    def __init__(
        self,
        project_root: Path,
        language_backend: str = "auto",
        max_input_chars: int = 2000,
    ):
        self.project_root = project_root
        self.max_input_chars = max_input_chars
        self.language_backend = "unavailable"
        self.language_model: Any = None
        self.language_tokenizer: Any = None
        self.transformer_device: Any = None
        self.intent_model: Any = None
        self.ai_support = GeminiSupport()
        translation_config_path = project_root / "configs/translation.yaml"
        translation_config = load_config(translation_config_path) if translation_config_path.is_file() else {}
        self.translator = IndicTransTranslator(project_root, translation_config)

        baseline_path = project_root / "models/language_id_baseline.joblib"
        transformer_path = project_root / "models/language_id_transformer"
        if language_backend == "baseline" or (language_backend == "auto" and baseline_path.is_file()):
            if baseline_path.is_file():
                self.language_model = joblib.load(baseline_path)
                self.language_backend = "character_ngram_baseline"
        elif language_backend == "transformer" or (language_backend == "auto" and transformer_path.is_dir()):
            if transformer_path.is_dir():
                self._load_transformer(transformer_path)
                self.language_backend = "muril_transformer"
        elif language_backend != "auto":
            raise ValueError("language_backend must be 'auto', 'baseline', or 'transformer'")

        intent_path = project_root / "models/intent_baseline.joblib"
        if intent_path.is_file():
            self.intent_model = joblib.load(intent_path)
        self.normalizer = LexiconNormalizer.from_directory(project_root / "data/raw/normalization", "normalization")
        self.roman_kannada = RomanKannadaConverter(
            self.normalizer, project_root / "data/processed/language_id/train.jsonl",
        )

    def _load_transformer(self, model_dir: Path) -> None:
        """Load the locally fine-tuned transformer backend on the available device."""
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.language_tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
        self.language_model = AutoModelForTokenClassification.from_pretrained(model_dir, local_files_only=True)
        self.transformer_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.language_model.to(self.transformer_device)
        self.language_model.eval()

    def _language_predictions(self, tokens: list[str]) -> tuple[list[str], list[float]]:
        labels = ["unknown"] * len(tokens)
        confidences = [0.0] * len(tokens)
        eligible: list[int] = []
        for index, token in enumerate(tokens):
            if not any(char.isalnum() for char in token):
                labels[index] = "other"
                confidences[index] = 1.0
            elif token.isnumeric():
                labels[index] = "other"
                confidences[index] = 1.0
            elif KANNADA_RE.search(token):
                labels[index] = "kn"
                confidences[index] = 1.0
            elif token.casefold() in COMMON_ENGLISH_WORDS:
                labels[index] = "en"
                confidences[index] = 0.99
            elif self.language_backend != "unavailable":
                eligible.append(index)
        if not eligible:
            return labels, confidences

        if self.language_backend == "character_ngram_baseline":
            predictions = self.language_model.predict([tokens[index] for index in eligible])
            probabilities = self.language_model.predict_proba([tokens[index] for index in eligible]) if hasattr(self.language_model, "predict_proba") else None
            for position, (index, label) in enumerate(zip(eligible, predictions)):
                labels[index] = str(label)
                confidences[index] = float(max(probabilities[position])) if probabilities is not None else 0.6
            return labels, confidences

        import torch

        encoding = self.language_tokenizer(
            [tokens[index] for index in eligible], is_split_into_words=True,
            truncation=True, max_length=512, return_tensors="pt",
        )
        word_ids = encoding.word_ids(batch_index=0)
        with torch.no_grad():
            output = self.language_model(**{key: value.to(self.transformer_device) for key, value in encoding.items()})
        predictions = output.logits.argmax(dim=-1)[0].tolist()
        probabilities = output.logits.softmax(dim=-1)[0].max(dim=-1).values.tolist()
        id_to_label = self.language_model.config.id2label
        seen: set[int] = set()
        for subword_index, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen or word_id >= len(eligible):
                continue
            seen.add(word_id)
            labels[eligible[word_id]] = str(id_to_label[predictions[subword_index]])
            confidences[eligible[word_id]] = float(probabilities[subword_index])
        return labels, confidences

    @staticmethod
    def _polish_known_english_frame(
        source_text: str, translated_text: str, source_language: str, target_language: str,
    ) -> str:
        """Supply a natural English frame where colloquial Kannada omits the subject."""
        source = source_language.casefold()
        target = target_language.casefold()
        route_question = unicode_regex.fullmatch(
            r"\s*[\w\u0c80-\u0cff]+(?:[- ](?:ಗೆ|ge))\s+(?:ಹೇಗೆ|hege)\s+(?:ಹೋಗ\w*|hog\w*)\s*[?!.]*\s*",
            source_text.strip(), flags=unicode_regex.IGNORECASE,
        )
        if route_question and source in {"kn", "kannada", "kan_knda"} and target in {"en", "english", "eng_latn"}:
            destination = extract_entities(source_text)["destination"]
            if destination:
                return f"How do I get to {destination['value']}?"
        do_suffix = unicode_regex.search(
            r"(?i)(?:^|\s)(?:maadabeku|madabeku|maadbeku|madbeku)\s*[.!?]*$",
            source_text.strip(),
        )
        if source not in {"kn", "kannada", "kan_knda"} or target not in {"en", "english", "eng_latn"} or not do_suffix:
            return translated_text
        activity = extract_entities(source_text)["activity"]
        if not activity:
            return translated_text
        action = {
            "assignment": "completed", "appointment": "scheduled", "class": "held",
            "exam": "taken", "homework": "completed", "interview": "held",
            "meeting": "held", "office": "attended", "presentation": "given",
            "project": "completed", "report": "prepared", "shopping": "done",
            "task": "completed", "work": "done",
        }.get(activity["value"].casefold())
        if not action:
            return translated_text
        date = extract_entities(source_text)["date"]
        when = f" {date['value']}" if date else ""
        article = "An" if activity["value"].casefold().startswith(("appointment", "exam")) else "A"
        return f"{article} {activity['value']} needs to be {action}{when}."

    def process(
        self,
        text: str,
        request_translation: bool = False,
        source_language: str = "kn",
        target_language: str = "en",
    ) -> dict[str, Any]:
        """Run the current pipeline and return structured outputs and availability."""
        started = time.perf_counter()
        if not isinstance(text, str) or not text.strip():
            return {"error": "Enter non-empty text.", "input": text if isinstance(text, str) else None}
        if len(text) > self.max_input_chars:
            return {"error": f"Input exceeds the {self.max_input_chars}-character limit.", "input": text[:self.max_input_chars]}

        matches = list(TOKEN_RE.finditer(text))
        split_tokens: list[tuple[str, int, int, bool]] = []
        for match in matches:
            split_tokens.extend(_split_kannada_suffix(match.group(), match.start()))
        labels, confidences = self._language_predictions([part[0] for part in split_tokens])
        token_output = []
        for (token, start, end, is_suffix), label, confidence in zip(split_tokens, labels, confidences):
            token_output.append({
                "text": token, "language": "kn" if is_suffix else label,
                "start": start, "end": end, "confidence": 1.0 if is_suffix else confidence,
            })
        normalized = self.roman_kannada.convert(text, token_output)
        intent_result: dict[str, Any] | None = None
        if self.intent_model is not None:
            intent = str(self.intent_model.predict([text])[0])
            intent_result = {"label": intent}
            if hasattr(self.intent_model, "predict_proba"):
                probabilities = self.intent_model.predict_proba([text])[0]
                best = int(probabilities.argmax())
                intent_result["confidence"] = float(probabilities[best])
        translation: dict[str, Any] | None = None
        if request_translation:
            try:
                source_text = normalized["text"] if source_language.casefold() in {"kn", "kannada", "kan_knda"} else text
                translated_text = self.translator.translate(
                    text, source_language, target_language, normalized_source=source_text,
                )
                translated_text = self._polish_known_english_frame(
                    text, translated_text, source_language, target_language,
                )
                translation = {
                    "status": "translated", "source_language": source_language,
                    "target_language": target_language, "text": translated_text,
                }
            except TranslationUnavailableError as error:
                translation = {"status": "unavailable", "message": str(error)}

        entities = extract_entities(text)
        uncertainty = decide_support(
            text, token_output, float(normalized.get("coverage", 0)), entities,
            model_available=self.language_backend != "unavailable",
        )
        support: dict[str, Any] | None = None
        applied_language_tokens: set[str] = set()
        applied_entities: set[str] = set()
        if uncertainty["needed"]:
            support_payload = {
                "original_text": text,
                "tokens": uncertainty["tokens"],
                "normalized_text": normalized["text"],
                "intent": intent_result.get("label") if intent_result else None,
                "entities": [value for value in entities.values() if value],
            }
            support = self.ai_support.cross_check(support_payload, uncertainty["web_search"])
            if support.get("status") == "available":
                # Apply only high-confidence label/entity evidence. Gemini has no translation/intent fields.
                support_variants = {
                    str(item.get("token", "")).casefold(): str(item.get("likely_form", ""))
                    for item in support.get("romanization_support", [])
                    if item.get("confidence", 0) >= 0.97
                }
                if support_variants and float(normalized.get("coverage", 0)) < 0.35:
                    variant_result = self.roman_kannada.apply_supported_variants(text, token_output, support_variants)
                    if variant_result["applied"]:
                        normalized["text"] = variant_result["text"]
                        support["applied_romanization"] = variant_result["applied"]
                for item in support.get("token_support", []):
                    if item.get("confidence", 0) >= 0.95 and item.get("suggested_label") in {"kn", "en"}:
                        token = next((t for t in token_output if t["text"].casefold() == item["token"].casefold()), None)
                        if token and token["confidence"] < 0.85:
                            token["language"] = item["suggested_label"]
                            token["ai_supported"] = True
                            applied_language_tokens.add(str(item["token"]).casefold())
                destination = entities.get("destination")
                web_evidence_available = not uncertainty["web_search"] or bool(support.get("sources"))
                for item in support.get("entity_support", []):
                    name = str(item.get("text", ""))
                    if not (web_evidence_available and item.get("confidence", 0) >= 0.95):
                        continue
                    start = text.casefold().find(name.casefold())
                    if start < 0:
                        continue
                    detail = {"value": name, "entity_type": item["entity_type"], "start": start,
                              "end": start + len(name), "confidence": item["confidence"], "ai_supported": True}
                    named_entities = entities.setdefault("named_entities", [])
                    if not any(existing.get("value", "").casefold() == name.casefold()
                               for existing in named_entities):
                        named_entities.append(detail)
                    if item.get("entity_type") == "location" and (
                        destination is None or destination.get("value", "").casefold() == name.casefold()
                    ):
                        entities["destination"] = {"value": name, "raw": name, "start": start,
                                                    "end": start + len(name)}
                    applied_entities.add(name.casefold())
        if support is not None:
            statuses_support = support.get("status", "unavailable")
            if support.get("status") == "available":
                support["token_support"] = [
                    {**item, "applied": str(item.get("token", "")).casefold() in applied_language_tokens}
                    for item in support.get("token_support", [])
                ]
                support["entity_support"] = [
                    {**item, "applied": bool(item.get("confidence", 0) >= 0.95
                                              and str(item.get("text", "")).casefold() in applied_entities)}
                    for item in support.get("entity_support", [])
                ]
            support["triggered_by"] = [
                "uncertain word/entity" if item.get("language") == "other" or item.get("confidence", 1) < 0.72
                else "possible proper noun" for item in uncertainty["tokens"]
            ]
        statuses = {
            "language_id": "available" if self.language_backend != "unavailable" else "model_missing",
            "normalization": "available",
            "intent": "available" if self.intent_model is not None else "awaiting_labeled_data_and_training",
            "entity_extraction": "rules_only_not_yet_evaluated",
            "translation": translation["status"] if request_translation and translation else "not_requested",
            "ai_support": support["status"] if support else "not_needed",
        }
        return {
            "input": text,
            "tokens": token_output,
            "normalized_kannada": normalized["text"],
            "normalization_coverage": normalized["coverage"],
            "spelling_corrections": normalized.get("corrections", []),
            "intent": intent_result,
            "entities": entities,
            "ai_support": support,
            "translation": translation,
            "component_status": statuses,
            "language_model": self.language_backend,
            "processing_time_ms": round((time.perf_counter() - started) * 1000, 2),
        }
