from __future__ import annotations

import json
import sys
import types
from typing import Any

import pytest

from src.support.ai_support import GeminiSupport, _CACHE
from src.support.decision import decide_support
from src.support.schemas import validate_support


@pytest.mark.parametrize("sentence", [
    "Majestic-ge hege hogodu?", "Nale meeting-ge hogbeku", "Nale hgabeku",
    "Google-ge hogbeku", "Amazon alli order madidini", "NMIT-ge hogbeku",
    "Python kalibeku", "Bengaluru-ge hogbeku", "Xyloquartz-ge hogbeku",
    "ನಾನು ಮನೆಗೆ ಹೋಗುತ್ತೇನೆ", "I will go to office", "Nale class attend madbeku",
    "NMIT-ge hogbeku", "Google-ge hogbeku", "Majestic-ge hege hogodu?",
    "Nale hgabeku", "Python kalibeku",
])
def test_project_examples_produce_a_bounded_support_decision(sentence: str) -> None:
    # Decision tests never call Gemini; they only check trigger selection and limits.
    tokens = [{"text": word, "language": "unknown", "confidence": 0.0}
              for word in sentence.replace("?", "").split()]
    decision = decide_support(sentence, tokens, 0.0, {})
    assert isinstance(decision["needed"], bool)
    assert len(decision["tokens"]) <= 8


def test_confident_local_result_skips_gemini() -> None:
    decision = decide_support("meeting hogbeku", [
        {"text": "meeting", "language": "en", "confidence": .99},
        {"text": "hogbeku", "language": "kn", "confidence": .98},
    ], .95, {})
    assert decision == {"needed": False, "tokens": [], "web_search": False}


def test_uncertain_proper_noun_requests_grounding() -> None:
    decision = decide_support("Majestic-ge hege hogodu?", [
        {"text": "Majestic", "language": "other", "confidence": .4},
        {"text": "ge", "language": "kn", "confidence": 1.0},
        {"text": "hege", "language": "kn", "confidence": .95},
    ], .5, {})
    assert decision["needed"] and decision["web_search"]
    assert [item["text"] for item in decision["tokens"]] == ["Majestic"]


def test_romanization_variant_requests_language_support_without_web() -> None:
    decision = decide_support("Nale hgabeku", [
        {"text": "Nale", "language": "unknown", "confidence": 0.0},
        {"text": "hgabeku", "language": "unknown", "confidence": 0.0},
    ], 0.0, {}, model_available=False)
    assert decision["needed"]
    assert [item["text"] for item in decision["tokens"]] == ["hgabeku"]
    assert decision["web_search"] is False


def test_high_confidence_romanization_hint_uses_local_transliterator() -> None:
    from pathlib import Path
    from src.models.normalizer import LexiconNormalizer
    from src.models.roman_kannada import RomanKannadaConverter

    converter = RomanKannadaConverter(LexiconNormalizer([], "normalization"))
    result = converter.apply_supported_variants("hgabeku", [
        {"text": "hgabeku", "start": 0, "end": 7, "confidence": 0.2},
    ], {"hgabeku": "hogbeku"})
    assert result["applied"]
    assert "ಬೇಕು" in result["text"]


def test_missing_key_degrades_safely() -> None:
    result = GeminiSupport(api_key="").cross_check({"tokens": []}, False)
    assert result["status"] == "unavailable"
    assert not result["token_support"]


def test_invalid_json_is_rejected() -> None:
    with pytest.raises(ValueError):
        validate_support("not-json", {"unknown"}, False)


def test_support_rejects_unrequested_token_and_invalid_confidence() -> None:
    result = validate_support({
        "support_needed": True, "overall_confidence": 1.0,
        "token_support": [
            {"token": "Majestic", "original_label": "other", "suggested_label": "location",
             "confidence": .99, "reason": "candidate entity", "web_checked": True},
            {"token": "invented", "original_label": "other", "suggested_label": "en",
             "confidence": .99, "reason": "", "web_checked": False},
        ],
        "entity_support": [{"text": "Majestic", "entity_type": "location", "confidence": .99}],
        "romanization_support": [], "normalization_support": {"suggestion": "", "confidence": 0},
        "pipeline_notes": [],
    }, {"majestic"}, False)
    assert len(result["token_support"]) == 1
    assert result["token_support"][0]["web_checked"] is False


def test_api_and_search_failures_do_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    # Fake SDK ensures the failure path is exercised without contacting Google.
    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")

    class BadModels:
        def generate_content(self, **kwargs: Any) -> Any:
            raise RuntimeError("offline")

    class Client:
        def __init__(self, **kwargs: Any):
            self.models = BadModels()

    genai.Client = Client  # type: ignore[attr-defined]
    genai.types = types.SimpleNamespace(
        Tool=lambda **kwargs: kwargs,
        GoogleSearch=lambda: object(),
        GenerateContentConfig=lambda **kwargs: kwargs,
    )
    google.genai = genai  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    _CACHE.clear()
    result = GeminiSupport(api_key="fake").cross_check({"tokens": [{"text": "NMIT"}]}, True)
    assert result["status"] == "unavailable"


def test_cache_avoids_duplicate_request(monkeypatch: pytest.MonkeyPatch) -> None:
    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    calls = {"count": 0}

    class Response:
        text = json.dumps({
            "support_needed": True, "overall_confidence": .97,
            "token_support": [{"token": "NMIT", "original_label": "other", "suggested_label": "organization",
                               "confidence": .97, "reason": "institution", "web_checked": False}],
            "entity_support": [{"text": "NMIT", "entity_type": "institution", "confidence": .97}],
            "romanization_support": [], "normalization_support": {"suggestion": "", "confidence": 0},
            "pipeline_notes": ["Likely an institution."],
        })
        candidates = []

    class Models:
        def generate_content(self, **kwargs: Any) -> Any:
            calls["count"] += 1
            return Response()

    class Client:
        def __init__(self, **kwargs: Any): self.models = Models()

    genai.Client = Client  # type: ignore[attr-defined]
    genai.types = types.SimpleNamespace(Tool=lambda **kwargs: kwargs,
                                       GoogleSearch=lambda: object(),
                                       GenerateContentConfig=lambda **kwargs: kwargs)
    google.genai = genai  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    _CACHE.clear()
    service = GeminiSupport(api_key="fake")
    payload = {"original_text": "NMIT-ge hogbeku", "tokens": [{"text": "NMIT", "language": "other"}]}
    first = service.cross_check(payload, True)
    second = service.cross_check(payload, True)
    assert first["status"] == second["status"] == "available"
    assert second["cached"] is True and calls["count"] == 1
