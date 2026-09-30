"""Validation for Gemini's supporting-only JSON response."""

from __future__ import annotations

from typing import Any

SUPPORT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "support_needed": {"type": "BOOLEAN"},
        "overall_confidence": {"type": "NUMBER"},
        "token_support": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "token": {"type": "STRING"}, "original_label": {"type": "STRING"},
            "suggested_label": {"type": "STRING"}, "confidence": {"type": "NUMBER"},
            "reason": {"type": "STRING"}, "web_checked": {"type": "BOOLEAN"},
        }, "required": ["token", "original_label", "suggested_label", "confidence", "reason", "web_checked"]}},
        "entity_support": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "text": {"type": "STRING"}, "entity_type": {"type": "STRING"},
            "confidence": {"type": "NUMBER"},
        }, "required": ["text", "entity_type", "confidence"]}},
        "romanization_support": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "token": {"type": "STRING"}, "likely_form": {"type": "STRING"},
            "confidence": {"type": "NUMBER"},
        }, "required": ["token", "likely_form", "confidence"]}},
        "normalization_support": {"type": "OBJECT", "properties": {
            "suggestion": {"type": "STRING"}, "confidence": {"type": "NUMBER"},
        }, "required": ["suggestion", "confidence"]},
        "pipeline_notes": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["support_needed", "overall_confidence", "token_support", "entity_support",
                 "romanization_support", "normalization_support", "pipeline_notes"],
}


def validate_support(data: Any, candidate_tokens: set[str], web_used: bool) -> dict[str, Any]:
    """Drop malformed or unsupported entries; never accept free-form answer text."""
    if not isinstance(data, dict):
        raise ValueError("Gemini support response must be a JSON object")

    def confidence(item: dict[str, Any]) -> bool:
        value = item.get("confidence")
        return isinstance(value, (int, float)) and 0 <= value <= 1

    tokens = []
    for item in data.get("token_support", []):
        if (isinstance(item, dict) and str(item.get("token", "")).casefold() in candidate_tokens
                and item.get("suggested_label") in {"kn", "en", "other", "location", "person", "organization", "brand", "technical_term", "romanized_kannada"}
                and confidence(item)):
            tokens.append({**item, "web_checked": bool(item.get("web_checked")) and web_used})
    entities = []
    allowed_types = {"location", "person", "organization", "brand", "institution", "technical_term"}
    for item in data.get("entity_support", []):
        if (isinstance(item, dict) and str(item.get("text", "")).casefold() in candidate_tokens
                and str(item.get("entity_type", "")).casefold() in allowed_types and confidence(item)):
            entities.append(item)
    romans = [item for item in data.get("romanization_support", []) if isinstance(item, dict)
              and str(item.get("token", "")).casefold() in candidate_tokens and confidence(item)]
    normalization = data.get("normalization_support", {})
    if not isinstance(normalization, dict) or not confidence(normalization):
        normalization = {"suggestion": "", "confidence": 0.0}
    notes = data.get("pipeline_notes", [])
    needed = bool(data.get("support_needed"))
    return {
        "support_needed": needed,
        "overall_confidence": min(1.0, max(0.0, float(data.get("overall_confidence", 0)))) if isinstance(data.get("overall_confidence"), (int, float)) else 0,
        "token_support": tokens if needed else [], "entity_support": entities if needed else [],
        "romanization_support": romans if needed else [],
        "normalization_support": normalization,
        "pipeline_notes": [str(note)[:200] for note in notes[:4] if isinstance(note, str)] if isinstance(notes, list) else [],
        "sources": [],
    }
