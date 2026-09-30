"""Conservative rules for deciding when local results need a second opinion."""

from __future__ import annotations

import re
from typing import Any

COMMON_WORDS = {
    "a", "an", "and", "are", "at", "be", "but", "class", "do", "for", "go", "hege",
    "hogbeku", "hogodu", "how", "i", "in", "is", "it", "kalibeku", "meeting", "me", "my",
    "nale", "naale", "of", "on", "please", "the", "to", "tomorrow", "what", "where", "with",
    "you", "nanu", "naan", "manege", "hoguttene", "madidini", "maadidini", "alli", "idu",
}
PROPER_CONTEXT = re.compile(r"(?:-ge|\sge|ige|alli|\s+hege\s+hog|\bto\b)", re.I)
TECH_CONTEXT = re.compile(r"\b(?:kalibeku|learn|install|coding|program(?:ming)?|technical)\b", re.I)
ROMANIZED_ENDING = re.compile(r"(?:beku|beka|idini|idini|tini|thini|tane|tini|odu|odu|illa|ide)$", re.I)


def decide_support(text: str, tokens: list[dict[str, Any]], normalization_coverage: float,
                   entities: dict[str, Any], model_available: bool = True) -> dict[str, Any]:
    """Return only tokens whose local labels or entity identity are plausibly uncertain."""
    candidates: list[dict[str, Any]] = []
    raw_tokens = [t for t in tokens if t.get("language") not in {"other"} and str(t.get("text", "")).isalnum()]
    for token in tokens:
        word = str(token.get("text", ""))
        folded = word.casefold()
        if not word or not any(ch.isalnum() for ch in word):
            continue
        if folded in COMMON_WORDS:
            continue
        if token.get("language") == "other" or (
            model_available and float(token.get("confidence", 1.0)) < 0.72
        ):
            candidates.append(token)
            continue
        # Capitalized names in destination/locative contexts merit entity cross-checking.
        if (word[:1].isupper() and folded not in COMMON_WORDS
                and (not model_available or PROPER_CONTEXT.search(text) or TECH_CONTEXT.search(text)
                     or entities.get("destination"))):
            candidates.append(token)
        elif not model_available and ROMANIZED_ENDING.search(folded) and folded not in COMMON_WORDS:
            # Unknown Romanized inflections merit linguistic support, but never web search.
            candidates.append(token)
        elif model_available and normalization_coverage < 0.35 and token.get("language") == "unknown":
            candidates.append(token)
    # Split suffixes are retained as context, but not queried as stand-alone candidates.
    candidates = [t for t in candidates if str(t.get("text", "")).casefold() not in {"ge", "alli", "ige"}]
    unique: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        unique[str(candidate["text"]).casefold()] = candidate
    return {
        "needed": bool(unique),
        "tokens": list(unique.values())[:8],
        "web_search": any(
            str(item["text"])[:1].isupper()
            and str(item["text"]).casefold() not in COMMON_WORDS
            for item in unique.values()
        ),
    }
