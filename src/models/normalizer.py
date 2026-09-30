"""Conservative lookup baseline for manually verified Kannada conversion pairs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.data.normalization import load_pairs

TOKEN_PATTERN = re.compile(r"\w+(?:[-'][\w]+)*|\s+|[^\w\s]", flags=re.UNICODE)


class LexiconNormalizer:
    """Apply exact curated sentence pairs and one-word mappings; preserve unknown text."""

    def __init__(self, pairs: list[dict[str, str]], mode: str):
        if mode not in {"normalization", "transliteration"}:
            raise ValueError("mode must be 'normalization' or 'transliteration'")
        self.mode = mode
        target_field = "target_kannada" if mode == "normalization" else "target_transliteration"
        self.sentences: dict[str, str] = {}
        self.words: dict[str, str] = {}
        for pair in pairs:
            source, target = pair["source"], pair[target_field]
            if not target:
                continue
            self.sentences[source.casefold()] = target
            if len(source.split()) == 1:
                self.words[source.casefold()] = target

    @classmethod
    def from_directory(cls, raw_dir: Path, mode: str = "normalization") -> "LexiconNormalizer":
        """Load all manual pair files from a directory tree."""
        pairs, issues = load_pairs(raw_dir)
        if issues:
            raise ValueError("Invalid manual normalization data: " + "; ".join(issues[:5]))
        return cls(pairs, mode)

    def convert(self, text: str) -> dict[str, Any]:
        """Return converted text and indicate how many lexical tokens were uncovered."""
        if not text.strip():
            return {"text": text, "matched_tokens": 0, "unmatched_tokens": 0, "coverage": 1.0}
        exact = self.sentences.get(text.casefold())
        if exact is not None:
            return {"text": exact, "matched_tokens": 1, "unmatched_tokens": 0, "coverage": 1.0}
        matched = 0
        unmatched = 0
        output: list[str] = []
        for segment in TOKEN_PATTERN.findall(text):
            if segment.isspace() or not any(character.isalnum() for character in segment):
                output.append(segment)
                continue
            replacement = self.words.get(segment.casefold())
            if replacement is None:
                output.append(segment)
                unmatched += 1
            else:
                output.append(replacement)
                matched += 1
        total = matched + unmatched
        return {
            "text": "".join(output), "matched_tokens": matched, "unmatched_tokens": unmatched,
            "coverage": matched / total if total else 1.0,
        }
