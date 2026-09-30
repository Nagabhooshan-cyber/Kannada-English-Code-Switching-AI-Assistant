"""Convert Romanized Kannada words to Kannada script without changing English words."""

from __future__ import annotations

import re
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.models.normalizer import LexiconNormalizer

KANNADA_RE = re.compile(r"[\u0c80-\u0cff]")

# Common Kanglish spellings are not one-to-one phonetic spellings. These overrides
# preserve the customary Kannada forms; less common words use the spelling rules below.
COMMON_WORDS = {
    "nale": "ನಾಳೆ", "naale": "ನಾಳೆ", "ninne": "ನಿನ್ನೆ", "ivattu": "ಇವತ್ತು",
    "nanu": "ನಾನು", "naanu": "ನಾನು", "neenu": "ನೀನು", "avanu": "ಅವನು", "kannada": "ಕನ್ನಡ",
    "avalu": "ಅವಳು", "namaskara": "ನಮಸ್ಕಾರ", "namaste": "ನಮಸ್ಕಾರ",
    "hegiddiya": "ಹೇಗಿದ್ದೀಯಾ", "hegiddira": "ಹೇಗಿದ್ದೀರಾ", "hege": "ಹೇಗೆ",
    "chennagide": "ಚೆನ್ನಾಗಿದೆ", "chennagiddini": "ಚೆನ್ನಾಗಿದ್ದೀನಿ", "tumba": "ತುಂಬಾ",
    "beku": "ಬೇಕು", "beda": "ಬೇಡ", "hogbeku": "ಹೋಗಬೇಕು", "barbeku": "ಬರಬೇಕು",
    "maadabeku": "ಮಾಡಬೇಕು", "madabeku": "ಮಾಡಬೇಕು", "maadbeku": "ಮಾಡಬೇಕು", "madbeku": "ಮಾಡಬೇಕು",
    "hogtini": "ಹೋಗ್ತೀನಿ", "hogodu": "ಹೋಗೋದು", "hogodhu": "ಹೋಗೋದು",
    "bartini": "ಬರ್ತೀನಿ", "maadi": "ಮಾಡಿ", "madi": "ಮಾಡಿ",
    "kodi": "ಕೊಡಿ", "heli": "ಹೇಳಿ", "yenu": "ಏನು", "enu": "ಏನು", "yaake": "ಯಾಕೆ",
    "ella": "ಎಲ್ಲ", "ide": "ಇದೆ", "illa": "ಇಲ್ಲ", "idu": "ಇದು", "adu": "ಅದು",
    "nimage": "ನಿಮಗೆ", "nanage": "ನನಗೆ", "mane": "ಮನೆ", "oota": "ಊಟ",
    "gantege": "ಗಂಟೆಗೆ", "gantge": "ಗಂಟೆಗೆ", "gante": "ಗಂಟೆ", "ganthe": "ಗಂಟೆ",
}
# These are established colloquial spellings that map to a common written form,
# separate from typo detection, which uses the project's labeled training vocabulary.
COMMON_KANGLISH_FORMS = {
    "hogabeku": "hogbeku", "madabeku": "maadabeku", "madbeku": "maadabeku",
    "maadbeku": "maadabeku",
}
KANNADA_LOANWORDS = {
    "office": "ಆಫೀಸ್", "meeting": "ಮೀಟಿಂಗ್", "school": "ಸ್ಕೂಲ್",
    "class": "ಕ್ಲಾಸ್", "bus": "ಬಸ್", "project": "ಪ್ರಾಜೆಕ್ಟ್", "report": "ರಿಪೋರ್ಟ್",
    "majestic": "ಮೆಜೆಸ್ಟಿಕ್",
}
KANNADA_PLACE_NAMES = {
    "udupi": "ಉಡುಪಿ",
    "majestic": "ಮೆಜೆಸ್ಟಿಕ್",
}

VOWELS = {
    "aa": ("ಆ", "ಾ"), "ii": ("ಈ", "ೀ"), "ee": ("ಈ", "ೀ"), "uu": ("ಊ", "ೂ"),
    "oo": ("ಊ", "ೂ"), "ai": ("ಐ", "ೈ"), "au": ("ಔ", "ೌ"),
    "a": ("ಅ", ""), "i": ("ಇ", "ಿ"), "e": ("ಎ", "ೆ"), "u": ("ಉ", "ು"),
    "o": ("ಒ", "ೊ"),
}
CONSONANTS = {
    "ksh": "ಕ್ಷ", "kh": "ಖ", "gh": "ಘ", "ch": "ಚ", "jh": "ಝ", "th": "ಥ",
    "dh": "ಧ", "ph": "ಫ", "bh": "ಭ", "sh": "ಶ", "ng": "ಙ", "ny": "ಞ",
    "ṭ": "ಟ", "ḍ": "ಡ", "ṇ": "ಣ", "t": "ತ", "d": "ದ", "n": "ನ", "p": "ಪ",
    "b": "ಬ", "m": "ಮ", "y": "ಯ", "r": "ರ", "l": "ಲ", "v": "ವ", "w": "ವ",
    "s": "ಸ", "h": "ಹ", "k": "ಕ", "g": "ಗ", "c": "ಚ", "j": "ಜ", "f": "ಫ",
}


def transliterate_word(word: str) -> str:
    """Use common Kanglish word forms, then a conservative phonetic fallback."""
    key = word.casefold()
    key = COMMON_KANGLISH_FORMS.get(key, key)
    if key in COMMON_WORDS:
        return COMMON_WORDS[key]
    if not key.isascii() or not key.isalpha():
        return word
    output: list[str] = []
    index = 0
    pending_consonant = False
    while index < len(key):
        vowel_key = next((v for v in sorted(VOWELS, key=len, reverse=True) if key.startswith(v, index)), None)
        if vowel_key:
            independent, sign = VOWELS[vowel_key]
            if pending_consonant:
                output.append(sign)
                pending_consonant = False
            else:
                output.append(independent)
            index += len(vowel_key)
            continue
        consonant_key = next((c for c in sorted(CONSONANTS, key=len, reverse=True) if key.startswith(c, index)), None)
        if consonant_key:
            if pending_consonant:
                output.append("್")
            output.append(CONSONANTS[consonant_key])
            pending_consonant = True
            index += len(consonant_key)
            continue
        return word
    # Kannada consonants carry an implicit vowel unless another consonant follows.
    return "".join(output)


class RomanKannadaConverter:
    """Correct likely typos against training vocabulary, then transliterate Kannada tokens."""

    def __init__(self, lexicon: LexiconNormalizer, train_data_path: Path | None = None):
        self.lexicon = lexicon
        self.vocabulary = Counter()
        if train_data_path and train_data_path.is_file():
            for line in train_data_path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for word, label in zip(record.get("tokens", []), record.get("labels", [])):
                    if label == "kn" and str(word).isascii() and str(word).isalpha():
                        self.vocabulary[str(word).casefold()] += 1
        for word in COMMON_WORDS:
            self.vocabulary.setdefault(word, 100)

    @staticmethod
    def _distance_one(left: str, right: str) -> bool:
        """Return true for one insertion, deletion, substitution, or adjacent swap."""
        if abs(len(left) - len(right)) > 1:
            return False
        if len(left) == len(right):
            diffs = [i for i, (a, b) in enumerate(zip(left, right)) if a != b]
            if len(diffs) == 1:
                return True
            return (
                len(diffs) == 2 and diffs[1] == diffs[0] + 1
                and left[diffs[0]] == right[diffs[1]]
                and left[diffs[1]] == right[diffs[0]]
            )
        short, long = (left, right) if len(left) < len(right) else (right, left)
        i = j = skips = 0
        while i < len(short) and j < len(long):
            if short[i] == long[j]:
                i += 1
                j += 1
            else:
                skips += 1
                j += 1
                if skips > 1:
                    return False
        return True

    def _correct_spelling(self, word: str) -> tuple[str, str | None]:
        key = word.casefold()
        if len(key) < 5 or key in self.vocabulary:
            return key, None
        candidates = [
            candidate for candidate in self.vocabulary
            if self._distance_one(key, candidate)
        ]
        if len(candidates) != 1:
            return key, None
        return candidates[0], candidates[0]

    def convert(self, text: str, tokens: list[dict[str, Any]]) -> dict[str, Any]:
        exact = self.lexicon.sentences.get(text.casefold())
        if exact is not None:
            return {"text": exact, "matched_tokens": 1, "unmatched_tokens": 0,
                    "corrections": [], "coverage": 1.0}
        replacements: dict[tuple[int, int], str] = {}
        corrections: list[dict[str, str]] = []
        matched = unmatched = 0
        for token in tokens:
            raw = token["text"]
            is_kannada = token["language"] == "kn"
            is_loanword = raw.casefold() in KANNADA_LOANWORDS
            is_place_name = bool(re.match(r"(?:-ge|\s+ge|ige)", text[token["end"]:], re.IGNORECASE))
            if (not is_kannada and not is_loanword) or not any(ch.isalpha() for ch in raw):
                continue
            if KANNADA_RE.search(raw):
                continue
            corrected_word, matched_word = (
                self._correct_spelling(raw) if is_kannada and not is_place_name else (raw.casefold(), None)
            )
            corrected_word = COMMON_KANGLISH_FORMS.get(corrected_word, corrected_word)
            replacement = (
                KANNADA_PLACE_NAMES.get(raw.casefold())
                or KANNADA_LOANWORDS.get(raw.casefold())
                or self.lexicon.words.get(raw.casefold())
                or transliterate_word(corrected_word)
            )
            if KANNADA_RE.search(replacement):
                replacements[(token["start"], token["end"])] = replacement
                matched += 1
                if corrected_word != raw.casefold() or matched_word is not None:
                    corrections.append({
                        "input": raw,
                        "matched_training_word": matched_word or raw.casefold(),
                        "corrected_word": corrected_word,
                        "kannada_script": replacement,
                    })
            else:
                unmatched += 1
        output: list[str] = []
        cursor = 0
        previous_token: dict[str, Any] | None = None
        for token in tokens:
            start, end = token["start"], token["end"]
            gap = text[cursor:start]
            if (
                gap == "-"
                and previous_token is not None
                and (previous_token["start"], previous_token["end"]) in replacements
                and (start, end) in replacements
                and str(token["text"]).casefold() in {"ge", "ige"}
            ):
                # Join a Kannada case marker directly to a Kannada-rendered
                # English loanword: "meeting-ge" -> "ಮೀಟಿಂಗ್‌ಗೆ".
                gap = ""
            output.append(gap)
            output.append(replacements.get((start, end), text[start:end]))
            cursor = end
            previous_token = token
        output.append(text[cursor:])
        result = "".join(output)
        total = matched + unmatched
        return {"text": result, "matched_tokens": matched, "unmatched_tokens": unmatched,
                "corrections": corrections,
                "coverage": matched / total if total else 1.0}

    def apply_supported_variants(
        self, text: str, tokens: list[dict[str, Any]], variants: dict[str, str], confidence_threshold: float = 0.97,
    ) -> dict[str, Any]:
        """Render only high-confidence AI-suggested Romanized forms through the local transliterator."""
        replacements: list[tuple[int, int, str]] = []
        applied: list[dict[str, str]] = []
        for token in tokens:
            raw = str(token.get("text", ""))
            suggestion = variants.get(raw.casefold())
            if (not suggestion or token.get("confidence", 1.0) >= 0.85
                    or not suggestion.isascii() or not suggestion.isalpha()):
                continue
            kannada = transliterate_word(suggestion)
            if KANNADA_RE.search(kannada):
                replacements.append((int(token["start"]), int(token["end"]), kannada))
                applied.append({"input": raw, "romanized_form": suggestion, "kannada_script": kannada})
        result = text
        for start, end, replacement in sorted(replacements, reverse=True):
            result = result[:start] + replacement + result[end:]
        return {"text": result, "applied": applied}
