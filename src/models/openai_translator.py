"""OpenAI-backed translation adapter for lightweight hosted deployments."""

from __future__ import annotations

import json
import os
from typing import Any

from src.models.translator import TranslationUnavailableError


class OpenAITranslator:
    """Translate Kannada, Romanized Kannada, English, and mixed input via OpenAI."""

    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        normalized_source: str | None = None,
    ) -> str:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise TranslationUnavailableError(
                "OpenAI translation is not configured. Add a valid OPENAI_API_KEY in the hosting service's private environment settings."
            )

        source_names = {"kn": "Kannada", "en": "English"}
        source = source_names.get(source_language.casefold(), source_language)
        target = source_names.get(target_language.casefold(), target_language)
        if source_language.casefold() == target_language.casefold():
            return text

        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, timeout=45.0, max_retries=1)
            response = client.responses.create(
                model=self.model,
                instructions=(
                    "Translate the user's text faithfully into the requested target language. "
                    "The input may contain colloquial Kannada written in Latin letters, Kannada script, "
                    "English, or code-switching. Use natural grammar, preserve names and meaning, "
                    "do not invent details, and return only the translation."
                ),
                input=json.dumps(
                    {
                        "source_language": source,
                        "target_language": target,
                        "original_text": text,
                        "Kannada_script_hint": normalized_source if normalized_source != text else None,
                    },
                    ensure_ascii=False,
                ),
                store=False,
            )
            translated = str(response.output_text or "").strip()
        except Exception as error:
            raise TranslationUnavailableError(
                "OpenAI translation failed. Check the private API key, model setting, and API account availability in the hosting service."
            ) from error

        if not translated:
            raise TranslationUnavailableError("OpenAI returned an empty translation. Please retry.")
        return translated
