"""Gemini-backed, optional second opinion with bounded in-memory caching."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

from src.support.prompts import SYSTEM_PROMPT
from src.support.schemas import SUPPORT_SCHEMA, validate_support
from src.support.web_grounding import grounding_sources

logger = logging.getLogger(__name__)
_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_TTL = 24 * 60 * 60
_CACHE_LIMIT = 256


def _cache_key(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class GeminiSupport:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

    def cross_check(self, payload: dict[str, Any], web_search: bool) -> dict[str, Any]:
        """Return support fields or a safe unavailable state; never block local inference."""
        candidates = {str(item.get("text", "")).casefold() for item in payload.get("tokens", [])}
        if not self.api_key:
            return {"status": "unavailable", "message": "AI support is not configured.", "token_support": [],
                    "entity_support": [], "romanization_support": [], "pipeline_notes": [], "sources": []}
        cache_payload = {"context": payload, "web_search": web_search, "model": self.model}
        key = _cache_key(cache_payload)
        now = time.monotonic()
        with _CACHE_LOCK:
            cached = _CACHE.get(key)
            if cached and now - cached[0] < _CACHE_TTL:
                return {**cached[1], "cached": True}
            if cached:
                _CACHE.pop(key, None)
        try:
            from google import genai
            from google.genai import types

            tools = [types.Tool(google_search=types.GoogleSearch())] if web_search else None
            config = types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=SUPPORT_SCHEMA,
                tools=tools,
            )
            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=self.model,
                contents=json.dumps(cache_payload, ensure_ascii=False),
                config=config,
            )
            raw = getattr(response, "text", None)
            parsed = json.loads(raw) if raw else None
            support = validate_support(parsed, candidates, bool(web_search and grounding_sources(response)))
            support["sources"] = grounding_sources(response) if web_search else []
            support["status"] = "available"
            support["cached"] = False
            with _CACHE_LOCK:
                if len(_CACHE) >= _CACHE_LIMIT:
                    oldest = min(_CACHE, key=lambda k: _CACHE[k][0])
                    _CACHE.pop(oldest, None)
                _CACHE[key] = (now, support)
            return support
        except Exception as error:  # API, grounding, SDK and parsing failures never break the local pipeline.
            logger.warning("Gemini support unavailable (%s)", type(error).__name__)
            return {"status": "unavailable", "message": "AI support is temporarily unavailable.",
                    "token_support": [], "entity_support": [], "romanization_support": [],
                    "pipeline_notes": [], "sources": []}
