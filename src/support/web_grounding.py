"""Extract verifiable citation metadata from Gemini Search grounding results."""

from __future__ import annotations

from typing import Any


def grounding_sources(response: Any) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    try:
        candidates = getattr(response, "candidates", None) or []
        metadata = getattr(candidates[0], "grounding_metadata", None) if candidates else None
        chunks = getattr(metadata, "grounding_chunks", None) or []
        for chunk in chunks:
            web = getattr(chunk, "web", None)
            uri = getattr(web, "uri", None)
            if isinstance(uri, str) and uri.startswith(("https://", "http://")):
                sources.append({"url": uri, "title": str(getattr(web, "title", "") or uri)[:200]})
    except (AttributeError, IndexError, TypeError):
        return []
    return sources[:5]
