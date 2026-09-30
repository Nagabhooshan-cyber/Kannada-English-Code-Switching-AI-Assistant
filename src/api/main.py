"""FastAPI service exposing the assistant pipeline."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse

from src.inference.pipeline import AssistantPipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)

app = FastAPI(
    title="Kannada-English Code-Switching Assistant",
    description="Analyze Romanized Kannada, Kannada-script, and English text; optionally request translation.",
    version="0.1.0",
)


class ProcessRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000, description="Text to analyze")
    translate: bool = False
    source_language: Literal["kn", "en"] = "kn"
    target_language: Literal["kn", "en"] = "en"


@lru_cache(maxsize=1)
def get_pipeline() -> AssistantPipeline:
    """Construct pipeline once per application worker."""
    backend = os.getenv("LANGUAGE_BACKEND", "auto").strip().casefold()
    if backend not in {"auto", "baseline", "transformer"}:
        raise RuntimeError("LANGUAGE_BACKEND must be auto, baseline, or transformer")
    return AssistantPipeline(PROJECT_ROOT, language_backend=backend)


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the HTTP service is responding without loading models."""
    return {"status": "ok", "service": "kannada-english-assistant"}


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """Serve the end-user page."""
    return FileResponse(PROJECT_ROOT / "src/api/static/index.html")


@app.post("/v1/process")
def process(request: ProcessRequest) -> dict[str, Any]:
    """Run language identification, normalization, entity extraction, and optional translation."""
    if request.source_language == request.target_language and request.translate:
        raise HTTPException(status_code=422, detail="Source and target languages must differ for translation.")
    try:
        result = get_pipeline().process(
            request.text,
            request_translation=request.translate,
            source_language=request.source_language,
            target_language=request.target_language,
        )
        return result
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
