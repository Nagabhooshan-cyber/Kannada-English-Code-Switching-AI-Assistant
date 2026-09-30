"""Lazy IndicTrans2 inference adapter for Kannada and English translation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.models.normalizer import LexiconNormalizer

KANNADA_SCRIPT_RE = re.compile(r"[\u0c80-\u0cff]")
LATIN_RE = re.compile(r"[A-Za-z]")


class TranslationUnavailableError(RuntimeError):
    """Raised when translation cannot run due to missing data, packages, or model access."""


class IndicTransTranslator:
    """Translate English/Kannada text through AI4Bharat's pretrained IndicTrans2 models.

    Models and the IndicTransToolkit are loaded only on the first translation request.
    Romanized Kannada input can be preprocessed by the pipeline's script converter.
    """

    CHECKPOINTS = {
        ("en", "kn"): "ai4bharat/indictrans2-en-indic-dist-200M",
        ("kn", "en"): "ai4bharat/indictrans2-indic-en-dist-200M",
    }

    def __init__(self, project_root: Path, config: dict[str, Any] | None = None):
        self.project_root = project_root
        self.config = config or {}
        # HF_TOKEN is read by huggingface_hub from the process environment.
        # Load the local, git-ignored .env so developers can keep it out of source.
        try:
            from dotenv import load_dotenv

            load_dotenv(self.project_root / ".env", override=False)
        except ImportError:
            # CLI-based `hf auth login` remains supported without python-dotenv.
            pass
        self.max_input_chars = int(self.config.get("max_input_chars", 2000))
        self._loaded: dict[tuple[str, str], tuple[Any, Any, Any, Any]] = {}
        self._romanizer = LexiconNormalizer.from_directory(project_root / "data/raw/normalization", "transliteration")

    @staticmethod
    def _canonical_language(language: str) -> str:
        aliases = {
            "en": "en", "english": "en", "eng_latn": "en",
            "kn": "kn", "kannada": "kn", "kan_knda": "kn",
        }
        try:
            return aliases[language.strip().casefold()]
        except (AttributeError, KeyError) as error:
            raise TranslationUnavailableError(
                "Supported source/target languages are English ('en') and Kannada ('kn')."
            ) from error

    def _load_direction(self, source: str, target: str) -> tuple[Any, Any, Any, Any]:
        direction = (source, target)
        if direction in self._loaded:
            return self._loaded[direction]
        checkpoint = self.config.get("models", {}).get(f"{source}_to_{target}") or self.CHECKPOINTS.get(direction)
        if checkpoint is None:
            raise TranslationUnavailableError("Only English↔Kannada translation is configured.")
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            try:
                from IndicTransToolkit import IndicProcessor
            except ImportError:
                try:
                    from IndicTransToolkit.IndicTransToolkit import IndicProcessor
                except ImportError:
                    from src.models.indic_processor_fallback import IndicProcessor
        except ImportError as error:
            raise TranslationUnavailableError(
                "Optional translation dependencies are unavailable. Install requirements-translation.txt, then restart the application."
            ) from error

        cache_path = self.project_root / self.config.get("cache_dir", "models/translation_cache")
        cache_path.mkdir(parents=True, exist_ok=True)
        repo_cache = cache_path / f"models--{checkpoint.replace('/', '--')}" / "snapshots"
        cache_ready = any(
            (snapshot / "tokenizer_config.json").is_file()
            and any((snapshot / weights).is_file() for weights in ("model.safetensors", "pytorch_model.bin"))
            for snapshot in repo_cache.iterdir()
        ) if repo_cache.is_dir() else False
        device_name = str(self.config.get("device", "auto"))
        device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
        try:
            load_options = {
                "trust_remote_code": True,
                "cache_dir": cache_path,
                # Avoid slow online metadata retries after the large checkpoint is cached.
                "local_files_only": cache_ready,
            }
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, **load_options)
            model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, **load_options)
            model.to(device)
            model.eval()
            processor = IndicProcessor(inference=True)
        except Exception as error:
            raise TranslationUnavailableError(
                f"Could not load {checkpoint}. Accept the checkpoint's Hugging Face access conditions and ensure it is downloaded. Details: {error}"
            ) from error
        self._loaded[direction] = (tokenizer, model, processor, device)
        return self._loaded[direction]

    def translate(
        self, text: str, source_language: str, target_language: str,
        normalized_source: str | None = None,
    ) -> str:
        """Translate text, raising a clear error when preprocessing or model access is unavailable."""
        if not isinstance(text, str) or not text.strip():
            raise TranslationUnavailableError("Translation input must be non-empty text.")
        if len(text) > self.max_input_chars:
            raise TranslationUnavailableError(f"Translation input exceeds {self.max_input_chars} characters.")
        source = self._canonical_language(source_language)
        target = self._canonical_language(target_language)
        if source == target:
            return text

        prepared_text = normalized_source if normalized_source is not None else text
        if normalized_source is None and source == "kn" and not KANNADA_SCRIPT_RE.search(text) and LATIN_RE.search(text):
            converted = self._romanizer.convert(text)
            if converted["matched_tokens"] == 0:
                raise TranslationUnavailableError(
                    "No Kannada-script words were detected for translation. Check the language detection result or enter Kannada script."
                )
            prepared_text = converted["text"]

        tokenizer, model, processor, device = self._load_direction(source, target)
        import torch

        language_codes = self.config.get("language_codes", {})
        source_code = language_codes.get(source, "eng_Latn" if source == "en" else "kan_Knda")
        target_code = language_codes.get(target, "eng_Latn" if target == "en" else "kan_Knda")
        try:
            try:
                prepared = processor.preprocess_batch([prepared_text], src_lang=source_code, tgt_lang=target_code, visualize=False)
            except TypeError:
                prepared = processor.preprocess_batch([prepared_text], src_lang=source_code, tgt_lang=target_code)
            model_inputs = tokenizer(
                prepared, padding="longest", truncation=True,
                max_length=int(self.config.get("max_length", 256)), return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                generated = model.generate(
                    **model_inputs, num_beams=int(self.config.get("num_beams", 5)),
                    num_return_sequences=1, max_length=int(self.config.get("max_length", 256)),
                    # Current HF generation creates an empty cache tuple that this checkpoint's
                    # custom decoder cannot handle. Disable KV caching for compatible inference.
                    use_cache=bool(self.config.get("use_cache", False)),
                )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            translated = processor.postprocess_batch(decoded, lang=target_code)
        except Exception as error:
            raise TranslationUnavailableError(f"IndicTrans2 inference failed: {error}") from error
        if not translated or not str(translated[0]).strip():
            raise TranslationUnavailableError("IndicTrans2 returned an empty translation.")
        return str(translated[0])
