"""Small pure-Python IndicTrans2 processor for environments without the Cython toolkit.

This follows the IndicTransToolkit inference preprocessing path for Kannada/English:
normalize and tokenize Kannada, transliterate it to the shared Devanagari representation,
add the model language tags, then detokenize English output. It is intentionally limited
to the two languages used by this project.
"""

from __future__ import annotations

from typing import Sequence


class IndicProcessor:
    """Portable Kannada-English subset of IndicTransToolkit's inference processor."""

    def __init__(self, inference: bool = True):
        try:
            from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
            from indicnlp.tokenize import indic_detokenize, indic_tokenize
            from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
            from sacremoses import MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer
        except ImportError as error:
            raise ImportError(
                "The Windows-compatible IndicTrans2 preprocessing dependencies are missing. "
                "Install requirements-translation.txt."
            ) from error

        self.inference = inference
        self._indic_detokenize = indic_detokenize
        self._indic_tokenize = indic_tokenize
        self._transliterator = UnicodeIndicTransliterator()
        self._normalizer = IndicNormalizerFactory().get_normalizer("kn")
        self._english_normalizer = MosesPunctNormalizer()
        self._english_tokenizer = MosesTokenizer(lang="en")
        self._english_detokenizer = MosesDetokenizer(lang="en")

    def preprocess_batch(
        self,
        batch: Sequence[str],
        src_lang: str,
        tgt_lang: str | None = None,
        is_target: bool = False,
        visualize: bool = False,
    ) -> list[str]:
        del visualize
        if src_lang not in {"kan_Knda", "eng_Latn"} or tgt_lang not in {"kan_Knda", "eng_Latn"}:
            raise ValueError("Portable preprocessing supports Kannada and English only.")
        processed: list[str] = []
        for sentence in batch:
            normalized = sentence.strip()
            if src_lang == "eng_Latn":
                normalized = self._english_normalizer.normalize(normalized)
                tokens = self._english_tokenizer.tokenize(normalized, escape=False)
                converted = " ".join(tokens)
            else:
                normalized = self._normalizer.normalize(normalized)
                tokens = self._indic_tokenize.trivial_tokenize(normalized, "kn")
                converted = self._transliterator.transliterate(" ".join(tokens), "kn", "hi")
                converted = converted.replace(" ् ", "्")
            processed.append(converted if is_target else f"{src_lang} {tgt_lang} {converted}")
        return processed

    def postprocess_batch(
        self,
        sentences: Sequence[str],
        lang: str = "eng_Latn",
        visualize: bool = False,
        num_return_sequences: int = 1,
    ) -> list[str]:
        del visualize, num_return_sequences
        if lang not in {"kan_Knda", "eng_Latn"}:
            raise ValueError("Portable postprocessing supports Kannada and English only.")
        outputs: list[str] = []
        for sentence in sentences:
            if lang == "eng_Latn":
                outputs.append(self._english_detokenizer.detokenize(sentence.split()))
            else:
                kannada = self._transliterator.transliterate(sentence, "hi", "kn")
                outputs.append(self._indic_detokenize.trivial_detokenize(kannada, "kn"))
        return outputs
