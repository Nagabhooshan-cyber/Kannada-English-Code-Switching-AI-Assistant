# Kannada–English Code-Switching AI Assistant

A local web assistant and research pipeline for Romanized Kannada, Kannada-script, English, and code-mixed text. It provides word-level language identification, Kannada-script normalization, rule-based entity extraction, optional translation, and an uncertainty-triggered Gemini support layer. The local pipeline remains the primary system.

## Current phase

The repository includes data preparation and training scripts, local API/UI, starter normalization and intent components, and evaluation reports from prior runs. Raw datasets and trained model artifacts are intentionally excluded. Add the licensed CoLI-Kanglish files to `data/raw/coli_kanglish/` to reproduce the language-ID preparation and training results.

Supported source formats are CSV, TSV, JSON, and JSONL. The loader recognizes common token/label column names (`tokens`/`labels`, `words`/`tags`, etc.) or sentence text with a whitespace-aligned label sequence. For other schemas, set column names in `configs/data.yaml`. Raw input files are only read, never modified.

## Setup

Requires Python 3.11+. Create and activate a virtual environment, then install the lightweight data-phase dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The base installation includes the FastAPI site, optional Gemini support, and the OpenAI SDK for API-backed translation. IndicTrans2's large local inference dependencies remain optional and are listed in `requirements-translation.txt`.

For local Hugging Face access, copy `.env.example` to `.env` and put your read token in `HF_TOKEN`, or authenticate with `hf auth login`. The `.env` file is ignored by Git. Never paste a token into source code, a notebook, a public issue, or a chat. For hosted deployments, configure `HF_TOKEN` in the hosting provider's private environment/secrets settings; do not upload `.env`.

### Optional Gemini AI support

The local models and rules remain the primary language identification, normalization, intent, and entity components. Translation uses OpenAI by default to avoid loading a large model on small hosted instances. After local analysis, a small uncertainty check may ask Gemini for a second opinion about unknown/low-confidence words or likely named entities. Gemini cannot supply the user-facing translation, intent, or a conversational answer. High-confidence support may refine a language label or add a supported named entity; weak or unavailable support leaves local results in place.

Copy `.env.example` to `.env` and add a Gemini API key obtained from Google AI Studio as `GEMINI_API_KEY`. Optionally set `GEMINI_MODEL`; the default is `gemini-3.8-flash`. The key is read only by the backend and must never be committed, placed in browser code, or included in a deployment image. `.env` is gitignored. Google Search grounding is enabled only when a likely proper noun or entity needs outside identification; ordinary confident words do not trigger search. Search results are used as entity evidence, not as a general search or question-answering feature.

AI support is skipped when the local pipeline is confident. Repeated identical support requests are cached in process memory for 24 hours (up to 256 entries). If the key is missing, Gemini/Search fails, or the response is invalid, local processing continues unchanged. Only the submitted text and the limited uncertain-token context are sent to Google on triggered requests. API and grounding usage may be billed; check Google's current pricing and usage limits before enabling it. Do not submit sensitive text when external support is enabled. Results can still misclassify ambiguous names or Romanized spellings; verify important outputs.

## Prepare CoLI-Kanglish

```powershell
python scripts/prepare_data.py
```

The script discovers source files recursively, validates token-label alignment, removes exact duplicate sentence/label sequences, creates `train.jsonl`, `validation.jsonl`, and `test.jsonl` under `data/processed/language_id/`, and writes exploration statistics to `reports/data_exploration/coli_kanglish_summary.json`. It groups records by `source_id` (or source file) where possible. If that grouping cannot populate all partitions, it records a seeded record-level fallback in the report; use distinct source metadata for stronger leakage controls.

The exploration report includes example/token counts, observed class and token distributions, common tokens grouped by source label, mixed-label frequency, punctuation counts, exact duplicate count, and a bounded near-duplicate sample. The near-duplicate scan caps oversized first-token buckets at 300 unique sentences; its count is therefore a sampled diagnostic, not a corpus-wide exhaustive statistic.

## Project status

Implemented: data preparation and exploration, character n-gram and MuRIL training scripts, starter normalization/intent/entity components, OpenAI API translation, optional local IndicTrans2 translation, a local web interface/API, and optional Gemini support. Evaluation summaries are in `reports/`. This checkout does not include source datasets or trained model weights, so training and some model-backed features require the user to supply the licensed data and model artifacts. Docker preparation is included; hosted translation needs a private `OPENAI_API_KEY` secret.

Run the baseline with:

```powershell
python scripts/train_language_id_baseline.py
```

It trains only on `train.jsonl`, evaluates on validation and the derived holdout, and—when configured and available—evaluates `data/raw/Test_withLabels_Kanglish.csv` separately. The model is saved to `models/language_id_baseline.joblib`; detailed per-class metrics and confusion matrices are saved to `reports/evaluation/language_id_baseline.json`.

Current measured results (character n-grams 2–5, default logistic regression):

| Evaluation set | Examples | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: | ---: |
| Validation split | 1,477 | 0.7292 | 0.4790 | 0.6973 |
| Derived holdout from training CSV | 1,477 | 0.7231 | 0.4684 | 0.6895 |
| Official labeled test CSV | 4,585 | 0.8035 | 0.4093 | 0.7808 |

The official test accuracy is higher than its macro F1 because performance is uneven across labels: the baseline did not predict the rare `location` class in that evaluation. Treat these as initial baseline measurements, not production performance. The data has one word and label per row, with no source IDs; the internal splits therefore use a seeded record-level fallback and cannot ensure speaker/comment-level separation.

## Transformer language-ID model

Fine-tune the transformer with:

```powershell
python scripts/train_language_id_transformer.py
```

The default checkpoint is `google/muril-base-cased`. The implementation aligns labels to each word's first subword and ignores later subwords and padding in the loss. The tokenizer and fine-tuned model are saved under `models/language_id_transformer/`; metrics are saved to `reports/evaluation/language_id_transformer.json`. The first run downloads the checkpoint from Hugging Face. Training on CPU is substantially slower than the baseline.

Measured MuRIL results (2 epochs, CPU, seed 42):

| Evaluation set | Examples | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: | ---: |
| Validation split | 1,477 | 0.7448 | 0.4096 | 0.6822 |
| Derived holdout from training CSV | 1,477 | 0.7414 | 0.4162 | 0.6726 |
| Official labeled test CSV | 4,585 | 0.8395 | 0.4320 | 0.7973 |

The transformer did not predict `location` or `name` on these evaluation sets and had low recall for `other`. Its official-test macro F1 is higher than the baseline's (0.4320 vs. 0.4093), while its validation macro F1 is lower (0.4096 vs. 0.4790). These are initial measurements; class imbalance and the record-level internal split remain limitations.

## Romanized Kannada normalization data and baseline

Add human-verified rows to `data/raw/normalization/manual_pairs_template.csv` or another CSV/JSONL file in `data/raw/normalization/`. CSV columns are `source`, `target_kannada`, `target_transliteration`, `source_id`, and `notes`. `target_kannada` is the normalized Kannada-script output; `target_transliteration` is an optional Kannada-script rendering that aims to preserve the original wording. Leave a target column blank when that annotation is unavailable. Do not use generated examples as ground truth.

Prepare separate normalization and transliteration splits with:

```powershell
python scripts/prepare_normalization_data.py
```

The lookup baseline can use exact sentence pairs and one-word pairs from the manual annotations:

```powershell
python scripts/normalize_text.py "Nale meeting-ge hogbeku" --mode normalization
python scripts/normalize_text.py "Nale meeting-ge hogbeku" --mode transliteration
```

The app now converts Romanized Kannada tokens into Kannada script using a local spelling converter. For a Kannada-labeled token absent from the vocabulary, it corrects a single insertion, deletion, substitution, or adjacent-letter swap only when exactly one Kannada-labeled training word matches. Corrections are returned with the matched training word and Kannada output. Common Kanglish forms have explicit Kannada-script outputs. Common English loanwords such as `office` are also rendered in Kannada script in the conversion result, while the original token's language label remains English. Other English tokens and punctuation are preserved. Larger or ambiguous spelling errors remain unchanged. This is script conversion, not a trained normalization model, and it does not rewrite colloquial Kannada into formal Kannada.

## Intent dataset and baseline

Add human-labeled utterances to `data/raw/intent/manual_intent_template.csv`. Aim for at least 300 diverse, preferably human-written examples per intent. Use one of the configured intent labels and put entity annotations in the `entities` column as a JSON object (or leave it blank). Keep `source_id` consistent for examples from the same source when that metadata is available.

Prepare deduplicated, source-grouped train/validation/test JSONL with:

```powershell
python scripts/prepare_intent_data.py
```

When the prepared splits contain examples from at least two intents, train and evaluate the word/character TF-IDF baseline with:

```powershell
python scripts/train_intent_baseline.py
```

The project currently includes an empty template only. No intent examples, model, or performance metrics have been fabricated.

## Entity extraction

The current rule-based extractor is available with:

```powershell
python scripts/extract_entities.py "Nale report complete madbeku"
```

It currently recognizes a small set of relative dates, numeric times, common activities, route destinations marked with `-ge`/`ge`, and query topics. Simple `place-ge hege hogodu?` route questions receive the natural English frame “How do I get to [place]?” Time expressions without AM/PM are marked ambiguous; extracted spans include character offsets. This is a starter component and has not been evaluated against a manually annotated entity set.

## Kannada–English translation

The app uses the OpenAI Responses API for translation by default, which avoids loading large model weights on small hosting instances. Copy `.env.example` to `.env`, set `OPENAI_API_KEY`, and optionally set `OPENAI_MODEL` (default: `gpt-4o-mini`). On Render, add these as private environment variables; never put the key in frontend code or commit it. Translation sends the submitted sentence and its local Kannada-script hint to OpenAI, so API usage may be billed and user text is shared with the provider. The API call sets `store=False`; review the provider's current data controls before sending sensitive text.

```powershell
python scripts/run_pipeline.py "Nale meeting-ge hogbeku" --translate --source-language kn --target-language en
```

For a local IndicTrans2 alternative, set `TRANSLATION_BACKEND=indictrans`, install `requirements-translation.txt`, accept the Hugging Face checkpoint access conditions, and configure `HF_TOKEN`. This backend downloads and runs large local checkpoints and is unsuitable for Render's 512 MB free instance. The hosted default (`TRANSLATION_BACKEND=openai`) does not need the IndicTrans2 dependencies.

## HTTP API

Start the API locally after installing `requirements.txt`:

```powershell
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/` for the user interface or `http://127.0.0.1:8000/docs` for interactive API documentation. The page accepts Romanized Kannada, Kannada script, English, or mixed text. Enter text and choose **Analyze text**; turn on **Translate this text** and select languages to request translation. `GET /health` checks that the service is responding. `POST /v1/process` accepts JSON such as:

```json
{
  "text": "Nale meeting-ge hogbeku",
  "translate": false,
  "source_language": "kn",
  "target_language": "en"
}
```

For deployment, use the start command `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT` where the hosting provider supplies `PORT`. Set `OPENAI_API_KEY` for hosted translation and `GEMINI_API_KEY` for optional uncertainty support. Set `HF_TOKEN` only when needed, and optionally configure `OPENAI_MODEL`, `GEMINI_MODEL`, or `LANGUAGE_BACKEND` in the host's private environment settings. Do not upload `.env`. The service code is in `src/api/main.py`.

The web interface and API are implemented. Model weights and datasets stay out of Git; a fresh clone therefore reports missing language models until they are supplied through an appropriate deployment artifact or model registry. The default hosted translation uses the OpenAI API; local IndicTrans2 remains optional and requires gated checkpoint access. Check dataset licenses before distributing dataset files separately.

### Container deployment preparation

The `Dockerfile` builds the HTTP API without copying local datasets, model weights, or `.env` credentials into the image. Build and run it locally with Docker:

```powershell
docker build -t kannada-english-assistant .
docker run --rm -p 8000:8000 kannada-english-assistant
```

Then open `http://127.0.0.1:8000/docs`. The container is portable to Docker-based hosts that provide a `PORT` environment variable. It intentionally starts without a trained language-ID model; provide a model artifact through the host's secure artifact or persistent-volume mechanism before expecting language-ID predictions. For uncertainty-triggered Gemini entity support, configure `GEMINI_API_KEY` as a private host secret. Never bake credentials into the image.

## Text pipeline

Run the integrated text pipeline with:

```powershell
python scripts/run_pipeline.py "Nale meeting-ge hogbeku, but first report complete madbeku"
```

The default uses the saved character n-gram language-ID baseline. Add `--language-backend transformer` to use the locally fine-tuned MuRIL model. The pipeline returns token spans and language labels, normalization coverage, intent/entities, component availability, and processing time. Intent remains unavailable until labeled intent data and a trained model are available. Romanized Kannada script conversion is available locally. Translation is optional and reports unavailable until its model and dependencies are configured. The current CoLI CSV has isolated word/tag rows, so language-ID quality on full utterances is limited by missing sentence context.
