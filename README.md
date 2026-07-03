---
title: SHL Assessment Recommender
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# SHL Assessment Recommender

Conversational FastAPI service for recommending SHL assessments from the provided catalog.

## Goal

Build a stateless `/chat` endpoint that:

- clarifies vague assessment requests
- recommends 1 to 10 catalog assessments
- refines recommendations when constraints change
- compares catalog assessments
- refuses off-topic requests
- returns a short catalog-grounded reason for each recommendation
- always returns the required schema

## Planned Retrieval

- SPLADE sparse retrieval for exact skills, technologies, and assessment names.
- Dense retrieval for semantic job-description queries.
- RRF fusion combines dense and sparse results.
- Voyage reranking is used when `VOYAGE_API_KEY` is available; otherwise the fused ranking is used.
- Gemini writes the natural-language reply when `GEMINI_API_KEY` is available; recommendations still come from catalog-backed code.
- One retrieval document per `entity_id`, because each catalog entity is one SHL assessment.
- Reranking at the assessment level so every recommendation maps directly to a valid catalog item.

Retrieval defaults:

```text
Dense top K: 30
SPLADE top K: 30
RRF top K: 30
Rerank/final top K: 10
RRF smoothing K: 60
```

## Setup

```bash
conda create -n shl-recommender python=3.11 -y
conda activate shl-recommender
pip install -r requirements.txt
```

If you already created a Conda environment for this project, activate that environment and run only:

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload
```

Then open:

- `GET http://127.0.0.1:8000/health`
- `POST http://127.0.0.1:8000/chat`
- `GET http://127.0.0.1:8000/` for the simple browser chat UI

Example `/chat` request:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "I am hiring a mid-level Java backend developer who works with stakeholders. Recommend SHL assessments."
    }
  ]
}
```

Each recommendation includes:

```json
{
  "name": "Catalog assessment name",
  "url": "Catalog URL",
  "test_type": "Catalog-derived test type",
  "reason": "Short match reason"
}
```

## Data

The catalog is already present at:

```text
data/raw/shl_product_catalog.json
```

This JSON is treated as the authoritative data source. The API should only recommend assessments whose name, URL, and test type come from this file.

Build normalized assessment documents:

```bash
python scripts/build_documents.py
```

This creates one retrieval chunk per `entity_id`.

Build dense Voyage embeddings:

```bash
python scripts/build_dense_embeddings.py
```

This also builds or updates the Chroma dense index. Use `--skip-chroma` only when you want to generate the `.npy` embedding file without touching Chroma.

Build SPLADE sparse vectors:

```bash
python scripts/build_splade_embeddings.py
```

Or run the full pipeline:

```bash
python scripts/build_index.py
```

Generated files:

```text
data/processed/assessment_documents.json
data/processed/dense_embeddings.npy
data/processed/dense_embeddings_meta.json
data/processed/splade_sparse_vectors.json
data/chroma/
```

Resume cache files:

```text
data/processed/dense_embeddings_cache.jsonl
data/processed/splade_sparse_vectors_cache.jsonl
data/processed/query_embeddings_cache.json
```

If an embedding run is interrupted, run the same command again. The script reads the cache and continues from missing `entity_id`s instead of embedding everything again. Use `--force` only when you intentionally want to rebuild a step from scratch.

Dense embeddings use Voyage and require:

```text
VOYAGE_API_KEY=...
```

Optional model settings:

```text
DENSE_EMBED_MODEL=voyage-4-large
RERANK_MODEL=rerank-2.5
SPLADE_MODEL=naver/splade-cocondenser-ensembledistil
GENERATION_MODEL=gemini-3.1-flash-lite
GEMINI_API_KEY=...
```

Useful build variants:

```bash
python scripts/build_index.py --skip-dense
python scripts/build_index.py --skip-chroma
python scripts/build_index.py --skip-splade
python scripts/build_dense_embeddings.py --force
python scripts/build_dense_embeddings.py --reset-chroma
python scripts/build_splade_embeddings.py --force
```

## Verify Retrieval And APIs

Run the API usage checker:

```bash
python scripts/check_api_usage.py
```

The full live pipeline is active when the output includes:

```text
voyage_embeddings_called=True
voyage_rerank_called=True
gemini_generation_called=True
```

For development, inspect retrieval stages with:

```bash
curl "http://127.0.0.1:8000/debug/retrieval?query=mid-level%20Java%20backend%20developer"
```

This returns dense top 30, SPLADE top 30, RRF top 30, and rerank top 10 results.

## Evaluation

Run the small behavior evaluation:

```bash
python scripts/run_eval.py
```

It reads:

```text
data/eval/questions.json
```

and writes local results to:

```text
data/eval/results.json
```

## Tests

```bash
pytest
```

The tests cover health, vague-query clarification, recommendation schema, off-topic refusal, and catalog-grounded comparison behavior.

## Deployment

Render is supported with `render.yaml`.

Required environment variables:

```text
VOYAGE_API_KEY=...
GEMINI_API_KEY=...
```

Render start command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```
