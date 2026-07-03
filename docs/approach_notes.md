# SHL Assessment Recommender: Approach Notes

## Page 1: System Overview

### Objective

This project implements a stateless conversational FastAPI service for recommending SHL assessments from the provided SHL product catalog. The service exposes `GET /health` for readiness and `POST /chat` for the official chat API. A simple browser UI is also available at `GET /` for manual testing.

The assistant stays strictly in scope. It only discusses SHL assessments and catalog-grounded comparisons, and it refuses general hiring advice, legal questions, salary questions, and prompt-injection attempts.

### Data Source And Processing

The authoritative source is:

```text
data/raw/shl_product_catalog.json
```

The system uses structured JSON parsing and normalization. It does not scrape external catalog pages. Catalog links are normalized into recommendation URLs, and catalog keys are mapped into readable assessment types such as Knowledge & Skills, Personality & Behavior, Ability & Aptitude, and Simulations.

Each catalog `entity_id` is treated as one assessment-level retrieval document. Hierarchical chunking is intentionally not used because each catalog item is already a compact structured assessment record. This keeps every retrieved item directly mapped to a valid SHL assessment.

Current processed corpus:

```text
377 assessment chunks
```

### Conversational Behavior

The chat controller supports the four required behaviors:

1. Clarifies vague queries before recommending. For example, `I need an assessment` is not enough, so the assistant asks for role, skills, seniority, and assessment type.

2. Recommends 1 to 10 assessments when enough context is available. Each recommendation includes a catalog-backed name, URL, test type, and short match reason.

3. Refines recommendations when the user changes constraints mid-conversation. For example, `Actually, add personality tests` updates the previous shortlist instead of starting from zero.

4. Compares assessments when asked. For example, `What is the difference between OPQ and GSA?` is answered using catalog fields, not the model's prior knowledge.

The `/chat` endpoint is stateless, so the client sends the full conversation history on every turn. The browser UI already handles this automatically.

<div style="page-break-after: always;"></div>

## Page 2: Retrieval, Generation, And Deployment

### Retrieval Pipeline

The production retrieval flow is:

```text
Voyage dense retrieval top 30
+ SPLADE sparse retrieval top 30
-> RRF fusion top 30
-> Voyage rerank top 10
-> Gemini reply generation
```

Dense retrieval uses Voyage `voyage-4-large` embeddings with Chroma. SPLADE retrieval uses `naver/splade-cocondenser-ensembledistil` for exact matching of technologies, skills, and assessment names. Reciprocal Rank Fusion combines the dense and sparse rankings, and Voyage reranking selects the final top results.

Default retrieval settings:

```text
Dense top K: 30
SPLADE top K: 30
RRF top K: 30
Final rerank top K: 10
RRF smoothing K: 60
```

Gemini is used only to write the natural-language assistant reply. The structured recommendation list is controlled by code and always comes from the SHL catalog.

### API Response

Each recommendation follows this structure:

```json
{
  "name": "Catalog assessment name",
  "url": "Catalog URL",
  "test_type": "Catalog-derived type",
  "reason": "Short catalog/retrieval match reason"
}
```

The `reason` field is generated locally from catalog and query overlap, not from model prior knowledge.

### Verification And Deployment

Pipeline verification:

```bash
python scripts/check_api_usage.py
```

Expected indicators:

```text
dense_retrieval_active=True
voyage_rerank_called=True
gemini_generation_called=True
```

Behavior evaluation and tests:

```bash
python scripts/run_eval.py
python -m pytest
```

Required deployment secrets:

```text
VOYAGE_API_KEY
GEMINI_API_KEY
```

For deployment, `data/chroma/` is rebuilt from committed dense embeddings rather than committed to Git. This keeps the repository clean while preserving dense retrieval in production. Hugging Face Docker Spaces use port `7860`; Render uses the platform-provided `$PORT`.
