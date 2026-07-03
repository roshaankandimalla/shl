# Approach Notes

## Data Source

Use the provided SHL product catalog JSON as the authoritative source. Recommendations must only use names, URLs, and test types from that catalog.

## Retrieval Plan

Each `entity_id` is treated as one assessment-level retrieval document. The catalog records are short structured JSON objects, so aggressive hierarchical chunking is unnecessary for the first version.

Hybrid retrieval:

- SPLADE sparse retrieval for exact assessment names, technologies, and skill terms.
- Voyage dense retrieval for semantic matching of job descriptions.
- Direct assessment-level scoring, preserving a one-to-one mapping between retrieved documents and valid recommendations.
- Final Voyage reranking with metadata and conversation constraint boosts.

## Controller Plan

The chat controller handles:

- vague query clarification
- recommendation
- refinement
- comparison
- off-topic refusal
- strict schema formatting
