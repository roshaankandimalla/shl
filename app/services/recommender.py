import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import (
    ASSESSMENT_DOCUMENTS_PATH,
    CHROMA_COLLECTION_NAME,
    CHROMA_DATA_DIR,
    DENSE_EMBED_MODEL,
    DENSE_TOP_K,
    QUERY_EMBEDDINGS_CACHE_PATH,
    RERANK_TOP_K,
    RERANK_MODEL,
    RRF_K,
    RRF_TOP_K,
    SPARSE_TOP_K,
    SPLADE_MODEL,
    SPLADE_VECTORS_PATH,
    VOYAGE_EMBEDDINGS_URL,
    VOYAGE_RERANK_URL,
)
from app.schemas import Recommendation
from app.services.catalog import Assessment, load_catalog


OFF_TOPIC_TERMS = {
    "salary",
    "legal",
    "lawsuit",
    "contract",
    "visa",
    "termination",
    "fire employee",
    "write a poem",
    "ignore previous",
    "system prompt",
    "prompt injection",
    "medical advice",
    "investment advice",
    "tax advice",
    "weather",
    "recipe",
    "movie",
}

VAGUE_TERMS = {
    "assessment",
    "test",
    "hire",
    "hiring",
    "candidate",
    "employee",
    "role",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "for",
    "i",
    "me",
    "need",
    "please",
    "recommend",
    "some",
    "the",
    "to",
    "want",
    "we",
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9+#.]+", text.lower())


@dataclass(frozen=True)
class RetrievalHit:
    entity_id: str
    score: float


class Recommender:
    def __init__(self) -> None:
        self.assessments = load_catalog()
        self.assessments_by_id = {assessment.entity_id: assessment for assessment in self.assessments}
        self.documents = self._load_documents()
        self.documents_by_id = {str(document["entity_id"]): document for document in self.documents}
        self.splade_payload = self._load_splade_payload()
        self.query_embedding_cache = self._load_query_embedding_cache()
        self._splade_tokenizer: Any | None = None
        self._splade_model: Any | None = None

    def is_off_topic(self, text: str) -> bool:
        lower = text.lower()
        return any(term in lower for term in OFF_TOPIC_TERMS)

    def is_vague(self, text: str) -> bool:
        tokens = set(tokenize(text))
        meaningful = tokens - VAGUE_TERMS - STOPWORDS
        return len(meaningful) < 2

    def _load_documents(self) -> list[dict[str, Any]]:
        if not ASSESSMENT_DOCUMENTS_PATH.exists():
            return []
        return json.loads(ASSESSMENT_DOCUMENTS_PATH.read_text(encoding="utf-8"))

    def _load_splade_payload(self) -> dict[str, Any] | None:
        if not SPLADE_VECTORS_PATH.exists():
            return None
        return json.loads(SPLADE_VECTORS_PATH.read_text(encoding="utf-8"))

    def _load_query_embedding_cache(self) -> dict[str, list[float]]:
        if not QUERY_EMBEDDINGS_CACHE_PATH.exists():
            return {}
        try:
            payload = json.loads(QUERY_EMBEDDINGS_CACHE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return {str(key): value for key, value in payload.items() if isinstance(value, list)}

    def _save_query_embedding_cache(self) -> None:
        QUERY_EMBEDDINGS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        QUERY_EMBEDDINGS_CACHE_PATH.write_text(
            json.dumps(self.query_embedding_cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _dense_query_embedding(self, query: str) -> list[float] | None:
        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            return None
        cache_key = f"{os.environ.get('DENSE_EMBED_MODEL', DENSE_EMBED_MODEL)}::{query.strip().lower()}"
        cached = self.query_embedding_cache.get(cache_key)
        if cached:
            return cached

        try:
            response = httpx.post(
                VOYAGE_EMBEDDINGS_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "input": [query],
                    "model": os.environ.get("DENSE_EMBED_MODEL", DENSE_EMBED_MODEL),
                    "input_type": "query",
                    "truncation": True,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            embedding = response.json()["data"][0]["embedding"]
            self.query_embedding_cache[cache_key] = embedding
            self._save_query_embedding_cache()
            return embedding
        except httpx.HTTPError:
            return None

    def dense_retrieve(self, query: str, top_k: int = DENSE_TOP_K) -> list[RetrievalHit]:
        if not CHROMA_DATA_DIR.exists():
            return []

        try:
            import chromadb
        except ImportError:
            return []

        embedding = self._dense_query_embedding(query)
        if embedding is None:
            return []

        client = chromadb.PersistentClient(path=str(CHROMA_DATA_DIR))
        try:
            collection = client.get_collection(CHROMA_COLLECTION_NAME)
        except Exception:
            return []

        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["distances"],
        )
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        return [RetrievalHit(entity_id=str(entity_id), score=-float(distance)) for entity_id, distance in zip(ids, distances)]

    def _load_splade_model(self) -> bool:
        if self._splade_tokenizer is not None and self._splade_model is not None:
            return True
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError:
            return False

        model_name = os.environ.get("SPLADE_MODEL", SPLADE_MODEL)
        try:
            self._splade_tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self._splade_model = AutoModelForMaskedLM.from_pretrained(
                model_name,
                local_files_only=True,
                use_safetensors=False,
            )
        except OSError:
            return False
        self._splade_model.eval()
        self._torch = torch
        return True

    def _splade_query_vector(self, query: str) -> dict[int, float]:
        if not self._load_splade_model():
            return {}
        torch = self._torch
        encoded = self._splade_tokenizer(
            [query],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = self._splade_model(**encoded)
            token_scores = torch.log1p(torch.relu(output.logits))
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            sparse_scores = (token_scores * attention_mask).max(dim=1).values[0]

        nonzero_indices = (sparse_scores > 0).nonzero(as_tuple=True)[0].tolist()
        return {int(index): float(sparse_scores[index].item()) for index in nonzero_indices}

    def sparse_retrieve(self, query: str, top_k: int = SPARSE_TOP_K) -> list[RetrievalHit]:
        if not self.splade_payload:
            return []
        query_vector = self._splade_query_vector(query)
        if not query_vector:
            return []

        scored: list[RetrievalHit] = []
        for vector in self.splade_payload.get("vectors", []):
            score = 0.0
            for index, value in zip(vector.get("indices", []), vector.get("values", [])):
                score += query_vector.get(int(index), 0.0) * float(value)
            if score > 0:
                scored.append(RetrievalHit(entity_id=str(vector["entity_id"]), score=score))

        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:top_k]

    def rrf_fuse(
        self,
        dense_hits: list[RetrievalHit],
        sparse_hits: list[RetrievalHit],
        top_k: int = RRF_TOP_K,
        rrf_k: int = RRF_K,
    ) -> list[RetrievalHit]:
        scores: dict[str, float] = {}
        for hits in (dense_hits, sparse_hits):
            for rank, hit in enumerate(hits, start=1):
                scores[hit.entity_id] = scores.get(hit.entity_id, 0.0) + 1.0 / (rrf_k + rank)

        fused = [RetrievalHit(entity_id=entity_id, score=score) for entity_id, score in scores.items()]
        fused.sort(key=lambda hit: hit.score, reverse=True)
        return fused[:top_k]

    def rerank_hits(self, query: str, hits: list[RetrievalHit], limit: int = RERANK_TOP_K) -> list[RetrievalHit]:
        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key or not hits:
            return hits[:limit]

        documents = [self.documents_by_id.get(hit.entity_id, {}).get("search_text", "") for hit in hits]
        if not any(documents):
            return hits[:limit]

        try:
            response = httpx.post(
                VOYAGE_RERANK_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "query": query,
                    "documents": documents,
                    "model": os.environ.get("RERANK_MODEL", RERANK_MODEL),
                    "top_k": min(limit, len(documents)),
                },
                timeout=30.0,
            )
            response.raise_for_status()
            reranked = []
            for item in response.json().get("data", []):
                index = int(item.get("index", 0))
                score = float(item.get("relevance_score", item.get("score", 0.0)))
                if 0 <= index < len(hits):
                    reranked.append(RetrievalHit(entity_id=hits[index].entity_id, score=score))
            return reranked or hits[:limit]
        except httpx.HTTPError:
            return hits[:limit]

    def retrieval_debug(self, query: str) -> dict[str, list[dict[str, Any]]]:
        dense_hits = self.dense_retrieve(query, top_k=DENSE_TOP_K)
        sparse_hits = self.sparse_retrieve(query, top_k=SPARSE_TOP_K)
        rrf_hits = self.rrf_fuse(dense_hits, sparse_hits, top_k=RRF_TOP_K)
        fallback_used = False
        if not rrf_hits:
            rrf_hits = self.lexical_retrieve(query, top_k=RRF_TOP_K)
            fallback_used = True
        reranked_hits = self.rerank_hits(query, rrf_hits, limit=RERANK_TOP_K)
        return {
            "dense_top_30": self._debug_hits(dense_hits),
            "splade_top_30": self._debug_hits(sparse_hits),
            "rrf_top_30": self._debug_hits(rrf_hits),
            "rerank_top_10": self._debug_hits(reranked_hits),
            "fallback_used": [{"lexical": fallback_used}],
        }

    def _debug_hits(self, hits: list[RetrievalHit]) -> list[dict[str, Any]]:
        rows = []
        for rank, hit in enumerate(hits, start=1):
            assessment = self.assessments_by_id.get(hit.entity_id)
            if assessment is None:
                continue
            rows.append(
                {
                    "rank": rank,
                    "entity_id": hit.entity_id,
                    "score": hit.score,
                    "name": assessment.name,
                    "url": assessment.url,
                    "test_type": assessment.test_type,
                }
            )
        return rows

    def lexical_retrieve(self, query: str, top_k: int) -> list[RetrievalHit]:
        query_terms = Counter(tokenize(query))
        scored: list[RetrievalHit] = []

        for assessment in self.assessments:
            doc_terms = Counter(tokenize(assessment.searchable_text))
            lexical_score = sum(query_terms[token] * doc_terms.get(token, 0) for token in query_terms)
            name_boost = 3.0 if any(token in tokenize(assessment.name) for token in query_terms) else 0.0
            type_boost = 1.0 if assessment.test_type.lower() in query.lower() else 0.0
            score = lexical_score + name_boost + type_boost
            if score > 0:
                scored.append(RetrievalHit(entity_id=assessment.entity_id, score=score))

        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:top_k]

    def recommend(self, query: str, limit: int = 10) -> list[Recommendation]:
        dense_hits = self.dense_retrieve(query, top_k=DENSE_TOP_K)
        sparse_hits = self.sparse_retrieve(query, top_k=SPARSE_TOP_K)
        hits = self.rrf_fuse(dense_hits, sparse_hits, top_k=RRF_TOP_K)

        if not hits:
            hits = self.lexical_retrieve(query, top_k=RRF_TOP_K)

        final_hits = self.rerank_hits(query, hits, limit=min(limit, RERANK_TOP_K))
        recommendations: list[Recommendation] = []
        for hit in final_hits:
            assessment = self.assessments_by_id.get(hit.entity_id)
            if assessment is None:
                continue
            recommendations.append(
                Recommendation(
                    name=assessment.name,
                    url=assessment.url,
                    test_type=assessment.test_type,
                    reason=self.explain_match(query, assessment),
                )
            )
        return recommendations

    def explain_match(self, query: str, assessment: Assessment) -> str:
        query_terms = set(tokenize(query)) - STOPWORDS - VAGUE_TERMS
        text_terms = set(tokenize(assessment.searchable_text))
        matched = sorted(term for term in query_terms & text_terms if len(term) > 2)
        if matched:
            return f"Matches catalog text for: {', '.join(matched[:5])}."
        if assessment.test_type.lower() in query.lower():
            return f"Matches the requested {assessment.test_type.lower()} assessment type."
        return "Selected from the SHL catalog by hybrid retrieval and reranking."

    def find_assessment(self, label: str) -> Assessment | None:
        label_tokens = set(tokenize(label))
        if not label_tokens:
            return None

        aliases = {
            "opq": "Occupational Personality Questionnaire OPQ32r",
            "gsa": "Global Skills Assessment",
        }
        normalized_label = label.strip().lower()
        if normalized_label in aliases:
            label = aliases[normalized_label]
            label_tokens = set(tokenize(label))

        best: tuple[int, Assessment] | None = None
        for assessment in self.assessments:
            name_tokens = set(tokenize(assessment.name))
            text_tokens = set(tokenize(assessment.searchable_text))
            score = 4 * len(label_tokens & name_tokens) + len(label_tokens & text_tokens)
            if label.lower() in assessment.name.lower():
                score += 10
            if score > 0 and (best is None or score > best[0]):
                best = (score, assessment)
        return best[1] if best else None

    def describe_assessment(self, assessment: Assessment) -> str:
        raw = assessment.raw
        description = str(raw.get("description", "")).strip()
        keys = raw.get("keys", [])
        job_levels = raw.get("job_levels", [])
        duration = raw.get("duration", "")
        parts = [
            f"{assessment.name} ({assessment.test_type})",
            f"URL: {assessment.url}",
        ]
        if description:
            parts.append(f"Catalog description: {description}")
        if keys:
            parts.append(f"Catalog categories: {', '.join(str(item) for item in keys)}")
        if job_levels:
            parts.append(f"Job levels: {', '.join(str(item) for item in job_levels)}")
        if duration:
            parts.append(f"Duration: {duration}")
        return "\n".join(parts)
