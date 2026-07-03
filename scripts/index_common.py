"""Shared paths and helpers for index-building scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import CHROMA_DATA_DIR, PROCESSED_DATA_DIR


DOCUMENTS_PATH = PROCESSED_DATA_DIR / "assessment_documents.json"
DENSE_EMBEDDINGS_PATH = PROCESSED_DATA_DIR / "dense_embeddings.npy"
DENSE_METADATA_PATH = PROCESSED_DATA_DIR / "dense_embeddings_meta.json"
DENSE_CACHE_PATH = PROCESSED_DATA_DIR / "dense_embeddings_cache.jsonl"
SPLADE_VECTORS_PATH = PROCESSED_DATA_DIR / "splade_sparse_vectors.json"
SPLADE_CACHE_PATH = PROCESSED_DATA_DIR / "splade_sparse_vectors_cache.jsonl"
CHROMA_COLLECTION_NAME = "shl_assessments"


def load_documents() -> list[dict[str, Any]]:
    if not DOCUMENTS_PATH.exists():
        raise FileNotFoundError(f"Missing {DOCUMENTS_PATH}. Run scripts/build_documents.py first.")
    return json.loads(DOCUMENTS_PATH.read_text(encoding="utf-8"))


def read_jsonl_cache(path: Path, model: str, extra_filters: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    filters = extra_filters or {}
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("model") != model:
            continue
        if any(record.get(key) != value for key, value in filters.items()):
            continue
        entity_id = str(record.get("entity_id", ""))
        if entity_id:
            records[entity_id] = record
    return records


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def entity_ids(documents: list[dict[str, Any]]) -> list[str]:
    return [str(document["entity_id"]) for document in documents]
