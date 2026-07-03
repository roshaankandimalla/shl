"""Build resumable Voyage dense embeddings and Chroma dense index."""

from __future__ import annotations

import argparse
import json
import os

import httpx
import numpy as np

from index_common import (
    CHROMA_COLLECTION_NAME,
    DENSE_CACHE_PATH,
    DENSE_EMBEDDINGS_PATH,
    DENSE_METADATA_PATH,
    append_jsonl,
    entity_ids,
    load_documents,
    read_jsonl_cache,
)
from app.config import CHROMA_DATA_DIR


DEFAULT_DENSE_MODEL = "voyage-4-large"
VOYAGE_EMBEDDINGS_URL = "https://api.voyageai.com/v1/embeddings"


def _batched(values: list[dict], batch_size: int) -> list[list[dict]]:
    return [values[index : index + batch_size] for index in range(0, len(values), batch_size)]


def _final_dense_is_current(documents: list[dict], model: str) -> bool:
    if not DENSE_EMBEDDINGS_PATH.exists() or not DENSE_METADATA_PATH.exists():
        return False
    metadata = json.loads(DENSE_METADATA_PATH.read_text(encoding="utf-8"))
    return metadata.get("model") == model and metadata.get("entity_ids") == entity_ids(documents)


def _write_final_outputs(documents: list[dict], cache: dict[str, dict], model: str) -> None:
    ordered_embeddings = [cache[str(document["entity_id"])]["embedding"] for document in documents]
    matrix = np.asarray(ordered_embeddings, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)

    np.save(DENSE_EMBEDDINGS_PATH, matrix)
    DENSE_METADATA_PATH.write_text(
        json.dumps(
            {
                "model": model,
                "count": len(documents),
                "dimension": int(matrix.shape[1]),
                "entity_ids": entity_ids(documents),
                "input_type": "document",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved dense embeddings to {DENSE_EMBEDDINGS_PATH}")
    print(f"Saved dense metadata to {DENSE_METADATA_PATH}")


def _chroma_metadata(document: dict) -> dict[str, str | int | float | bool]:
    metadata = document.get("metadata", {})
    return {
        "entity_id": str(document["entity_id"]),
        "name": document["name"],
        "url": document["url"],
        "test_type": document["test_type"],
        "keys": ", ".join(str(item) for item in metadata.get("keys", [])),
        "job_levels": ", ".join(str(item) for item in metadata.get("job_levels", [])),
        "duration": str(metadata.get("duration", "")),
        "remote": str(metadata.get("remote", "")),
        "adaptive": str(metadata.get("adaptive", "")),
    }


def build_chroma_index(documents: list[dict], embeddings: np.ndarray, model: str, batch_size: int, reset: bool) -> None:
    try:
        import chromadb
    except ImportError:
        print("Skipped Chroma index: missing dependency chromadb. Install requirements.txt first.")
        return

    if len(documents) != embeddings.shape[0]:
        raise ValueError(f"Document count {len(documents)} does not match embedding rows {embeddings.shape[0]}.")

    CHROMA_DATA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DATA_DIR))
    if reset:
        try:
            client.delete_collection(CHROMA_COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={
            "description": "SHL assessment dense index",
            "embedding_model": model,
            "hnsw:space": "cosine",
        },
    )

    ids = [str(document["entity_id"]) for document in documents]
    texts = [document["search_text"] for document in documents]
    metadatas = [_chroma_metadata(document) for document in documents]
    vectors = embeddings.astype(float).tolist()

    for start in range(0, len(documents), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            embeddings=vectors[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"Chroma upserted {min(end, len(documents))}/{len(documents)} assessment chunks")

    print(f"Chroma collection '{CHROMA_COLLECTION_NAME}' has {collection.count()} records.")
    print(f"Chroma data stored at {CHROMA_DATA_DIR}")


def _load_dense_matrix() -> np.ndarray:
    return np.load(DENSE_EMBEDDINGS_PATH)


def build_dense_embeddings(
    model: str,
    batch_size: int,
    timeout_seconds: float,
    force: bool = False,
    skip_chroma: bool = False,
    chroma_batch_size: int = 64,
    reset_chroma: bool = False,
) -> None:
    documents = load_documents()
    if _final_dense_is_current(documents, model) and not force:
        print(f"Dense embeddings already complete at {DENSE_EMBEDDINGS_PATH}")
        if not skip_chroma:
            build_chroma_index(
                documents=documents,
                embeddings=_load_dense_matrix(),
                model=model,
                batch_size=chroma_batch_size,
                reset=reset_chroma,
            )
        return

    if force and DENSE_CACHE_PATH.exists():
        DENSE_CACHE_PATH.unlink()

    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        print("Skipped dense embeddings: VOYAGE_API_KEY is not set.")
        return

    cache = read_jsonl_cache(DENSE_CACHE_PATH, model=model)
    remaining = [document for document in documents if str(document["entity_id"]) not in cache]
    print(f"Dense cache has {len(cache)}/{len(documents)} embeddings. Remaining: {len(remaining)}")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    with httpx.Client(timeout=timeout_seconds) as client:
        for batch_number, batch in enumerate(_batched(remaining, batch_size), start=1):
            response = client.post(
                VOYAGE_EMBEDDINGS_URL,
                headers=headers,
                json={
                    "input": [document["search_text"] for document in batch],
                    "model": model,
                    "input_type": "document",
                    "truncation": True,
                },
            )
            response.raise_for_status()
            payload = response.json()
            records = []
            for document, item in zip(batch, payload["data"], strict=True):
                records.append(
                    {
                        "entity_id": str(document["entity_id"]),
                        "model": model,
                        "embedding": item["embedding"],
                    }
                )
            append_jsonl(DENSE_CACHE_PATH, records)
            cache.update({record["entity_id"]: record for record in records})
            print(f"Dense batch {batch_number}: cached {len(cache)}/{len(documents)} embeddings")

    if len(cache) == len(documents):
        _write_final_outputs(documents, cache, model)
        if not skip_chroma:
            build_chroma_index(
                documents=documents,
                embeddings=_load_dense_matrix(),
                model=model,
                batch_size=chroma_batch_size,
                reset=reset_chroma,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Voyage dense embeddings with resume cache.")
    parser.add_argument("--model", default=os.environ.get("DENSE_EMBED_MODEL", DEFAULT_DENSE_MODEL))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--force", action="store_true", help="Ignore existing final/cache files and rebuild.")
    parser.add_argument("--skip-chroma", action="store_true", help="Build Voyage embeddings but skip Chroma upsert.")
    parser.add_argument("--chroma-batch-size", type=int, default=64)
    parser.add_argument("--reset-chroma", action="store_true", help="Delete and recreate the Chroma collection.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dense_embeddings(
        model=args.model,
        batch_size=args.batch_size,
        timeout_seconds=args.timeout_seconds,
        force=args.force,
        skip_chroma=args.skip_chroma,
        chroma_batch_size=args.chroma_batch_size,
        reset_chroma=args.reset_chroma,
    )


if __name__ == "__main__":
    main()
