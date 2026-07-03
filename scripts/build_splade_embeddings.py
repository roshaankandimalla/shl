"""Build resumable SPLADE sparse vectors for assessment chunks."""

from __future__ import annotations

import argparse
import json
import os

from index_common import (
    SPLADE_CACHE_PATH,
    SPLADE_VECTORS_PATH,
    append_jsonl,
    load_documents,
    read_jsonl_cache,
)


DEFAULT_SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"


def _batched(values: list[dict], batch_size: int) -> list[list[dict]]:
    return [values[index : index + batch_size] for index in range(0, len(values), batch_size)]


def _final_splade_is_current(documents: list[dict], model: str, max_length: int, top_k: int) -> bool:
    if not SPLADE_VECTORS_PATH.exists():
        return False
    payload = json.loads(SPLADE_VECTORS_PATH.read_text(encoding="utf-8"))
    return (
        payload.get("model") == model
        and payload.get("count") == len(documents)
        and payload.get("max_length") == max_length
        and payload.get("top_k") == top_k
    )


def _write_final_outputs(documents: list[dict], cache: dict[str, dict], model: str, max_length: int, top_k: int) -> None:
    vectors = []
    for document in documents:
        cached = cache[str(document["entity_id"])]
        vectors.append(
            {
                "entity_id": str(document["entity_id"]),
                "name": document["name"],
                "indices": cached["indices"],
                "tokens": cached["tokens"],
                "values": cached["values"],
            }
        )

    SPLADE_VECTORS_PATH.write_text(
        json.dumps(
            {
                "model": model,
                "count": len(vectors),
                "max_length": max_length,
                "top_k": top_k,
                "vectors": vectors,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved SPLADE sparse vectors to {SPLADE_VECTORS_PATH}")


def build_splade_embeddings(model_name: str, batch_size: int, max_length: int, top_k: int, force: bool = False) -> None:
    documents = load_documents()
    if _final_splade_is_current(documents, model_name, max_length, top_k) and not force:
        print(f"SPLADE vectors already complete at {SPLADE_VECTORS_PATH}")
        return

    if force and SPLADE_CACHE_PATH.exists():
        SPLADE_CACHE_PATH.unlink()

    try:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError as exc:
        print(f"Skipped SPLADE vectors: missing dependency {exc.name}.")
        return

    filters = {"max_length": max_length, "top_k": top_k}
    cache = read_jsonl_cache(SPLADE_CACHE_PATH, model=model_name, extra_filters=filters)
    remaining = [document for document in documents if str(document["entity_id"]) not in cache]
    print(f"SPLADE cache has {len(cache)}/{len(documents)} vectors. Remaining: {len(remaining)}")

    if remaining:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        model.eval()

        with torch.no_grad():
            for batch_number, batch in enumerate(_batched(remaining, batch_size), start=1):
                encoded = tokenizer(
                    [document["search_text"] for document in batch],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                output = model(**encoded)
                token_scores = torch.log1p(torch.relu(output.logits))
                attention_mask = encoded["attention_mask"].unsqueeze(-1)
                sparse_scores = (token_scores * attention_mask).max(dim=1).values

                records = []
                for document, row in zip(batch, sparse_scores, strict=True):
                    positive_count = int((row > 0).sum().item())
                    keep = min(top_k, positive_count)
                    if keep == 0:
                        indices: list[int] = []
                        values: list[float] = []
                    else:
                        values_tensor, indices_tensor = torch.topk(row, k=keep)
                        indices = [int(index) for index in indices_tensor.tolist()]
                        values = [float(value) for value in values_tensor.tolist()]
                    records.append(
                        {
                            "entity_id": str(document["entity_id"]),
                            "model": model_name,
                            "max_length": max_length,
                            "top_k": top_k,
                            "indices": indices,
                            "tokens": tokenizer.convert_ids_to_tokens(indices),
                            "values": values,
                        }
                    )

                append_jsonl(SPLADE_CACHE_PATH, records)
                cache.update({record["entity_id"]: record for record in records})
                print(f"SPLADE batch {batch_number}: cached {len(cache)}/{len(documents)} vectors")

    if len(cache) == len(documents):
        _write_final_outputs(documents, cache, model_name, max_length, top_k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SPLADE sparse vectors with resume cache.")
    parser.add_argument("--model", default=os.environ.get("SPLADE_MODEL", DEFAULT_SPLADE_MODEL))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--force", action="store_true", help="Ignore existing final/cache files and rebuild.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_splade_embeddings(
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        top_k=args.top_k,
        force=args.force,
    )


if __name__ == "__main__":
    main()
