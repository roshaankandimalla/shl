"""Run the full index build pipeline.

This is a small orchestrator. Run the individual scripts when you want a
specific step:
- scripts/build_documents.py
- scripts/build_dense_embeddings.py
- scripts/build_splade_embeddings.py
"""

from __future__ import annotations

import argparse
import os

from build_dense_embeddings import build_dense_embeddings
from build_documents import build_documents
from build_splade_embeddings import build_splade_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SHL retrieval chunks and embeddings.")
    parser.add_argument("--skip-dense", action="store_true", help="Skip Voyage dense embeddings.")
    parser.add_argument("--skip-chroma", action="store_true", help="Skip Chroma dense index inside dense build.")
    parser.add_argument("--skip-splade", action="store_true", help="Skip SPLADE sparse vectors.")
    parser.add_argument("--force-documents", action="store_true", help="Rebuild assessment chunks.")
    parser.add_argument("--force-dense", action="store_true", help="Rebuild dense embedding cache and outputs.")
    parser.add_argument("--reset-chroma", action="store_true", help="Delete and recreate the Chroma collection.")
    parser.add_argument("--force-splade", action="store_true", help="Rebuild SPLADE cache and outputs.")
    parser.add_argument("--dense-model", default=None)
    parser.add_argument("--splade-model", default=None)
    parser.add_argument("--dense-batch-size", type=int, default=64)
    parser.add_argument("--chroma-batch-size", type=int, default=64)
    parser.add_argument("--splade-batch-size", type=int, default=8)
    parser.add_argument("--splade-max-length", type=int, default=512)
    parser.add_argument("--splade-top-k", type=int, default=256)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_documents(force=args.force_documents)

    if not args.skip_dense:
        build_dense_embeddings(
            model=args.dense_model or os.environ.get("DENSE_EMBED_MODEL", "voyage-4-large"),
            batch_size=args.dense_batch_size,
            timeout_seconds=args.timeout_seconds,
            force=args.force_dense,
            skip_chroma=args.skip_chroma,
            chroma_batch_size=args.chroma_batch_size,
            reset_chroma=args.reset_chroma,
        )

    if not args.skip_splade:
        build_splade_embeddings(
            model_name=args.splade_model or os.environ.get("SPLADE_MODEL", "naver/splade-cocondenser-ensembledistil"),
            batch_size=args.splade_batch_size,
            max_length=args.splade_max_length,
            top_k=args.splade_top_k,
            force=args.force_splade,
        )


if __name__ == "__main__":
    main()
