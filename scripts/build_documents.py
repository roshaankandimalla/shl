"""Build normalized assessment-level retrieval chunks.

Each catalog `entity_id` is one SHL assessment and one retrieval chunk.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from index_common import DOCUMENTS_PATH
from app.services.catalog import Assessment, load_catalog


def _metadata(assessment: Assessment) -> dict[str, Any]:
    raw = assessment.raw
    return {
        "job_levels": raw.get("job_levels", []),
        "languages": raw.get("languages", []),
        "duration": raw.get("duration", ""),
        "remote": raw.get("remote", ""),
        "adaptive": raw.get("adaptive", ""),
        "keys": raw.get("keys", []),
        "status": raw.get("status", ""),
    }


def assessment_to_document(assessment: Assessment) -> dict[str, Any]:
    return {
        "entity_id": assessment.entity_id,
        "name": assessment.name,
        "url": assessment.url,
        "test_type": assessment.test_type,
        "search_text": assessment.searchable_text,
        "metadata": _metadata(assessment),
    }


def build_documents(force: bool = False) -> list[dict[str, Any]]:
    if DOCUMENTS_PATH.exists() and not force:
        documents = json.loads(DOCUMENTS_PATH.read_text(encoding="utf-8"))
        print(f"Using existing {len(documents)} assessment chunks from {DOCUMENTS_PATH}")
        return documents

    assessments = load_catalog()
    documents = [assessment_to_document(assessment) for assessment in assessments]

    DOCUMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCUMENTS_PATH.write_text(json.dumps(documents, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(documents)} assessment chunks to {DOCUMENTS_PATH}")
    return documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build normalized SHL assessment chunks.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if the output already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_documents(force=args.force)


if __name__ == "__main__":
    main()
