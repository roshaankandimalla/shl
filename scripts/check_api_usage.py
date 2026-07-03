import asyncio
import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas import Message
from app.services.chat_controller import ChatController


calls: list[str] = []

original_sync_post = httpx.Client.post
original_async_post = httpx.AsyncClient.post
original_module_post = httpx.post


def record_call(url: object) -> None:
    text = str(url)
    if "voyageai.com" in text:
        calls.append("voyage_rerank" if "rerank" in text else "voyage_embeddings")
    if "generativelanguage.googleapis.com" in text:
        calls.append("gemini_generation")


def sync_post(self: httpx.Client, url: object, *args: object, **kwargs: object) -> httpx.Response:
    record_call(url)
    return original_sync_post(self, url, *args, **kwargs)


def module_post(url: object, *args: object, **kwargs: object) -> httpx.Response:
    record_call(url)
    return original_module_post(url, *args, **kwargs)


async def async_post(
    self: httpx.AsyncClient, url: object, *args: object, **kwargs: object
) -> httpx.Response:
    record_call(url)
    return await original_async_post(self, url, *args, **kwargs)


async def main() -> None:
    httpx.Client.post = sync_post
    httpx.AsyncClient.post = async_post
    httpx.post = module_post

    print(f"env_loaded: voyage={bool(os.environ.get('VOYAGE_API_KEY'))} gemini={bool(os.environ.get('GEMINI_API_KEY'))}")

    controller = ChatController()
    query = (
        "I am hiring a mid-level Java backend developer who works with "
        "stakeholders. Recommend SHL assessments."
    )
    dense_hits = controller.recommender.dense_retrieve(query, top_k=3)
    response = controller.respond(
        [
            Message(
                role="user",
                content=query,
            )
        ]
    )

    print(f"recommendations={len(response.recommendations)}")
    print(f"dense_retrieval_active={bool(dense_hits)}")
    print(f"voyage_embeddings_called={'voyage_embeddings' in calls}")
    print(f"voyage_rerank_called={'voyage_rerank' in calls}")
    print(f"gemini_generation_called={'gemini_generation' in calls}")
    print(f"api_calls={','.join(sorted(set(calls)))}")
    print(f"reply_preview={response.reply[:180].replace(chr(10), ' ')}")


if __name__ == "__main__":
    asyncio.run(main())
