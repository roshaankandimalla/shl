import os

import httpx

from app.config import GEMINI_API_BASE, GENERATION_MODEL
from app.schemas import Recommendation


def recommendation_reply(recommendations: list[Recommendation], context: str, refined: bool = False) -> str:
    fallback = (
        f"Updated shortlist: here are {len(recommendations)} SHL assessments that match the revised constraints."
        if refined
        else f"Here are {len(recommendations)} SHL assessments that best match the role and constraints you shared."
    )
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or not recommendations:
        return fallback

    names = "\n".join(f"- {item.name} ({item.test_type}): {item.reason}" for item in recommendations)
    prompt = (
        "Write one concise sentence introducing this SHL assessment shortlist. "
        "Do not add assessment names not listed. Do not add URLs. "
        f"User context: {context}\nShortlist:\n{names}"
    )
    model = os.environ.get("GENERATION_MODEL", GENERATION_MODEL)
    url = f"{GEMINI_API_BASE}/{model}:generateContent?key={api_key}"

    try:
        response = httpx.post(
            url,
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=20.0,
        )
        response.raise_for_status()
        candidates = response.json().get("candidates", [])
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        text = parts[0].get("text", "").strip() if parts else ""
        return text or fallback
    except httpx.HTTPError:
        return fallback
