import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import CATALOG_PATH


@dataclass(frozen=True)
class Assessment:
    entity_id: str
    name: str
    url: str
    test_type: str
    raw: dict[str, Any]

    @property
    def searchable_text(self) -> str:
        parts: list[str] = [self.name, self.test_type]
        for value in self.raw.values():
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, list):
                parts.extend(str(item) for item in value)
            elif isinstance(value, dict):
                parts.extend(str(item) for item in value.values())
        return " ".join(parts)


def _first_present(data: dict[str, Any], keys: tuple[str, ...], default: str = "") -> str:
    for key in keys:
        value = data.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return default


TEST_TYPE_CODES = {
    "Ability & Aptitude": "A",
    "Assessment Exercises": "E",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(value).strip()]


def _infer_test_type(data: dict[str, Any]) -> str:
    catalog_keys = _as_text_list(data.get("keys")) + _as_text_list(data.get("keys_raw"))
    codes = [TEST_TYPE_CODES[key] for key in TEST_TYPE_CODES if key in catalog_keys]
    return "".join(codes)


def _unwrap_catalog(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("assessments", "products", "data", "items", "catalog"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def load_catalog(path: Path = CATALOG_PATH) -> list[Assessment]:
    if not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = _unwrap_catalog(payload)
    assessments: list[Assessment] = []

    for row in rows:
        entity_id = _first_present(row, ("entity_id", "id", "product_id"))
        name = _first_present(row, ("name", "assessment_name", "title", "product_name"))
        url = _first_present(row, ("url", "link", "product_url"))
        test_type = _first_present(row, ("test_type", "type", "assessment_type")) or _infer_test_type(row)
        if not name or not url:
            continue
        assessments.append(Assessment(entity_id=entity_id, name=name, url=url, test_type=test_type, raw=row))

    return assessments
