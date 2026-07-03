from fastapi.testclient import TestClient

import app.services.chat_controller as chat_module
from app.main import app
from app.main import controller
from app.schemas import Recommendation


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_schema_for_vague_query() -> None:
    response = client.post("/chat", json={"messages": [{"role": "user", "content": "I need an assessment"}]})
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"reply", "recommendations", "end_of_conversation"}
    assert payload["recommendations"] == []
    assert payload["end_of_conversation"] is False


def test_chat_recommends_between_one_and_ten(monkeypatch) -> None:
    def fake_recommend(query: str, limit: int = 10) -> list[Recommendation]:
        return [
            Recommendation(
                name="Java 8",
                url="https://example.com/java-8",
                test_type="Knowledge & Skills",
                reason="Matches catalog text for: java.",
            )
        ]

    monkeypatch.setattr(controller.recommender, "recommend", fake_recommend)
    monkeypatch.setattr(
        chat_module,
        "recommendation_reply",
        lambda recommendations, context, refined=False: "Here are 1 SHL assessments that best match the role and constraints you shared.",
    )

    response = client.post(
        "/chat",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "I am hiring a mid-level Java backend developer who works with stakeholders.",
                }
            ]
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert 1 <= len(payload["recommendations"]) <= 10
    assert payload["recommendations"][0]["reason"]


def test_off_topic_refusal() -> None:
    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Give me legal advice about terminating an employee."}]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["recommendations"] == []
    assert "only help with SHL assessment" in payload["reply"]


def test_compare_opq_and_gsa_is_grounded() -> None:
    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "What is the difference between OPQ and GSA?"}]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["recommendations"] == []
    assert "catalog-grounded comparison" in payload["reply"]
    assert "URL:" in payload["reply"]
