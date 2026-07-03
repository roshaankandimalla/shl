import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas import Message
from app.services.chat_controller import ChatController


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = BASE_DIR / "data" / "eval" / "questions.json"
DEFAULT_OUTPUT = BASE_DIR / "data" / "eval" / "results.json"


def load_questions(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def score_case(expectation: str, payload: dict) -> bool:
    reply = payload["reply"].lower()
    recommendation_count = len(payload["recommendations"])
    if expectation == "recommendations":
        return 1 <= recommendation_count <= 10
    if expectation == "clarification":
        return recommendation_count == 0 and any(word in reply for word in ("share", "seniority", "role", "skills"))
    if expectation == "comparison":
        return recommendation_count == 0 and "catalog-grounded comparison" in reply
    if expectation == "refusal":
        return recommendation_count == 0 and "only help with shl assessment" in reply
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small SHL recommender behavior evaluation.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    controller = ChatController()
    results = []
    started = perf_counter()

    for case in load_questions(args.questions):
        messages = [Message(**message) for message in case["messages"]]
        case_started = perf_counter()
        response = controller.respond(messages)
        payload = response.model_dump()
        elapsed_ms = round((perf_counter() - case_started) * 1000, 2)
        passed = score_case(case.get("expectation", ""), payload)
        results.append(
            {
                "id": case["id"],
                "expectation": case.get("expectation", ""),
                "passed": passed,
                "elapsed_ms": elapsed_ms,
                "response": payload,
            }
        )
        status = "PASS" if passed else "FAIL"
        print(f"{status} {case['id']} ({elapsed_ms} ms)")

    passed_count = sum(1 for result in results if result["passed"])
    output = {
        "summary": {
            "passed": passed_count,
            "total": len(results),
            "elapsed_ms": round((perf_counter() - started) * 1000, 2),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
