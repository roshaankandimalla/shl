import re

from app.schemas import ChatResponse, Message
from app.services.generator import recommendation_reply
from app.services.recommender import Recommender


SENIORITY_PATTERN = re.compile(
    r"\b(entry[- ]?level|junior|graduate|mid[- ]?level|mid|senior|lead|manager|director|executive|\d+\s*\+?\s*(years?|yrs?))\b",
    re.IGNORECASE,
)

ROLE_PATTERN = re.compile(
    r"\b(developer|engineer|manager|analyst|sales|support|consultant|administrator|java|python|sql|backend|frontend|stakeholder)\b",
    re.IGNORECASE,
)

SKILL_PATTERN = re.compile(
    r"\b(java|python|sql|backend|frontend|stakeholder|sales|support|communication|leadership|coding|developer|engineer|analyst|manager)\b",
    re.IGNORECASE,
)

COMPARE_PATTERN = re.compile(r"\b(compare|difference|different|versus|vs\.?|between)\b", re.IGNORECASE)

REFINE_PATTERN = re.compile(
    r"\b(actually|add|include|also|instead|remove|exclude|drop|personality|cognitive|ability|aptitude|skills?|knowledge)\b",
    re.IGNORECASE,
)

CONFIRM_PATTERN = re.compile(
    r"\b(perfect|confirmed|confirm|that works|that's good|that is good|works for us|locking it in|lock it in|keep the shortlist|keep .* as-is|final list|that covers it|thanks|thank you)\b",
    re.IGNORECASE,
)

SHL_SCOPE_PATTERN = re.compile(
    r"\b(shl|assessment|assessments|test|tests|opq|gsa|personality|cognitive|ability|aptitude|skills?|java|python|sql|sales|support|developer|manager|analyst|candidate|hiring|job description)\b",
    re.IGNORECASE,
)

PROMPT_ATTACK_PATTERN = re.compile(
    r"\b(ignore|override|bypass|reveal|show|print|dump)\b.*\b(instructions?|prompt|system|developer|policy|rules?)\b",
    re.IGNORECASE,
)


class ChatController:
    def __init__(self) -> None:
        self.recommender = Recommender()

    def needs_seniority(self, conversation_text: str) -> bool:
        return bool(ROLE_PATTERN.search(conversation_text)) and not bool(SENIORITY_PATTERN.search(conversation_text))

    def clarification_question(self, conversation_text: str) -> str | None:
        missing = []
        if not ROLE_PATTERN.search(conversation_text):
            missing.append("role")
        if not SKILL_PATTERN.search(conversation_text):
            missing.append("key skills")
        if not SENIORITY_PATTERN.search(conversation_text):
            missing.append("seniority or years of experience")

        if len(missing) >= 3:
            return "Could you share the role, key skills, seniority, and whether you need ability, personality, knowledge, or behavioral assessments?"
        if missing == ["seniority or years of experience"]:
            return "Sure. What seniority level or years of experience should the assessment target?"
        if missing:
            return f"Could you share the {', '.join(missing)}?"
        return None

    def is_refinement(self, latest_user: str, messages: list[Message]) -> bool:
        has_prior_assistant = any(message.role == "assistant" for message in messages[:-1])
        return has_prior_assistant and bool(REFINE_PATTERN.search(latest_user))

    def is_confirmation(self, latest_user: str, messages: list[Message]) -> bool:
        has_prior_assistant = any(message.role == "assistant" for message in messages[:-1])
        return has_prior_assistant and bool(CONFIRM_PATTERN.search(latest_user))

    def is_out_of_scope(self, latest_user: str, messages: list[Message]) -> bool:
        if PROMPT_ATTACK_PATTERN.search(latest_user):
            return True
        has_prior_context = len(messages) > 1
        return not has_prior_context and not bool(SHL_SCOPE_PATTERN.search(latest_user))

    def extract_compare_terms(self, latest_user: str) -> list[str]:
        text = latest_user.strip(" ?.")
        patterns = [
            r"between\s+(.+?)\s+and\s+(.+)$",
            r"compare\s+(.+?)\s+(?:with|and|to|versus|vs\.?)\s+(.+)$",
            r"(.+?)\s+(?:versus|vs\.?)\s+(.+)$",
            r"difference\s+between\s+(.+?)\s+and\s+(.+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return [self.clean_compare_term(match.group(1)), self.clean_compare_term(match.group(2))]

        known_terms = []
        for assessment in self.recommender.assessments:
            if assessment.name.lower() in text.lower():
                known_terms.append(assessment.name)
        for alias in ("OPQ", "GSA"):
            if alias.lower() in text.lower() and alias not in known_terms:
                known_terms.append(alias)
        return known_terms[:2]

    def clean_compare_term(self, value: str) -> str:
        cleaned = re.sub(
            r"\b(what|is|the|difference|different|compare|between|assessment|assessments|test|tests|shl|please)\b",
            " ",
            value,
            flags=re.IGNORECASE,
        )
        return re.sub(r"\s+", " ", cleaned).strip(" ?.,")

    def compare(self, latest_user: str) -> ChatResponse:
        terms = [term for term in self.extract_compare_terms(latest_user) if term]
        if len(terms) < 2:
            return ChatResponse(
                reply="Which two SHL assessments would you like me to compare?",
                recommendations=[],
                end_of_conversation=False,
            )

        left = self.recommender.find_assessment(terms[0])
        right = self.recommender.find_assessment(terms[1])
        if left is None or right is None:
            return ChatResponse(
                reply="I could not find both assessments in the SHL catalog. Please use their catalog names or abbreviations.",
                recommendations=[],
                end_of_conversation=False,
            )

        reply = (
            "Here is a catalog-grounded comparison.\n\n"
            f"{self.recommender.describe_assessment(left)}\n\n"
            f"{self.recommender.describe_assessment(right)}"
        )
        return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)

    def respond(self, messages: list[Message]) -> ChatResponse:
        user_messages = [message.content for message in messages if message.role == "user"]
        latest_user = user_messages[-1] if user_messages else ""
        conversation_text = "\n".join(user_messages)

        if self.recommender.is_off_topic(latest_user) or self.is_out_of_scope(latest_user, messages):
            return ChatResponse(
                reply="I can only help with SHL assessment recommendations from the provided catalog.",
                recommendations=[],
                end_of_conversation=False,
            )

        if COMPARE_PATTERN.search(latest_user):
            return self.compare(latest_user)

        clarification = self.clarification_question(conversation_text)
        if self.recommender.is_vague(conversation_text) or (
            clarification and not self.is_refinement(latest_user, messages)
        ):
            return ChatResponse(
                reply=clarification
                or "Could you share the role, key skills, seniority, and whether you need ability, personality, knowledge, or behavioral assessments?",
                recommendations=[],
                end_of_conversation=False,
            )

        recommendations = self.recommender.recommend(conversation_text, limit=10)
        if not recommendations:
            return ChatResponse(
                reply="I need a little more detail to match this to SHL assessments. What role and skills should the assessment focus on?",
                recommendations=[],
                end_of_conversation=False,
            )

        return ChatResponse(
            reply=recommendation_reply(
                recommendations=recommendations,
                context=conversation_text,
                refined=self.is_refinement(latest_user, messages),
            ),
            recommendations=recommendations,
            end_of_conversation=self.is_confirmation(latest_user, messages),
        )
