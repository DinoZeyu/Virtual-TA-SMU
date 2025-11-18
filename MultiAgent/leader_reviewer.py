# MultiAgent/leader_reviewer.py

from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core.models import UserMessage
from .utils import extract_text_clean


# ======================================================
#  Leader (Router)
# ======================================================
def build_leader_agent() -> OllamaChatCompletionClient:
    """
    Leader model: Classifies the query into:
      - math
      - physics
      - chemistry
      - web
    """
    return OllamaChatCompletionClient(
        model="mistral:7b",
        model_info={
            "name": "mistral:7b",
            "type": "ollama",
            "format": "text",
            "json_output": False,
            "vision": False,
            "function_calling": False,
        },
    )


# ======================================================
#  Reviewer
# ======================================================
def build_reviewer_agent() -> OllamaChatCompletionClient:
    """
    Reviewer model: checks correctness and fixes wrong answers.
    """
    return OllamaChatCompletionClient(
        model="gemma3:12b",
        model_info={
            "name": "gemma3:12b",
            "type": "ollama",
            "format": "text",
            "json_output": False,
            "vision": False,
            "function_calling": False,
        },
    )


# ======================================================
#  Classification
# ======================================================
async def classify_subject(leader: OllamaChatCompletionClient, query: str) -> str:
    """
    Use the leader model to classify a query.
    """
    prompt = (
        "You are a routing model.\n"
        "Classify the following question into EXACTLY ONE category:\n"
        " - Math\n"
        " - Physics\n"
        " - Chemistry\n"
        " - Web\n\n"
        "Use 'Web' for questions requiring real-time or internet data.\n"
        "Respond with ONE WORD ONLY.\n\n"
        f"Question: {query}"
    )

    resp = await leader.create([UserMessage(content=prompt, source="user")])
    text = extract_text_clean(resp).lower()

    if "math" in text:
        return "math"
    if "phys" in text:
        return "physics"
    if "chem" in text:
        return "chemistry"
    if "web" in text or "search" in text:
        return "web"

    return "unknown"


# ======================================================
#  Reviewer Logic
# ======================================================
async def review_answer(
    reviewer: OllamaChatCompletionClient,
    question: str,
    answer: str,
) -> str:
    """
    Reviewer validates the answer.

    If correct → "APPROVED"
    If wrong → corrected answer:
        English explanation
        $$ ... $$
    """
    prompt = (
        "You are a strict academic reviewer.\n\n"
        "TASK:\n"
        "1. Read the user's question and the proposed answer.\n"
        "2. Decide if the answer is fully correct.\n\n"
        "IF the answer is correct:\n"
        "  - Reply ONLY with: APPROVED\n\n"
        "IF the answer is NOT correct:\n"
        "  - Reply ONLY with a corrected answer in this exact format:\n"
        "      [English explanation]\n"
        "      $$ ... $$\n"
        "  - The LaTeX block must contain ONLY math symbols.\n"
        "  - Never output HTML or Markdown fences.\n\n"
        f"Question:\n{question}\n\n"
        f"Proposed answer:\n{answer}\n"
    )

    resp = await reviewer.create([UserMessage(content=prompt, source="user")])
    return extract_text_clean(resp)
