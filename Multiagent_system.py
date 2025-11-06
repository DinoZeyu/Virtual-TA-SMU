import argparse
import asyncio
from MultiAgent import build_math_agent, build_physics_agent, build_chemistry_agent
from autogen_core.models import UserMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient

math_agent = build_math_agent()
physics_agent = build_physics_agent()
chemistry_agent = build_chemistry_agent()

leader_client = OllamaChatCompletionClient(model="mistral:7b")
reviewer_client = OllamaChatCompletionClient(
    model="gemma3:12b",
    model_info={
        "name": "gemma3:12b",
        "type": "ollama",
        "format": "text",
        "json_output": False,
        "vision": False,
        "function_calling": False
    }
)

def extract_text(resp):
    """Extract text content from various response formats."""
    if isinstance(resp, str):
        return resp.strip()
    
    # Try common attribute names
    for attr in ("output_text", "text"):
        if hasattr(resp, attr):
            return getattr(resp, attr).strip()
    
    # Try nested paths
    for path in (("content", 0, "text"), ("outputs", 0, "content", 0, "text")):
        try:
            val = resp
            for key in path:
                val = val[key] if isinstance(key, int) else getattr(val, key)
            return val.strip()
        except:
            pass
    
    return str(resp).strip()

async def classify_subject(query: str) -> str:
    """Classify the query into math, physics, chemistry, or unknown."""
    prompt = (
        "Classify this question as Math, Physics, or Chemistry. "
        "Reply with only one word.\n\n" + query
    )
    resp = await leader_client.create([UserMessage(content=prompt, source="user")])
    text = extract_text(resp).lower()
    
    for subject in ["math", "phys", "chem"]:
        if subject in text:
            return subject.replace("phys", "physics").replace("chem", "chemistry")
    return "unknown"

async def answer_query(query: str) -> str:
    """Route query to specialist, then review the answer."""
    subject = await classify_subject(query)
    
    # Get specialist answer
    agents = {"math": math_agent, "physics": physics_agent, "chemistry": chemistry_agent}
    
    if subject in agents:
        out = await asyncio.to_thread(agents[subject].invoke, {"input": query})
        answer = out.get("answer", "").strip()
    else:
        # Fallback: use reviewer directly
        resp = await reviewer_client.create([UserMessage(
            content=f"Answer this question step-by-step:\n\n{query}", 
            source="user"
        )])
        return extract_text(resp)
    
    # Review the answer
    review_prompt = (
        f"Question: {query}\nAnswer: {answer}\n\n"
        f"If this answer is correct, reply 'APPROVED'. "
        f"Otherwise, provide the corrected answer."
    )
    resp = await reviewer_client.create([UserMessage(content=review_prompt, source="user")])
    review = extract_text(resp)
    
    return answer if "APPROVED" in review.upper() else review

async def main():
    parser = argparse.ArgumentParser(description="Multi-agent system with Ollama + AutoGen")
    parser.add_argument("--query", type=str, required=True, help="Question to answer")
    args = parser.parse_args()
    
    result = await answer_query(args.query)
    print(result)
    
    await leader_client.close()
    await reviewer_client.close()

if __name__ == "__main__":
    asyncio.run(main())