import os
import argparse
from langchain_core.messages import HumanMessage
from MultiAgent import load_local_model, build_math_agent, build_physics_agent, build_chemistry_agent


# Load built agents
math_agent = build_math_agent()
physics_agent = build_physics_agent()
chemistry_agent = build_chemistry_agent()


# Load the leader model to decide use which agent
leader_model = load_local_model("mistralai/Mistral-7B-Instruct-v0.2")

# Prompts
leader_instruction = (
    "You are a leader agent responsible for routing questions "
    "to one of three specialized assistants: Math, Physics, or Chemistry. "
    "Your task is ONLY to classify the following user query into one of these subjects. "
    "Reply with exactly one word: Math, Physics, or Chemistry."
)

fallback_instruction = (
    "You are a knowledgeable general reasoning assistant. "
    "If a question is not clearly about Math, Physics, or Chemistry, "
    "answer it directly in a clear, step-by-step way."
)


# Classification
def classify_subject(query: str) -> str:
    """Classify a user query as Math / Physics / Chemistry."""
    full_prompt = f"{leader_instruction}\n\nUser query:\n{query}"
    result = leader_model.invoke([HumanMessage(content=full_prompt)])
    subject = result.content.strip().lower()
    if "math" in subject:
        return "math"
    elif "phys" in subject:
        return "physics"
    elif "chem" in subject:
        return "chemistry"
    else:
        return "unknown"


# QAuestion-Answer
def answer_query(query: str):
    """Route query to the correct agent or fall back to leader model reasoning."""
    subject = classify_subject(query)
    print(f"\n🔎 Leader classified subject: {subject.capitalize()}")

    if subject == "math":
        return math_agent.invoke({"input": query})["answer"]

    elif subject == "physics":
        return physics_agent.invoke({"input": query})["answer"]

    elif subject == "chemistry":
        return chemistry_agent.invoke({"input": query})["answer"]

    else:
        print("Leader could not classify — solving directly...")
        full_prompt = f"{fallback_instruction}\n\nQuestion:\n{query}"
        result = leader_model.invoke([HumanMessage(content=full_prompt)])
        return result.content.strip()


## Running
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Multi-Agent Reasoning System with a single query."
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Question to be answered by the multi-agent system (in quotes).",
    )
    args = parser.parse_args()
    user_query = args.query.strip()

    print(f"\n🧭 Query: {user_query}")
    print("🤖 Thinking...\n")

    try:
        answer = answer_query(user_query)
        print(f"Final Answer:\n{answer}\n")
    except Exception as e:
        print(f"Error: {e}\n")
