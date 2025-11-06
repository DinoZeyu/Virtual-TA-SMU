import streamlit as st
import asyncio
from MultiAgent import build_math_agent, build_physics_agent, build_chemistry_agent
from autogen_core.models import UserMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient


# ============================================================
# Initialize all agents (cached to avoid reloading)
# ============================================================
@st.cache_resource
def initialize_agents():
    return {
        "math": build_math_agent(),
        "physics": build_physics_agent(),
        "chemistry": build_chemistry_agent(),
        "leader": OllamaChatCompletionClient(model="mistral:7b"),
        "reviewer": OllamaChatCompletionClient(
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
    }

agents = initialize_agents()


# ============================================================
# Helper: extract model text safely
# ============================================================
def extract_text(resp):
    """Extract text from any Ollama or Autogen response cleanly."""
    # 1. String case
    if isinstance(resp, str):
        return resp.strip()

    # 2. Common attributes
    for attr in ("output_text", "text"):
        if hasattr(resp, attr):
            val = getattr(resp, attr)
            if isinstance(val, str):
                return val.strip()

    # 3. Nested content path
    for path in (("outputs", 0, "content", 0, "text"), ("content", 0, "text")):
        try:
            val = resp
            for key in path:
                val = val[key] if isinstance(key, int) else getattr(val, key)
            if isinstance(val, str):
                return val.strip()
        except Exception:
            continue

    # ✅ 4. Handle `CreateResult(content=...)`
    if hasattr(resp, "content"):
        val = resp.content
        if isinstance(val, str):
            return val.strip()
        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], str):
            return val[0].strip()
        return str(val).strip()

    # 5. Fallback
    return str(resp).strip()

# ============================================================
# Core multi-agent logic
# ============================================================
async def classify_subject(query: str) -> str:
    prompt = (
        "Classify this question as Math, Physics, or Chemistry. "
        "Reply with only one word.\n\n" + query
    )
    resp = await agents["leader"].create([UserMessage(content=prompt, source="user")])
    text = extract_text(resp).lower()
    for subject in ["math", "phys", "chem"]:
        if subject in text:
            return subject.replace("phys", "physics").replace("chem", "chemistry")
    return "unknown"


async def answer_query(query: str) -> str:
    subject = await classify_subject(query)
    specialist_agents = {
        "math": agents["math"],
        "physics": agents["physics"],
        "chemistry": agents["chemistry"]
    }

    if subject in specialist_agents:
        out = await asyncio.to_thread(specialist_agents[subject].invoke, {"input": query})
        answer = out.get("answer", "").strip()
    else:
        resp = await agents["reviewer"].create([UserMessage(
            content=f"Answer this question step-by-step:\n\n{query}", 
            source="user"
        )])
        return extract_text(resp)
    
    review_prompt = (
        f"Question: {query}\nAnswer: {answer}\n\n"
        f"If this answer is correct, reply 'APPROVED'. "
        f"Otherwise, provide the corrected answer."
    )
    resp = await agents["reviewer"].create([UserMessage(content=review_prompt, source="user")])
    review = extract_text(resp)
    
    return answer if "APPROVED" in review.upper() else review


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="Multi-Agent Academic Assistant", page_icon="🤖")
st.title("Multi-Agent Academic Assistant")

with st.sidebar:
    st.header("System Information")
    st.success("✓ Math Agent")
    st.success("✓ Physics Agent")
    st.success("✓ Chemistry Agent")
    st.info("📋 Leader: Mistral 7B")
    st.info("✅ Reviewer: Gemma3 12B")
    st.markdown("---")
    st.markdown("""
    **How it works:**
    
    1️⃣ Leader classifies your question  
    2️⃣ Specialist answers  
    3️⃣ Reviewer checks correctness  
    """)


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a Math, Physics, or Chemistry question...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Processing your question..."):
            try:
                # ✅ Safe event loop handling (fixed version)
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                if loop.is_running():
                    assistant_response = asyncio.run(answer_query(user_input))
                else:
                    assistant_response = loop.run_until_complete(answer_query(user_input))

            except RuntimeError as e:
                # if event loop closed, reinit
                if "closed" in str(e).lower():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    assistant_response = loop.run_until_complete(answer_query(user_input))
                else:
                    raise e
            except Exception as e:
                assistant_response = f"Error: {str(e)}"
        
        st.markdown(assistant_response)
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})