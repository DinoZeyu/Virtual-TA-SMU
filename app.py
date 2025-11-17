import streamlit as st
import asyncio
import re
import html
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------------------------
# Import Agents
# ---------------------------
from MultiAgent import (
    build_math_agent,
    build_physics_agent,
    build_chemistry_agent
)

from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core.models import UserMessage
from langchain_community.tools.tavily_search import TavilySearchResults


# =======================================================
# Safe text extractor (prevents leftover HTML artifacts)
# =======================================================

def extract_text(resp):
    if isinstance(resp, str):
        return resp.strip()

    for attr in ("output_text", "text"):
        if hasattr(resp, attr):
            return getattr(resp, attr).strip()

    return str(resp).strip()


def clean_html_artifacts(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


# =======================================================
# Unified Async Runner
# =======================================================

async def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return await coro
        return loop.run_until_complete(coro)
    except RuntimeError:
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop.run_until_complete(coro)


# =======================================================
# Initialize Agents
# =======================================================

@st.cache_resource
def initialize_agents():
    return {
        "math": build_math_agent(),
        "physics": build_physics_agent(),
        "chemistry": build_chemistry_agent(),

        "leader": OllamaChatCompletionClient(model="mistral:7b"),
        "reviewer": OllamaChatCompletionClient(model="gemma3:12b"),
    }

agents = initialize_agents()


# =======================================================
# Web Search Agent
# =======================================================

tavily = TavilySearchResults(k=5)

async def fetch_web_results(q):
    return await asyncio.to_thread(lambda: tavily.run(q))

async def web_answer(query):
    results = await fetch_web_results(query)
    content = "\n\n".join([r["content"] for r in results if "content" in r])

    prompt = f"""
Use ONLY the information below:

{content}

User question:
{query}
"""

    resp = await agents["reviewer"].create([UserMessage(content=prompt, source="user")])
    return extract_text(resp)


# =======================================================
# Classification
# =======================================================

async def classify_subject(query):
    prompt = f"""
Classify into ONE word: Math, Physics, Chemistry, Web
Query: {query}
"""

    resp = await agents["leader"].create([UserMessage(content=prompt, source="user")])
    t = extract_text(resp).lower()

    if "math" in t: return "math"
    if "phys" in t: return "physics"
    if "chem" in t: return "chemistry"
    if "web" in t: return "web"
    return "math"  # default fallback


# =======================================================
# Main Answer Pipeline
# =======================================================

async def answer_query(query, agent_override="Auto"):

    # Manual override
    if agent_override in ["Math", "Physics", "Chemistry"]:
        forced = agent_override.lower()
        out = await asyncio.to_thread(
            agents[forced].invoke, {"input": query}
        )
        return extract_text(out["answer"])

    # Auto mode
    subject = await classify_subject(query)

    if subject == "web":
        return await web_answer(query)

    out = await asyncio.to_thread(
        agents[subject].invoke, {"input": query}
    )
    return extract_text(out["answer"])


# =======================================================
# Streamlit UI
# =======================================================

st.set_page_config(page_title="Multi-Agent Assistant", page_icon="🤖")

st.title("🤖 Multi-Agent Assistant")
st.caption("Math • Physics • Chemistry • Web Search")

# Sidebar
with st.sidebar:
    st.markdown("### 🛠 System Status")

    st.markdown(
        """
    <div style="background:#e8f5e9;padding:15px;border-radius:10px;margin-bottom:10px;">
        <p style="margin: 0;"><b>✓ Math Agent</b></p>
        <p style="margin: 0;"><b>✓ Physics Agent</b></p>
        <p style="margin: 0;"><b>✓ Chemistry Agent</b></p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div style="background:#e3f2fd;padding:15px;border-radius:10px;margin-bottom:10px;">
        <b>📋 Leader:</b> Mistral-7B
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div style="background:#fff3e0;padding:15px;border-radius:10px;margin-bottom:10px;">
        <b>🔍 Reviewer:</b> Gemma-3-12B
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 🎛 Manual Agent Selection")

    agent_choice = st.radio(
        "Choose agent (optional):",
        ["Auto", "Math", "Physics", "Chemistry"],
        index=0,
    )


# Chat State
if "messages" not in st.session_state:
    st.session_state.messages = []


def render_message(role, content):

    content = clean_html_artifacts(content)

    parts = re.split(r'(\$\$[\s\S]*?\$\$)', content)

    for part in parts:
        if part.startswith("$$") and part.endswith("$$"):
            st.latex(part.strip("$"))
        else:
            bubble = "#DCF8C6" if role == "user" else "#F1F0F0"
            align = "right" if role == "user" else "left"
            st.markdown(
                f"""
                <div style="background:{bubble};padding:12px;border-radius:12px;
                            margin:8px 0;max-width:70%;float:{align};">
                    {html.escape(part)}
                </div>
                <div style="clear:both;"></div>
                """,
                unsafe_allow_html=True
            )


for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"])


# Input
user_input = st.chat_input("Ask a question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    render_message("user", user_input)

    with st.spinner("Thinking..."):
        assistant_response = asyncio.run(
            run_async(answer_query(user_input, agent_choice))
        )

    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    render_message("assistant", assistant_response)
