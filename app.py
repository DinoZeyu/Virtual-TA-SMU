import streamlit as st
import asyncio
import re
import html
from autogen_core.models import UserMessage

# ----- Import Multi-Agent System -----
from MultiAgent import (
    build_math_agent,
    build_physics_agent,
    build_chemistry_agent,
    build_leader_agent,
    build_reviewer_agent,
    classify_subject,
    review_answer,
)
from MultiAgent.utils import clean_html


# ============================================================
# Chat bubble renderer
# ============================================================
def render_message(role, content):
    content = clean_html(content)
    parts = re.split(r"(\$\$[\s\S]*?\$\$)", content)

    for part in parts:
        if part.startswith("$$") and part.endswith("$$"):
            st.latex(part.strip("$"))
            continue

        if not part.strip():
            continue

        safe = html.escape(part)
        color = "#DCF8C6" if role == "user" else "#F1F0F0"
        align = "right" if role == "user" else "left"

        st.markdown(
            f"""
            <div style="background:{color};
                        padding:12px; border-radius:12px;
                        margin:8px 0; max-width:80%; float:{align};">
                {safe}
            </div>
            <div style="clear:both;"></div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# Build agents once
# ============================================================
@st.cache_resource
def initialize_agents():
    return {
        "math": build_math_agent(),
        "physics": build_physics_agent(),
        "chemistry": build_chemistry_agent(),
        "leader": build_leader_agent(),
        "reviewer": build_reviewer_agent(),
    }

agents = initialize_agents()


# ============================================================
# Main reasoning pipeline
# ============================================================
async def answer_query(query: str, agent_override="Auto"):

    # manual override
    if agent_override in ["Math", "Physics", "Chemistry"]:
        forced = agent_override.lower()
        predicted = await classify_subject(agents["leader"], query)

        if predicted == forced:
            out = await asyncio.to_thread(agents[forced].invoke, {"input": query})
            answer = clean_html(out.get("answer", ""))
            return answer

    # auto classification
    subject = await classify_subject(agents["leader"], query)

    # Subject agents
    if subject in ["math", "physics", "chemistry"]:
        out = await asyncio.to_thread(agents[subject].invoke, {"input": query})
        answer = clean_html(out.get("answer", ""))

    # Web (NO reviewer used here)
    elif subject == "web":
        from langchain_community.tools.tavily_search import TavilySearchResults
        tavily = TavilySearchResults(k=5)

        web_data = await asyncio.to_thread(tavily.run, query)
        content = "\n".join([item.get("content", "") for item in web_data])

        return clean_html(
            f"Based on online information:\n\n{content}\n\n(Answer generated from web search)"
        )

    # Unknown → reviewer answers directly
    else:
        resp = await agents["reviewer"].create([
            UserMessage(content=f"Answer step-by-step:\n{query}", source="user")
        ])
        answer = clean_html(resp.output_text)

    # Reviewer validation
    review_result = await review_answer(
        agents["reviewer"],
        query,
        answer
    )

    if "APPROVED" in review_result.upper():
        return answer

    return clean_html(review_result)


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="Multi-Agent Academic Assistant", page_icon="🤖")

st.markdown("""
<div style="text-align:center;margin-bottom:20px;">
    <h1 style="color:#4CAF50;">🤖 Multi-Agent Academic Assistant</h1>
    <p style="font-size:18px;color:#555;">
        Math • Physics • Chemistry • Web Search
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🎛 Manual Agent Selection")
    agent_choice = st.radio(
        "Choose specific agent (optional):",
        ["Auto", "Math", "Physics", "Chemistry"],
        index=0
    )

    st.markdown("### 🛠 System Status")
    st.success("Math Agent Loaded")
    st.success("Physics Agent Loaded")
    st.success("Chemistry Agent Loaded")
    st.info("Leader: Mistral-7B")
    st.warning("Reviewer: Gemma-3-12B")

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"])

# Chat input
user_input = st.chat_input("Ask a question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    render_message("user", user_input)

    with st.spinner("Thinking..."):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(
            answer_query(user_input, agent_override=agent_choice)
        )

    st.session_state.messages.append({"role": "assistant", "content": response})
    render_message("assistant", response)
