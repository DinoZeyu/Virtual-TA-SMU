# MultiAgent/multi_agent.py

from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from .utils import load_local_model, load_faiss_retriever


def _build_agent(faiss_dir: str, system_text: str):
    """
    Helper to build a RAG agent:
      - local HF model (wrapped by load_local_model)
      - FAISS retriever (load_faiss_retriever)
      - stuff chain + retrieval chain
    """
    llm = load_local_model()
    retriever = load_faiss_retriever(faiss_dir)

    # IMPORTANT: only {context} and {input} are template variables
    system_prompt = system_text + "\n\nRetrieved examples:\n{context}\n"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    return rag_chain


def build_math_agent():
    """Math RAG agent: explanation + one LaTeX formula."""
    system_text = (
        "You are a careful math reasoning tutor.\n"
        "You solve school and undergraduate level math problems step by step.\n\n"
        "OUTPUT FORMAT:\n"
        "1. First write a clear explanation in plain English only. "
        "Do not use LaTeX, HTML, code fences, or markdown tables in this part.\n"
        "2. Then on a new line write ONE LaTeX block for the final key formula or answer, "
        "RULES FOR THE LATEX BLOCK:\n"
        "- Only math symbols, no English sentences.\n"
        "- No surrounding backticks or code fences.\n"
        "- Do not repeat the whole explanation inside the LaTeX block.\n"
        "- Never output HTML tags such as <div> or style attributes."
    )

    return _build_agent("faiss_index_math", system_text)


def build_physics_agent():
    """Physics RAG agent: explanation + one LaTeX equation."""
    system_text = (
        "You are a physics problem-solving tutor.\n"
        "You use the retrieved examples and standard physics formulas to answer questions "
        "about mechanics, electromagnetism and basic university physics.\n\n"
        "OUTPUT FORMAT:\n"
        "1. First give a clear step-by-step explanation in plain English. "
        "Explain units and physical meaning when helpful. "
        "Do not use LaTeX or HTML here.\n"
        "2. Then on a new line write ONE LaTeX block with the key final equation or result, "
        "RULES FOR THE LATEX BLOCK:\n"
        "- Only physics symbols and numbers, no English sentences.\n"
        "- No markdown code fences.\n"
        "- Never output HTML tags."
    )

    return _build_agent("faiss_index_physics", system_text)


def build_chemistry_agent():
    """Chemistry RAG agent: explanation + one LaTeX chemical equation."""
    system_text = (
        "You are a chemistry tutor.\n"
        "You explain reactions, stoichiometry, acids and bases, and similar topics.\n\n"
        "OUTPUT FORMAT:\n"
        "1. First write a clear English explanation of the chemical process. "
        "Do not use LaTeX or HTML in this explanation.\n"
        "2. Then on a new line write ONE LaTeX block containing the main balanced "
        "RULES FOR THE LATEX BLOCK:\n"
        "- Only symbols for elements, molecules and arrows, no English sentences.\n"
        "- You may use LaTeX commands like \\rightarrow or subscripts such as H_2O.\n"
        "- Do not output HTML or markdown code fences."
    )

    return _build_agent("faiss_index_chemistry", system_text)
