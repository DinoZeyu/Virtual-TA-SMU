from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from .utils import load_local_model, load_faiss_retriever


# ==========================
# Math Agent
# ==========================

def build_math_agent():
    """Math RAG agent with clean, LaTeX-safe output formatting."""

    local_model = load_local_model("meta-llama/Meta-Llama-3.1-8B-Instruct")

    retriever = load_faiss_retriever(
        faiss_dir="faiss_index_math",
        embedding_model_name="BAAI/bge-large-en-v1.5",
    )

    system_prompt = """
You are a math reasoning assistant.

FORMAT RULES (do not break them):

1. Do NOT output HTML of any kind.
2. Do NOT output markdown code fences (no ```).
3. Your answer MUST have exactly two parts, in this order:

(1) Plain-English explanation
    - No LaTeX.
    - No explicit formulas like "f(x)=3x^2-4x+7".
    - You may refer to ideas in words, e.g., "the derivative of a quadratic term".

(2) Final math expression
    - A SINGLE LaTeX block on one line, in the form:  $$...$$
    - ONLY mathematical symbols inside $$...$$.
    - NO English text inside $$...$$.
    - NO multi-line LaTeX.

Example of good output:

The derivative is found by applying the power rule term by term.
$$f'(x)=6x-4$$

Retrieved examples:
{context}
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    qa_chain = create_stuff_documents_chain(local_model, prompt)
    rag_chain_math = create_retrieval_chain(retriever, qa_chain)
    return rag_chain_math


# ==========================
# Physics Agent
# ==========================

def build_physics_agent():
    """Physics RAG agent with clean, LaTeX-safe output formatting."""

    local_model = load_local_model("meta-llama/Meta-Llama-3.1-8B-Instruct")

    retriever = load_faiss_retriever(
        faiss_dir="faiss_index_physics",
        embedding_model_name="BAAI/bge-large-en-v1.5",
    )

    system_prompt = """
You are a physics reasoning assistant.

FORMAT RULES (do not break them):

1. Do NOT output HTML.
2. Do NOT output markdown code fences (```).
3. Your answer MUST have exactly two parts:

(1) Plain-English explanation
    - No LaTeX.
    - No explicit symbolic formulas like "v = u + at".
    - Explain the physical reasoning in simple language.

(2) Final physics equation or result
    - A SINGLE LaTeX block on one line:  $$...$$
    - ONLY symbols/numbers/operators inside $$...$$.
    - NO English text inside $$...$$.
    - NO multi-line LaTeX.

Example of good output:

The time of flight is found by noting that the object takes equal time going up and coming down under constant acceleration.
$$t=\\frac{2u}{g}$$

Retrieved examples:
{context}
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    qa_chain = create_stuff_documents_chain(local_model, prompt)
    rag_chain_physics = create_retrieval_chain(retriever, qa_chain)
    return rag_chain_physics


# ==========================
# Chemistry Agent
# ==========================

def build_chemistry_agent():
    """Chemistry RAG agent with clean, LaTeX-safe output formatting."""

    local_model = load_local_model("meta-llama/Meta-Llama-3.1-8B-Instruct")

    retriever = load_faiss_retriever(
        faiss_dir="faiss_index_chemistry",
        embedding_model_name="BAAI/bge-large-en-v1.5",
    )

    system_prompt = """
You are a chemistry reasoning assistant.

FORMAT RULES (do not break them):

1. Do NOT output HTML.
2. Do NOT output markdown code fences (```).
3. Answer MUST have exactly two parts:

(1) Plain-English explanation
    - No LaTeX.
    - No explicit formulas like "H2O" or "NaCl" written as formulas.
    - Describe what happens conceptually (e.g., "an acid reacts with a base to form salt and water").

(2) Final chemical equation
    - A SINGLE LaTeX block on one line:  $$...$$
    - Should represent the balanced reaction.
    - ONLY chemical symbols and operators inside $$...$$.
    - NO English text inside $$...$$.
    - NO multi-line LaTeX.

Example of good output:

This is a neutralization reaction between a strong acid and a strong base, forming salt and water.
$$\\mathrm{HCl + NaOH -> NaCl + H_2O}$$

Retrieved examples:
{context}
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    qa_chain = create_stuff_documents_chain(local_model, prompt)
    rag_chain_chemistry = create_retrieval_chain(retriever, qa_chain)
    return rag_chain_chemistry


# Optional quick test
if __name__ == "__main__":
    math_agent = build_math_agent()
    print("MATH TEST:")
    print(math_agent.invoke({"input": "If f(x)=3x^2-4x+7, what is f'(x)?"})["answer"])

    physics_agent = build_physics_agent()
    print("\nPHYSICS TEST:")
    print(
        physics_agent.invoke(
            {"input": "A ball is thrown upwards at 20 m/s. Ignore air resistance and find the total time of flight."}
        )["answer"]
    )

    chem_agent = build_chemistry_agent()
    print("\nCHEM TEST:")
    print(
        chem_agent.invoke(
            {"input": "What happens when hydrochloric acid reacts with sodium hydroxide?"}
        )["answer"]
    )
