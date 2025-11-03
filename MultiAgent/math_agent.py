from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from .utils import load_local_model, load_faiss_retriever


def build_math_agent():
    """Builds and returns a LangChain RAG chain for math reasoning (GSM8K-based)."""
    local_model = load_local_model("meta-llama/Meta-Llama-3.1-8B-Instruct")

    retriever = load_faiss_retriever(
        faiss_dir="faiss_index_math",
        embedding_model_name="BAAI/bge-large-en-v1.5"
    )

    system_prompt = (
        "You are a math reasoning assistant.\n"
        "Use the retrieved GSM8K examples below to guide your step-by-step reasoning.\n"
        "If no relevant examples apply, reason it out yourself.\n\n"
        "Retrieved examples:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    qa_chain = create_stuff_documents_chain(local_model, prompt)
    rag_chain_math = create_retrieval_chain(retriever, qa_chain)

    return rag_chain_math


# Quick test
if __name__ == "__main__":
    rag_agent = build_math_agent()
    query = "If 8 workers can finish a task in 6 days, how many workers are needed to finish it in 3 days?"
    response = rag_agent.invoke({"input": query})
    print("\n🤖 Final Answer:")
    print(response["answer"])
