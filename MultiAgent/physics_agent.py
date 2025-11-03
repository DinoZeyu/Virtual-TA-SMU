from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from .utils import load_local_model, load_faiss_retriever


def build_physics_agent():
    """Physics reasoning agent using FAISS knowledge base."""
    local_model = load_local_model("meta-llama/Meta-Llama-3.1-8B-Instruct")
    
    retriever = load_faiss_retriever(
        faiss_dir="faiss_index_physics",
        embedding_model_name="BAAI/bge-large-en-v1.5"
    )

    system_prompt = (
        "You are a physics problem-solving assistant.\n"
        "Use the retrieved examples below to reason about the question clearly.\n"
        "Explain units, formulas, and steps as needed.\n\n"
        "Retrieved examples:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    qa_chain = create_stuff_documents_chain(local_model, prompt)
    return create_retrieval_chain(retriever, qa_chain)


if __name__ == "__main__":
    agent = build_physics_agent()
    query = "A ball is thrown upward with velocity 20 m/s. How long until it lands back?"
    print(agent.invoke({"input": query})["answer"])
