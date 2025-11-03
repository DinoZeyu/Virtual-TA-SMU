from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from .utils import load_local_model, load_faiss_retriever


def build_chemistry_agent():
    """Chemistry reasoning agent using FAISS knowledge base."""
    local_model = load_local_model("meta-llama/Meta-Llama-3.1-8B-Instruct")
    
    retriever = load_faiss_retriever(
        faiss_dir="faiss_index_chemistry",
        embedding_model_name="BAAI/bge-large-en-v1.5"
    )

    system_prompt = (
        "You are a chemistry assistant.\n"
        "Use the retrieved examples below to analyze chemical reactions, equations, and concepts step by step.\n"
        "If the question is conceptual, explain it clearly.\n\n"
        "Retrieved examples:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    qa_chain = create_stuff_documents_chain(local_model, prompt)
    return create_retrieval_chain(retriever, qa_chain)


if __name__ == "__main__":
    agent = build_chemistry_agent()
    query = "What happens when hydrochloric acid reacts with sodium hydroxide?"
    print(agent.invoke({"input": query})["answer"])
