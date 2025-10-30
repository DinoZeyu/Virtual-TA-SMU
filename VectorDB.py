import os
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize the embedding model
model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {"device": "cuda"}                
encode_kwargs = {"normalize_embeddings": True}
embedding_model= HuggingFaceEmbeddings(
    model_name=model_name, 
    model_kwargs=model_kwargs, 
    encode_kwargs=encode_kwargs,
)

# Chunk the text into manageable pieces for embedding
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=[
        "\n\n", "\n", "Q:", "A:", ".", "?", "!", ":", ";", " "
    ],
    length_function=len,
)

# Build the vector database from txt files in math, physics, and chemistry directories
def build_faiss_for_txt(subject: str, txt_path: str):

    # Load txt as LangChain Document objects
    loader = TextLoader(txt_path, encoding="utf-8")
    docs = loader.load()
    splits = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embedding_model)

    # Save index to subject-specific directory
    index_dir = f"faiss_index_{subject.lower()}"
    vectorstore.save_local(index_dir)
    print(f"Saved index → {index_dir}")


data_path = {
    "Math": "Datasets/Math/math_corpus.txt",
    "Physics": "Datasets/Physics/physics_corpus.txt",
    "Chemistry": "Datasets/Chemistry/chemistry_corpus.txt",
}

if __name__ == "__main__":
    for subject, path in data_path.items():
        build_faiss_for_txt(subject, path)
    print("All FAISS indexes built successfully using split_documents().")