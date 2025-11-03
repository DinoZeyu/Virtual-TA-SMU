import os
import warnings
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chat_models.base import BaseChatModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import PrivateAttr
from typing import List, Any, Optional


# Wrap Huggingface models into compatiable 
class HFChatWrapper(BaseChatModel):
    _pipeline: Any = PrivateAttr()

    def __init__(self, pipeline, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self._pipeline = pipeline
        self._tokenizer = tokenizer

    @property
    def _llm_type(self) -> str:
        return "huggingface_pipeline"

    def _generate(self, messages: Any, stop=None, **kwargs) -> ChatResult:
        if isinstance(messages, tuple):
            messages = messages[0]
        elif isinstance(messages, list) and len(messages) == 1 and isinstance(messages[0], tuple):
            messages = messages[0][0]

        chat = []
        for m in messages:
            if isinstance(m, dict):
                role = m.get("role", "user")
                content = m.get("content", "")
            else:
                role = "user" if isinstance(m, HumanMessage) else "assistant"
                content = getattr(m, "content", "")
            chat.append({"role": role, "content": content})

        prompt = self._tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        out = self._pipeline(prompt, return_full_text=False, **kwargs)[0]["generated_text"].strip()
        generation = ChatGeneration(message=AIMessage(content=out))
        return ChatResult(generations=[generation])



# Load models
def load_local_model(
    model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    dtype: torch.dtype = torch.bfloat16,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
    top_p: float = 0.9,
) -> HFChatWrapper:
    """
    Load any Hugging Face causal LM model as a local pipeline wrapped for LangChain.
    """

    warnings.filterwarnings("ignore")
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype=dtype,
        token=hf_token,
    )

    # Ensure padding setup
    tokenizer.pad_token = tokenizer.eos_token

    # Build text-generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        device_map="auto",
    )

    print(f"Loaded local model: {model_id}")
    return HFChatWrapper(pipe, tokenizer)


def load_faiss_retriever(
    faiss_dir: str,
    embedding_model_name: str = "BAAI/bge-large-en-v1.5",
    device: str = "cuda",
    normalize_embeddings: bool = True,
    k: int = 3,
):
    """
    Load a FAISS index and return a LangChain retriever object.

    Args:
        faiss_dir (str): Directory containing index.faiss + index.pkl
        embedding_model_name (str): Embedding model used to build the index
        device (str): Device for embeddings ('cuda' or 'cpu')
        normalize_embeddings (bool): Whether to normalize vectors
        k (int): Number of retrieved documents

    Returns:
        retriever (BaseRetriever): LangChain retriever ready for use
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))      
    project_root = os.path.abspath(os.path.join(base_dir, ".."))  
    full_faiss_path = os.path.join(project_root, faiss_dir)   

    if not os.path.exists(full_faiss_path):
        raise FileNotFoundError(f"FAISS directory not found: {full_faiss_path}")
    if not os.path.exists(os.path.join(full_faiss_path, "index.faiss")):
        raise FileNotFoundError(f"Missing index.faiss inside: {full_faiss_path}")
    if not os.path.exists(os.path.join(full_faiss_path, "index.pkl")):
        raise FileNotFoundError(f"Missing index.pkl inside: {full_faiss_path}")

  
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": normalize_embeddings}
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )


    vectorstore = FAISS.load_local(
        full_faiss_path,
        embedding_model,
        allow_dangerous_deserialization=True,
    )

    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})