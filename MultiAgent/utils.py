# MultiAgent/utils.py

import os
import warnings
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# -------------------------------------------------------
# 1️⃣ Extract clean text from Leader/Reviewer output
# -------------------------------------------------------
def extract_text_clean(resp) -> str:
    """Extract clean text from Autogen/Ollama responses (robust)."""

    # 1. plain string
    if isinstance(resp, str):
        return resp.strip()

    # 2. common direct attributes
    for key in ("output_text", "text", "content"):
        if hasattr(resp, key):
            val = getattr(resp, key)
            if isinstance(val, str):
                return val.strip()

    # 3. autogen nested list: resp.content → list of dict
    try:
        c = resp.content
        if isinstance(c, list) and len(c) > 0:
            first = c[0]
            if isinstance(first, dict) and "text" in first:
                return first["text"].strip()
    except:
        pass

    # 4. raw dict content
    try:
        if "content" in resp and isinstance(resp["content"], list):
            entry = resp["content"][0]
            if "text" in entry:
                return entry["text"].strip()
    except:
        pass

    # 5. fallback
    return str(resp).strip()



# -------------------------------------------------------
# 2️⃣ Clean leftover HTML from RAG output
# -------------------------------------------------------
def clean_html(text: str) -> str:
    """Remove accidental HTML emitted by models."""
    import re
    text = re.sub(r"</?div[^>]*>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()



# -------------------------------------------------------
# 3️⃣ HF Model Wrapper for LangChain
# -------------------------------------------------------
from langchain.chat_models.base import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import PrivateAttr


class HFChatWrapper(BaseChatModel):
    """Wrap HuggingFace pipeline → LangChain compatible."""
    _pipeline: any = PrivateAttr()
    _tokenizer: any = PrivateAttr()

    def __init__(self, pipeline, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self._pipeline = pipeline
        self._tokenizer = tokenizer

    @property
    def _llm_type(self) -> str:
        return "hf_pipeline"

    def _generate(self, messages, **kwargs) -> ChatResult:
        # Convert LangChain messages → chat template
        chat_list = []
        for m in messages:
            if isinstance(m, HumanMessage):
                chat_list.append({"role": "user", "content": m.content})
            else:
                chat_list.append({"role": "assistant", "content": m.content})

        prompt = self._tokenizer.apply_chat_template(
            chat_list, tokenize=False, add_generation_prompt=True
        )

        output = self._pipeline(
            prompt, max_new_tokens=512, return_full_text=False
        )[0]["generated_text"]

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=output))])



def load_local_model(
    model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    dtype: torch.dtype = torch.bfloat16,
):
    """Load local HF model & wrap it."""
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=dtype,
        token=hf_token,
    )

    gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0.1,                   
    top_p=1.0,                         
    do_sample=False,                   
    device_map="auto",
)

    return HFChatWrapper(gen, tokenizer)


# -------------------------------------------------------
# 4️⃣ FAISS retriever
# -------------------------------------------------------
def load_faiss_retriever(faiss_dir: str):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    final_dir = os.path.join(project_root, faiss_dir)

    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectordb = FAISS.load_local(
        final_dir,
        embedding,
        allow_dangerous_deserialization=True,
    )

    return vectordb.as_retriever(search_kwargs={"k": 3})
