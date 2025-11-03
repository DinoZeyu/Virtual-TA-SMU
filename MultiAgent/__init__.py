"""
Multi-Agent Package
===================
This package provides multiple specialized RAG-based domain agents (Math, Physics, Chemistry)
for the Virtual-TA-SMU project.

Each agent uses:
- A local fine-tuned Llama model for reasoning.
- A FAISS vector store for domain-specific retrieval.
- Shared model / retriever loading utilities from utils.py.
"""

# --- Import individual agent builders ---
from .math_agent import build_math_agent
from .physics_agent import build_physics_agent
from .chemistry_agent import build_chemistry_agent

# --- Import shared utilities ---
from .utils import load_local_model, load_faiss_retriever, HFChatWrapper

# --- Define what is exposed when using "from MultiAgent import *" ---
__all__ = [
    "build_math_agent",
    "build_physics_agent",
    "build_chemistry_agent",
    "load_local_model",
    "load_faiss_retriever",
    "HFChatWrapper",
]
