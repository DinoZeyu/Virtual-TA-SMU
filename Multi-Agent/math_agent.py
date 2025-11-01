import os
import warnings
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langgraph.prebuilt import create_react_agent   # ✅ keep this as required
from langchain.tools import Tool
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import PrivateAttr
from typing import List, Any, Optional

# =========================================================
# 0️⃣ General setup
# =========================================================
warnings.filterwarnings("ignore")
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# =========================================================
# 1️⃣ Load model as agent engine
# =========================================================
model_id = "meta-llama/Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.bfloat16,
    token=hf_token,
)
tokenizer.pad_token = tokenizer.eos_token

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,   # give enough room for reasoning + answer
    temperature=0.3,      # mild randomness for natural phrasing
    top_p=0.9,            # typical nucleus sampling
    do_sample=True,       # enable sampling (needed for natural text)
    repetition_penalty=1.1,   # avoids infinite repeats
    no_repeat_ngram_size=3,   # blocks phrase echoes
    device_map="auto",
)

# =========================================================
# 2️⃣ Define Hugging Face → LangChain wrapper
# =========================================================
class HFChatWrapper(BaseChatModel):
    """A LangChain-compatible wrapper around a Hugging Face text-generation pipeline."""
    _pipeline: Any = PrivateAttr()

    def __init__(self, pipeline, **kwargs):
        super().__init__(**kwargs)
        self._pipeline = pipeline

    @property
    def _llm_type(self) -> str:
        return "huggingface_pipeline"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> ChatResult:
        prompt = (
    "You are a helpful assistant. Solve the question step by step "
    "and give the final numeric result clearly at the end.\n\n"
    "Question: " + "\n".join(m.content for m in messages) + "\nAnswer:"
)


        gen_kwargs = {}
        if "max_new_tokens" in kwargs:
            gen_kwargs["max_new_tokens"] = kwargs.pop("max_new_tokens")
        if "temperature" in kwargs:
            gen_kwargs["temperature"] = kwargs.pop("temperature")
        if "top_p" in kwargs:
            gen_kwargs["top_p"] = kwargs.pop("top_p")
        if "do_sample" in kwargs:
            gen_kwargs["do_sample"] = kwargs.pop("do_sample")

        if stop:
            gen_kwargs["stop"] = stop

        result = self._pipeline(prompt, **gen_kwargs)[0]["generated_text"]
        generation = ChatGeneration(message=AIMessage(content=result))
        return ChatResult(generations=[generation])

    def bind_tools(self, tools: list[Any]):
        return self



# =========================================================
# 3️⃣ Define calculator tool
# =========================================================
def calculator(expression: str) -> str:
    """Evaluate a basic math expression safely."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

calc_tool = Tool(
    name="Calculator",
    description="Evaluates math expressions like '2+2' or '3*(5+2)'.",
    func=calculator,
)

toolkit = [calc_tool]


# Build Agent with local model and defined tool
local_model = HFChatWrapper(pipe)
agent_executor = create_react_agent(local_model, toolkit)


# Query Test
query = "What is (5 * 3) + (60 / 10)?"
response = agent_executor.invoke({"messages": [HumanMessage(content=query)]})
answer = response["messages"][-1].content


print("\n🤖 Final Answer:")
print(answer)