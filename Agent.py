from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_tavily import TavilySearch
from huggingface_hub import login
from langchain.chat_models.base import BaseChatModel
from typing import List, Any
import os

# =========================================================
# 1️⃣ 环境变量
# =========================================================
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

# =========================================================
# 2️⃣ 定义兼容包装类（修正版）
# =========================================================
class HFChatWrapper(BaseChatModel):
    """让 HuggingFace pipeline 兼容 LangChain agent"""
    pipeline: Any  # ✅ 显式声明字段

    def __init__(self, pipeline):
        super().__init__(pipeline=pipeline)  # ✅ 调用父类 init
        self.pipeline = pipeline

    @property
    def _llm_type(self) -> str:
        return "huggingface_pipeline"

    def _generate(self, messages: List[HumanMessage], **kwargs) -> Any:
        prompt = "\n".join([m.content for m in messages])
        output = self.pipeline(prompt, max_new_tokens=256)[0]["generated_text"]
        return AIMessage(content=output)

    async def _agenerate(self, messages: List[HumanMessage], **kwargs) -> Any:
        return self._generate(messages, **kwargs)

# =========================================================
# 3️⃣ 加载 Llama 3 模型
# =========================================================
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipe = pipeline(
    "text-generation",
    model=AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", token=hf_token),
    tokenizer=AutoTokenizer.from_pretrained(model_id, token=hf_token),
    max_new_tokens=256,
)

llm = HFChatWrapper(pipe)

# =========================================================
# 4️⃣ 定义工具
# =========================================================
search_tool = TavilySearch(max_results=2)
tools = [search_tool]

# =========================================================
# 5️⃣ 创建智能体
# =========================================================
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant that uses search when needed.",
)

# =========================================================
# 6️⃣ 测试调用
# =========================================================
query = "What kind of clothes should we wear in Dallas?"
response = agent.invoke({"input": query})
print(response["output"])
