import os
from pathlib import Path
from dotenv import load_dotenv

_ = load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# OpenAI 兼容 LLM 配置（所有节点共用一个最强模型；详见 .env.example）
LLM_BASE_URL = os.environ.get("LLM_BASE_URL") or "https://api.openai.com/v1"
LLM_API_KEY = os.environ.get("LLM_API_KEY")
LLM_MODEL = os.environ.get("LLM_MODEL")
LLM_API_MODE = (
    os.environ.get("LLM_API_MODE", "responses").strip().lower().replace("-", "_")
)
LLM_OUTPUT_VERSION = os.environ.get("LLM_OUTPUT_VERSION", "responses/v1")

# 长期记忆默认模板：用户画像 (User Profile)
DEFAULT_USER_PROFILE = {
  "name": "User", # [示例] "Michael"
  "identity": "用户", # [示例] "CS研究生"
  "physical_stats": {
    "height": 0, # [示例] 180 (cm)
    "weight": 0, # [示例] 75 (kg)
    "age": 0,    # [示例] 24
    "injuries": [] # [示例] ["膝盖轻微疼痛", "左肩不适"]
  },
  "dietary_context": {
    "provider": "Self", # [示例] "Mother" 或 "外卖"
    "preferences": [], # [示例] ["喜欢吃肉", "不吃香菜"]
    "goal": "健康"     # [示例] "增肌" 或 "减脂"
  },
  "mental_state": {
    "stress_sources": [], # [示例] ["论文Deadline", "工作压力"]
    "relaxation_preference": "" # [示例] "打游戏" 或 "看电影"
  }
}

# 持久化画像存储文件
PROFILE_STORE_PATH = os.environ.get("PROFILE_STORE_PATH", "profile_store.json")

# 情节记忆存储文件（每用户最近 N 轮对话摘要，跨 thread 持久化）
EPISODE_STORE_PATH = os.environ.get("EPISODE_STORE_PATH", "episode_store.json")

# 本地知识库目录
KNOWLEDGE_BASE_DIR = os.environ.get("KNOWLEDGE_BASE_DIR", "knowledge_base")
KNOWLEDGE_BASE_SHARED_SUBDIR = os.environ.get("KNOWLEDGE_BASE_SHARED_SUBDIR", "shared")
KNOWLEDGE_BASE_AGENT_SUBDIRS = {
  "trainer": os.environ.get("KNOWLEDGE_BASE_TRAINER_SUBDIR", "trainer"),
  "nutritionist": os.environ.get("KNOWLEDGE_BASE_NUTRITIONIST_SUBDIR", "nutritionist"),
  "wellness": os.environ.get("KNOWLEDGE_BASE_WELLNESS_SUBDIR", "wellness"),
  "general": os.environ.get("KNOWLEDGE_BASE_GENERAL_SUBDIR", "general"),
}

# RAG: Retrieve & Re-rank 配置（默认针对 8GB 显存端侧优化）
# 默认使用 BAAI/bge-m3:多语言(支持 zh+en 100+ 语言)、8192 长上下文、
# 中英跨语言检索原生支持。项目知识库混有中文笔记和 WHO/USDA 英文语料,
# bge-m3 是能同时兼顾两者的最佳选择。需要在 zh-only、极低显存场景下换回
# bge-small-zh-v1.5 可通过环境变量覆盖。
#
# Reranker 默认使用 BAAI/bge-reranker-v2-m3：基于 bge-m3 架构，与 embedding
# 模型同源，原生支持中英文跨语言重排，效果远优于 bge-reranker-base。
RAG_EMBED_MODEL_NAME = os.environ.get("RAG_EMBED_MODEL_NAME", "BAAI/bge-m3")
RAG_RERANK_MODEL_NAME = os.environ.get("RAG_RERANK_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
RAG_DEVICE = os.environ.get("RAG_DEVICE", "auto")
RAG_HF_HOME = (
    os.environ.get("RAG_HF_HOME")
    or os.environ.get("HF_HOME")
    or str(PROJECT_ROOT / "hf_cache")
)
RAG_HF_HUB_CACHE = (
    os.environ.get("RAG_HF_HUB_CACHE")
    or os.environ.get("HUGGINGFACE_HUB_CACHE")
    or str(Path(RAG_HF_HOME) / "hub")
)
RAG_FALLBACK_EMBED_MODEL_NAME = os.environ.get(
    "RAG_FALLBACK_EMBED_MODEL_NAME", "BAAI/bge-small-zh-v1.5"
)

# 第一阶段召回数量（向量检索 Top-K）
RAG_RETRIEVE_TOP_K = int(os.environ.get("RAG_RETRIEVE_TOP_K", "12"))
# 第二阶段重排后返回数量
RAG_FINAL_TOP_K = int(os.environ.get("RAG_FINAL_TOP_K", "4"))

# 编码和重排批大小（端侧可调，4060 8GB 默认较稳）
RAG_EMBED_BATCH_SIZE = int(os.environ.get("RAG_EMBED_BATCH_SIZE", "32"))
RAG_RERANK_BATCH_SIZE = int(os.environ.get("RAG_RERANK_BATCH_SIZE", "16"))
