import os
from pathlib import Path
from dotenv import load_dotenv

_ = load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TRUTHY = {"1", "true", "yes"}

# OpenAI 兼容 LLM 配置（所有节点共用一个最强模型；详见 .env.example）
LLM_BASE_URL = os.environ.get("LLM_BASE_URL") or "https://api.openai.com/v1"
LLM_API_KEY = os.environ.get("LLM_API_KEY")
LLM_MODEL = os.environ.get("LLM_MODEL")
LLM_API_MODE = (
    os.environ.get("LLM_API_MODE", "responses").strip().lower().replace("-", "_")
)
LLM_OUTPUT_VERSION = os.environ.get("LLM_OUTPUT_VERSION", "responses/v1")
LLM_DISABLE_THINKING = os.environ.get("LLM_DISABLE_THINKING", "false").lower() in _TRUTHY

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
  },
  "response_style": {
    "tone": "",      # [示例] "concise" / "warm" / "direct"
    "humor": "",     # [示例] "light" / "none"
    "formality": "", # [示例] "casual" / "formal"
    "language": ""   # [示例] "zh" / "en"
  }
}

# 持久化画像存储文件
PROFILE_STORE_PATH = os.environ.get("PROFILE_STORE_PATH", "profile_store.json")

# LangGraph checkpoint / observability / health logs paths. Existing defaults stay
# project-root relative for local development; Docker Compose overrides them to /app/data.
SQLITE_DB_PATH = os.environ.get("SQLITE_DB_PATH", "checkpoints.db")
OBSERVABILITY_DB_PATH = os.environ.get("OBSERVABILITY_DB_PATH", "observability.db")
HEALTH_LOGS_DB_PATH = os.environ.get("HEALTH_LOGS_DB_PATH", "health_logs.db")
DEFAULT_TIMEZONE = os.environ.get("DEFAULT_TIMEZONE", "Asia/Shanghai")

# 情节记忆存储文件（每用户最近 N 轮对话摘要，跨 thread 持久化）
EPISODE_STORE_PATH = os.environ.get("EPISODE_STORE_PATH", "episode_store.json")
EPISODE_SEMANTIC_RETRIEVAL_ENABLED = (
  os.environ.get("EPISODE_SEMANTIC_RETRIEVAL_ENABLED", "true").lower() in _TRUTHY
)
EPISODE_EMBED_ON_WRITE_ENABLED = (
  os.environ.get("EPISODE_EMBED_ON_WRITE_ENABLED", "false").lower() in _TRUTHY
)
EPISODE_SEMANTIC_MIN_COUNT = int(os.environ.get("EPISODE_SEMANTIC_MIN_COUNT", "8"))
EPISODE_SEMANTIC_TOP_K = int(os.environ.get("EPISODE_SEMANTIC_TOP_K", "3"))
EPISODE_INDEX_DIR = os.environ.get(
  "EPISODE_INDEX_DIR",
  str(Path.home() / ".health_guide_indices" / "episodes"),
)

# 本地知识库目录
KNOWLEDGE_BASE_DIR = os.environ.get("KNOWLEDGE_BASE_DIR", "knowledge_base")
KNOWLEDGE_BASE_AGENT_SUBDIRS = {
  "trainer": os.environ.get("KNOWLEDGE_BASE_TRAINER_SUBDIR", "trainer"),
  "nutritionist": os.environ.get("KNOWLEDGE_BASE_NUTRITIONIST_SUBDIR", "nutritionist"),
  "psychologist": os.environ.get("KNOWLEDGE_BASE_PSYCHOLOGIST_SUBDIR", "psychologist"),
  "doctor": os.environ.get("KNOWLEDGE_BASE_DOCTOR_SUBDIR", "doctor"),
  # Safety KB is consulted by Critic before review.
  "safety": os.environ.get("KNOWLEDGE_BASE_SAFETY_SUBDIR", "safety"),
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


# === 社区 MCP 工具服务器（可选）===
# 三个开关默认全 false，老用户拉新版无感升级；显式 opt-in 才会 spawn 子进程。
MCP_TRAINER_ENABLED = (
    os.environ.get("MCP_TRAINER_ENABLED", "false").lower() in _TRUTHY
)
MCP_NUTRITIONIST_ENABLED = (
    os.environ.get("MCP_NUTRITIONIST_ENABLED", "false").lower() in _TRUTHY
)
MCP_DOCTOR_ENABLED = (
    os.environ.get("MCP_DOCTOR_ENABLED", os.environ.get("MCP_CRITIC_ENABLED", "false")).lower() in _TRUTHY
)
# Backward compatibility: older .env files used MCP_CRITIC_ENABLED for medical-mcp.
MCP_CRITIC_ENABLED = MCP_DOCTOR_ENABLED
# Nutritionist MCP（jlfwong/food-data-central-mcp-server）未发到 npm，
# 需先 `bash scripts/setup_mcp_servers.sh` clone+install，再把打印出的
# src/index.ts 绝对路径填到这里。
MCP_USDA_SCRIPT_PATH = os.environ.get("MCP_USDA_SCRIPT_PATH", "")
USDA_API_KEY = os.environ.get("USDA_API_KEY", "")
# medical-mcp 1.0.8 的 npm bin 链接缺 shebang，npx 起不来；setup 脚本把它装到
# 固定目录后我们直接 `node <build/index.js>` 绕开 npx 的 exec 路径。
MCP_MEDICAL_SCRIPT_PATH = os.environ.get("MCP_MEDICAL_SCRIPT_PATH", "")
# wger 自 1.0.0 起就在启动时强制要 auth；没配 key 时 startup 会跳过该 server
# 并打印一条警告。免费 key 在 wger.de 注册账号后的 API 设置页生成。
WGER_API_KEY = os.environ.get("WGER_API_KEY", "")
# wger-mcp 的 zod schema 比当前 wger.de API 旧（variations 字段会回 undefined
# 触发校验失败）；setup 脚本会把 wger-mcp 装到固定路径并 sed 把 variations
# 改成 .optional()。该变量指向打 patch 后的 dist/index.js。
MCP_WGER_SCRIPT_PATH = os.environ.get("MCP_WGER_SCRIPT_PATH", "")
# 90s 默认是为首次冷启动留余量：medical-mcp ~10-20s 拉包 + 启动 ~5s，国内代理慢时
# 30s 不够。命中本地 npx 缓存后实际只用 1-2s。
MCP_STARTUP_TIMEOUT_SEC = int(os.environ.get("MCP_STARTUP_TIMEOUT_SEC", "90"))

# === 多模态 Vision（可选）===
VISION_ENABLED = os.environ.get("VISION_ENABLED", "true").lower() in _TRUTHY
VISION_PROVIDER = os.environ.get("VISION_PROVIDER", "disabled").strip().lower()
VISION_BASE_URL = os.environ.get("VISION_BASE_URL") or os.environ.get("LLM_BASE_URL") or ""
VISION_API_KEY = os.environ.get("VISION_API_KEY") or os.environ.get("LLM_API_KEY") or ""
VISION_MODEL = os.environ.get("VISION_MODEL", "")
VISION_TIMEOUT_SEC = int(os.environ.get("VISION_TIMEOUT_SEC", "60"))

# === 微信 iLink / ClawBot（可选）===
WECHAT_ILINK_BASE_URL = os.environ.get(
    "WECHAT_ILINK_BASE_URL",
    "https://ilinkai.weixin.qq.com",
).rstrip("/")
WECHAT_BOT_TOKEN = os.environ.get("WECHAT_BOT_TOKEN", "")
WECHAT_APP_ID = os.environ.get("WECHAT_APP_ID", "")
WECHAT_APP_SECRET = os.environ.get("WECHAT_APP_SECRET", "")
WECHAT_POLL_TIMEOUT_SEC = int(os.environ.get("WECHAT_POLL_TIMEOUT_SEC", "30"))
WECHAT_WORKER_IDLE_SEC = float(os.environ.get("WECHAT_WORKER_IDLE_SEC", "1"))
WECHAT_ENDPOINT_QRCODE = os.environ.get("WECHAT_ENDPOINT_QRCODE", "/v1/bot/qrcode")
WECHAT_ENDPOINT_QRCODE_STATUS = os.environ.get(
    "WECHAT_ENDPOINT_QRCODE_STATUS",
    "/v1/bot/qrcode/status",
)
WECHAT_ENDPOINT_UPDATES = os.environ.get("WECHAT_ENDPOINT_UPDATES", "/v1/messages/updates")
WECHAT_ENDPOINT_SEND = os.environ.get("WECHAT_ENDPOINT_SEND", "/v1/messages/send")
WECHAT_ENDPOINT_PUSH = os.environ.get("WECHAT_ENDPOINT_PUSH", "/v1/messages/push")
WECHAT_ENDPOINT_MEDIA = os.environ.get("WECHAT_ENDPOINT_MEDIA", "/v1/media/{media_id}")

# === 备份（可选）===
BACKUP_DIR = os.environ.get("BACKUP_DIR", "backups")
BACKUP_RETENTION_DAYS = int(os.environ.get("BACKUP_RETENTION_DAYS", "14"))
BACKUP_INTERVAL_HOURS = float(os.environ.get("BACKUP_INTERVAL_HOURS", "24"))
OSS_ACCESS_KEY_ID = os.environ.get("OSS_ACCESS_KEY_ID", "")
OSS_ACCESS_KEY_SECRET = os.environ.get("OSS_ACCESS_KEY_SECRET", "")
OSS_BUCKET = os.environ.get("OSS_BUCKET", "")
OSS_ENDPOINT = os.environ.get("OSS_ENDPOINT", "")
OSS_PREFIX = os.environ.get("OSS_PREFIX", "health-guide-backup")
