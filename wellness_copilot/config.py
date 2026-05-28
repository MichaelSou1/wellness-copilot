import os
from pathlib import Path
from dotenv import load_dotenv

_ = load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TRUTHY = {"1", "true", "yes"}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    text = raw.strip().lower()
    if text == "":
        return bool(default)
    return text in _TRUTHY


def _api_mode(name: str, default: str = "responses") -> str:
    raw = os.environ.get(name)
    value = raw if raw and raw.strip() else default
    return value.strip().lower().replace("-", "_")


# OpenAI 兼容 LLM 配置。LLM_* 是“其它文本节点”的默认模型；
# Orchestrator 和 multimodal_processor 可用各自前缀单独覆盖。
LLM_BASE_URL = os.environ.get("LLM_BASE_URL") or "https://api.openai.com/v1"
LLM_API_KEY = os.environ.get("LLM_API_KEY")
LLM_MODEL = os.environ.get("LLM_MODEL")
LLM_API_MODE = _api_mode("LLM_API_MODE", "responses")
LLM_OUTPUT_VERSION = os.environ.get("LLM_OUTPUT_VERSION", "responses/v1")
LLM_DISABLE_THINKING = _env_bool("LLM_DISABLE_THINKING", False)

# 父 agent / Orchestrator：默认继承 LLM_*，需要更强/更慢模型时单独配置。
ORCHESTRATOR_LLM_BASE_URL = os.environ.get("ORCHESTRATOR_LLM_BASE_URL") or LLM_BASE_URL
ORCHESTRATOR_LLM_API_KEY = os.environ.get("ORCHESTRATOR_LLM_API_KEY") or LLM_API_KEY
ORCHESTRATOR_LLM_MODEL = os.environ.get("ORCHESTRATOR_LLM_MODEL") or LLM_MODEL
ORCHESTRATOR_LLM_API_MODE = _api_mode("ORCHESTRATOR_LLM_API_MODE", LLM_API_MODE)
ORCHESTRATOR_LLM_OUTPUT_VERSION = os.environ.get("ORCHESTRATOR_LLM_OUTPUT_VERSION") or LLM_OUTPUT_VERSION
ORCHESTRATOR_LLM_DISABLE_THINKING = _env_bool("ORCHESTRATOR_LLM_DISABLE_THINKING", LLM_DISABLE_THINKING)

# RAG 评测集生成专用 LLM：scripts/generate_eval_dataset.py 使用。
# 默认继承 LLM_*，需要用便宜/批量友好的模型生成 query 时可单独覆盖。
EVAL_DATASET_LLM_BASE_URL = os.environ.get("EVAL_DATASET_LLM_BASE_URL") or LLM_BASE_URL
EVAL_DATASET_LLM_API_KEY = os.environ.get("EVAL_DATASET_LLM_API_KEY") or LLM_API_KEY
EVAL_DATASET_LLM_MODEL = os.environ.get("EVAL_DATASET_LLM_MODEL") or LLM_MODEL
EVAL_DATASET_LLM_API_MODE = _api_mode("EVAL_DATASET_LLM_API_MODE", LLM_API_MODE)
EVAL_DATASET_LLM_OUTPUT_VERSION = os.environ.get("EVAL_DATASET_LLM_OUTPUT_VERSION") or LLM_OUTPUT_VERSION
EVAL_DATASET_LLM_DISABLE_THINKING = _env_bool("EVAL_DATASET_LLM_DISABLE_THINKING", LLM_DISABLE_THINKING)

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

# === Backend MVP API / queue / observability ===
BACKEND_API_KEY = os.environ.get("BACKEND_API_KEY") or os.environ.get("WELLNESS_API_KEY", "")
BACKEND_DB_PATH = os.environ.get("BACKEND_DB_PATH", HEALTH_LOGS_DB_PATH)
BACKEND_JOB_LEASE_SEC = int(os.environ.get("BACKEND_JOB_LEASE_SEC", "300"))
BACKEND_OUTBOX_LEASE_SEC = int(os.environ.get("BACKEND_OUTBOX_LEASE_SEC", "120"))
BACKEND_WORKER_IDLE_SEC = float(os.environ.get("BACKEND_WORKER_IDLE_SEC", "1"))
BACKEND_OUTBOX_IDLE_SEC = float(os.environ.get("BACKEND_OUTBOX_IDLE_SEC", "1"))
BACKEND_MAX_AGENT_RETRIES = int(os.environ.get("BACKEND_MAX_AGENT_RETRIES", "3"))
BACKEND_SQLITE_BUSY_TIMEOUT_MS = int(os.environ.get("BACKEND_SQLITE_BUSY_TIMEOUT_MS", "10000"))
BACKEND_SYNC_TIMEOUT_SEC = float(os.environ.get("BACKEND_SYNC_TIMEOUT_SEC", "35"))
BACKEND_MAX_PENDING_JOBS = int(os.environ.get("BACKEND_MAX_PENDING_JOBS", "50"))
BACKEND_MAX_RUNNING_JOBS = int(os.environ.get("BACKEND_MAX_RUNNING_JOBS", "8"))
BACKEND_PREWARM_RAG = _env_bool("BACKEND_PREWARM_RAG", False)
BACKEND_PREWARM_RAG_QUERY = os.environ.get("BACKEND_PREWARM_RAG_QUERY", "健康建议")
BACKEND_RETRY_DELAYS_SEC = tuple(
    int(part.strip())
    for part in os.environ.get("BACKEND_RETRY_DELAYS_SEC", "30,60,120").split(",")
    if part.strip()
)
FAKE_AGENT_MODE = _env_bool("FAKE_AGENT_MODE", False)
FAKE_AGENT_DELAY_MS = int(os.environ.get("FAKE_AGENT_DELAY_MS", "150"))

# === Apple Calendar / iCloud CalDAV（可选）===
# 使用 Apple ID 的 App 专用密码，不要使用 Apple ID 主密码。
ICLOUD_CALDAV_URL = os.environ.get("ICLOUD_CALDAV_URL") or "https://caldav.icloud.com"
ICLOUD_USERNAME = os.environ.get("ICLOUD_USERNAME", "")
ICLOUD_APP_SPECIFIC_PASSWORD = os.environ.get("ICLOUD_APP_SPECIFIC_PASSWORD", "")
ICLOUD_CALENDAR_NAME = os.environ.get("ICLOUD_CALENDAR_NAME", "")

# 情节记忆存储文件（每用户最近 N 轮对话摘要，跨 thread 持久化）
EPISODE_STORE_PATH = os.environ.get("EPISODE_STORE_PATH", "episode_store.json")
EPISODE_SEMANTIC_RETRIEVAL_ENABLED = _env_bool("EPISODE_SEMANTIC_RETRIEVAL_ENABLED", True)
EPISODE_EMBED_ON_WRITE_ENABLED = _env_bool("EPISODE_EMBED_ON_WRITE_ENABLED", False)
EPISODE_SEMANTIC_MIN_COUNT = int(os.environ.get("EPISODE_SEMANTIC_MIN_COUNT", "8"))
EPISODE_SEMANTIC_TOP_K = int(os.environ.get("EPISODE_SEMANTIC_TOP_K", "3"))
EPISODE_INDEX_DIR = os.environ.get(
  "EPISODE_INDEX_DIR",
  str(Path.home() / ".wellness_copilot_indices" / "episodes"),
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

# RAG 长文档增强：
# - hybrid retrieval: dense + 轻量 BM25 共同进入候选池，提升数字、食物名、
#   专有名词、列表项等精确词命中。
# - fine PDF chunking: PDF 结构块使用更小的 max length，并尽量不把多个列表项/
#   表格块塞进同一个 child chunk。
# - parent expansion: 先召回 PDF section parent，再把该 section 内最相关的 child
#   chunks 补进候选池；默认以 gated rescue 方式启用，避免同章节噪声常驻排序。
# - parent rerank context: 是否把 parent excerpt 注入 reranker 输入；默认关闭，
#   避免章节长上下文稀释 child chunk 的局部证据。
# - parent score fusion: 是否让 parent score 参与候选/最终分数；默认保留但低权重。
# - neighbor context: 最终返回 PDF 命中时补前后相邻 chunk，降低答案跨 chunk 边界的风险。
RAG_HYBRID_RETRIEVAL_ENABLED = _env_bool("RAG_HYBRID_RETRIEVAL_ENABLED", True)
RAG_BM25_TOP_K = int(os.environ.get("RAG_BM25_TOP_K", "24"))
RAG_BM25_SCORE_WEIGHT = float(os.environ.get("RAG_BM25_SCORE_WEIGHT", "0.18"))
RAG_RERANK_POOL_MULTIPLIER = int(os.environ.get("RAG_RERANK_POOL_MULTIPLIER", "2"))
RAG_RERANK_POOL_MAX = int(os.environ.get("RAG_RERANK_POOL_MAX", "30"))
RAG_PDF_FINE_CHUNKING_ENABLED = _env_bool("RAG_PDF_FINE_CHUNKING_ENABLED", True)
RAG_PDF_FINE_CHUNK_MAX_CHARS = int(os.environ.get("RAG_PDF_FINE_CHUNK_MAX_CHARS", "320"))
RAG_PDF_PARENT_EXPANSION_ENABLED = _env_bool("RAG_PDF_PARENT_EXPANSION_ENABLED", True)
RAG_PDF_PARENT_RESCUE_ENABLED = _env_bool("RAG_PDF_PARENT_RESCUE_ENABLED", True)
RAG_PDF_PARENT_RESCUE_LOOKAHEAD = int(os.environ.get("RAG_PDF_PARENT_RESCUE_LOOKAHEAD", "10"))
RAG_PDF_PARENT_RESCUE_MIN_PDF_CANDIDATES = int(
    os.environ.get("RAG_PDF_PARENT_RESCUE_MIN_PDF_CANDIDATES", "2")
)
RAG_PDF_PARENT_RESCUE_MIN_PARENT_SCORE = float(
    os.environ.get("RAG_PDF_PARENT_RESCUE_MIN_PARENT_SCORE", "0.56")
)
RAG_PDF_PARENT_RERANK_CONTEXT_ENABLED = _env_bool(
    "RAG_PDF_PARENT_RERANK_CONTEXT_ENABLED",
    False,
)
RAG_PDF_PARENT_SCORE_FUSION_ENABLED = _env_bool(
    "RAG_PDF_PARENT_SCORE_FUSION_ENABLED",
    True,
)
RAG_PDF_SECTION_PARENT_TOP_K = int(os.environ.get("RAG_PDF_SECTION_PARENT_TOP_K", "3"))
RAG_PDF_SECTION_CHILD_TOP_K = int(os.environ.get("RAG_PDF_SECTION_CHILD_TOP_K", "8"))
RAG_PDF_SECTION_PARENT_MAX_CHARS = int(os.environ.get("RAG_PDF_SECTION_PARENT_MAX_CHARS", "2400"))
RAG_PDF_SECTION_SCORE_WEIGHT = float(os.environ.get("RAG_PDF_SECTION_SCORE_WEIGHT", "0.05"))
RAG_PDF_RERANK_PARENT_CONTEXT_CHARS = int(os.environ.get("RAG_PDF_RERANK_PARENT_CONTEXT_CHARS", "120"))
RAG_PDF_NEIGHBOR_CHUNKS = int(os.environ.get("RAG_PDF_NEIGHBOR_CHUNKS", "1"))

# 编码和重排批大小（端侧可调，4060 8GB 默认较稳）
RAG_EMBED_BATCH_SIZE = int(os.environ.get("RAG_EMBED_BATCH_SIZE", "32"))
RAG_RERANK_BATCH_SIZE = int(os.environ.get("RAG_RERANK_BATCH_SIZE", "16"))


# === 社区 MCP 工具服务器（可选）===
# 三个开关默认全 false，老用户拉新版无感升级；显式 opt-in 才会 spawn 子进程。
MCP_TRAINER_ENABLED = _env_bool("MCP_TRAINER_ENABLED", False)
MCP_NUTRITIONIST_ENABLED = _env_bool("MCP_NUTRITIONIST_ENABLED", False)
MCP_DOCTOR_ENABLED = _env_bool("MCP_DOCTOR_ENABLED", _env_bool("MCP_CRITIC_ENABLED", False))
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

# === 多模态 VLM / multimodal_processor（可选）===
# MULTIMODAL_LLM_* 是新配置名；VISION_* 作为旧配置名保留兼容。
_LEGACY_VISION_ENABLED = _env_bool("VISION_ENABLED", True)
MULTIMODAL_LLM_ENABLED = _env_bool("MULTIMODAL_LLM_ENABLED", _LEGACY_VISION_ENABLED)
MULTIMODAL_LLM_PROVIDER = (
    os.environ.get("MULTIMODAL_LLM_PROVIDER")
    or os.environ.get("VISION_PROVIDER")
    or "disabled"
).strip().lower()
MULTIMODAL_LLM_BASE_URL = (
    os.environ.get("MULTIMODAL_LLM_BASE_URL")
    or os.environ.get("VISION_BASE_URL")
    or LLM_BASE_URL
    or ""
)
MULTIMODAL_LLM_API_KEY = (
    os.environ.get("MULTIMODAL_LLM_API_KEY")
    or os.environ.get("VISION_API_KEY")
    or LLM_API_KEY
    or ""
)
MULTIMODAL_LLM_MODEL = os.environ.get("MULTIMODAL_LLM_MODEL") or os.environ.get("VISION_MODEL", "")
MULTIMODAL_LLM_TIMEOUT_SEC = int(
    os.environ.get("MULTIMODAL_LLM_TIMEOUT_SEC")
    or os.environ.get("VISION_TIMEOUT_SEC", "60")
)
MULTIMODAL_LLM_API_MODE = _api_mode("MULTIMODAL_LLM_API_MODE", "chat_completions")

# Backward-compatible aliases for older code/config/docs.
VISION_ENABLED = MULTIMODAL_LLM_ENABLED
VISION_PROVIDER = MULTIMODAL_LLM_PROVIDER
VISION_BASE_URL = MULTIMODAL_LLM_BASE_URL
VISION_API_KEY = MULTIMODAL_LLM_API_KEY
VISION_MODEL = MULTIMODAL_LLM_MODEL
VISION_TIMEOUT_SEC = MULTIMODAL_LLM_TIMEOUT_SEC

# === 微信 iLink / ClawBot（可选）===
WECHAT_ILINK_BASE_URL = os.environ.get(
    "WECHAT_ILINK_BASE_URL",
    "https://ilinkai.weixin.qq.com",
).rstrip("/")
WECHAT_QR_BASE_URL = os.environ.get("WECHAT_QR_BASE_URL", WECHAT_ILINK_BASE_URL).rstrip("/")
WECHAT_CDN_BASE_URL = os.environ.get(
    "WECHAT_CDN_BASE_URL",
    "https://novac2c.cdn.weixin.qq.com/c2c",
).rstrip("/")
WECHAT_ILINK_APP_ID = os.environ.get("WECHAT_ILINK_APP_ID", "bot")
WECHAT_CHANNEL_VERSION = os.environ.get("WECHAT_CHANNEL_VERSION", "0.1.0")
WECHAT_BOT_TYPE = int(os.environ.get("WECHAT_BOT_TYPE", "3"))
WECHAT_BOT_TOKEN = os.environ.get("WECHAT_BOT_TOKEN", "")
WECHAT_POLL_TIMEOUT_SEC = int(os.environ.get("WECHAT_POLL_TIMEOUT_SEC", "30"))
WECHAT_WORKER_IDLE_SEC = float(os.environ.get("WECHAT_WORKER_IDLE_SEC", "1"))
WECHAT_ENDPOINT_QRCODE = os.environ.get("WECHAT_ENDPOINT_QRCODE", "/ilink/bot/get_bot_qrcode")
WECHAT_ENDPOINT_QRCODE_STATUS = os.environ.get(
    "WECHAT_ENDPOINT_QRCODE_STATUS",
    "/ilink/bot/get_qrcode_status",
)
WECHAT_ENDPOINT_UPDATES = os.environ.get("WECHAT_ENDPOINT_UPDATES", "/ilink/bot/getupdates")
WECHAT_ENDPOINT_SEND = os.environ.get("WECHAT_ENDPOINT_SEND", "/ilink/bot/sendmessage")
WECHAT_ENDPOINT_PUSH = os.environ.get("WECHAT_ENDPOINT_PUSH", WECHAT_ENDPOINT_SEND)
WECHAT_ENDPOINT_MEDIA = os.environ.get("WECHAT_ENDPOINT_MEDIA", "")

# === 备份（可选）===
BACKUP_DIR = os.environ.get("BACKUP_DIR", "backups")
BACKUP_RETENTION_DAYS = int(os.environ.get("BACKUP_RETENTION_DAYS", "14"))
BACKUP_INTERVAL_HOURS = float(os.environ.get("BACKUP_INTERVAL_HOURS", "24"))
OSS_ACCESS_KEY_ID = os.environ.get("OSS_ACCESS_KEY_ID", "")
OSS_ACCESS_KEY_SECRET = os.environ.get("OSS_ACCESS_KEY_SECRET", "")
OSS_BUCKET = os.environ.get("OSS_BUCKET", "")
OSS_ENDPOINT = os.environ.get("OSS_ENDPOINT", "")
OSS_PREFIX = os.environ.get("OSS_PREFIX", "wellness-copilot-backup")
