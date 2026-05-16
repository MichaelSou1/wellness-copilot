import re
import os
import json
import hashlib
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

from .config import (
    KNOWLEDGE_BASE_DIR,
    KNOWLEDGE_BASE_AGENT_SUBDIRS,
    RAG_EMBED_BATCH_SIZE,
    RAG_EMBED_MODEL_NAME,
    RAG_FALLBACK_EMBED_MODEL_NAME,
    RAG_FINAL_TOP_K,
    RAG_HF_HOME,
    RAG_HF_HUB_CACHE,
    RAG_RERANK_BATCH_SIZE,
    RAG_RERANK_MODEL_NAME,
    RAG_RETRIEVE_TOP_K,
    RAG_DEVICE,
)


# 目前支持的知识语料文件类型。
# - .md / .txt: 直接按 UTF-8 读取
# - .pdf: 通过 pypdf 按页提取文本,用 \f (form-feed) 作为分页分隔符,
#         便于后续 chunk 归页
# - .docx: 通过 python-docx 提取段落文本
SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf", ".docx"}
PAGE_SEPARATOR = "\f"

# 句末边界正则：中文句末标点，或英文句末标点后接空白/行尾。
# 用于切分时的"软边界对齐"，避免在句子中间硬切。
_SENT_END_RE = re.compile(r'[。！？]|[.!?](?=[ \t\n]|$)')


@dataclass
class Chunk:
    chunk_id: str
    source: str
    text: str
    page_range: Optional[str] = None  # 仅 PDF 有值, 例如 "3" 或 "3-4"


# Module-level caches so all LocalKnowledgeBase instances share one copy of each model.
_EMBED_MODEL_CACHE: Dict[str, object] = {}
_RERANK_MODEL_CACHE: Dict[str, object] = {}
_RERANK_DISABLED: Dict[str, str] = {}  # model_name -> reason string when load failed

_KNOWN_CHECKPOINT_MODES = {
    "BAAI/bge-m3": "bin",
    "BAAI/bge-reranker-v2-m3": "safetensors",
    "BAAI/bge-small-zh-v1.5": "safetensors",
}


class LocalKnowledgeBase:
    """本地 RAG：两阶段检索（Dense Retrieve + Cross-Encoder Re-rank）。"""

    def __init__(
        self,
        kb_dir: str = KNOWLEDGE_BASE_DIR,
        chunk_size: int = 420,
        overlap: int = 100,
        recursive: bool = True,
        boundary_look_back: int = 120,
        min_chunk_chars: int = 30,
    ):
        self.kb_dir = Path(kb_dir)
        self.cache_dir = self.kb_dir / ".index_cache"
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.recursive = recursive
        self.boundary_look_back = boundary_look_back
        self.min_chunk_chars = min_chunk_chars
        self.chunks: List[Chunk] = []

        # 运行时对象（延迟加载，减少冷启动）
        self._np = None
        self._embed_model = None
        self._embed_model_name = ""  # actual model name used (may be fallback)
        self._rerank_model = None
        self._device = None
        self._rerank_disabled_reason = ""

        # 索引缓存
        self._chunk_embeddings = None
        self._faiss_index = None
        self._faiss_disabled_reason = ""
        self._fingerprint = None
        self._ready = False

    def _read_documents(self) -> List[Dict[str, str]]:
        if not self.kb_dir.exists():
            return []

        docs = []
        iter_files = self.kb_dir.rglob("*") if self.recursive else self.kb_dir.glob("*")
        for path in sorted(iter_files):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
                continue

            source = str(path.relative_to(self.kb_dir)).replace("\\", "/")

            if suffix == ".pdf":
                content = self._read_pdf(path)
                file_type = "pdf"
            elif suffix == ".docx":
                content = self._read_docx(path)
                file_type = "text"
            else:
                try:
                    content = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    content = path.read_text(encoding="utf-8", errors="ignore")
                file_type = "text"

            if not content or not content.strip():
                # 扫描版 PDF / 空文件等情况:跳过,避免污染索引
                continue

            docs.append({"source": source, "text": content, "file_type": file_type})
        return docs

    @staticmethod
    def _read_pdf(path: Path) -> str:
        """按页提取 PDF 文本,用 form-feed (\\f) 作为页分隔符。

        设计说明:
        - 用 pypdf 进行纯 Python 解析,无需系统依赖,部署方便。
        - 使用 \\f 作为分页符,后续 chunker 可以通过计算 \\f 出现次数推算 chunk 所在页范围,
          而又不会因为分页符干扰 embedding 文本语义(\\f 不携带语义,BGE 类 tokenizer 会直接忽略)。
        - 对单页文本做轻量清洗:合并重复换行、去除行尾空白,减少 PDF 提取时的噪声。
        - 扫描版 PDF 会得到空字符串 —— 由调用方负责跳过;OCR 不在本项目范围内。
        """
        try:
            pypdf = importlib.import_module("pypdf")
        except ImportError as e:
            raise ImportError(
                "解析 PDF 需要 pypdf,请先执行: pip install pypdf"
            ) from e

        try:
            reader = pypdf.PdfReader(str(path))
        except Exception as e:
            print(f"[RAG][warn] 无法解析 PDF: {path} ({e.__class__.__name__}: {e})")
            return ""

        pages_text: List[str] = []
        for page in reader.pages:
            try:
                raw = page.extract_text() or ""
            except Exception:
                raw = ""
            # 轻量清洗:压缩重复空行,去掉行尾空白
            cleaned = re.sub(r"[ \t]+\n", "\n", raw)
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
            # 页尾补一个换行,防止后续 chunk 跨页时两页文字被直接拼接成无意义长词
            if cleaned:
                cleaned += "\n"
            pages_text.append(cleaned)

        return PAGE_SEPARATOR.join(pages_text)

    @staticmethod
    def _read_docx(path: Path) -> str:
        """提取 Word (.docx) 文件的段落文本，段落间以换行分隔。

        设计说明:
        - 使用 python-docx 进行纯 Python 解析，无需 LibreOffice 等系统依赖。
        - 仅提取正文段落（Paragraph）；表格单元格内容也一并提取，避免遗漏结构化数据。
        - 不处理嵌入图片/图表等非文本内容（OCR 不在本项目范围）。
        """
        try:
            docx = importlib.import_module("docx")
        except ImportError as e:
            raise ImportError(
                "解析 Word 文档需要 python-docx，请先执行: pip install python-docx"
            ) from e

        try:
            doc = docx.Document(str(path))
        except Exception as e:
            print(f"[RAG][warn] 无法解析 Word 文档: {path} ({e.__class__.__name__}: {e})")
            return ""

        parts: List[str] = []

        # 正文段落
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                parts.append(text)

        # 表格单元格
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    text = cell.text.strip()
                    if text:
                        parts.append(text)

        return "\n".join(parts)

    def _docs_fingerprint(self, docs: List[Dict[str, str]], effective_embed_model: str = "") -> str:
        payload = {
            "docs": docs,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "boundary_look_back": self.boundary_look_back,
            "min_chunk_chars": self.min_chunk_chars,
            # Use the model that was actually loaded, not just the configured name.
            # This prevents a stale cache hit when the fallback model was used previously.
            "embed_model": effective_embed_model or RAG_EMBED_MODEL_NAME,
        }
        s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def _resolve_device(self) -> str:
        if RAG_DEVICE != "auto":
            return RAG_DEVICE

        try:
            torch = importlib.import_module("torch")
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _lazy_import_numpy(self):
        if self._np is None:
            self._np = importlib.import_module("numpy")
        return self._np

    @staticmethod
    def _configure_hf_cache_env():
        Path(RAG_HF_HOME).mkdir(parents=True, exist_ok=True)
        Path(RAG_HF_HUB_CACHE).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", RAG_HF_HOME)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", RAG_HF_HUB_CACHE)

    @staticmethod
    def _torch_loads_bin_safely() -> bool:
        try:
            torch = importlib.import_module("torch")
        except ImportError:
            return False

        version = torch.__version__.split("+", 1)[0]
        parts = []
        for raw_part in version.split(".")[:2]:
            digits = "".join(ch for ch in raw_part if ch.isdigit())
            parts.append(int(digits or 0))
        while len(parts) < 2:
            parts.append(0)
        return tuple(parts) >= (2, 6)

    @staticmethod
    def _can_load_model_checkpoint(model_name: str, model_path: Optional[str]) -> bool:
        mode = _KNOWN_CHECKPOINT_MODES.get(model_name)
        if mode == "safetensors":
            return True
        if mode == "bin":
            return LocalKnowledgeBase._torch_loads_bin_safely()

        if model_path:
            path = Path(model_path)
            if any(path.glob("*.safetensors")):
                return True
            if any(path.glob("*.bin")):
                return LocalKnowledgeBase._torch_loads_bin_safely()

        return True

    @staticmethod
    def _hub_cache_candidates() -> List[Path]:
        candidates = [
            Path(RAG_HF_HUB_CACHE),
        ]

        env_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
        if env_cache:
            candidates.append(Path(env_cache))

        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            candidates.append(Path(hf_home) / "hub")

        candidates.append(Path.home() / ".cache" / "huggingface" / "hub")

        unique = []
        seen = set()
        for path in candidates:
            resolved = str(path.expanduser())
            if resolved in seen:
                continue
            seen.add(resolved)
            unique.append(path.expanduser())
        return unique

    @staticmethod
    def _resolve_local_model_path(
        model_name: str,
        required_files: Optional[List[str]] = None,
    ) -> Optional[str]:
        if not model_name:
            return None

        direct = Path(model_name)
        if direct.exists():
            if required_files and any(not (direct / name).exists() for name in required_files):
                return None
            return str(direct)

        if "/" not in model_name:
            return None

        repo_dir_name = f"models--{model_name.replace('/', '--')}"
        for hub_cache in LocalKnowledgeBase._hub_cache_candidates():
            repo_dir = hub_cache / repo_dir_name
            snapshots_dir = repo_dir / "snapshots"
            if not snapshots_dir.exists():
                continue

            ref_main = repo_dir / "refs" / "main"
            if ref_main.exists():
                try:
                    revision = ref_main.read_text(encoding="utf-8").strip()
                    resolved = snapshots_dir / revision
                    if resolved.exists():
                        if required_files and any(not (resolved / name).exists() for name in required_files):
                            continue
                        return str(resolved)
                except Exception as e:
                    print(f"[RAG][warn] Failed to read refs/main for {model_name}: {e}")

            snapshots = sorted(p for p in snapshots_dir.iterdir() if p.is_dir())
            for resolved in reversed(snapshots):
                if required_files and any(not (resolved / name).exists() for name in required_files):
                    continue
                return str(resolved)
        return None

    @staticmethod
    def _model_label(requested_name: str, resolved_path: Optional[str]) -> str:
        return resolved_path or requested_name

    def _load_embed_model(self, sentence_transformers):
        if self._embed_model is not None:
            return

        # Use module-level cache: all KB instances share one SentenceTransformer
        cache_key = f"{RAG_EMBED_MODEL_NAME}|{RAG_FALLBACK_EMBED_MODEL_NAME}"
        if cache_key in _EMBED_MODEL_CACHE:
            self._embed_model, self._embed_model_name = _EMBED_MODEL_CACHE[cache_key]
            return

        embed_required_files = ["modules.json", "config.json"]
        primary_path = self._resolve_local_model_path(
            RAG_EMBED_MODEL_NAME,
            required_files=embed_required_files,
        )
        fallback_path = self._resolve_local_model_path(
            RAG_FALLBACK_EMBED_MODEL_NAME,
            required_files=embed_required_files,
        )

        candidates = []
        if primary_path:
            if self._can_load_model_checkpoint(RAG_EMBED_MODEL_NAME, primary_path):
                candidates.append((RAG_EMBED_MODEL_NAME, primary_path))
            elif fallback_path:
                print(
                    "[RAG][warn] Embedding model "
                    f"{RAG_EMBED_MODEL_NAME} is cached but requires torch>=2.6; "
                    f"fallback to {RAG_FALLBACK_EMBED_MODEL_NAME}."
                )
        elif not fallback_path and self._can_load_model_checkpoint(RAG_EMBED_MODEL_NAME, None):
            candidates.append((RAG_EMBED_MODEL_NAME, None))

        if (
            RAG_FALLBACK_EMBED_MODEL_NAME != RAG_EMBED_MODEL_NAME
            and fallback_path
        ):
            candidates.append((RAG_FALLBACK_EMBED_MODEL_NAME, fallback_path))

        last_error = None
        for requested_name, resolved_path in candidates:
            try:
                model_ref = self._model_label(requested_name, resolved_path)
                model = sentence_transformers.SentenceTransformer(
                    model_ref,
                    device=self._device,
                )
                if requested_name != RAG_EMBED_MODEL_NAME:
                    print(
                        "[RAG][warn] Embedding model "
                        f"{RAG_EMBED_MODEL_NAME} unavailable locally; "
                        f"fallback to {requested_name}."
                    )
                self._embed_model = model
                self._embed_model_name = requested_name
                _EMBED_MODEL_CACHE[cache_key] = (model, requested_name)
                return
            except Exception as e:
                last_error = e
                print(
                    "[RAG][warn] Failed to load embedding model "
                    f"{requested_name}: {e.__class__.__name__}: {e}"
                )

        raise RuntimeError(
            "Unable to load any embedding model. "
            f"Tried primary={RAG_EMBED_MODEL_NAME}, "
            f"fallback={RAG_FALLBACK_EMBED_MODEL_NAME}."
        ) from last_error

    def _lazy_load_reranker(self, sentence_transformers):
        if self._rerank_model is not None or self._rerank_disabled_reason:
            return

        # Use module-level cache: all KB instances share one CrossEncoder
        if RAG_RERANK_MODEL_NAME in _RERANK_MODEL_CACHE:
            self._rerank_model = _RERANK_MODEL_CACHE[RAG_RERANK_MODEL_NAME]
            return
        if RAG_RERANK_MODEL_NAME in _RERANK_DISABLED:
            self._rerank_disabled_reason = _RERANK_DISABLED[RAG_RERANK_MODEL_NAME]
            return

        rerank_path = self._resolve_local_model_path(
            RAG_RERANK_MODEL_NAME,
            required_files=["config.json"],
        )
        model_ref = self._model_label(RAG_RERANK_MODEL_NAME, rerank_path)
        try:
            model = sentence_transformers.CrossEncoder(
                model_ref,
                device=self._device,
                max_length=512,
            )
            self._rerank_model = model
            _RERANK_MODEL_CACHE[RAG_RERANK_MODEL_NAME] = model
        except Exception as e:
            reason = f"{e.__class__.__name__}: {e}"
            self._rerank_disabled_reason = reason
            _RERANK_DISABLED[RAG_RERANK_MODEL_NAME] = reason
            print(
                "[RAG][warn] Failed to load reranker "
                f"{RAG_RERANK_MODEL_NAME}; falling back to dense-only retrieval. "
                f"Reason: {reason}"
            )

    def _lazy_load_models(self, require_reranker: bool = True):
        if self._embed_model is not None and (
            self._rerank_model is not None or not require_reranker or self._rerank_disabled_reason
        ):
            return

        self._configure_hf_cache_env()
        sentence_transformers = importlib.import_module("sentence_transformers")
        torch = importlib.import_module("torch")

        self._device = self._resolve_device()
        use_fp16 = self._device == "cuda"

        self._load_embed_model(sentence_transformers)
        if require_reranker:
            self._lazy_load_reranker(sentence_transformers)

        # 端侧优化：GPU 场景启用半精度，减少显存和延迟
        if use_fp16:
            try:
                self._embed_model.half()
                if self._rerank_model is not None and hasattr(self._rerank_model, "model"):
                    self._rerank_model.model.half()
            except Exception:
                pass

        # 端侧优化：开启推理优化（若可用）
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    def _cache_meta_path(self) -> Path:
        return self.cache_dir / "index_meta.json"

    def _cache_embeddings_path(self) -> Path:
        return self.cache_dir / "embeddings.npy"

    def _cache_faiss_path(self) -> Path:
        return self.cache_dir / "index.faiss"

    def _cache_chunks_path(self) -> Path:
        return self.cache_dir / "chunks.json"

    def _lazy_import_faiss(self):
        try:
            return importlib.import_module("faiss")
        except Exception as e:
            if not self._faiss_disabled_reason:
                self._faiss_disabled_reason = f"{e.__class__.__name__}: {e}"
                print(
                    "[RAG][warn] FAISS unavailable; falling back to NumPy dense retrieval. "
                    f"{self._faiss_disabled_reason}"
                )
            return None

    def _build_faiss_index(self):
        if self._chunk_embeddings is None or len(self._chunk_embeddings) == 0:
            self._faiss_index = None
            return None

        faiss = self._lazy_import_faiss()
        if faiss is None:
            self._faiss_index = None
            return None

        try:
            np = self._lazy_import_numpy()
            embeddings = np.ascontiguousarray(self._chunk_embeddings, dtype=np.float32)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            self._faiss_index = index
            return index
        except Exception as e:
            self._faiss_index = None
            self._faiss_disabled_reason = f"{e.__class__.__name__}: {e}"
            print(
                "[RAG][warn] Failed to build FAISS index; falling back to NumPy dense retrieval. "
                f"{self._faiss_disabled_reason}"
            )
            return None

    def _try_load_faiss_index(self) -> bool:
        faiss_path = self._cache_faiss_path()
        if not faiss_path.exists():
            return False

        faiss = self._lazy_import_faiss()
        if faiss is None:
            return False

        try:
            self._faiss_index = faiss.read_index(str(faiss_path))
            return True
        except Exception as e:
            self._faiss_index = None
            print(
                "[RAG][warn] Failed to load FAISS index cache; rebuilding from embeddings. "
                f"{e.__class__.__name__}: {e}"
            )
            return False

    def _save_faiss_index(self):
        if self._faiss_index is None:
            return

        faiss = self._lazy_import_faiss()
        if faiss is None:
            return

        try:
            faiss.write_index(self._faiss_index, str(self._cache_faiss_path()))
        except Exception as e:
            print(
                "[RAG][warn] Failed to save FAISS index cache; retrieval can still use in-memory index. "
                f"{e.__class__.__name__}: {e}"
            )

    def _try_load_cache(self) -> bool:
        np = self._lazy_import_numpy()
        meta_path = self._cache_meta_path()
        emb_path = self._cache_embeddings_path()
        chunks_path = self._cache_chunks_path()
        if not (meta_path.exists() and emb_path.exists() and chunks_path.exists()):
            return False

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("fingerprint") != self._fingerprint:
                return False

            chunk_items = json.loads(chunks_path.read_text(encoding="utf-8"))
            self.chunks = [Chunk(**item) for item in chunk_items]
            self._chunk_embeddings = np.load(str(emb_path))
            if not self._try_load_faiss_index():
                self._build_faiss_index()
            return True
        except Exception:
            return False

    def _save_cache(self):
        np = self._lazy_import_numpy()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_meta_path().write_text(
            json.dumps({"fingerprint": self._fingerprint}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._cache_chunks_path().write_text(
            json.dumps([c.__dict__ for c in self.chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        np.save(str(self._cache_embeddings_path()), self._chunk_embeddings)
        self._save_faiss_index()

    def _split_text(self, text: str):
        """切分文本为 chunks。返回 [(piece, page_range_or_None), ...]。

        切分策略（优先级从高到低）：
        1. Markdown 标题行（^#{1,6} ）作为 section 硬边界，chunk 不跨 section，
           避免将不同主题段落混入同一个 chunk。
        2. 在 section 内做滑动窗口；窗口末尾向前查找最近断句点，
           查找范围为 boundary_look_back（默认 120 字符）：
             - 优先级 1：段落分隔符 \\n\\n
             - 优先级 2：中英文句末标点（。！？ / . ! ? + 空白）
             - 优先级 3：单个换行 \\n
           找不到断点则退回字符硬切（与原策略相同）。
        3. 长度 < min_chunk_chars 的碎片直接丢弃，避免噪声 chunk 污染索引。

        PDF 文件因含有 PAGE_SEPARATOR，会先建立「字符偏移 → 页码」映射，
        chunk 产生后反推页码区间用于 citation，与原有行为完全兼容。
        """
        has_pages = PAGE_SEPARATOR in text
        page_numbers: Optional[List[int]] = None

        if has_pages:
            cleaned_chars: List[str] = []
            page_numbers = []
            current_page = 1
            for ch in text:
                if ch == PAGE_SEPARATOR:
                    current_page += 1
                    continue
                cleaned_chars.append(ch)
                page_numbers.append(current_page)
            cleaned = "".join(cleaned_chars)
        else:
            cleaned = re.sub(r"\n{3,}", "\n\n", text).strip()

        if not cleaned.strip():
            return []

        # 按 Markdown 标题行划出 section 边界（非 Markdown 文件只有一个 section）
        heading_positions = [0]
        for m in re.finditer(r'^#{1,6}\s', cleaned, re.MULTILINE):
            if m.start() > 0:
                heading_positions.append(m.start())
        heading_positions.append(len(cleaned))

        look_back = self.boundary_look_back
        results: List = []

        for si in range(len(heading_positions) - 1):
            sec_start = heading_positions[si]
            sec_end = heading_positions[si + 1]
            sec_text = cleaned[sec_start:sec_end]
            n = len(sec_text)
            start = 0

            while start < n:
                end = min(start + self.chunk_size, n)

                # 在末尾 look_back 窗口内寻找最佳断句位置（仅在非末尾块时执行）
                if end < n:
                    win_start = max(start, end - look_back)

                    # 优先级 1：段落分隔
                    p = sec_text.rfind('\n\n', win_start, end)
                    if p > start:
                        end = p + 2
                    else:
                        # 优先级 2：句末标点（中英文）
                        best = -1
                        for m in _SENT_END_RE.finditer(sec_text, win_start, end):
                            if m.end() > start:
                                best = m.end()  # 取最靠右的命中
                        if best > start:
                            end = best
                        else:
                            # 优先级 3：单个换行
                            p = sec_text.rfind('\n', win_start, end)
                            if p > start:
                                end = p + 1

                piece = sec_text[start:end].strip()
                if len(piece) >= self.min_chunk_chars:
                    global_start = sec_start + start
                    global_end = sec_start + end - 1
                    if page_numbers is not None:
                        last_idx = len(page_numbers) - 1
                        ps = page_numbers[min(global_start, last_idx)]
                        pe = page_numbers[min(global_end, last_idx)]
                        page_range = str(ps) if ps == pe else f"{ps}-{pe}"
                    else:
                        page_range = None
                    results.append((piece, page_range))

                if end >= n:
                    break
                start = max(start + 1, end - self.overlap)

        return results

    def clear_cache(self):
        if self.cache_dir.exists():
            for p in self.cache_dir.rglob("*"):
                if p.is_file():
                    p.unlink()
            for p in sorted(self.cache_dir.rglob("*"), reverse=True):
                if p.is_dir():
                    p.rmdir()

        self._chunk_embeddings = None
        self._faiss_index = None
        self.chunks = []
        self._ready = False

    def build(self, force_rebuild: bool = False):
        if force_rebuild:
            self.clear_cache()

        docs = self._read_documents()

        # Fingerprint is computed before model load for cache lookup.
        # We use a provisional fingerprint (primary model name) to check the cache.
        # If the cache matches, we accept it; if not, we load the model, then
        # recompute the fingerprint with the effective model name before saving.
        provisional_fp = self._docs_fingerprint(docs)
        self._fingerprint = provisional_fp

        chunks: List[Chunk] = []
        for d in docs:
            pieces = self._split_text(d["text"])
            for i, (piece_text, page_range) in enumerate(pieces):
                if page_range:
                    chunk_id = f"{d['source']}#p{page_range}-chunk-{i+1}"
                else:
                    chunk_id = f"{d['source']}#chunk-{i+1}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        source=d["source"],
                        text=piece_text,
                        page_range=page_range,
                    )
                )

        self.chunks = chunks

        if not self.chunks:
            self._chunk_embeddings = None
            self._faiss_index = None
            self._ready = True
            return

        # 优先加载缓存索引，加速冷启动
        if self._try_load_cache():
            self._ready = True
            return

        self._lazy_load_models(require_reranker=False)

        # Recompute fingerprint with effective model name in case fallback was used
        if self._embed_model_name and self._embed_model_name != RAG_EMBED_MODEL_NAME:
            self._fingerprint = self._docs_fingerprint(docs, self._embed_model_name)
        np = self._lazy_import_numpy()

        texts = [c.text for c in self.chunks]
        embeddings = self._embed_model.encode(
            texts,
            batch_size=RAG_EMBED_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self._chunk_embeddings = np.asarray(embeddings, dtype=np.float32)
        self._build_faiss_index()
        self._save_cache()
        self._ready = True

    def get_index_stats(self) -> Dict[str, int]:
        return {
            "doc_count": len(self._read_documents()),
            "chunk_count": len(self.chunks),
            "cache_exists": int(self.cache_dir.exists()),
            "faiss_index_loaded": int(self._faiss_index is not None),
        }

    def _dense_topk(self, query_emb, top_k: int):
        np = self._lazy_import_numpy()
        top_k = max(1, min(top_k, len(self.chunks)))

        if self._faiss_index is not None:
            query_vec = np.ascontiguousarray([query_emb], dtype=np.float32)
            scores, indices = self._faiss_index.search(query_vec, top_k)
            pairs = [
                (int(i), float(score))
                for i, score in zip(indices[0], scores[0])
                if int(i) >= 0
            ]
            return pairs

        sim_scores = self._chunk_embeddings @ query_emb
        candidate_idx = np.argpartition(-sim_scores, top_k - 1)[:top_k]
        candidate_idx = sorted(
            candidate_idx.tolist(), key=lambda i: float(sim_scores[i]), reverse=True
        )
        return [(int(i), float(sim_scores[i])) for i in candidate_idx]

    def retrieve_stages(
        self,
        query: str,
        stage1_k: int = RAG_RETRIEVE_TOP_K,
        stage2_k: int = RAG_FINAL_TOP_K,
    ) -> Dict[str, List[Dict[str, object]]]:
        """返回两阶段检索的原始结果，用于对 Embedding 召回与 Rerank 质量分别做评测。

        - stage1: 仅使用 embedding 的稠密检索 Top-K（按余弦相似度排序）
        - stage2: 对 stage1 的候选做 cross-encoder 重排后的 Top-K（按 rerank 分排序）

        与 `retrieve()` 不同，这里不做任何分数融合或 boost，保持原始分数以便公平对比两个阶段。
        """
        empty: Dict[str, List[Dict[str, object]]] = {"stage1": [], "stage2": []}
        if not self._ready:
            self.build()

        if not self.chunks or self._chunk_embeddings is None:
            return empty

        query = (query or "").strip()
        if not query:
            return empty

        self._lazy_load_models(require_reranker=False)

        stage1_k = max(1, min(stage1_k, len(self.chunks)))
        stage2_k = max(1, min(stage2_k, stage1_k))

        # Stage-1: Dense Retrieve
        query_emb = self._embed_model.encode(
            [query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        candidates = self._dense_topk(query_emb, stage1_k)
        candidate_idx = [i for i, _dense in candidates]
        dense_scores = {i: dense for i, dense in candidates}

        stage1_results: List[Dict[str, object]] = []
        for rank, i in enumerate(candidate_idx, start=1):
            c = self.chunks[i]
            stage1_results.append(
                {
                    "rank": rank,
                    "chunk_id": c.chunk_id,
                    "source": c.source,
                    "page_range": c.page_range,
                    "dense_score": round(dense_scores[i], 4),
                }
            )

        stage2_results: List[Dict[str, object]] = []
        self._lazy_load_models(require_reranker=True)
        if self._rerank_model is None:
            for rank, i in enumerate(candidate_idx[:stage2_k], start=1):
                c = self.chunks[i]
                stage2_results.append(
                    {
                        "rank": rank,
                        "chunk_id": c.chunk_id,
                        "source": c.source,
                        "page_range": c.page_range,
                        "dense_score": round(dense_scores[i], 4),
                        "rerank_score": None,
                    }
                )
            return {"stage1": stage1_results, "stage2": stage2_results}

        # Stage-2: Re-rank
        pairs = [[query, self.chunks[i].text] for i in candidate_idx]
        rerank_scores = self._rerank_model.predict(
            pairs,
            batch_size=RAG_RERANK_BATCH_SIZE,
            show_progress_bar=False,
        )

        combined = []
        for i, rr in zip(candidate_idx, rerank_scores):
            combined.append((i, dense_scores[i], float(rr)))
        combined.sort(key=lambda x: x[2], reverse=True)

        for rank, (i, dense, rr) in enumerate(combined[:stage2_k], start=1):
            c = self.chunks[i]
            stage2_results.append(
                {
                    "rank": rank,
                    "chunk_id": c.chunk_id,
                    "source": c.source,
                    "page_range": c.page_range,
                    "dense_score": round(dense, 4),
                    "rerank_score": round(rr, 4),
                }
            )

        return {"stage1": stage1_results, "stage2": stage2_results}

    def retrieve(self, query: str, top_k: int = RAG_FINAL_TOP_K) -> List[Dict[str, str]]:
        if not self._ready:
            self.build()

        if not self.chunks or self._chunk_embeddings is None:
            return []

        query = (query or "").strip()
        if not query:
            return []

        self._lazy_load_models(require_reranker=False)

        # Stage-1: Dense Retrieve
        query_emb = self._embed_model.encode(
            [query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        retrieve_k = min(max(top_k, RAG_RETRIEVE_TOP_K), len(self.chunks))
        candidates = self._dense_topk(query_emb, retrieve_k)
        candidate_idx = [i for i, _dense in candidates]
        dense_scores = {i: dense for i, dense in candidates}

        combined = []
        self._lazy_load_models(require_reranker=True)
        if self._rerank_model is None:
            for i in candidate_idx:
                dense = dense_scores[i]
                combined.append((i, dense, None, dense))
        else:
            # Stage-2: Re-rank
            pairs = [[query, self.chunks[i].text] for i in candidate_idx]
            rerank_scores = self._rerank_model.predict(
                pairs,
                batch_size=RAG_RERANK_BATCH_SIZE,
                show_progress_bar=False,
            )

            for i, rr in zip(candidate_idx, rerank_scores):
                dense = dense_scores[i]
                rerank = float(rr)
                # 组合分数（以重排分为主，保留召回分作微调）
                final_score = rerank + 0.15 * dense
                combined.append((i, dense, rerank, final_score))

        combined.sort(key=lambda x: x[3], reverse=True)
        final_k = min(max(1, top_k), len(combined))
        selected = combined[:final_k]

        results = []
        for idx, dense_score, rerank_score, final_score in selected:
            c = self.chunks[idx]
            results.append(
                {
                    "chunk_id": c.chunk_id,
                    "source": c.source,
                    "page_range": c.page_range,
                    "score": round(final_score, 4),
                    "dense_score": round(dense_score, 4),
                    "rerank_score": round(rerank_score, 4) if rerank_score is not None else None,
                    "content": c.text,
                }
            )
        return results
