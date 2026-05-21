import re
import os
import json
import hashlib
import importlib
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from .config import (
    KNOWLEDGE_BASE_DIR,
    KNOWLEDGE_BASE_AGENT_SUBDIRS,
    RAG_EMBED_BATCH_SIZE,
    RAG_EMBED_MODEL_NAME,
    RAG_FALLBACK_EMBED_MODEL_NAME,
    RAG_FINAL_TOP_K,
    RAG_BM25_SCORE_WEIGHT,
    RAG_BM25_TOP_K,
    RAG_HF_HOME,
    RAG_HF_HUB_CACHE,
    RAG_HYBRID_RETRIEVAL_ENABLED,
    RAG_PDF_FINE_CHUNK_MAX_CHARS,
    RAG_PDF_FINE_CHUNKING_ENABLED,
    RAG_PDF_NEIGHBOR_CHUNKS,
    RAG_PDF_PARENT_EXPANSION_ENABLED,
    RAG_PDF_PARENT_RESCUE_ENABLED,
    RAG_PDF_PARENT_RESCUE_LOOKAHEAD,
    RAG_PDF_PARENT_RESCUE_MIN_PARENT_SCORE,
    RAG_PDF_PARENT_RESCUE_MIN_PDF_CANDIDATES,
    RAG_PDF_PARENT_RERANK_CONTEXT_ENABLED,
    RAG_PDF_PARENT_SCORE_FUSION_ENABLED,
    RAG_PDF_RERANK_PARENT_CONTEXT_CHARS,
    RAG_PDF_SECTION_CHILD_TOP_K,
    RAG_PDF_SECTION_PARENT_MAX_CHARS,
    RAG_PDF_SECTION_PARENT_TOP_K,
    RAG_PDF_SECTION_SCORE_WEIGHT,
    RAG_RERANK_BATCH_SIZE,
    RAG_RERANK_MODEL_NAME,
    RAG_RERANK_POOL_MAX,
    RAG_RERANK_POOL_MULTIPLIER,
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
CHUNKER_VERSION = "structure-v5-hybrid-fine-pdf-gated-parent"

# 句末边界正则：中文句末标点，或英文句末标点后接空白/行尾。
# 用于切分时的"软边界对齐"，避免在句子中间硬切。
_SENT_END_RE = re.compile(r'[。！？]|[.!?](?=[ \t\n]|$)')
_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+\S")
_MARKDOWN_LIST_ITEM_RE = re.compile(r"^\s{0,6}(?:[-*+•‣◦]|\d{1,3}[.)])\s+")
_MARKDOWN_TABLE_RE = re.compile(r"^\s*\|.*\|\s*$")
_MARKDOWN_RULE_RE = re.compile(r"^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_WORD_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+-]{1,}")
_PDF_CHAPTER_HEADING_RE = re.compile(
    r"^第[一二三四五六七八九十百千万零〇两\d]+[章节篇部]\s*\S*"
)
_PDF_ENGLISH_NUMBERED_HEADING_RE = re.compile(
    r"^(?:chapter|part|section)\s+[0-9ivxlcdm]+(?:[.:\-\s]+|$)\S*",
    re.IGNORECASE,
)
_PDF_PAGE_ARTIFACT_RE = re.compile(
    r"^(?:Dietary Guidelines for Americans,?\s*\d{4}.*\|\s*\d+|\d+)$",
    re.IGNORECASE,
)


@dataclass
class Chunk:
    chunk_id: str
    source: str
    text: str
    page_range: Optional[str] = None  # 仅 PDF 有值, 例如 "3" 或 "3-4"
    section_id: str = ""
    section_path: str = ""
    section_index: int = 0
    is_pdf: bool = False


@dataclass
class SectionParent:
    section_id: str
    source: str
    section_path: str
    text: str
    page_range: Optional[str] = None
    child_chunk_ids: List[str] = None


@dataclass
class _TextBlock:
    text: str
    start: int
    end: int  # exclusive offset in the cleaned document
    kind: str = "paragraph"
    heading_level: int = 0


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
        self.section_parents: List[SectionParent] = []

        # 运行时对象（延迟加载，减少冷启动）
        self._np = None
        self._embed_model = None
        self._embed_model_name = ""  # actual model name used (may be fallback)
        self._rerank_model = None
        self._device = None
        self._rerank_disabled_reason = ""

        # 索引缓存
        self._chunk_embeddings = None
        self._section_embeddings = None
        self._faiss_index = None
        self._faiss_disabled_reason = ""
        self._bm25_doc_counts: List[Counter] = []
        self._bm25_doc_lengths: List[int] = []
        self._bm25_doc_freq: Counter = Counter()
        self._bm25_avgdl = 0.0
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
        - 对单页文本做 PDF 专用清洗:优先保留版面空行,合并硬换行,去掉页眉/页码噪声,
          让后续结构化 chunker 能识别标题、段落和列表。
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
                plain = page.extract_text() or ""
            except Exception:
                plain = ""
            layout = ""
            try:
                layout = page.extract_text(extraction_mode="layout") or ""
            except Exception:
                layout = ""

            raw = LocalKnowledgeBase._choose_pdf_extraction(plain, layout)
            cleaned = LocalKnowledgeBase._normalize_pdf_page_text(raw)
            # 页尾补一个换行,防止后续 chunk 跨页时两页文字被直接拼接成无意义长词
            if cleaned:
                cleaned += "\n"
            pages_text.append(cleaned)

        return PAGE_SEPARATOR.join(pages_text)

    @staticmethod
    def _visible_char_count(text: str) -> int:
        return sum(1 for ch in text if not ch.isspace())

    @staticmethod
    def _choose_pdf_extraction(plain: str, layout: str) -> str:
        """Prefer pypdf layout mode unless it produces pathological whitespace output."""
        if not layout.strip():
            return plain
        if not plain.strip():
            return layout

        plain_visible = LocalKnowledgeBase._visible_char_count(plain)
        layout_visible = LocalKnowledgeBase._visible_char_count(layout)
        if layout_visible < max(20, int(plain_visible * 0.75)):
            return plain

        max_reasonable_len = max(len(plain) * 4, layout_visible * 6, 2000)
        if len(layout) > max_reasonable_len:
            return plain

        return layout

    @staticmethod
    def _repair_spaced_letters(line: str) -> str:
        """Undo common PDF cover/title extraction like 'D i e t a r y'."""
        parts = re.split(r"(\s{2,})", line.strip())
        repaired: List[str] = []
        for part in parts:
            if not part or part.isspace():
                repaired.append(" " if part else part)
                continue
            tokens = part.split()
            if len(tokens) >= 3:
                compactable = sum(1 for token in tokens if re.fullmatch(r"[A-Za-z0-9]", token))
                if compactable / len(tokens) >= 0.8:
                    repaired.append("".join(tokens))
                    continue
            repaired.append(part)
        return "".join(repaired).strip()

    @staticmethod
    def _is_pdf_page_artifact_line(line: str) -> bool:
        return bool(_PDF_PAGE_ARTIFACT_RE.match(line.strip()))

    @staticmethod
    def _line_ends_sentence(line: str) -> bool:
        return bool(re.search(r"[。！？.!?;；:：]$|[”’\")\]]$", line.strip()))

    @staticmethod
    def _join_pdf_lines(left: str, right: str) -> str:
        left = left.rstrip()
        right = right.lstrip()
        if not left:
            return right
        if not right:
            return left
        if _CJK_RE.search(left[-1:]) and _CJK_RE.search(right[:1]):
            return left + right
        if left[-1:] in "-‐‑‒–—":
            return left[:-1] + right
        return left + " " + right

    @staticmethod
    def _looks_like_structural_heading(line: str) -> bool:
        stripped = re.sub(r"\s+", " ", line.strip())
        if not stripped:
            return False
        if _MARKDOWN_LIST_ITEM_RE.match(stripped) or LocalKnowledgeBase._is_table_line(stripped):
            return False
        if _MARKDOWN_HEADING_RE.match(stripped):
            return True
        if _PDF_CHAPTER_HEADING_RE.match(stripped):
            return True
        if _PDF_ENGLISH_NUMBERED_HEADING_RE.match(stripped):
            return True
        if stripped in {"小贴士", "营养健康食谱"}:
            return True
        if len(stripped) > 90:
            return False
        if re.search(r"[。！？.!?;；:：][”’\")\]]?$", stripped):
            return False

        if _CJK_RE.search(stripped):
            cjk_count = len(_CJK_RE.findall(stripped))
            return 2 <= cjk_count <= 34

        words = re.findall(r"[A-Za-z][A-Za-z'’-]*", stripped)
        if not (2 <= len(words) <= 10):
            return False
        titleish = sum(
            1
            for word in words
            if word[:1].isupper() or word.isupper() or len(word) <= 3
        )
        return titleish / len(words) >= 0.7

    @staticmethod
    def _looks_like_explicit_pdf_heading(line: str) -> bool:
        stripped = re.sub(r"\s+", " ", line.strip())
        return bool(
            _PDF_CHAPTER_HEADING_RE.match(stripped)
            or _PDF_ENGLISH_NUMBERED_HEADING_RE.match(stripped)
            or stripped in {"小贴士", "营养健康食谱"}
        )

    @staticmethod
    def _heading_level_for_line(line: str) -> int:
        stripped = line.strip()
        markdown = _MARKDOWN_HEADING_RE.match(stripped)
        if markdown:
            return len(markdown.group(1))
        if _PDF_CHAPTER_HEADING_RE.match(stripped):
            return 1
        if _PDF_ENGLISH_NUMBERED_HEADING_RE.match(stripped):
            return 2
        if stripped in {"小贴士", "营养健康食谱"}:
            return 3
        return 2

    @staticmethod
    def _split_embedded_pdf_bullets(line: str) -> List[str]:
        normalized = line.strip()
        if not normalized:
            return []
        if not _MARKDOWN_LIST_ITEM_RE.match(normalized):
            return [normalized]
        return [
            part.strip()
            for part in re.split(r"\s+(?=[+•‣◦]\s+)", normalized)
            if part.strip()
        ]

    @staticmethod
    def _normalize_pdf_page_text(raw: str) -> str:
        """Normalize one extracted PDF page while preserving structural blank lines."""
        text = raw.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[\u00a0\u2000-\u200a\u202f\u205f\u3000]", " ", text)
        text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

        lines: List[str] = []
        for raw_line in text.splitlines():
            repaired = LocalKnowledgeBase._repair_spaced_letters(raw_line)
            repaired = re.sub(r"[ \t]+", " ", repaired).strip()
            if not repaired:
                lines.append("")
                continue
            for line in LocalKnowledgeBase._split_embedded_pdf_bullets(repaired):
                if not LocalKnowledgeBase._is_pdf_page_artifact_line(line):
                    lines.append(line)

        blocks: List[str] = []
        current = ""
        current_kind = ""

        def flush_current() -> None:
            nonlocal current, current_kind
            if current.strip():
                blocks.append(current.strip())
            current = ""
            current_kind = ""

        for idx, line in enumerate(lines):
            if not line:
                flush_current()
                continue

            prev_blank = idx == 0 or not lines[idx - 1]
            next_blank = idx == len(lines) - 1 or not lines[idx + 1]
            is_pdf_heading = (
                LocalKnowledgeBase._looks_like_explicit_pdf_heading(line)
                or (
                    prev_blank
                    and next_blank
                    and LocalKnowledgeBase._looks_like_structural_heading(line)
                )
            )
            if is_pdf_heading:
                flush_current()
                blocks.append(line)
                continue

            is_list_item = bool(_MARKDOWN_LIST_ITEM_RE.match(line))
            if is_list_item:
                flush_current()
                current = line
                current_kind = "list_item"
                continue

            if not current:
                current = line
                current_kind = "paragraph"
                continue

            if current_kind == "list_item" or not LocalKnowledgeBase._line_ends_sentence(current):
                current = LocalKnowledgeBase._join_pdf_lines(current, line)
            else:
                flush_current()
                current = line
                current_kind = "paragraph"

        flush_current()
        return re.sub(r"\n{3,}", "\n\n", "\n\n".join(blocks)).strip()

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
            "chunker_version": CHUNKER_VERSION,
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

    def _cache_sections_path(self) -> Path:
        return self.cache_dir / "sections.json"

    def _cache_section_embeddings_path(self) -> Path:
        return self.cache_dir / "section_embeddings.npy"

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
        sections_path = self._cache_sections_path()
        section_emb_path = self._cache_section_embeddings_path()
        if not (meta_path.exists() and emb_path.exists() and chunks_path.exists()):
            return False

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("fingerprint") != self._fingerprint:
                return False

            chunk_items = json.loads(chunks_path.read_text(encoding="utf-8"))
            self.chunks = [Chunk(**item) for item in chunk_items]
            self._chunk_embeddings = np.load(str(emb_path))
            if sections_path.exists():
                section_items = json.loads(sections_path.read_text(encoding="utf-8"))
                self.section_parents = [SectionParent(**item) for item in section_items]
            else:
                self.section_parents = []
            if section_emb_path.exists():
                self._section_embeddings = np.load(str(section_emb_path))
            else:
                self._section_embeddings = None
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
        self._cache_sections_path().write_text(
            json.dumps([s.__dict__ for s in self.section_parents], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if self._section_embeddings is not None:
            np.save(str(self._cache_section_embeddings_path()), self._section_embeddings)
        self._save_faiss_index()

    @staticmethod
    def _make_text_block(raw: str, start: int, kind: str, heading_level: int = 0) -> Optional[_TextBlock]:
        stripped = raw.strip()
        if not stripped:
            return None
        leading = len(raw) - len(raw.lstrip())
        trailing_end = len(raw.rstrip())
        return _TextBlock(
            text=stripped,
            start=start + leading,
            end=start + trailing_end,
            kind=kind,
            heading_level=heading_level,
        )

    @staticmethod
    def _is_table_line(line: str) -> bool:
        stripped = line.strip()
        return bool(stripped and (_MARKDOWN_TABLE_RE.match(line) or stripped.count("|") >= 2))

    @staticmethod
    def _lexical_tokens(text: str) -> List[str]:
        """Dependency-free tokens for BM25 over mixed Chinese/English health docs."""
        raw = text or ""
        words = [w.lower() for w in _WORD_TOKEN_RE.findall(raw) if len(w) >= 2]
        cjk_chars = _CJK_RE.findall(raw)
        cjk_bigrams = [
            "".join(cjk_chars[i : i + 2])
            for i in range(max(0, len(cjk_chars) - 1))
        ]
        cjk_trigrams = [
            "".join(cjk_chars[i : i + 3])
            for i in range(max(0, len(cjk_chars) - 2))
        ]
        return words + cjk_bigrams + cjk_trigrams

    def _build_lexical_index(self) -> None:
        self._bm25_doc_counts = []
        self._bm25_doc_lengths = []
        self._bm25_doc_freq = Counter()
        self._bm25_avgdl = 0.0

        for chunk in self.chunks:
            lexical_text = chunk.text
            if chunk.section_path:
                lexical_text = f"{chunk.section_path}\n{lexical_text}"
            counts = Counter(self._lexical_tokens(lexical_text))
            self._bm25_doc_counts.append(counts)
            doc_len = sum(counts.values())
            self._bm25_doc_lengths.append(doc_len)
            if counts:
                self._bm25_doc_freq.update(counts.keys())

        total_len = sum(self._bm25_doc_lengths)
        self._bm25_avgdl = total_len / len(self._bm25_doc_lengths) if self._bm25_doc_lengths else 0.0

    def _bm25_score_doc(self, query_counts: Counter, idx: int) -> float:
        if (
            not query_counts
            or idx < 0
            or idx >= len(self._bm25_doc_counts)
            or not self._bm25_doc_lengths
        ):
            return 0.0
        doc_counts = self._bm25_doc_counts[idx]
        doc_len = self._bm25_doc_lengths[idx]
        if not doc_counts or doc_len <= 0:
            return 0.0

        k1 = 1.5
        b = 0.75
        n_docs = max(1, len(self._bm25_doc_counts))
        score = 0.0
        for token, qf in query_counts.items():
            tf = doc_counts.get(token, 0)
            if tf <= 0:
                continue
            df = self._bm25_doc_freq.get(token, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1 - b + b * doc_len / max(1.0, self._bm25_avgdl))
            score += idf * ((tf * (k1 + 1)) / denom) * min(1.0, 0.5 + 0.5 * qf)
        return float(score)

    def _bm25_topk(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        if (
            not RAG_HYBRID_RETRIEVAL_ENABLED
            or top_k <= 0
            or not self._bm25_doc_counts
        ):
            return []

        query_counts = Counter(self._lexical_tokens(query))
        if not query_counts:
            return []

        scored = [
            (idx, self._bm25_score_doc(query_counts, idx))
            for idx in range(len(self._bm25_doc_counts))
        ]
        scored = [(idx, score) for idx, score in scored if score > 0]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: max(1, min(top_k, len(scored)))]

    @staticmethod
    def _join_blocks(blocks: List[_TextBlock]) -> str:
        parts: List[str] = []
        prev: Optional[_TextBlock] = None
        for block in blocks:
            text = block.text.strip()
            if not text:
                continue
            if not parts:
                parts.append(text)
            elif prev and prev.kind == "list_item" and block.kind == "list_item":
                parts.append("\n" + text)
            else:
                parts.append("\n\n" + text)
            prev = block
        return "".join(parts).strip()

    @staticmethod
    def _parse_text_blocks(cleaned: str) -> List[_TextBlock]:
        """Parse Markdown-ish documents into stable structural blocks."""
        blocks: List[_TextBlock] = []
        pending_parts: List[str] = []
        pending_start = 0
        pending_kind = "paragraph"

        def flush_pending() -> None:
            nonlocal pending_parts, pending_start, pending_kind
            block = LocalKnowledgeBase._make_text_block(
                "".join(pending_parts),
                pending_start,
                pending_kind,
            )
            if block is not None:
                blocks.append(block)
            pending_parts = []

        def start_pending(raw_line: str, line_start: int, kind: str) -> None:
            nonlocal pending_parts, pending_start, pending_kind
            pending_parts = [raw_line]
            pending_start = line_start
            pending_kind = kind

        offset = 0
        for raw_line in cleaned.splitlines(keepends=True):
            line_start = offset
            offset += len(raw_line)
            stripped = raw_line.strip()

            if not stripped:
                flush_pending()
                continue

            heading_match = _MARKDOWN_HEADING_RE.match(raw_line)
            is_structural_heading = bool(
                heading_match or LocalKnowledgeBase._looks_like_structural_heading(stripped)
            )
            is_rule = bool(_MARKDOWN_RULE_RE.match(raw_line))
            is_table = LocalKnowledgeBase._is_table_line(raw_line)
            is_list_item = bool(_MARKDOWN_LIST_ITEM_RE.match(raw_line))

            if is_structural_heading:
                flush_pending()
                block = LocalKnowledgeBase._make_text_block(
                    raw_line,
                    line_start,
                    "heading",
                    heading_level=LocalKnowledgeBase._heading_level_for_line(stripped),
                )
                if block is not None:
                    blocks.append(block)
                continue

            if is_rule:
                flush_pending()
                continue

            if is_table:
                if pending_parts and pending_kind == "table":
                    pending_parts.append(raw_line)
                else:
                    flush_pending()
                    start_pending(raw_line, line_start, "table")
                continue

            if is_list_item:
                flush_pending()
                start_pending(raw_line, line_start, "list_item")
                continue

            if pending_parts and pending_kind == "list_item" and raw_line[:1].isspace():
                pending_parts.append(raw_line)
                continue

            if pending_parts and pending_kind == "paragraph":
                pending_parts.append(raw_line)
            else:
                flush_pending()
                start_pending(raw_line, line_start, "paragraph")

        flush_pending()
        return blocks

    @staticmethod
    def _sections_from_blocks(blocks: List[_TextBlock]) -> List[Tuple[List[_TextBlock], List[_TextBlock]]]:
        """Return (heading_path, content_blocks) sections without emitting heading-only noise."""
        sections: List[Tuple[List[_TextBlock], List[_TextBlock]]] = []
        heading_stack: List[_TextBlock] = []
        content: List[_TextBlock] = []

        def flush_content() -> None:
            nonlocal content
            if content:
                sections.append((list(heading_stack), content))
                content = []

        for block in blocks:
            if block.kind == "heading":
                flush_content()
                heading_stack = [
                    h for h in heading_stack if h.heading_level < block.heading_level
                ]
                heading_stack.append(block)
            else:
                content.append(block)

        flush_content()
        if not sections and heading_stack:
            sections.append(([], heading_stack))
        return sections

    def _snap_fallback_end(self, text: str, start: int, end: int) -> int:
        if end >= len(text):
            return end

        win_start = max(start, end - self.boundary_look_back)
        p = text.rfind("\n\n", win_start, end)
        if p > start:
            return p + 2

        best = -1
        for m in _SENT_END_RE.finditer(text, win_start, end):
            if m.end() > start:
                best = m.end()
        if best > start:
            return best

        p = text.rfind("\n", win_start, end)
        if p > start:
            return p + 1
        return end

    @staticmethod
    def _skip_inline_space(text: str, start: int) -> int:
        while start < len(text) and text[start] in " \t\n\r":
            start += 1
        return start

    def _snap_fallback_start(self, text: str, start: int, previous_end: int) -> int:
        if start <= 0:
            return 0

        boundary_limit = min(previous_end, start + self.boundary_look_back)
        p = text.find("\n\n", start, boundary_limit)
        if p != -1:
            return self._skip_inline_space(text, p + 2)

        for m in _SENT_END_RE.finditer(text, start, boundary_limit):
            return self._skip_inline_space(text, m.end())

        whitespace_limit = min(len(text), start + 40)
        for idx in range(start, whitespace_limit):
            if text[idx].isspace():
                return self._skip_inline_space(text, idx)

        if text[start - 1:start].isalnum() and text[start:start + 1].isalnum():
            word_end = start
            while word_end < whitespace_limit and text[word_end:word_end + 1].isalnum():
                word_end += 1
            if word_end < whitespace_limit:
                return word_end

        return start

    def _split_oversize_block(self, block: _TextBlock, max_chars: int) -> List[_TextBlock]:
        """Fallback splitter for a single structural block that is longer than max_chars."""
        text = block.text
        n = len(text)
        if n <= max_chars:
            return [block]

        max_chars = max(1, max_chars)
        pieces: List[_TextBlock] = []
        start = 0
        while start < n:
            end = min(start + max_chars, n)
            end = self._snap_fallback_end(text, start, end)
            if end <= start:
                end = min(start + max_chars, n)

            raw_piece = text[start:end]
            piece = raw_piece.strip()
            if piece:
                leading = len(raw_piece) - len(raw_piece.lstrip())
                trailing_end = len(raw_piece.rstrip())
                pieces.append(
                    _TextBlock(
                        text=piece,
                        start=block.start + start + leading,
                        end=block.start + start + trailing_end,
                        kind=block.kind,
                        heading_level=block.heading_level,
                    )
                )

            if end >= n:
                break
            next_start = max(start + 1, end - self.overlap)
            start = self._snap_fallback_start(text, next_start, end)

        return pieces

    def _page_range_for_blocks(
        self,
        blocks: List[_TextBlock],
        page_numbers: Optional[List[int]],
    ) -> Optional[str]:
        if page_numbers is None or not blocks:
            return None
        last_idx = len(page_numbers) - 1
        start = max(0, min(block.start for block in blocks))
        end = max(block.end for block in blocks) - 1
        ps = page_numbers[min(start, last_idx)]
        pe = page_numbers[min(max(end, start), last_idx)]
        return str(ps) if ps == pe else f"{ps}-{pe}"

    def _pack_section_blocks(
        self,
        prefix_blocks: List[_TextBlock],
        content_blocks: List[_TextBlock],
        page_numbers: Optional[List[int]],
        is_pdf: bool = False,
    ) -> List[Tuple[str, Optional[str]]]:
        max_chars = self.chunk_size
        if is_pdf and RAG_PDF_FINE_CHUNKING_ENABLED:
            max_chars = max(self.min_chunk_chars, min(self.chunk_size, RAG_PDF_FINE_CHUNK_MAX_CHARS))
        prefix_text = self._join_blocks(prefix_blocks)
        prefix_len = len(prefix_text)
        prefix_separator = 2 if prefix_text else 0
        fallback_limit = max(1, max_chars - prefix_len - prefix_separator)

        expanded_blocks: List[_TextBlock] = []
        for block in content_blocks:
            candidate_text = self._join_blocks(prefix_blocks + [block])
            if len(candidate_text) > max_chars and len(block.text) > fallback_limit:
                expanded_blocks.extend(self._split_oversize_block(block, fallback_limit))
            else:
                expanded_blocks.append(block)

        if not expanded_blocks:
            text = self._join_blocks(prefix_blocks)
            if len(text) >= self.min_chunk_chars:
                return [(text, self._page_range_for_blocks(prefix_blocks, page_numbers))]
            return []

        results: List[Tuple[str, Optional[str]]] = []
        start = 0
        n = len(expanded_blocks)
        while start < n:
            end = start
            current: List[_TextBlock] = []

            while end < n:
                candidate = current + [expanded_blocks[end]]
                candidate_text = self._join_blocks(prefix_blocks + candidate)
                next_block = expanded_blocks[end]
                if (
                    is_pdf
                    and RAG_PDF_FINE_CHUNKING_ENABLED
                    and current
                    and (
                        next_block.kind in {"list_item", "table"}
                        or current[-1].kind in {"list_item", "table"}
                    )
                ):
                    break
                if candidate and len(candidate_text) <= max_chars:
                    current = candidate
                    end += 1
                    continue
                if not current:
                    current = candidate
                    end += 1
                break

            chunk_blocks = prefix_blocks + current
            piece = self._join_blocks(chunk_blocks)
            if len(piece) >= self.min_chunk_chars:
                page_blocks = current or prefix_blocks
                results.append((piece, self._page_range_for_blocks(page_blocks, page_numbers)))

            if end >= n:
                break

            overlap_start = len(current)
            if self.overlap > 0:
                for idx in range(len(current) - 1, -1, -1):
                    if len(self._join_blocks(current[idx:])) >= self.overlap:
                        overlap_start = idx
                        break

            next_start = start + overlap_start
            if next_start <= start:
                next_start = start + 1
            start = next_start

        return results

    @staticmethod
    def _section_path_from_prefix(prefix_blocks: List[_TextBlock], source: str = "") -> str:
        headings = [
            re.sub(r"\s+", " ", block.text).strip("# ").strip()
            for block in prefix_blocks
            if block.text.strip()
        ]
        headings = [heading for heading in headings if heading]
        if headings:
            return " > ".join(headings)
        if source:
            return Path(source).stem
        return ""

    def _split_text_with_metadata(
        self,
        text: str,
        source: str = "",
        file_type: str = "",
    ) -> List[Dict[str, object]]:
        """切分文本，并为 parent-child 检索保留 section metadata。返回 dict 列表。

        切分策略（优先级从高到低）：
        1. 先解析 Markdown-ish 文档结构：标题层级、段落、列表项和表格。
           chunk 优先在结构块边界切分，并把标题路径带入所在 section 的 chunk。
        2. 只有单个结构块本身超过 chunk_size 时，才 fallback 到 max-length
           切分；fallback 仍会优先对齐段落、句末标点和单换行。
        3. section 内相邻 chunk 按 overlap 回退到前一个或多个结构块；
           oversize fallback 块使用字符级 overlap。
        4. 长度 < min_chunk_chars 的碎片直接丢弃，避免噪声 chunk 污染索引。

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

        blocks = self._parse_text_blocks(cleaned)
        sections = self._sections_from_blocks(blocks)

        results: List[Dict[str, object]] = []
        is_pdf = bool(has_pages or file_type == "pdf" or source.lower().endswith(".pdf"))
        for section_index, (prefix_blocks, content_blocks) in enumerate(sections, start=1):
            section_path = self._section_path_from_prefix(prefix_blocks, source)
            section_id = f"{source}#section-{section_index}" if source else f"section-{section_index}"
            for piece, page_range in self._pack_section_blocks(
                prefix_blocks,
                content_blocks,
                page_numbers,
                is_pdf=is_pdf,
            ):
                results.append(
                    {
                        "text": piece,
                        "page_range": page_range,
                        "section_id": section_id,
                        "section_path": section_path,
                        "section_index": section_index,
                        "is_pdf": is_pdf,
                    }
                )
        return results

    def _split_text(self, text: str):
        """切分文本为 chunks。返回 [(piece, page_range_or_None), ...]。"""
        return [
            (str(piece["text"]), piece.get("page_range"))
            for piece in self._split_text_with_metadata(text)
        ]

    def clear_cache(self):
        if self.cache_dir.exists():
            for p in self.cache_dir.rglob("*"):
                if p.is_file():
                    p.unlink()
            for p in sorted(self.cache_dir.rglob("*"), reverse=True):
                if p.is_dir():
                    p.rmdir()

        self._chunk_embeddings = None
        self._section_embeddings = None
        self._faiss_index = None
        self.chunks = []
        self.section_parents = []
        self._ready = False

    @staticmethod
    def _parse_page_range(page_range: Optional[str]) -> Optional[Tuple[int, int]]:
        if not page_range:
            return None
        text = str(page_range).strip()
        match = re.match(r"^(\d+)(?:-(\d+))?$", text)
        if not match:
            return None
        start = int(match.group(1))
        end = int(match.group(2) or start)
        return (min(start, end), max(start, end))

    @classmethod
    def _merge_page_ranges(cls, page_ranges: List[Optional[str]]) -> Optional[str]:
        parsed = [cls._parse_page_range(value) for value in page_ranges]
        parsed = [value for value in parsed if value is not None]
        if not parsed:
            return None
        start = min(value[0] for value in parsed)
        end = max(value[1] for value in parsed)
        return str(start) if start == end else f"{start}-{end}"

    def _build_section_parents(self) -> List[SectionParent]:
        grouped: Dict[Tuple[str, str], List[Chunk]] = {}
        for chunk in self.chunks:
            if not chunk.is_pdf or not chunk.section_id:
                continue
            grouped.setdefault((chunk.source, chunk.section_id), []).append(chunk)

        parents: List[SectionParent] = []
        for (_source, _section_id), children in grouped.items():
            children = sorted(children, key=lambda chunk: chunk.section_index)
            first = children[0]
            body_parts: List[str] = []
            for child in children:
                text = child.text.strip()
                if text:
                    body_parts.append(text)
            parent_text = "\n\n".join(body_parts)
            if first.section_path and first.section_path not in parent_text[:300]:
                parent_text = f"{first.section_path}\n\n{parent_text}"
            parent_text = parent_text[: max(200, RAG_PDF_SECTION_PARENT_MAX_CHARS)].strip()
            if not parent_text:
                continue
            parents.append(
                SectionParent(
                    section_id=first.section_id,
                    source=first.source,
                    section_path=first.section_path,
                    text=parent_text,
                    page_range=self._merge_page_ranges([child.page_range for child in children]),
                    child_chunk_ids=[child.chunk_id for child in children],
                )
            )
        return parents

    def _chunk_index_by_id(self) -> Dict[str, int]:
        return {chunk.chunk_id: idx for idx, chunk in enumerate(self.chunks)}

    def _section_parent_by_id(self) -> Dict[str, SectionParent]:
        return {parent.section_id: parent for parent in self.section_parents}

    @staticmethod
    def _bounded_rerank_pool_size(base_k: int, n_chunks: int) -> int:
        multiplier = max(1, RAG_RERANK_POOL_MULTIPLIER)
        pool_max = max(base_k, RAG_RERANK_POOL_MAX)
        return max(1, min(n_chunks, max(base_k, min(base_k * multiplier, pool_max))))

    @staticmethod
    def _parent_score_weight() -> float:
        if not RAG_PDF_PARENT_SCORE_FUSION_ENABLED:
            return 0.0
        return max(0.0, RAG_PDF_SECTION_SCORE_WEIGHT)

    @staticmethod
    def _bm25_score_weight() -> float:
        if not RAG_HYBRID_RETRIEVAL_ENABLED:
            return 0.0
        return max(0.0, RAG_BM25_SCORE_WEIGHT)

    def _candidate_retrieval_score(self, data: Dict[str, float]) -> float:
        return (
            float(data.get("dense_score", 0.0))
            + self._bm25_score_weight() * float(data.get("bm25_score", 0.0))
            + self._parent_score_weight() * float(data.get("parent_section_score", 0.0))
        )

    def _is_pdf_chunk_idx(self, idx: int) -> bool:
        if idx < 0 or idx >= len(self.chunks):
            return False
        chunk = self.chunks[idx]
        return bool(chunk.is_pdf or chunk.page_range or chunk.source.lower().endswith(".pdf"))

    def _should_use_pdf_parent_rescue(
        self,
        section_scores,
        current_ranked: List[Tuple[int, Dict[str, float]]],
    ) -> bool:
        if not RAG_PDF_PARENT_RESCUE_ENABLED:
            return True
        if section_scores is None or len(section_scores) == 0:
            return False
        best_parent_score = float(section_scores.max())
        if best_parent_score < RAG_PDF_PARENT_RESCUE_MIN_PARENT_SCORE:
            return False

        lookahead = max(1, RAG_PDF_PARENT_RESCUE_LOOKAHEAD)
        min_pdf = max(0, RAG_PDF_PARENT_RESCUE_MIN_PDF_CANDIDATES)
        pdf_count = sum(
            1
            for idx, _data in current_ranked[:lookahead]
            if self._is_pdf_chunk_idx(int(idx))
        )
        return pdf_count < min_pdf

    def _pdf_section_candidate_scores(
        self,
        query_emb,
        base_scores: Dict[int, float],
        current_ranked: Optional[List[Tuple[int, Dict[str, float]]]] = None,
    ) -> Dict[int, Dict[str, float]]:
        if (
            not RAG_PDF_PARENT_EXPANSION_ENABLED
            or not self.section_parents
            or self._section_embeddings is None
            or RAG_PDF_SECTION_PARENT_TOP_K <= 0
            or RAG_PDF_SECTION_CHILD_TOP_K <= 0
        ):
            return {}

        np = self._lazy_import_numpy()
        section_scores = self._section_embeddings @ query_emb
        if not self._should_use_pdf_parent_rescue(section_scores, current_ranked or []):
            return {}

        top_k = min(RAG_PDF_SECTION_PARENT_TOP_K, len(self.section_parents))
        section_idx = np.argpartition(-section_scores, top_k - 1)[:top_k]
        section_idx = sorted(
            section_idx.tolist(),
            key=lambda idx: float(section_scores[idx]),
            reverse=True,
        )
        chunk_lookup = self._chunk_index_by_id()
        expanded: Dict[int, Dict[str, float]] = {}
        for parent_idx in section_idx:
            parent = self.section_parents[parent_idx]
            parent_score = float(section_scores[parent_idx])
            child_indices = [
                chunk_lookup[chunk_id]
                for chunk_id in (parent.child_chunk_ids or [])
                if chunk_id in chunk_lookup
            ]
            if not child_indices:
                continue
            child_indices = sorted(
                child_indices,
                key=lambda idx: float(base_scores.get(idx, self._chunk_embeddings[idx] @ query_emb)),
                reverse=True,
            )[:RAG_PDF_SECTION_CHILD_TOP_K]
            for idx in child_indices:
                dense = float(base_scores.get(idx, self._chunk_embeddings[idx] @ query_emb))
                expanded[idx] = {
                    "dense_score": dense,
                    "bm25_score": 0.0,
                    "parent_section_score": parent_score,
                    "retrieval_score": dense + self._parent_score_weight() * parent_score,
                }
        return expanded

    def _dense_candidates_with_pdf_sections(
        self,
        query_emb,
        dense_k: int,
        pool_k: Optional[int] = None,
        query: str = "",
    ):
        base_k = max(1, min(dense_k, len(self.chunks)))
        pool_k = max(base_k, min(pool_k or base_k, len(self.chunks)))
        dense_candidate_k = pool_k if RAG_HYBRID_RETRIEVAL_ENABLED else base_k
        base_candidates = self._dense_topk(query_emb, max(base_k, dense_candidate_k))
        combined: Dict[int, Dict[str, float]] = {
            idx: {
                "dense_score": dense,
                "bm25_score": 0.0,
                "parent_section_score": 0.0,
                "retrieval_score": dense,
            }
            for idx, dense in base_candidates
        }
        base_scores = {idx: dense for idx, dense in base_candidates}

        bm25_candidates = self._bm25_topk(query, RAG_BM25_TOP_K)
        top_bm25 = bm25_candidates[0][1] if bm25_candidates else 0.0
        for idx, score in bm25_candidates:
            dense = float(base_scores.get(idx, self._chunk_embeddings[idx] @ query_emb))
            data = combined.setdefault(
                idx,
                {
                    "dense_score": dense,
                    "bm25_score": 0.0,
                    "parent_section_score": 0.0,
                    "retrieval_score": dense,
                },
            )
            data["dense_score"] = max(float(data.get("dense_score", dense)), dense)
            if top_bm25 > 0:
                data["bm25_score"] = max(float(data.get("bm25_score", 0.0)), score / top_bm25)

        for data in combined.values():
            data["retrieval_score"] = self._candidate_retrieval_score(data)

        current_ranked = sorted(
            combined.items(),
            key=lambda item: item[1]["retrieval_score"],
            reverse=True,
        )

        for idx, data in self._pdf_section_candidate_scores(
            query_emb,
            base_scores,
            current_ranked=current_ranked,
        ).items():
            existing = combined.get(idx)
            if existing is None:
                combined[idx] = data
            else:
                existing["dense_score"] = max(existing["dense_score"], data["dense_score"])
                existing["parent_section_score"] = max(
                    existing.get("parent_section_score", 0.0),
                    data.get("parent_section_score", 0.0),
                )

        for data in combined.values():
            data["retrieval_score"] = self._candidate_retrieval_score(data)

        ranked = sorted(
            combined.items(),
            key=lambda item: item[1]["retrieval_score"],
            reverse=True,
        )
        return [
            (
                int(idx),
                float(data["dense_score"]),
                float(data.get("parent_section_score", 0.0)),
                float(data["retrieval_score"]),
            )
            for idx, data in ranked[:pool_k]
        ]

    def _rerank_text_for_chunk(self, idx: int) -> str:
        chunk = self.chunks[idx]
        if not (chunk.is_pdf or chunk.section_id or chunk.section_path):
            return chunk.text

        parts: List[str] = []
        if chunk.section_path:
            parts.append(f"Section: {chunk.section_path}")

        parent = self._section_parent_by_id().get(chunk.section_id)
        if RAG_PDF_PARENT_RERANK_CONTEXT_ENABLED and parent and parent.text:
            parent_excerpt = parent.text[: max(0, RAG_PDF_RERANK_PARENT_CONTEXT_CHARS)].strip()
            if parent_excerpt:
                parts.append(f"Parent context:\n{parent_excerpt}")

        parts.append(f"Child chunk:\n{chunk.text}")
        return "\n\n".join(part for part in parts if part.strip())

    def _result_for_chunk(
        self,
        idx: int,
        rank: int,
        dense_score: float,
        rerank_score: Optional[float] = None,
        score: Optional[float] = None,
        parent_section_score: float = 0.0,
        include_content: bool = False,
    ) -> Dict[str, object]:
        c = self.chunks[idx]
        result: Dict[str, object] = {
            "rank": rank,
            "chunk_id": c.chunk_id,
            "source": c.source,
            "page_range": c.page_range,
            "section_id": c.section_id,
            "section_path": c.section_path,
            "dense_score": round(dense_score, 4),
        }
        if parent_section_score:
            result["parent_section_score"] = round(parent_section_score, 4)
        if rerank_score is not None:
            result["rerank_score"] = round(rerank_score, 4)
        elif rerank_score is None and score is not None:
            result["rerank_score"] = None
        if score is not None:
            result["score"] = round(score, 4)
        if include_content:
            result["content"] = c.text
        return result

    def _pdf_neighbor_indices(self, idx: int, radius: int) -> List[int]:
        if radius <= 0 or idx < 0 or idx >= len(self.chunks):
            return [idx]
        current = self.chunks[idx]
        if not (current.is_pdf or current.page_range or current.source.lower().endswith(".pdf")):
            return [idx]
        indices: List[int] = []
        for neighbor_idx in range(max(0, idx - radius), min(len(self.chunks), idx + radius + 1)):
            if self.chunks[neighbor_idx].source == current.source:
                indices.append(neighbor_idx)
        return indices or [idx]

    def _attach_pdf_context(self, result: Dict[str, object], include_content: bool = False) -> Dict[str, object]:
        chunk_id = str(result.get("chunk_id") or "")
        lookup = self._chunk_index_by_id()
        if chunk_id not in lookup:
            return result
        idx = lookup[chunk_id]
        context_indices = self._pdf_neighbor_indices(idx, RAG_PDF_NEIGHBOR_CHUNKS)
        if context_indices == [idx]:
            result["context_chunk_ids"] = [chunk_id]
            result["context_page_range"] = result.get("page_range")
            return result

        context_chunks = [self.chunks[i] for i in context_indices]
        result["context_chunk_ids"] = [chunk.chunk_id for chunk in context_chunks]
        result["context_page_range"] = self._merge_page_ranges(
            [chunk.page_range for chunk in context_chunks]
        )
        if include_content:
            pieces = []
            for chunk in context_chunks:
                label = f"[chunk: {chunk.chunk_id}"
                if chunk.page_range:
                    label += f" | page: {chunk.page_range}"
                label += "]"
                pieces.append(f"{label}\n{chunk.text.strip()}")
            result["content"] = "\n\n".join(pieces).strip()
        return result

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
            pieces = self._split_text_with_metadata(
                d["text"],
                source=d["source"],
                file_type=d.get("file_type", ""),
            )
            for i, piece in enumerate(pieces):
                piece_text = str(piece["text"])
                page_range = piece.get("page_range")
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
                        section_id=str(piece.get("section_id") or ""),
                        section_path=str(piece.get("section_path") or ""),
                        section_index=int(piece.get("section_index") or 0),
                        is_pdf=bool(piece.get("is_pdf")),
                    )
                )

        self.chunks = chunks
        self.section_parents = self._build_section_parents()
        self._build_lexical_index()

        if not self.chunks:
            self._chunk_embeddings = None
            self._section_embeddings = None
            self._faiss_index = None
            self._ready = True
            return

        # 优先加载缓存索引，加速冷启动
        if self._try_load_cache():
            self._build_lexical_index()
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
        if self.section_parents:
            section_texts = [s.text for s in self.section_parents]
            section_embeddings = self._embed_model.encode(
                section_texts,
                batch_size=RAG_EMBED_BATCH_SIZE,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self._section_embeddings = np.asarray(section_embeddings, dtype=np.float32)
        else:
            self._section_embeddings = None
        self._build_faiss_index()
        self._save_cache()
        self._ready = True

    def get_index_stats(self) -> Dict[str, int]:
        return {
            "doc_count": len(self._read_documents()),
            "chunk_count": len(self.chunks),
            "section_parent_count": len(self.section_parents),
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

        - stage1: embedding 稠密检索 Top-K；PDF 会额外用 section parent 召回扩展候选池。
        - stage2: 对 stage1 的候选做 cross-encoder 重排后的 Top-K（按 rerank 分排序）。

        与 `retrieve()` 不同，这里不做最终 rerank+dense 分数融合，便于分别观察候选池和重排器。
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

        rerank_pool_k = self._bounded_rerank_pool_size(stage1_k, len(self.chunks))
        candidates = self._dense_candidates_with_pdf_sections(
            query_emb,
            dense_k=stage1_k,
            pool_k=rerank_pool_k,
            query=query,
        )
        candidate_idx = [i for i, _dense, _parent, _retrieval in candidates]
        dense_scores = {i: dense for i, dense, _parent, _retrieval in candidates}
        parent_scores = {i: parent for i, _dense, parent, _retrieval in candidates}
        retrieval_scores = {i: retrieval for i, _dense, _parent, retrieval in candidates}

        stage1_results: List[Dict[str, object]] = []
        for rank, i in enumerate(candidate_idx[:stage1_k], start=1):
            stage1_results.append(
                self._attach_pdf_context(
                    self._result_for_chunk(
                        idx=i,
                        rank=rank,
                        dense_score=dense_scores[i],
                        score=retrieval_scores[i],
                        parent_section_score=parent_scores.get(i, 0.0),
                    )
                )
            )

        stage2_results: List[Dict[str, object]] = []
        self._lazy_load_models(require_reranker=True)
        if self._rerank_model is None:
            for rank, i in enumerate(candidate_idx[:stage2_k], start=1):
                stage2_results.append(
                    self._attach_pdf_context(
                        self._result_for_chunk(
                            idx=i,
                            rank=rank,
                            dense_score=dense_scores[i],
                            rerank_score=None,
                            score=retrieval_scores.get(i, dense_scores[i]),
                            parent_section_score=parent_scores.get(i, 0.0),
                        )
                    )
                )
            return {"stage1": stage1_results, "stage2": stage2_results}

        # Stage-2: Re-rank
        pairs = [[query, self._rerank_text_for_chunk(i)] for i in candidate_idx]
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
            stage2_results.append(
                self._attach_pdf_context(
                    self._result_for_chunk(
                        idx=i,
                        rank=rank,
                        dense_score=dense,
                        rerank_score=rr,
                        parent_section_score=parent_scores.get(i, 0.0),
                    )
                )
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
        rerank_pool_k = self._bounded_rerank_pool_size(retrieve_k, len(self.chunks))
        candidates = self._dense_candidates_with_pdf_sections(
            query_emb,
            dense_k=retrieve_k,
            pool_k=rerank_pool_k,
            query=query,
        )
        candidate_idx = [i for i, _dense, _parent, _retrieval in candidates]
        dense_scores = {i: dense for i, dense, _parent, _retrieval in candidates}
        parent_scores = {i: parent for i, _dense, parent, _retrieval in candidates}
        retrieval_scores = {i: retrieval for i, _dense, _parent, retrieval in candidates}

        combined = []
        self._lazy_load_models(require_reranker=True)
        if self._rerank_model is None:
            for i in candidate_idx:
                dense = dense_scores[i]
                combined.append((i, dense, None, retrieval_scores.get(i, dense)))
        else:
            # Stage-2: Re-rank
            pairs = [[query, self._rerank_text_for_chunk(i)] for i in candidate_idx]
            rerank_scores = self._rerank_model.predict(
                pairs,
                batch_size=RAG_RERANK_BATCH_SIZE,
                show_progress_bar=False,
            )

            for i, rr in zip(candidate_idx, rerank_scores):
                dense = dense_scores[i]
                rerank = float(rr)
                # 组合分数（以重排分为主，保留召回分作微调）
                final_score = (
                    rerank
                    + 0.15 * dense
                    + self._parent_score_weight() * parent_scores.get(i, 0.0)
                )
                combined.append((i, dense, rerank, final_score))

        combined.sort(key=lambda x: x[3], reverse=True)
        final_k = min(max(1, top_k), len(combined))
        selected = combined[:final_k]

        results = []
        for idx, dense_score, rerank_score, final_score in selected:
            result = self._result_for_chunk(
                idx=idx,
                rank=len(results) + 1,
                dense_score=dense_score,
                rerank_score=rerank_score,
                score=final_score,
                parent_section_score=parent_scores.get(idx, 0.0),
                include_content=True,
            )
            results.append(self._attach_pdf_context(result, include_content=True))
        return results
