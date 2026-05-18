"""Semantic episodic memory index.

The write path is best-effort from ``episode_store.append_episode``. The read
path in TurnStart combines recent episodes with semantically similar older
episodes so follow-up threads can recall details that are no longer recent.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .config import EPISODE_INDEX_DIR
from .episode_store import episode_id, episode_index_text, get_all_episodes
from .rag import Chunk, LocalKnowledgeBase


def _safe_user_dir(user_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", user_id or "default_user").strip("_")
    return cleaned or "default_user"


class EpisodeMemory:
    def __init__(self, user_id: str):
        self.user_id = user_id or "default_user"
        self.base_dir = Path(EPISODE_INDEX_DIR).expanduser() / _safe_user_dir(self.user_id)
        self.ids_path = self.base_dir / "ids.json"
        self.vecs_path = self.base_dir / "vecs.npy"
        self.index_path = self.base_dir / "index.faiss"
        self._kb = LocalKnowledgeBase()
        self._items: List[Dict[str, Any]] = []
        self._loaded = False

    def _load_items(self) -> List[Dict[str, Any]]:
        if not self.ids_path.exists():
            return []
        try:
            data = json.loads(self.ids_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _load(self) -> None:
        if self._loaded:
            return
        self._items = self._load_items()
        np = self._kb._lazy_import_numpy()
        if self.vecs_path.exists() and self._items:
            try:
                self._kb._chunk_embeddings = np.load(str(self.vecs_path))
            except Exception:
                self._kb._chunk_embeddings = None
        self._kb.chunks = [
            Chunk(
                chunk_id=item.get("episode_id") or item.get("id") or f"episode-{idx}",
                source="episode",
                text=item.get("text") or "",
            )
            for idx, item in enumerate(self._items)
        ]
        if self._kb._chunk_embeddings is not None and len(self._items):
            faiss = self._kb._lazy_import_faiss()
            if faiss is not None and self.index_path.exists():
                try:
                    self._kb._faiss_index = faiss.read_index(str(self.index_path))
                except Exception:
                    self._kb._faiss_index = None
            if self._kb._faiss_index is None:
                self._kb._build_faiss_index()
        self._loaded = True

    def _save(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.ids_path.write_text(
            json.dumps(self._items, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        np = self._kb._lazy_import_numpy()
        if self._kb._chunk_embeddings is not None:
            np.save(str(self.vecs_path), self._kb._chunk_embeddings)
        faiss = self._kb._lazy_import_faiss()
        if faiss is not None and self._kb._faiss_index is not None:
            try:
                faiss.write_index(self._kb._faiss_index, str(self.index_path))
            except Exception:
                pass

    def _rebuild_runtime_index(self) -> None:
        self._kb.chunks = [
            Chunk(
                chunk_id=item.get("episode_id") or item.get("id") or f"episode-{idx}",
                source="episode",
                text=item.get("text") or "",
            )
            for idx, item in enumerate(self._items)
        ]
        self._kb._faiss_index = None
        if self._kb._chunk_embeddings is not None and len(self._items):
            self._kb._build_faiss_index()

    def index_episode(
        self,
        episode_id_value: str,
        text: str,
        episode: Optional[Dict[str, Any]] = None,
    ) -> None:
        text = (text or "").strip()
        if not episode_id_value or not text:
            return
        self._load()
        self._kb._lazy_load_models(require_reranker=False)
        np = self._kb._lazy_import_numpy()
        vector = self._kb._embed_model.encode(
            [text],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        existing_vecs = []
        if self._kb._chunk_embeddings is not None:
            existing_vecs = list(self._kb._chunk_embeddings)
        keep = [
            (item, vec)
            for item, vec in zip(self._items, existing_vecs)
            if (item.get("episode_id") or item.get("id")) != episode_id_value
        ]
        self._items = [item for item, _ in keep]
        vectors = [vec for _, vec in keep]

        self._items.append(
            {
                "episode_id": episode_id_value,
                "text": text,
                "episode": episode or {},
            }
        )
        vectors.append(vector)
        self._kb._chunk_embeddings = np.asarray(vectors, dtype=np.float32)
        self._rebuild_runtime_index()
        self._save()

    def retrieve_similar(
        self,
        query: str,
        top_k: int = 3,
        exclude_ids: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []
        exclude_ids = exclude_ids or set()
        self._load()
        if not self._items or self._kb._chunk_embeddings is None:
            return []
        self._kb._lazy_load_models(require_reranker=False)
        query_emb = self._kb._embed_model.encode(
            [query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        candidates = self._kb._dense_topk(query_emb, min(len(self._items), top_k + len(exclude_ids) + 3))
        results: List[Dict[str, Any]] = []
        for idx, score in candidates:
            item = self._items[idx]
            eid = item.get("episode_id") or item.get("id") or ""
            if eid in exclude_ids:
                continue
            ep = dict(item.get("episode") or {})
            if not ep:
                ep = {
                    "id": eid,
                    "ts": "",
                    "query": item.get("text", "")[:120],
                    "experts": [],
                    "gist": item.get("text", "")[:400],
                }
            ep.setdefault("id", eid)
            ep["_memory_source"] = "相关"
            ep["_memory_score"] = round(float(score), 4)
            results.append(ep)
            if len(results) >= top_k:
                break
        return results

    def rebuild_from_store(self) -> int:
        episodes = get_all_episodes(self.user_id)
        self._items = []
        self._loaded = True
        self._kb._chunk_embeddings = None
        self._kb._faiss_index = None
        count = 0
        for ep in episodes:
            ep = dict(ep)
            eid = ep.get("id") or episode_id(ep)
            ep["id"] = eid
            text = episode_index_text(ep)
            if not text:
                continue
            self.index_episode(eid, text, episode=ep)
            count += 1
        return count
