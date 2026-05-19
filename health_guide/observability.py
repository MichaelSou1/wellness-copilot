import json
import sqlite3
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from .config import OBSERVABILITY_DB_PATH


@dataclass
class TurnRecord:
    thread_id: str
    turn_index: int
    route: str
    user_query: str
    final_answer: str
    tools_used: List[str]
    retrieval_hits: int
    citations_count: int
    latency_ms: float
    actuation_count: int = 0
    vision_calls: int = 0
    wechat_msgs_in: int = 0
    wechat_msgs_out: int = 0


class ObservabilityTracker:
    def __init__(self, db_path: str = OBSERVABILITY_DB_PATH):
        self.db_path = db_path
        path = Path(db_path)
        if path.parent and str(path.parent) not in {"", "."}:
            path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._ensure_table()

    def _ensure_table(self):
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS turn_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                route TEXT,
                user_query TEXT,
                final_answer TEXT,
                tools_used TEXT,
                retrieval_hits INTEGER,
                citations_count INTEGER,
                latency_ms REAL,
                actuation_count INTEGER DEFAULT 0,
                vision_calls INTEGER DEFAULT 0,
                wechat_msgs_in INTEGER DEFAULT 0,
                wechat_msgs_out INTEGER DEFAULT 0
            )
            """
        )
        existing = {
            row[1]
            for row in cur.execute("PRAGMA table_info(turn_metrics)").fetchall()
        }
        for name, ddl in {
            "actuation_count": "ALTER TABLE turn_metrics ADD COLUMN actuation_count INTEGER DEFAULT 0",
            "vision_calls": "ALTER TABLE turn_metrics ADD COLUMN vision_calls INTEGER DEFAULT 0",
            "wechat_msgs_in": "ALTER TABLE turn_metrics ADD COLUMN wechat_msgs_in INTEGER DEFAULT 0",
            "wechat_msgs_out": "ALTER TABLE turn_metrics ADD COLUMN wechat_msgs_out INTEGER DEFAULT 0",
        }.items():
            if name not in existing:
                cur.execute(ddl)
        self._conn.commit()

    def log_turn(self, record: TurnRecord):
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO turn_metrics (
                created_at, thread_id, turn_index, route, user_query, final_answer,
                tools_used, retrieval_hits, citations_count, latency_ms,
                actuation_count, vision_calls, wechat_msgs_in, wechat_msgs_out
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(),
                record.thread_id,
                record.turn_index,
                record.route,
                record.user_query,
                record.final_answer,
                json.dumps(record.tools_used, ensure_ascii=False),
                record.retrieval_hits,
                record.citations_count,
                record.latency_ms,
                record.actuation_count,
                record.vision_calls,
                record.wechat_msgs_in,
                record.wechat_msgs_out,
            ),
        )
        self._conn.commit()

    def get_thread_summary(self, thread_id: str) -> Dict[str, float]:
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT route, tools_used, retrieval_hits, citations_count, latency_ms
            FROM turn_metrics
            WHERE thread_id = ?
            ORDER BY turn_index ASC
            """,
            (thread_id,),
        )
        rows = cur.fetchall()
        if not rows:
            return {
                "turn_count": 0,
                "avg_latency_ms": 0,
                "retrieval_hit_rate": 0,
                "citation_rate": 0,
                "routes": {},
                "tool_counts": {},
            }

        latencies = [r[4] for r in rows if r[4] is not None]
        retrieval_flags = [1 if (r[2] or 0) > 0 else 0 for r in rows]
        citation_flags = [1 if (r[3] or 0) > 0 else 0 for r in rows]

        route_counts: Dict[str, int] = {}
        tool_counts: Dict[str, int] = {}

        for route, tools_json, _, _, _ in rows:
            route_key = route or "Unknown"
            route_counts[route_key] = route_counts.get(route_key, 0) + 1

            try:
                tools = json.loads(tools_json or "[]")
            except Exception:
                tools = []
            for t in tools:
                tool_counts[t] = tool_counts.get(t, 0) + 1

        return {
            "turn_count": len(rows),
            "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0,
            "retrieval_hit_rate": round(sum(retrieval_flags) / len(rows), 3),
            "citation_rate": round(sum(citation_flags) / len(rows), 3),
            "routes": route_counts,
            "tool_counts": tool_counts,
        }

    def export_thread_report(self, thread_id: str, report_path: str = "reports/latest_metrics.json") -> str:
        summary = self.get_thread_summary(thread_id)
        out = Path(report_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(out)
