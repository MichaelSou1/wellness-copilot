import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    from health_guide.config import KNOWLEDGE_BASE_DIR, KNOWLEDGE_BASE_AGENT_SUBDIRS
    from health_guide.rag import LocalKnowledgeBase

    parser = argparse.ArgumentParser(description="Offline prebuild for local RAG embeddings/index cache.")
    parser.add_argument("--kb-dir", default=KNOWLEDGE_BASE_DIR, help="Knowledge base directory path")
    parser.add_argument("--chunk-size", type=int, default=420, help="Chunk size")
    parser.add_argument("--overlap", type=int, default=100, help="Chunk overlap")
    parser.add_argument("--boundary-look-back", type=int, default=120,
                        help="Max chars to search back for a sentence boundary (default: 120)")
    parser.add_argument("--min-chunk-chars", type=int, default=30,
                        help="Discard chunks shorter than this many characters (default: 30)")
    parser.add_argument(
        "--agent",
        default="",
        help="Optional agent namespace to prebuild (trainer/nutritionist/psychologist/safety). Empty means build all.",
    )
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild: clear cache and regenerate")
    parser.add_argument(
        "--stats-out",
        default="reports/rag_index_stats.json",
        help="Output path for index stats JSON",
    )
    args = parser.parse_args()

    kb_root = Path(args.kb_dir)
    agents_to_build = [args.agent] if args.agent else list(KNOWLEDGE_BASE_AGENT_SUBDIRS.keys())

    stats = {}
    for agent in agents_to_build:
        subdir = KNOWLEDGE_BASE_AGENT_SUBDIRS.get(agent, agent)
        kb_dir = str(kb_root / subdir)
        kb = LocalKnowledgeBase(
            kb_dir=kb_dir,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            boundary_look_back=args.boundary_look_back,
            min_chunk_chars=args.min_chunk_chars,
        )
        kb.build(force_rebuild=args.rebuild)
        stats[agent] = kb.get_index_stats()
        print(f"[RAG Index] {agent}: {stats[agent]}")

    payload = {
        "per_agent_stats": stats,
        "kb_dir": str(kb_root),
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "boundary_look_back": args.boundary_look_back,
        "min_chunk_chars": args.min_chunk_chars,
        "agent": args.agent or "all",
        "force_rebuild": args.rebuild,
    }

    out = Path(args.stats_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[RAG Index] Build complete")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[RAG Index] Stats written to: {out}")


if __name__ == "__main__":
    main()
