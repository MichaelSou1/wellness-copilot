import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from health_guide.rag import Chunk, LocalKnowledgeBase, PAGE_SEPARATOR, SectionParent  # noqa: E402


def _chunk_texts(text: str, **kwargs):
    kb = LocalKnowledgeBase(min_chunk_chars=1, **kwargs)
    return [piece for piece, _page_range in kb._split_text(text)]


def test_markdown_structure_drives_chunk_boundaries_and_overlap():
    text = """# Guide

## Strength
- Alpha strength item keeps its own structural block.
- Beta strength item is reused as block overlap.
- Gamma strength item starts the next part.

## Nutrition
Protein guidance belongs to a separate section.
"""

    chunks = _chunk_texts(text, chunk_size=125, overlap=35)

    assert any(chunk.startswith("# Guide\n\n## Strength") for chunk in chunks)
    assert any(chunk.startswith("# Guide\n\n## Nutrition") for chunk in chunks)
    assert all(not ("## Strength" in chunk and "## Nutrition" in chunk) for chunk in chunks)
    assert any(
        "Beta strength item" in chunks[i] and "Beta strength item" in chunks[i + 1]
        for i in range(len(chunks) - 1)
    )


def test_oversize_block_falls_back_to_length_split_with_character_overlap():
    text = "".join(chr(65 + (i % 26)) for i in range(140))

    chunks = _chunk_texts(
        text,
        chunk_size=50,
        overlap=10,
        boundary_look_back=5,
    )

    assert len(chunks) > 1
    assert all(len(chunk) <= 50 for chunk in chunks)
    assert chunks[0][-10:] == chunks[1][:10]


def test_page_ranges_survive_structural_chunking():
    text = "第一页说明需要保留引用页码。" + PAGE_SEPARATOR + "第二页继续说明同一个结构块。"
    kb = LocalKnowledgeBase(chunk_size=200, overlap=20, min_chunk_chars=1)

    pieces = kb._split_text(text)

    assert pieces == [("第一页说明需要保留引用页码。第二页继续说明同一个结构块。", "1-2")]


def test_pdf_normalization_merges_hard_wrapped_lines_and_keeps_headings():
    raw = """第一章  示例标题

这是一个中文 PDF 段落，第一行被硬切
到第二行继续说明。

小贴士

1. 第一条建议也可能被硬切
到下一行。
"""

    cleaned = LocalKnowledgeBase._normalize_pdf_page_text(raw)
    blocks = LocalKnowledgeBase._parse_text_blocks(cleaned)

    assert "硬切到第二行" in cleaned
    assert "硬切\n到第二行" not in cleaned
    assert [(block.kind, block.text) for block in blocks[:3]] == [
        ("heading", "第一章 示例标题"),
        ("paragraph", "这是一个中文 PDF 段落，第一行被硬切到第二行继续说明。"),
        ("heading", "小贴士"),
    ]
    assert blocks[3].kind == "list_item"
    assert "被硬切到下一行" in blocks[3].text


def test_page_range_ignores_repeated_heading_prefix_page():
    text = "# Title\n" + PAGE_SEPARATOR + "正文在第二页，标题只是语义前缀。"
    kb = LocalKnowledgeBase(chunk_size=200, overlap=20, min_chunk_chars=1)

    pieces = kb._split_text(text)

    assert pieces == [("# Title\n\n正文在第二页，标题只是语义前缀。", "2")]


def test_pdf_chunks_keep_section_metadata_for_parent_child_retrieval():
    text = "第一章 总则\n\n第一页正文说明。 " + PAGE_SEPARATOR + "第二页继续说明。"
    kb = LocalKnowledgeBase(chunk_size=200, overlap=20, min_chunk_chars=1)

    pieces = kb._split_text_with_metadata(text, source="guide.pdf", file_type="pdf")

    assert pieces
    assert pieces[0]["is_pdf"] is True
    assert pieces[0]["section_id"] == "guide.pdf#section-1"
    assert "第一章 总则" in pieces[0]["section_path"]
    assert pieces[0]["page_range"] == "1-2"


def test_pdf_fine_chunking_splits_adjacent_list_items():
    text = """第一章 营养建议

1. 早餐建议包含全谷物和蛋白质。

2. 午餐建议包含蔬菜和适量主食。
"""
    kb = LocalKnowledgeBase(chunk_size=420, overlap=20, min_chunk_chars=1)

    pieces = kb._split_text_with_metadata(text, source="guide.pdf", file_type="pdf")
    texts = [str(piece["text"]) for piece in pieces]

    assert len(texts) == 2
    assert "早餐建议" in texts[0]
    assert "午餐建议" not in texts[0]
    assert "午餐建议" in texts[1]
    assert all(text.startswith("第一章 营养建议") for text in texts)


def test_pdf_neighbor_context_expands_final_content():
    kb = LocalKnowledgeBase(min_chunk_chars=1)
    kb.chunks = [
        Chunk("guide.pdf#p1-chunk-1", "guide.pdf", "第一页", "1", is_pdf=True),
        Chunk("guide.pdf#p2-chunk-2", "guide.pdf", "第二页", "2", is_pdf=True),
        Chunk("guide.pdf#p3-chunk-3", "guide.pdf", "第三页", "3", is_pdf=True),
    ]
    result = {
        "chunk_id": "guide.pdf#p2-chunk-2",
        "source": "guide.pdf",
        "page_range": "2",
        "content": "第二页",
    }

    expanded = kb._attach_pdf_context(result, include_content=True)

    assert expanded["context_chunk_ids"] == [
        "guide.pdf#p1-chunk-1",
        "guide.pdf#p2-chunk-2",
        "guide.pdf#p3-chunk-3",
    ]
    assert expanded["context_page_range"] == "1-3"
    assert "第一页" in expanded["content"]
    assert "第三页" in expanded["content"]


def test_pdf_parent_candidates_survive_beyond_visible_dense_top_k():
    import numpy as np
    import health_guide.rag as rag_module

    kb = LocalKnowledgeBase(min_chunk_chars=1)
    kb.chunks = [
        Chunk("other.md#chunk-1", "other.md", "高分普通片段"),
        Chunk(
            "guide.pdf#p2-chunk-2",
            "guide.pdf",
            "低分但属于高相关章节的片段",
            "2",
            section_id="guide.pdf#section-1",
            section_path="糖尿病饮食",
            is_pdf=True,
        ),
    ]
    kb.section_parents = [
        SectionParent(
            section_id="guide.pdf#section-1",
            source="guide.pdf",
            section_path="糖尿病饮食",
            text="糖尿病饮食 parent context",
            page_range="2",
            child_chunk_ids=["guide.pdf#p2-chunk-2"],
        )
    ]
    kb._chunk_embeddings = np.asarray([[0.9, 0.0], [0.1, 0.0]], dtype=np.float32)
    kb._section_embeddings = np.asarray([[1.0, 0.0]], dtype=np.float32)
    kb._dense_topk = lambda _query_emb, _top_k: [(0, 0.9)]

    old_expansion = rag_module.RAG_PDF_PARENT_EXPANSION_ENABLED
    old_fusion = rag_module.RAG_PDF_PARENT_SCORE_FUSION_ENABLED
    old_weight = rag_module.RAG_PDF_SECTION_SCORE_WEIGHT
    try:
        rag_module.RAG_PDF_PARENT_EXPANSION_ENABLED = True
        rag_module.RAG_PDF_PARENT_SCORE_FUSION_ENABLED = True
        rag_module.RAG_PDF_SECTION_SCORE_WEIGHT = 0.05
        candidates = kb._dense_candidates_with_pdf_sections(
            np.asarray([1.0, 0.0], dtype=np.float32),
            dense_k=1,
            pool_k=2,
        )
    finally:
        rag_module.RAG_PDF_PARENT_EXPANSION_ENABLED = old_expansion
        rag_module.RAG_PDF_PARENT_SCORE_FUSION_ENABLED = old_fusion
        rag_module.RAG_PDF_SECTION_SCORE_WEIGHT = old_weight

    assert [idx for idx, _dense, _parent, _score in candidates] == [0, 1]
    assert candidates[1][2] == 1.0
    assert abs(candidates[1][3] - 0.15) < 1e-6


def test_pdf_parent_expansion_can_be_disabled():
    import numpy as np
    import health_guide.rag as rag_module

    kb = LocalKnowledgeBase(min_chunk_chars=1)
    kb.chunks = [
        Chunk("other.md#chunk-1", "other.md", "高分普通片段"),
        Chunk(
            "guide.pdf#p2-chunk-2",
            "guide.pdf",
            "低分但属于高相关章节的片段",
            "2",
            section_id="guide.pdf#section-1",
            section_path="糖尿病饮食",
            is_pdf=True,
        ),
    ]
    kb.section_parents = [
        SectionParent(
            section_id="guide.pdf#section-1",
            source="guide.pdf",
            section_path="糖尿病饮食",
            text="糖尿病饮食 parent context",
            page_range="2",
            child_chunk_ids=["guide.pdf#p2-chunk-2"],
        )
    ]
    kb._chunk_embeddings = np.asarray([[0.9, 0.0], [0.1, 0.0]], dtype=np.float32)
    kb._section_embeddings = np.asarray([[1.0, 0.0]], dtype=np.float32)
    kb._dense_topk = lambda _query_emb, _top_k: [(0, 0.9)]

    old_expansion = rag_module.RAG_PDF_PARENT_EXPANSION_ENABLED
    try:
        rag_module.RAG_PDF_PARENT_EXPANSION_ENABLED = False
        candidates = kb._dense_candidates_with_pdf_sections(
            np.asarray([1.0, 0.0], dtype=np.float32),
            dense_k=1,
            pool_k=2,
        )
    finally:
        rag_module.RAG_PDF_PARENT_EXPANSION_ENABLED = old_expansion

    assert [idx for idx, _dense, _parent, _score in candidates] == [0]


def test_pdf_parent_score_fusion_can_be_disabled():
    import numpy as np
    import health_guide.rag as rag_module

    kb = LocalKnowledgeBase(min_chunk_chars=1)
    kb.chunks = [
        Chunk("other.md#chunk-1", "other.md", "高分普通片段"),
        Chunk(
            "guide.pdf#p2-chunk-2",
            "guide.pdf",
            "低分但属于高相关章节的片段",
            "2",
            section_id="guide.pdf#section-1",
            section_path="糖尿病饮食",
            is_pdf=True,
        ),
    ]
    kb.section_parents = [
        SectionParent(
            section_id="guide.pdf#section-1",
            source="guide.pdf",
            section_path="糖尿病饮食",
            text="糖尿病饮食 parent context",
            page_range="2",
            child_chunk_ids=["guide.pdf#p2-chunk-2"],
        )
    ]
    kb._chunk_embeddings = np.asarray([[0.9, 0.0], [0.1, 0.0]], dtype=np.float32)
    kb._section_embeddings = np.asarray([[1.0, 0.0]], dtype=np.float32)
    kb._dense_topk = lambda _query_emb, _top_k: [(0, 0.9)]

    old_expansion = rag_module.RAG_PDF_PARENT_EXPANSION_ENABLED
    old_fusion = rag_module.RAG_PDF_PARENT_SCORE_FUSION_ENABLED
    try:
        rag_module.RAG_PDF_PARENT_EXPANSION_ENABLED = True
        rag_module.RAG_PDF_PARENT_SCORE_FUSION_ENABLED = False
        candidates = kb._dense_candidates_with_pdf_sections(
            np.asarray([1.0, 0.0], dtype=np.float32),
            dense_k=1,
            pool_k=2,
        )
    finally:
        rag_module.RAG_PDF_PARENT_EXPANSION_ENABLED = old_expansion
        rag_module.RAG_PDF_PARENT_SCORE_FUSION_ENABLED = old_fusion

    assert candidates[1][2] == 1.0
    assert abs(candidates[1][3] - 0.1) < 1e-6


def test_hybrid_bm25_adds_lexical_candidate_to_pool():
    import numpy as np
    import health_guide.rag as rag_module

    kb = LocalKnowledgeBase(min_chunk_chars=1)
    kb.chunks = [
        Chunk("general.md#chunk-1", "general.md", "普通健康建议"),
        Chunk("nutrition.md#chunk-2", "nutrition.md", "维生素A 可以来自胡萝卜"),
    ]
    kb._chunk_embeddings = np.asarray([[0.9, 0.0], [0.1, 0.0]], dtype=np.float32)
    kb._dense_topk = lambda _query_emb, _top_k: [(0, 0.9)]
    kb._build_lexical_index()

    old_hybrid = rag_module.RAG_HYBRID_RETRIEVAL_ENABLED
    old_top_k = rag_module.RAG_BM25_TOP_K
    try:
        rag_module.RAG_HYBRID_RETRIEVAL_ENABLED = True
        rag_module.RAG_BM25_TOP_K = 4
        candidates = kb._dense_candidates_with_pdf_sections(
            np.asarray([1.0, 0.0], dtype=np.float32),
            dense_k=1,
            pool_k=2,
            query="维生素A 从哪里来",
        )
    finally:
        rag_module.RAG_HYBRID_RETRIEVAL_ENABLED = old_hybrid
        rag_module.RAG_BM25_TOP_K = old_top_k

    assert [idx for idx, _dense, _parent, _score in candidates] == [0, 1]


def test_gated_parent_rescue_skips_when_pdf_candidates_are_present():
    import numpy as np
    import health_guide.rag as rag_module

    kb = LocalKnowledgeBase(min_chunk_chars=1)
    kb.chunks = [
        Chunk(
            "guide.pdf#p1-chunk-1",
            "guide.pdf",
            "已有 PDF 候选",
            "1",
            section_id="guide.pdf#section-0",
            section_path="已有章节",
            is_pdf=True,
        ),
        Chunk(
            "guide.pdf#p2-chunk-2",
            "guide.pdf",
            "parent rescue 才会加入的片段",
            "2",
            section_id="guide.pdf#section-1",
            section_path="糖尿病饮食",
            is_pdf=True,
        ),
    ]
    kb.section_parents = [
        SectionParent(
            section_id="guide.pdf#section-1",
            source="guide.pdf",
            section_path="糖尿病饮食",
            text="糖尿病饮食 parent context",
            page_range="2",
            child_chunk_ids=["guide.pdf#p2-chunk-2"],
        )
    ]
    kb._chunk_embeddings = np.asarray([[0.9, 0.0], [0.1, 0.0]], dtype=np.float32)
    kb._section_embeddings = np.asarray([[1.0, 0.0]], dtype=np.float32)
    kb._dense_topk = lambda _query_emb, _top_k: [(0, 0.9)]
    kb._build_lexical_index()

    old_expansion = rag_module.RAG_PDF_PARENT_EXPANSION_ENABLED
    old_rescue = rag_module.RAG_PDF_PARENT_RESCUE_ENABLED
    old_lookahead = rag_module.RAG_PDF_PARENT_RESCUE_LOOKAHEAD
    old_min_pdf = rag_module.RAG_PDF_PARENT_RESCUE_MIN_PDF_CANDIDATES
    try:
        rag_module.RAG_PDF_PARENT_EXPANSION_ENABLED = True
        rag_module.RAG_PDF_PARENT_RESCUE_ENABLED = True
        rag_module.RAG_PDF_PARENT_RESCUE_LOOKAHEAD = 2
        rag_module.RAG_PDF_PARENT_RESCUE_MIN_PDF_CANDIDATES = 1
        candidates = kb._dense_candidates_with_pdf_sections(
            np.asarray([1.0, 0.0], dtype=np.float32),
            dense_k=1,
            pool_k=2,
        )
    finally:
        rag_module.RAG_PDF_PARENT_EXPANSION_ENABLED = old_expansion
        rag_module.RAG_PDF_PARENT_RESCUE_ENABLED = old_rescue
        rag_module.RAG_PDF_PARENT_RESCUE_LOOKAHEAD = old_lookahead
        rag_module.RAG_PDF_PARENT_RESCUE_MIN_PDF_CANDIDATES = old_min_pdf

    assert [idx for idx, _dense, _parent, _score in candidates] == [0]


def test_pdf_rerank_text_excludes_section_parent_by_default():
    kb = LocalKnowledgeBase(min_chunk_chars=1)
    kb.chunks = [
        Chunk(
            "guide.pdf#p2-chunk-2",
            "guide.pdf",
            "这里是 child chunk。",
            "2",
            section_id="guide.pdf#section-1",
            section_path="糖尿病饮食",
            is_pdf=True,
        )
    ]
    kb.section_parents = [
        SectionParent(
            section_id="guide.pdf#section-1",
            source="guide.pdf",
            section_path="糖尿病饮食",
            text="这里是 parent context。",
            page_range="2",
            child_chunk_ids=["guide.pdf#p2-chunk-2"],
        )
    ]

    rerank_text = kb._rerank_text_for_chunk(0)

    assert "Section: 糖尿病饮食" in rerank_text
    assert "Parent context:" not in rerank_text
    assert "这里是 parent context。" not in rerank_text
    assert "Child chunk:" in rerank_text
    assert "这里是 child chunk。" in rerank_text


def test_pdf_rerank_text_can_include_section_parent():
    import health_guide.rag as rag_module

    kb = LocalKnowledgeBase(min_chunk_chars=1)
    kb.chunks = [
        Chunk(
            "guide.pdf#p2-chunk-2",
            "guide.pdf",
            "这里是 child chunk。",
            "2",
            section_id="guide.pdf#section-1",
            section_path="糖尿病饮食",
            is_pdf=True,
        )
    ]
    kb.section_parents = [
        SectionParent(
            section_id="guide.pdf#section-1",
            source="guide.pdf",
            section_path="糖尿病饮食",
            text="这里是 parent context。",
            page_range="2",
            child_chunk_ids=["guide.pdf#p2-chunk-2"],
        )
    ]

    old_context = rag_module.RAG_PDF_PARENT_RERANK_CONTEXT_ENABLED
    try:
        rag_module.RAG_PDF_PARENT_RERANK_CONTEXT_ENABLED = True
        rerank_text = kb._rerank_text_for_chunk(0)
    finally:
        rag_module.RAG_PDF_PARENT_RERANK_CONTEXT_ENABLED = old_context

    assert "Section: 糖尿病饮食" in rerank_text
    assert "Parent context:" in rerank_text
    assert "这里是 parent context。" in rerank_text
    assert "Child chunk:" in rerank_text
    assert "这里是 child chunk。" in rerank_text
