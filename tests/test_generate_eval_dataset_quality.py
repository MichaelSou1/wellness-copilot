import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_eval_dataset import (  # noqa: E402
    CharNgramBM25,
    GeneratedSample,
    _cap_chunks_per_file,
    _filter_chunks_for_generation,
    _filter_generated_samples,
    _literal_overlap_score,
    _supporting_span_trace_score,
)


def _args(**overrides):
    base = {
        "min_chunk_len": 20,
        "chunk_quality_filter": True,
        "min_content_char_ratio": 0.55,
        "max_table_line_ratio": 0.65,
        "max_duplicate_line_ratio": 0.55,
        "min_question_len": 6,
        "max_question_len": 90,
        "max_literal_overlap": 0.72,
        "require_supporting_span": True,
        "min_support_trace": 0.78,
        "downsample_easy_bm25": False,
        "easy_bm25_score_ratio": 2.4,
        "easy_bm25_min_overlap": 0.38,
        "llm_verify": False,
        "require_llm_verdict": False,
        "questions_per_chunk": 2,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_literal_overlap_flags_copy_paste_question():
    chunk = "成年人每周至少进行150分钟中等强度有氧活动，并配合力量训练。"
    pasted = "成年人每周至少进行150分钟中等强度有氧活动吗？"
    natural = "平时运动量想达标，一周大概要动多久？"

    assert _literal_overlap_score(pasted, chunk) > 0.72
    assert _literal_overlap_score(natural, chunk) < 0.72


def test_supporting_span_must_trace_to_chunk():
    chunk = "睡前减少咖啡因摄入，保持规律作息，有助于改善入睡困难。"

    assert _supporting_span_trace_score("保持规律作息，有助于改善入睡困难。", chunk) == 1.0
    assert _supporting_span_trace_score("每天进行高强度间歇训练。", chunk) < 0.78


def test_chunk_filter_skips_table_like_noise():
    chunks = [
        {
            "agent": "nutritionist",
            "chunk_id": "ok.md#chunk-1",
            "source": "ok.md",
            "text": "规律饮食可以帮助控制总能量摄入。早餐、午餐和晚餐都应包含优质蛋白。",
        },
        {
            "agent": "nutritionist",
            "chunk_id": "table.md#chunk-1",
            "source": "table.md",
            "text": "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |",
        },
    ]

    kept, reasons = _filter_chunks_for_generation(chunks, _args())

    assert [chunk["source"] for chunk in kept] == ["ok.md"]
    assert reasons["mostly_table_or_rule_lines"] == 1


def test_file_level_chunk_cap_limits_long_sources_before_global_sampling():
    chunks = [
        {
            "agent": "nutritionist",
            "chunk_id": f"book.pdf#chunk-{idx}",
            "source": "book.pdf",
            "text": f"长 PDF 片段 {idx}，包含足够的正文内容。",
        }
        for idx in range(15)
    ]
    chunks.extend(
        {
            "agent": "nutritionist",
            "chunk_id": f"short.md#chunk-{idx}",
            "source": "short.md",
            "text": f"短文档片段 {idx}，包含足够的正文内容。",
        }
        for idx in range(3)
    )

    capped, stats = _cap_chunks_per_file(chunks, max_chunks_per_file=10, seed=7)

    by_source = {}
    for chunk in capped:
        by_source.setdefault(chunk["source"], 0)
        by_source[chunk["source"]] += 1

    assert len(capped) == 13
    assert by_source == {"book.pdf": 10, "short.md": 3}
    assert stats["capped_file_count"] == 1
    assert stats["dropped_chunk_count"] == 5


def test_file_level_chunk_cap_can_be_disabled():
    chunks = [
        {
            "agent": "nutritionist",
            "chunk_id": f"book.pdf#chunk-{idx}",
            "source": "book.pdf",
            "text": f"长 PDF 片段 {idx}，包含足够的正文内容。",
        }
        for idx in range(12)
    ]

    capped, stats = _cap_chunks_per_file(chunks, max_chunks_per_file=0, seed=7)

    assert capped == chunks
    assert stats["enabled"] is False


def test_generated_sample_filter_rejects_leakage_and_accepts_traceable_sample():
    chunk = {
        "agent": "trainer",
        "chunk_id": "cardio.md#chunk-1",
        "source": "cardio.md",
        "text": "成年人每周建议累计150到300分钟中等强度有氧运动。可以拆成多次完成。",
    }
    bm25 = CharNgramBM25([chunk])
    candidates = [
        GeneratedSample(
            query="成年人每周建议累计150到300分钟中等强度有氧运动吗？",
            answer="每周150到300分钟。",
            supporting_span="成年人每周建议累计150到300分钟中等强度有氧运动。",
        ),
        GeneratedSample(
            query="一周有氧运动想达标，大概要安排多久？",
            answer="每周累计150到300分钟中等强度有氧运动。",
            supporting_span="成年人每周建议累计150到300分钟中等强度有氧运动。",
            question_type="scenario",
        ),
    ]

    accepted, rejected = _filter_generated_samples(
        candidates,
        chunk,
        _args(),
        bm25,
        seen_questions=set(),
    )

    assert rejected["literal_overlap_high"] == 1
    assert len(accepted) == 1
    assert accepted[0][0].query.startswith("一周有氧运动")


def test_prefilter_can_avoid_mutating_seen_questions():
    chunk = {
        "agent": "trainer",
        "chunk_id": "cardio.md#chunk-1",
        "source": "cardio.md",
        "text": "成年人每周建议累计150到300分钟中等强度有氧运动。可以拆成多次完成。",
    }
    bm25 = CharNgramBM25([chunk])
    seen_questions = set()
    candidates = [
        GeneratedSample(
            query="一周有氧运动想达标，大概要安排多久？",
            answer="每周累计150到300分钟中等强度有氧运动。",
            supporting_span="成年人每周建议累计150到300分钟中等强度有氧运动。",
        ),
        GeneratedSample(
            query="一周有氧运动想达标，大概要安排多久？",
            answer="每周累计150到300分钟中等强度有氧运动。",
            supporting_span="成年人每周建议累计150到300分钟中等强度有氧运动。",
        ),
    ]

    accepted, rejected = _filter_generated_samples(
        candidates,
        chunk,
        _args(llm_verify=True),
        bm25,
        seen_questions=seen_questions,
        apply_llm_verdicts=False,
        mark_seen=False,
        limit=2,
    )

    assert len(accepted) == 1
    assert rejected["duplicate_question"] == 1
    assert seen_questions == set()
