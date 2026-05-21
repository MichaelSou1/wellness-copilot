import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_rag import EvalSample, _is_relevant, _relaxed_stage_metrics, _stage_metrics
from scripts.evaluate_rag import load_dataset


def test_chunk_ground_truth_takes_precedence_over_source_match():
    sample = EvalSample(
        query="q",
        agent="nutritionist",
        relevant_sources={"nutritionist/diet.md"},
        relevant_chunk_ids={"nutritionist:diet.md#chunk-2"},
    )

    same_source_wrong_chunk = {
        "source": "diet.md",
        "chunk_id": "diet.md#chunk-1",
    }
    exact_chunk = {
        "source": "diet.md",
        "chunk_id": "diet.md#chunk-2",
    }

    assert not _is_relevant(same_source_wrong_chunk, sample)
    assert _is_relevant(exact_chunk, sample)


def test_duplicate_source_matches_are_deduped_for_metrics():
    sample = EvalSample(
        query="q",
        agent="trainer",
        relevant_sources={"mobility.md"},
        relevant_chunk_ids=set(),
    )
    results = [
        {"source": "mobility.md", "chunk_id": "mobility.md#chunk-1"},
        {"source": "mobility.md", "chunk_id": "mobility.md#chunk-2"},
        {"source": "other.md", "chunk_id": "other.md#chunk-1"},
    ]

    metrics = _stage_metrics(results, sample, [3])

    assert metrics["recall@3"] == 1.0
    assert metrics["ndcg@3"] == 1.0
    assert metrics["map@3"] == 1.0


def test_metrics_do_not_exceed_one_for_precise_chunk_ground_truth():
    sample = EvalSample(
        query="q",
        agent="psychologist",
        relevant_sources={"stress.md"},
        relevant_chunk_ids={"stress.md#chunk-2"},
    )
    results = [
        {"source": "stress.md", "chunk_id": "stress.md#chunk-1"},
        {"source": "stress.md", "chunk_id": "stress.md#chunk-2"},
        {"source": "stress.md", "chunk_id": "stress.md#chunk-3"},
    ]

    metrics = _stage_metrics(results, sample, [3])

    assert metrics["first_relevant_rank"] == 2
    assert metrics["recall@3"] == 1.0
    assert metrics["ndcg@3"] <= 1.0
    assert metrics["map@3"] <= 1.0


def test_load_dataset_keeps_generated_metadata(tmp_path):
    dataset = tmp_path / "rag.jsonl"
    dataset.write_text(
        (
            '{"query":"膝盖不舒服还能做什么腿部训练？","agent":"trainer",'
            '"relevant_chunk_ids":["trainer:exercise_safety.md#chunk-1"],'
            '"relevant_sources":["trainer/exercise_safety.md"],'
            '"answer":"避免疼痛动作，优先选择低冲击训练。",'
            '"supporting_span":"疼痛时应避免加重症状的动作。",'
            '"question_type":"scenario","difficulty":"medium",'
            '"sample_kind":"single_hop_positive",'
            '"quality":{"literal_overlap":0.21}}\n'
        ),
        encoding="utf-8",
    )

    sample = load_dataset(dataset)[0]

    assert sample.question_type == "scenario"
    assert sample.difficulty == "medium"
    assert sample.sample_kind == "single_hop_positive"
    assert sample.answer
    assert sample.supporting_span
    assert sample.quality == {"literal_overlap": 0.21}


def test_relaxed_metrics_report_source_page_and_adjacent_hits():
    sample = EvalSample(
        query="q",
        agent="nutritionist",
        relevant_sources={"nutritionist:guide.pdf"},
        relevant_chunk_ids={"nutritionist:guide.pdf#p10-chunk-5"},
        page_ranges={"10"},
    )
    results = [
        {
            "source": "guide.pdf",
            "chunk_id": "guide.pdf#p9-10-chunk-4",
            "page_range": "9-10",
        },
        {
            "source": "other.pdf",
            "chunk_id": "other.pdf#p10-chunk-1",
            "page_range": "10",
        },
    ]

    metrics = _relaxed_stage_metrics(results, sample, [1])

    assert metrics["source_hit@1"] == 1.0
    assert metrics["page_hit@1"] == 1.0
    assert metrics["same_source_near_miss@1"] == 1.0
    assert metrics["same_page_or_adjacent_chunk_hit@1"] == 1.0
