import pytest

from rag.evaluation.metrics import latency_summary, ndcg_at_k, query_metrics, recall_at_k, reciprocal_rank


def test_recall_and_reciprocal_rank():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, {"b", "z"}, 1) == 0.0
    assert recall_at_k(ranked, {"b", "z"}, 5) == 0.5
    assert reciprocal_rank(ranked, {"c"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, {"z"}) == 0.0


def test_ndcg_is_one_for_perfect_ranking():
    assert ndcg_at_k(["a", "b", "x"], {"a", "b"}) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "a"], {"a"}) < 1.0


def test_query_metrics_keys():
    metrics = query_metrics(["a"], {"a"})
    assert metrics["hit@1"] == 1.0 and metrics["mrr"] == 1.0 and "ndcg@10" in metrics


def test_latency_summary():
    summary = latency_summary([10.0, 20.0, 30.0, 40.0])
    assert summary["avg"] == 25.0 and summary["max"] == 40.0
