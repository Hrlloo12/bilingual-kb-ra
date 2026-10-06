from rag.retrieval.fusion import weighted_rrf


def test_rrf_rewards_agreement_between_rankers():
    fused = weighted_rrf({"bm25": ["a", "b", "c"], "dense": ["b", "d", "a"]}, {"bm25": 1.0, "dense": 1.0})
    assert [item for item, _ in fused][:2] == ["b", "a"]


def test_rrf_weights_shift_preference():
    rankings = {"bm25": ["a", "x"], "dense": ["b", "y"]}
    assert weighted_rrf(rankings, {"bm25": 1.0, "dense": 2.0})[0][0] == "b"
    assert weighted_rrf(rankings, {"bm25": 2.0, "dense": 1.0})[0][0] == "a"


def test_rrf_limit():
    assert len(weighted_rrf({"bm25": ["a", "b", "c"]}, {}, limit=2)) == 2
