import pytest
from pytest import approx

from src.eval.retrieval_metrics import hit_at_k, precision_at_k, recall_at_k


def test_precision_at_k_caps_at_one_over_k_for_a_single_relevant_document():
    # The reason hit@k exists: perfect retrieval of a one-answer question still reads 0.333.
    assert precision_at_k(["doc-1", "doc-3", "doc-0"], ["doc-1"], k=3) == approx(1 / 3)


def test_hit_at_k_is_one_when_any_relevant_document_is_in_the_top_k():
    assert hit_at_k(["doc-3", "doc-1", "doc-0"], ["doc-1"], k=3) == 1.0
    assert hit_at_k(["doc-3", "doc-1", "doc-0"], ["doc-1"], k=1) == 0.0


def test_recall_at_k_counts_the_fraction_of_relevant_documents_found():
    assert recall_at_k(["doc-4", "doc-2", "doc-0"], ["doc-4", "doc-5"], k=3) == approx(0.5)
    assert recall_at_k(["doc-4", "doc-5", "doc-0"], ["doc-4", "doc-5"], k=3) == approx(1.0)


def test_hit_and_recall_are_not_applicable_when_nothing_should_be_retrieved():
    assert hit_at_k(["doc-0", "doc-1", "doc-2"], [], k=3) is None
    assert recall_at_k(["doc-0", "doc-1", "doc-2"], [], k=3) is None


@pytest.mark.parametrize("metric", [precision_at_k, hit_at_k, recall_at_k])
def test_every_retrieval_metric_rejects_non_positive_k(metric):
    with pytest.raises(ValueError):
        metric(["doc-0"], ["doc-0"], k=0)
