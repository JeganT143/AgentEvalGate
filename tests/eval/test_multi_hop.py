"""Gate: every multi-hop question must retrieve every document it needs.

A question like "How does the mitochondria's role relate to what plants do with
sunlight?" can only be answered well if search returns BOTH the mitochondria and the
photosynthesis documents. This checks exactly that - retrieval only, no generation and
no judge, so it is cheap and identical on every run. The measured baseline, why the
answers themselves aren't gated yet, and the bar are documented in src/eval/policy.py.
"""

import pytest

from src.config import get_settings
from src.eval.gate_report import GateReport, GateRow
from src.eval.golden_dataset import load_golden_examples
from src.eval.policy import RETRIEVAL_GATED_CATEGORIES, RETRIEVAL_RECALL_BAR
from src.eval.retrieval_metrics import recall_at_k
from src.rag.factory import build_demo_pipeline
from src.rag.pipeline import RAGPipeline
from src.rag.usage import track_usage

MULTI_HOP_EXAMPLES = [row for row in load_golden_examples() if row["category"] in RETRIEVAL_GATED_CATEGORIES]


@pytest.fixture(scope="module")
def pipeline() -> RAGPipeline:
    return build_demo_pipeline(get_settings())


@pytest.mark.parametrize("example", MULTI_HOP_EXAMPLES, ids=[e["id"] for e in MULTI_HOP_EXAMPLES])
def test_retrieves_every_required_document(example: dict, pipeline: RAGPipeline, gate_report: GateReport) -> None:
    common = {"example_id": example["id"], "category": example["category"], "check": "retrieval"}
    with track_usage() as meter:
        try:
            retrieved_ids = [r.document.id for r in pipeline.retrieve(example["query"])]
        except Exception as exc:
            gate_report.record(GateRow(**common, passed=False, error=f"{type(exc).__name__}: {exc}"[:300]))
            raise

    expected = example["expected_context_ids"]
    recall = recall_at_k(retrieved_ids, expected, pipeline.top_k)
    passed = recall >= RETRIEVAL_RECALL_BAR
    missing = sorted(set(expected) - set(retrieved_ids))
    gate_report.record(
        GateRow(
            **common, passed=passed, score=recall, threshold=RETRIEVAL_RECALL_BAR, cost_usd=meter.cost_usd,
            detail=f"top {pipeline.top_k}: {retrieved_ids}; needed {expected}"
            + (f"; missing {missing}" if missing else ""),
        )
    )
    assert passed, (
        f"{example['id']}: search returned {retrieved_ids} for {example['query']!r} but the answer needs "
        f"{expected} - missing {missing}"
    )
