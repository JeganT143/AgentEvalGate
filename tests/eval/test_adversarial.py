"""Gate: out-of-corpus adversarial questions must be declined, not answered.

The faithfulness gate (test_faithfulness.py) is scoped to `typical` because
faithfulness can't grade a correct "I don't know" - see the threshold comment there.
This closes part of that gap for `adversarial`: for a question with no relevant
document in the corpus, the only correct behaviour is to decline, and that's checked
deterministically (src/eval/refusal.py) - no judge call, no judge variance, ~1 cheap
generation call per example.

Baseline measured before this gate was added (2026-10-06, gpt-4o-mini, grounded
prompt): 11/11 such questions answered "I don't know." The hallucination-inducing
prompt from src/eval/generate_demo_runs.py ("never say you don't know") is exactly the
regression this catches.
"""

import pytest

from src.config import get_settings
from src.eval.gate_report import GateReport, GateRow
from src.eval.golden_dataset import load_golden_examples
from src.eval.refusal import is_refusal, must_decline
from src.rag.factory import build_demo_pipeline
from src.rag.pipeline import RAGPipeline
from src.rag.usage import track_usage

DECLINE_EXAMPLES = [row for row in load_golden_examples() if must_decline(row)]


@pytest.fixture(scope="module")
def pipeline() -> RAGPipeline:
    return build_demo_pipeline(get_settings())


@pytest.mark.parametrize("example", DECLINE_EXAMPLES, ids=[e["id"] for e in DECLINE_EXAMPLES])
def test_declines_out_of_corpus_question(example: dict, pipeline: RAGPipeline, gate_report: GateReport) -> None:
    try:
        with track_usage() as meter:
            answer = pipeline.answer(example["query"]).answer
    except Exception as exc:
        gate_report.record(
            GateRow(
                example_id=example["id"], category=example["category"], check="declines",
                passed=False, error=f"{type(exc).__name__}: {exc}"[:300],
            )
        )
        raise

    declined = is_refusal(answer)
    gate_report.record(
        GateRow(
            example_id=example["id"], category=example["category"], check="declines",
            passed=declined, detail=f"answer: {answer!r}", cost_usd=meter.cost_usd,
        )
    )
    assert declined, (
        f"{example['id']}: the corpus has no document that answers {example['query']!r}, "
        f"so the pipeline should decline - instead it answered {answer!r}"
    )
