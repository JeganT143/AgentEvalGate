import math

import pytest
from deepeval import assert_test
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from src.config import get_settings
from src.eval.gate_report import GateReport, GateRow
from src.eval.golden_dataset import load_golden_examples
from src.eval.metrics import score_example
from src.eval.policy import FAITHFULNESS_GATED_CATEGORIES, FAITHFULNESS_THRESHOLD
from src.rag.factory import build_demo_pipeline
from src.rag.pipeline import RAGPipeline
from src.rag.usage import UsageMeter, track_usage

# The threshold and its full, data-derived justification live in src/eval/policy.py,
# shared with the dashboard so the live demo grades against the exact same bar.


TYPICAL_EXAMPLES = [row for row in load_golden_examples() if row["category"] in FAITHFULNESS_GATED_CATEGORIES]


class RagasFaithfulnessMetric(BaseMetric):
    """DeepEval metric wrapping score_example()'s RAGAS faithfulness score.

    score_example() always computes context_precision alongside faithfulness
    (one call, no extra cost) - it's surfaced on self.context_precision and in
    self.reason for visibility, but deliberately NOT gated here: Day 2 / Step 6
    only measured and justified a threshold for faithfulness. Asserting on
    context_precision with an unjustified number would repeat the exact
    "arbitrary round number" mistake that step was written to avoid.

    Runs synchronously only (a_measure delegates to measure) because
    score_example() calls RAGAS's sync .score() wrapper, which explicitly
    raises if invoked from inside a running event loop - see Day 2 / Step 6.
    Callers must use assert_test(..., run_async=False).
    """

    def __init__(self, threshold: float = FAITHFULNESS_THRESHOLD):
        self.threshold = threshold
        self.context_precision: float | None = None

    @property
    def __name__(self) -> str:
        return "RAGAS Faithfulness"

    def measure(self, test_case: LLMTestCase) -> float:
        scores = score_example(
            query=test_case.input,
            retrieved_context=test_case.retrieval_context,
            answer=test_case.actual_output,
        )
        self.score = scores.faithfulness
        self.context_precision = scores.context_precision
        cp_display = (
            "nan" if isinstance(self.context_precision, float) and math.isnan(self.context_precision)
            else f"{self.context_precision:.3f}"
        )
        if isinstance(self.score, float) and math.isnan(self.score):
            # A naive `score >= threshold` silently evaluates to False for NaN with no
            # explanation - Day 2 / Step 5 found this happens for real (RAGAS extracts
            # zero checkable statements from some answers), so it's handled explicitly.
            self.success = False
            self.reason = (
                "faithfulness is NaN: RAGAS extracted zero checkable statements from "
                "the answer, so it cannot be scored - treated as a failure, not "
                f"silently ignored. context_precision={cp_display} (informational, not gated)."
            )
        else:
            self.success = self.score >= self.threshold
            self.reason = (
                f"faithfulness={self.score:.3f} (threshold={self.threshold}); "
                f"context_precision={cp_display} (informational, not gated)"
            )
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)


@pytest.fixture(scope="module")
def pipeline() -> RAGPipeline:
    # The embedder Settings.embedder selects - real OpenAI embeddings behind CachedEmbedder
    # by default (src/rag/factory.py::build_embedder), so the gate tests the same retrieval
    # the API and dashboard use.
    return build_demo_pipeline(get_settings())


def _gate_row(
    example: dict, metric: RagasFaithfulnessMetric, exc: Exception | None, meter: UsageMeter
) -> GateRow:
    if metric.score is None:
        # Never scored: the pipeline or the judge call itself blew up (bad API key, network,
        # quota) - recorded as an error so it can't be misread as a quality regression.
        error = f"{type(exc).__name__}: {exc}" if exc else "example was not scored"
        return GateRow(
            example_id=example["id"], category=example["category"], check="faithfulness",
            passed=False, threshold=metric.threshold, error=error[:300], cost_usd=meter.cost_usd,
        )
    return GateRow(
        example_id=example["id"], category=example["category"], check="faithfulness",
        passed=bool(metric.success), score=metric.score, threshold=metric.threshold,
        detail=metric.reason or "", context_precision=metric.context_precision, cost_usd=meter.cost_usd,
    )


@pytest.mark.parametrize("example", TYPICAL_EXAMPLES, ids=[e["id"] for e in TYPICAL_EXAMPLES])
def test_faithfulness(example: dict, pipeline: RAGPipeline, gate_report: GateReport) -> None:
    """Runs a `typical` golden example through the real pipeline and real judge,
    asserting RAGAS faithfulness >= FAITHFULNESS_THRESHOLD via DeepEval's assert_test.
    """
    metric = RagasFaithfulnessMetric()
    caught: Exception | None = None
    with track_usage() as meter:
        try:
            result = pipeline.answer(example["query"])
            test_case = LLMTestCase(
                input=example["query"],
                actual_output=result.answer,
                retrieval_context=[r.document.text for r in result.retrieved_context],
            )
            assert_test(test_case, [metric], run_async=False)
        except Exception as exc:
            caught = exc
            raise
        finally:
            gate_report.record(_gate_row(example, metric, caught, meter))
