"""What the gate would say about one live run - pure logic, no Streamlit, unit-tested.

The live demo used to show raw scores and leave the reader to guess whether 0.5641 or
a faithfulness of 1.000 was good. This turns a run into the same pass / block decision
the CI gate makes (src/eval/policy.py, src/eval/refusal.py), plus a plain-English why.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.eval.policy import (
    FAITHFULNESS_GATED_CATEGORIES,
    FAITHFULNESS_THRESHOLD,
    RETRIEVAL_GATED_CATEGORIES,
    RETRIEVAL_RECALL_BAR,
)
from src.eval.refusal import must_decline


@dataclass(frozen=True)
class Verdict:
    kind: str  # "pass" | "fail" | "info"
    title: str
    body: str


def needs_judge(example: dict) -> bool:
    """Decline checks are pattern-based - only the other questions need the (slow) AI judge.
    Multi-hop questions are gated on retrieval, but the judge still grades the answer for
    information."""
    return not must_decline(example)


def verdict_for(example: dict, run: dict, judged: dict | None) -> Verdict | None:
    """The gate's decision for this run, or None while the judge hasn't scored it yet."""
    if must_decline(example):
        if run["declined"]:
            return Verdict(
                "pass",
                "Would pass the gate: correctly declined",
                "None of the documents answers this question, and the model said so instead of guessing.",
            )
        return Verdict(
            "fail",
            "Would block the merge: answered a question it should have declined",
            "No document answers this question, so any answer the model gives is made up.",
        )

    if example["category"] in RETRIEVAL_GATED_CATEGORIES:
        needed = len(example["expected_context_ids"])
        if run["recall_at_k"] is not None and run["recall_at_k"] >= RETRIEVAL_RECALL_BAR:
            return Verdict(
                "pass",
                "Would pass the gate: search found "
                + ("both documents" if needed == 2 else f"all {needed} documents")
                + " this question needs",
                "Multi-hop questions are gated on search: every document the answer depends on must be "
                "in the top results. The answer's grounded score is tracked for information, not gated, "
                "because the model still declines many of these even with both documents in hand.",
            )
        return Verdict(
            "fail",
            "Would block the merge: search missed a document this question needs",
            "A multi-hop answer can't be right if search never returned one of the documents it depends on.",
        )

    if judged is None:
        return None

    score = judged["faithfulness"]
    if example["category"] in FAITHFULNESS_GATED_CATEGORIES:
        if isinstance(score, float) and math.isnan(score):
            return Verdict(
                "fail",
                "Would block the merge: the answer couldn't be graded",
                "The judge found no checkable claims in the answer (score NaN). The gate counts "
                "that as a failure rather than silently skipping it.",
            )
        if score >= FAITHFULNESS_THRESHOLD:
            body = (
                "Every claim the judge checked is backed by the retrieved documents."
                if score >= 1.0
                else "Enough of the answer's claims are backed by the retrieved documents to clear the bar."
            )
            return Verdict("pass", f"Would pass the gate: grounded score {score:.2f} ≥ {FAITHFULNESS_THRESHOLD:.2f}", body)
        return Verdict(
            "fail",
            f"Would block the merge: grounded score {score:.2f} < {FAITHFULNESS_THRESHOLD:.2f}",
            "Some claims in this answer are not supported by the retrieved documents.",
        )

    return Verdict(
        "info",
        "Tracked, not gated: trick question",
        "This question has a false premise, is ambiguous, or pits two documents against each other. "
        "Whether the answer is right is a judgement call, so compare it with the reference answer.",
    )
