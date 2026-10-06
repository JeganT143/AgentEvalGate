"""Deterministic "did the model decline?" check for out-of-corpus questions.

Faithfulness can't grade a correct refusal: "I don't know" contains no checkable
claims, so RAGAS scores it ~0 or NaN - exactly what tests/eval/test_faithfulness.py's
threshold comment says about the `adversarial` category. A refusal is a behaviour, not
a graded quality, so it gets a plain pattern check instead of a second LLM judge: free,
instant, and identical on every run (no judge variance to measure).
"""

from __future__ import annotations

import re

_REFUSAL_PATTERNS = re.compile(
    r"|".join(
        [
            r"\bi (?:do not|don't|dont) know\b",
            r"\b(?:does not|doesn't|do not|don't) (?:contain|include|provide|mention|specify|say|state)\b",
            r"\bnot (?:provided|mentioned|specified|stated|included|available|given)\b",
            r"\bno (?:information|mention|details?|data)\b",
            r"\b(?:cannot|can't|can not|unable to) (?:be )?(?:answer|determine|tell|say)",
            r"\bnot enough (?:information|context)\b",
            r"\b(?:unclear|ambiguous)\b",
            r"\b(?:please |could you )?clarify\b",
        ]
    ),
    re.IGNORECASE,
)


def is_refusal(answer: str) -> bool:
    """True if the answer declines to answer (or asks for clarification)."""
    # Normalise curly apostrophes - models emit "don’t" as often as "don't".
    return bool(_REFUSAL_PATTERNS.search(answer.replace("’", "'")))


def must_decline(example: dict) -> bool:
    """True for golden examples whose only correct behaviour is to decline.

    An adversarial question with no relevant document in the corpus
    (expected_context_ids == []) cannot be answered from the context, so any substantive
    answer is a hallucination by definition. Adversarial examples that DO have relevant
    documents (false premises, conflicting-context traps) need a judgement call about what
    the right answer is, so they are left out of this check rather than guessed at.
    """
    return example["category"] == "adversarial" and not example["expected_context_ids"]
