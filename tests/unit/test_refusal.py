import pytest

from src.eval.golden_dataset import load_golden_examples
from src.eval.refusal import is_refusal, must_decline


@pytest.mark.parametrize(
    "answer",
    [
        "I don't know.",
        "I don’t know.",  # curly apostrophe
        "I do not know the answer to that.",
        "The provided context does not contain this information.",
        "The context doesn't mention who won.",
        "That is not specified in the context.",
        "There is no information about that here.",
        "I cannot answer that from the context.",
        "This can't be determined from the given documents.",
        "The question is ambiguous - could you clarify what 'it' refers to?",
    ],
)
def test_declines_are_recognised(answer):
    assert is_refusal(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "Paris is the capital of France.",
        "FastAPI is a modern Python web framework for building APIs quickly with type hints.",
        "World War II ended in 1945.",
        "The Great Wall of China is larger overall, stretching over 13,000 miles.",
    ],
)
def test_substantive_answers_are_not_mistaken_for_declines(answer):
    assert not is_refusal(answer)


def test_must_decline_selects_only_adversarial_questions_with_nothing_to_retrieve():
    examples = load_golden_examples()
    selected = [e for e in examples if must_decline(e)]

    assert selected, "expected at least one out-of-corpus adversarial example"
    assert all(e["category"] == "adversarial" and e["expected_context_ids"] == [] for e in selected)
    # Adversarial traps that DO have relevant documents are judgement calls, not declines.
    assert not must_decline({"category": "adversarial", "expected_context_ids": ["doc-4", "doc-5"]})
    assert not must_decline({"category": "typical", "expected_context_ids": []})
