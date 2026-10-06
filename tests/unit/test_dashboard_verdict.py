from dashboard.verdict import needs_judge, verdict_for

TYPICAL = {"id": "typical-001", "category": "typical", "expected_context_ids": ["doc-1"]}
MULTI_HOP = {"id": "multihop-001", "category": "multi_hop", "expected_context_ids": ["doc-4", "doc-5"]}
OUT_OF_SCOPE = {"id": "adversarial-001", "category": "adversarial", "expected_context_ids": []}
TRAP = {"id": "adversarial-017", "category": "adversarial", "expected_context_ids": ["doc-4", "doc-5"]}


def test_typical_question_passes_at_or_above_the_gate_threshold():
    assert verdict_for(TYPICAL, {"declined": False}, {"faithfulness": 1.0}).kind == "pass"
    assert verdict_for(TYPICAL, {"declined": False}, {"faithfulness": 0.5}).kind == "pass"


def test_typical_question_blocks_below_threshold_and_on_nan():
    assert verdict_for(TYPICAL, {"declined": False}, {"faithfulness": 0.4}).kind == "fail"
    assert verdict_for(TYPICAL, {"declined": False}, {"faithfulness": float("nan")}).kind == "fail"


def test_typical_question_has_no_verdict_until_judged():
    assert verdict_for(TYPICAL, {"declined": False}, None) is None


def test_out_of_scope_question_is_decided_by_the_decline_check_without_a_judge():
    assert not needs_judge(OUT_OF_SCOPE)
    assert verdict_for(OUT_OF_SCOPE, {"declined": True}, None).kind == "pass"
    assert verdict_for(OUT_OF_SCOPE, {"declined": False}, None).kind == "fail"


def test_multi_hop_is_decided_by_retrieval_before_the_judge_has_run():
    assert needs_judge(MULTI_HOP)  # still graded, for information
    assert verdict_for(MULTI_HOP, {"declined": True, "recall_at_k": 1.0}, None).kind == "pass"
    assert verdict_for(MULTI_HOP, {"declined": False, "recall_at_k": 0.5}, None).kind == "fail"


def test_ungated_trick_questions_are_informational_not_pass_fail():
    assert needs_judge(TRAP)
    assert verdict_for(TRAP, {"declined": False, "recall_at_k": 1.0}, {"faithfulness": 1.0}).kind == "info"
