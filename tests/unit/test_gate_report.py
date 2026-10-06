import json
import math

from src.eval.gate_report import GateReport, GateRow
from src.eval.results_store import RunResult


def _row(example_id="typical-001", passed=True, score=1.0, error=None, check="faithfulness"):
    return GateRow(
        example_id=example_id, category="typical", check=check, passed=passed,
        score=score, threshold=0.5, detail="faithfulness=1.000", error=error,
    )


def test_all_passing_rows_produce_a_passed_report():
    report = GateReport()
    report.record(_row())
    report.record(_row("typical-002"))

    assert report.passed
    assert report.headline().startswith("PASSED")


def test_a_failing_row_blocks_and_is_listed_first_in_the_markdown():
    report = GateReport()
    report.record(_row("typical-001"))
    report.record(_row("typical-002", passed=False, score=0.4))

    assert not report.passed
    assert report.headline().startswith("BLOCKED - 1/2")
    table_rows = [line for line in report.to_markdown().splitlines() if line.startswith("| ")][1:]
    assert "typical-002" in table_rows[0]


def test_errors_are_reported_as_configuration_problems_not_regressions():
    report = GateReport()
    report.record(_row(passed=False, score=None, error="AuthenticationError: 401"))

    assert not report.passed
    assert report.errors and not report.failures
    assert "not an answer-quality regression" in report.headline()


def test_empty_report_never_passes():
    # A gate that ran zero examples (e.g. a broken filter) must not show green.
    assert not GateReport().passed


def test_run_result_round_trips_through_results_store_format(tmp_path):
    report = GateReport()
    report.record(_row(score=1.0))
    report.record(_row("typical-002", score=float("nan"), passed=False))
    report.record(_row("adversarial-001", check="declines", score=None))

    summary = tmp_path / "step_summary.md"
    report.write(tmp_path / "out", run_id="ci-123", judge_model="judge-x", step_summary=summary)

    data = json.loads((tmp_path / "out" / "ci-123.json").read_text())
    result = RunResult.from_dict(data)
    assert result.passed is False
    assert result.scores["faithfulness_mean"] == 1.0  # NaN excluded from the mean
    assert result.scores["decline_rate"] == 1.0
    assert [f["id"] for f in result.failing_examples] == ["typical-002"]
    assert "AgentEvalGate" in summary.read_text()
    assert not math.isnan(result.scores["faithfulness_mean"])


def test_cost_and_retrieval_rate_flow_into_the_run_result():
    report = GateReport()
    report.record(_row(score=1.0))
    report.record(GateRow(
        example_id="multihop-001", category="multi_hop", check="retrieval", passed=False,
        score=0.5, threshold=1.0, detail="missing ['doc-5']", cost_usd=0.0001,
    ))
    report.record(GateRow(
        example_id="multihop-002", category="multi_hop", check="retrieval", passed=True,
        score=1.0, threshold=1.0, cost_usd=0.0002,
    ))

    result = report.to_run_result("ci-1", "judge-x")
    assert result.cost_usd == round(0.0003, 6)
    assert result.scores["retrieval_pass_rate"] == 0.5
    assert result.failing_examples[0]["reason"].startswith("recall@k 0.50 < 1.0")
    assert "cost of this run: $0.0003" in report.to_markdown()
