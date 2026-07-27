import pytest

from src.eval.results_store import RunResult, append_run_result, load_all_run_results


def _make_result(run_id: str, timestamp: str, passed: bool = True) -> RunResult:
    return RunResult(
        run_id=run_id,
        timestamp=timestamp,
        dataset_version="v1",
        prompt_version="v1",
        judge_model="gpt-4o-mini-2024-07-18",
        scores={"faithfulness_mean": 0.97},
        passed=passed,
        failing_examples=[],
    )


def test_load_all_run_results_empty_when_no_results_dir(tmp_path):
    assert load_all_run_results(results_dir=tmp_path / "does-not-exist") == []


def test_append_then_load_round_trips_all_fields(tmp_path):
    result = RunResult(
        run_id="run-001",
        timestamp="2026-07-27T10:00:00Z",
        dataset_version="v1",
        prompt_version="v1",
        judge_model="gpt-4o-mini-2024-07-18",
        scores={"faithfulness_mean": 0.97, "context_precision_mean": 0.95},
        passed=True,
        failing_examples=[{"id": "typical-010", "reason": "below threshold"}],
        cost_usd=0.42,
        latency_ms={"p50": 800, "p95": 1200},
    )
    append_run_result(result, results_dir=tmp_path)

    loaded = load_all_run_results(results_dir=tmp_path)

    assert loaded == [result]


def test_json_on_disk_uses_pass_not_passed(tmp_path):
    append_run_result(_make_result("run-001", "2026-07-27T10:00:00Z"), results_dir=tmp_path)

    raw = (tmp_path / "run-001.json").read_text()
    import json

    data = json.loads(raw)
    assert "pass" in data
    assert "passed" not in data


def test_append_refuses_to_overwrite_existing_run_id(tmp_path):
    append_run_result(_make_result("run-001", "2026-07-27T10:00:00Z"), results_dir=tmp_path)

    with pytest.raises(FileExistsError):
        append_run_result(_make_result("run-001", "2026-07-27T11:00:00Z"), results_dir=tmp_path)


def test_load_all_run_results_sorted_oldest_to_newest(tmp_path):
    append_run_result(_make_result("run-b", "2026-07-27T12:00:00Z"), results_dir=tmp_path)
    append_run_result(_make_result("run-a", "2026-07-27T09:00:00Z"), results_dir=tmp_path)
    append_run_result(_make_result("run-c", "2026-07-27T15:00:00Z"), results_dir=tmp_path)

    loaded = load_all_run_results(results_dir=tmp_path)

    assert [r.run_id for r in loaded] == ["run-a", "run-b", "run-c"]
