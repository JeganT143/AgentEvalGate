"""Eval-gate session plumbing: one shared GateReport, written out when the run ends.

Every gated test records a row (see src/eval/gate_report.py). At session end the report
is printed in the terminal summary, appended to $GITHUB_STEP_SUMMARY when running in
GitHub Actions (so a red check shows a per-question table on the run page), and written
to $AEG_GATE_REPORT_DIR (default .cache/gate-report/) for CI to upload as an artifact.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from src.eval.gate_report import GateReport

_REPORT = GateReport()


@pytest.fixture(scope="session")
def gate_report() -> GateReport:
    return _REPORT


def pytest_sessionfinish(session, exitstatus) -> None:
    if not _REPORT.rows:
        return
    from src.config import get_settings

    run_id = os.environ.get("GITHUB_RUN_ID")
    run_id = f"ci-{run_id}" if run_id else time.strftime("local-%Y%m%d-%H%M%S", time.gmtime())
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    _REPORT.write(
        out_dir=Path(os.environ.get("AEG_GATE_REPORT_DIR", ".cache/gate-report")),
        run_id=run_id,
        judge_model=get_settings().judge_model,
        step_summary=Path(summary) if summary else None,
    )


def pytest_terminal_summary(terminalreporter) -> None:
    if _REPORT.rows:
        terminalreporter.section("AgentEvalGate")
        terminalreporter.write_line(_REPORT.headline())
