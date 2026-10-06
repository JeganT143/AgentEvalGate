"""Collects one row per gated golden example during an eval-gate run, and renders it.

Without this, a red gate only says "pytest failed" - finding out *which* question
regressed, by how much, and whether it was a quality drop or a broken API key meant
reading thousands of lines of tracebacks. The eval tests record into a GateReport;
tests/eval/conftest.py writes it to the GitHub Actions job summary (a table on the
run's page) and to JSON files CI uploads as an artifact - one of them already in
results_store's RunResult format, so a run can be dropped into results/ as-is.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.eval.results_store import RunResult


@dataclass(frozen=True)
class GateRow:
    example_id: str
    category: str
    check: str  # "faithfulness" | "declines" | "retrieval"
    passed: bool
    score: float | None = None
    threshold: float | None = None
    detail: str = ""
    # Informational only (never gated) - carried so a run's average can be recorded.
    context_precision: float | None = None
    # Real metered OpenAI spend for this example (src/rag/usage.py), judge included.
    cost_usd: float = 0.0
    # Set when the example couldn't be scored at all (auth failure, network error) - a
    # configuration problem, reported separately so it's never mistaken for a regression.
    error: str | None = None


@dataclass
class GateReport:
    rows: list[GateRow] = field(default_factory=list)

    def record(self, row: GateRow) -> None:
        self.rows.append(row)

    @property
    def errors(self) -> list[GateRow]:
        return [r for r in self.rows if r.error]

    @property
    def failures(self) -> list[GateRow]:
        return [r for r in self.rows if not r.passed and not r.error]

    @property
    def passed(self) -> bool:
        return bool(self.rows) and not self.failures and not self.errors

    @property
    def cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.rows)

    def headline(self) -> str:
        if not self.rows:
            return "No gated examples ran."
        if self.errors:
            first = self.errors[0].error
            return (
                f"ERROR - {len(self.errors)}/{len(self.rows)} examples could not be scored "
                f"(first error: {first}). This is a configuration/infrastructure problem, "
                "not an answer-quality regression."
            )
        if self.failures:
            return f"BLOCKED - {len(self.failures)}/{len(self.rows)} gated examples fell below the bar."
        return f"PASSED - all {len(self.rows)} gated examples met the bar."

    def to_markdown(self) -> str:
        icon = "✅" if self.passed else ("⚠️" if self.errors else "❌")
        lines = [f"## {icon} AgentEvalGate", "", f"**{self.headline()}**", ""]

        by_check: dict[str, list[GateRow]] = {}
        for row in self.rows:
            by_check.setdefault(row.check, []).append(row)
        for check, rows in by_check.items():
            ok = sum(r.passed for r in rows)
            lines.append(f"- `{check}`: {ok}/{len(rows)} passed")
        lines.append(f"- cost of this run: ${self.cost_usd:.4f} (metered OpenAI usage, judge included)")
        lines.append("")

        if self.rows:
            lines += [
                "| Result | Example | Category | Check | Score | Bar | Detail |",
                "|---|---|---|---|---|---|---|",
            ]
            # Failures and errors first - that's what someone opening a red run is looking for.
            for row in sorted(self.rows, key=lambda r: (r.passed, r.example_id)):
                status = "⚠️ error" if row.error else ("✅ pass" if row.passed else "❌ fail")
                detail = (row.error or row.detail).replace("|", "\\|").replace("\n", " ")
                lines.append(
                    f"| {status} | `{row.example_id}` | {row.category} | {row.check} | "
                    f"{_fmt(row.score)} | {_fmt(row.threshold)} | {detail[:160]} |"
                )
        return "\n".join(lines) + "\n"

    def to_json(self, run_id: str) -> str:
        return json.dumps(
            {"run_id": run_id, "headline": self.headline(), "rows": [asdict(r) for r in self.rows]}, indent=2
        )

    def to_run_result(
        self, run_id: str, judge_model: str, dataset_version: str = "v1", prompt_version: str = "current"
    ) -> RunResult:
        faithfulness = [
            r.score for r in self.rows
            if r.check == "faithfulness" and r.score is not None and not math.isnan(r.score)
        ]
        context_precision = [
            r.context_precision for r in self.rows
            if r.context_precision is not None and not math.isnan(r.context_precision)
        ]
        declines = [r for r in self.rows if r.check == "declines"]
        retrieval = [r for r in self.rows if r.check == "retrieval"]
        scores: dict = {}
        if faithfulness:
            scores["faithfulness_mean"] = round(statistics.mean(faithfulness), 4)
        if context_precision:
            scores["context_precision_mean"] = round(statistics.mean(context_precision), 4)
        if declines:
            scores["decline_rate"] = round(sum(r.passed for r in declines) / len(declines), 4)
        if retrieval:
            scores["retrieval_pass_rate"] = round(sum(r.passed for r in retrieval) / len(retrieval), 4)
        return RunResult(
            run_id=run_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            dataset_version=dataset_version,
            prompt_version=prompt_version,
            judge_model=judge_model,
            scores=scores,
            passed=self.passed,
            failing_examples=[
                {"id": r.example_id, "category": r.category, "reason": _reason(r)}
                for r in self.rows if not r.passed
            ],
            cost_usd=round(self.cost_usd, 6),
        )

    def write(self, out_dir: Path, run_id: str, judge_model: str, step_summary: Path | None = None) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "gate_report.json").write_text(self.to_json(run_id))
        (out_dir / f"{run_id}.json").write_text(
            json.dumps(self.to_run_result(run_id, judge_model).to_dict(), indent=2)
        )
        (out_dir / "summary.md").write_text(self.to_markdown())
        if step_summary is not None:
            with open(step_summary, "a") as f:
                f.write(self.to_markdown())


def _reason(row: GateRow) -> str:
    if row.error:
        return row.error
    if row.check == "faithfulness":
        return f"faithfulness {_fmt(row.score)} < threshold {row.threshold}; {row.detail}"
    if row.check == "declines":
        return f"should have declined; {row.detail}"
    if row.check == "retrieval":
        return f"recall@k {_fmt(row.score)} < {row.threshold}; {row.detail}"
    return row.detail


def _fmt(value: float | None) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return f"{value:.2f}"
