from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_RESULTS_DIR = Path("results")


@dataclass(frozen=True)
class RunResult:
    """One CI eval-gate run, matching the project plan's Run Result Schema
    (section 6.2) - a lightweight experiment-tracking record, not a full metrics
    dump. `scores`/`latency_ms` are left as plain dicts deliberately: their exact
    shape is the Eval Layer's concern, not the storage layer's.

    The Python attribute is `passed`, not `pass` (a reserved word) - serialized
    as "pass" in JSON to match the documented schema exactly.
    """

    run_id: str
    timestamp: str  # ISO 8601
    dataset_version: str
    prompt_version: str
    judge_model: str
    scores: dict
    passed: bool
    failing_examples: list[dict] = field(default_factory=list)
    cost_usd: float = 0.0
    latency_ms: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["pass"] = data.pop("passed")
        return data

    @classmethod
    def from_dict(cls, data: dict) -> RunResult:
        data = dict(data)
        data["passed"] = data.pop("pass")
        return cls(**data)


def append_run_result(result: RunResult, results_dir: Path = DEFAULT_RESULTS_DIR) -> Path:
    """Write one immutable JSON file for this run - results/<run_id>.json.

    Append-only: refuses to overwrite an existing run_id's file, so a recorded
    run can't silently be mutated after the fact - the whole point of a log.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{result.run_id}.json"
    if path.exists():
        raise FileExistsError(f"Run result already exists: {path} (run_id must be unique)")
    path.write_text(json.dumps(result.to_dict(), indent=2))
    return path


def load_all_run_results(results_dir: Path = DEFAULT_RESULTS_DIR) -> list[RunResult]:
    """Load every recorded run result, oldest to newest by timestamp.

    Non-recursive glob deliberately: results_dir may hold other kinds of records
    in subdirectories (e.g. judge_variance/summary.json) that aren't RunResults
    and must never be picked up here.
    """
    if not results_dir.exists():
        return []
    results = [
        RunResult.from_dict(json.loads(path.read_text())) for path in sorted(results_dir.glob("*.json"))
    ]
    return sorted(results, key=lambda r: r.timestamp)


DEFAULT_VARIANCE_SUMMARY_PATH = DEFAULT_RESULTS_DIR / "judge_variance" / "summary.json"


def load_judge_variance_summary(path: Path = DEFAULT_VARIANCE_SUMMARY_PATH) -> dict | None:
    """Load the committed judge-variance baseline (Day 2 / Step 5's real measurement,
    promoted to a small summary file - see internal/mentoring_notes.md, Day 4 / Step 3).

    Returns None if it doesn't exist rather than raising - a dashboard panel reading
    this should degrade to "not measured yet", not crash the whole page.
    """
    if not path.exists():
        return None
    return json.loads(path.read_text())
