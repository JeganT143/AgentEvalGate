from __future__ import annotations

import json
from pathlib import Path

DEFAULT_GOLDEN_PATHS: tuple[Path, ...] = (
    Path("data/golden/v0.jsonl"),
    Path("data/golden/v1.jsonl"),
    Path("data/golden/adversarial.jsonl"),
)


def load_golden_examples(paths: tuple[Path, ...] = DEFAULT_GOLDEN_PATHS) -> list[dict]:
    """Load every golden example across all dataset files, in on-disk line order.

    One shared loader so callers needing different slices (tests/eval/test_faithfulness.py's
    typical-only filter, the dashboard's full-dataset dropdown) filter this output themselves
    instead of duplicating JSONL-parsing logic - see internal/mentoring_notes.md, Day 8 / Step 1.
    """
    rows: list[dict] = []
    for path in paths:
        with open(path) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows
