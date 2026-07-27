import os

# src.api.main now constructs Settings() at import time (Day 6 / Step 4 - CORS needs
# dashboard_origin before the app can serve a single request), which requires
# AEG_API_KEY. Unit tests still make zero real LLM calls - the boundary is mocked, per
# CONTRIBUTING.md's contract - so a placeholder value here satisfies Settings'
# validation without a real key ever being needed or used.
os.environ.setdefault("AEG_API_KEY", "test-placeholder-key-unit-tests-only")
