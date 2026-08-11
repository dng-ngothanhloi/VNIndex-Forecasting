"""
conftest.py – shared pytest path setup for the tests/ package
==================================================================
Ensures both PROJECT_ROOT (so `helpers.*` imports resolve) and
PROJECT_ROOT/src (so `preprocessing.*` and `preprocess` imports resolve)
are importable, mirroring the sys.path convention already used by
src/preprocess.py and src/preprocess_steps/__init__.py.

No new pytest/hypothesis dependencies or configuration are introduced
here (Requirement 8.8, Requirement 9.5) — this is only path setup so the
new test files under tests/preprocessing/ and tests/ can import the
production package.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

for _path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
