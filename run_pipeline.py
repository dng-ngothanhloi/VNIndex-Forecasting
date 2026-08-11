"""
run_pipeline.py – DEPRECATED backward-compatible shim
=========================================================
The canonical experiment orchestration entry point has moved to
experiments/run_experiment.py as part of the architecture-convergence
task (research code now lives under src/, experiments/). This file is
kept only so existing muscle-memory invocations of
`python run_pipeline.py ...` keep working; it does not contain any
orchestration logic of its own.

Prefer: python experiments/run_experiment.py [same arguments]
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_experiment import main

if __name__ == "__main__":
    main()
