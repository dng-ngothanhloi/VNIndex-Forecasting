from __future__ import annotations

import os
import random
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from .common import find_project_root


# --- from step_01_setup.py ---
def run_setup(context: dict) -> dict:
    warnings.filterwarnings("ignore")
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 140)

    seed = 42
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    context.update({"seed": seed})

    # ===== TẠO THƯ MỤC logs/figures/ardl/ (ADAPTIVE PATH) =====
    # Tìm project root từ context hoặc relative path
    # src/forecasting/ardl/setup.py -> parents[3] is the project root.
    project_root = context.get("PROJECT_ROOT", Path(__file__).resolve().parents[3])
    figures_dir = project_root / "logs" / "figures" / "ardl"
    figures_dir.mkdir(parents=True, exist_ok=True)
    context["figures_dir"] = figures_dir
    print(f"[INFO] Figures directory: {figures_dir.resolve()}")
    # ===== KẾT THÚC =====

    print("ARDL step 1: setup complete")
    return context


# --- from step_02_find_project_root.py ---
def run_find_project_root(context: dict) -> dict:
    project_root = find_project_root()
    context["PROJECT_ROOT"] = project_root
    context["PCA_DIR"] = project_root / "data/processed/pca"
    context["CORE_DIR"] = project_root / "data/processed/core"
    print("ARDL step 2: project root ->", project_root)
    return context
