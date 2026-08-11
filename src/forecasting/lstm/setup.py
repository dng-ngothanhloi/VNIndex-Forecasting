from __future__ import annotations

import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


# --- from step_01_imports.py ---
def run_imports(context: dict) -> dict:
    """Imports and environment setup."""
    warnings.filterwarnings("ignore")
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 160)

    # Make the run as deterministic as TensorFlow allows.
    os.environ["PYTHONHASHSEED"] = "42"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(42)
    np.random.seed(42)
    try:
        tf.keras.utils.set_random_seed(42)
    except Exception:
        tf.random.set_seed(42)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    print("TensorFlow version:", tf.__version__)
    print("Deterministic ops:", os.environ.get("TF_DETERMINISTIC_OPS"))

    return context


# --- from step_02_paths.py ---
def _find_project_root() -> Path:
    """Find the project root that contains processed PCA files."""
    expected = Path("data/processed/pca/train_pca.csv")
    candidates = [
        Path.cwd(),
        Path.cwd().parent,
        Path.cwd().parent.parent,
    ]

    for root in candidates:
        if (root / expected).exists():
            return root

    if Path("/content").exists():
        for p in Path("/content").rglob("train_pca.csv"):
            if p.parent.name == "pca":
                root = p.parents[3]
                if (root / expected).exists():
                    return root

    raise FileNotFoundError(
        "Cannot find data/processed/pca/train_pca.csv. Put the notebook inside the project or mount the project folder in Colab."
    )


def run_paths(context: dict) -> dict:
    """Path configuration."""
    project_root = _find_project_root()
    pca_dir = project_root / "data/processed/pca"
    core_dir = project_root / "data/processed/core"

    print("PROJECT_ROOT:", project_root)
    print("PCA_DIR exists:", pca_dir.exists())
    print("CORE_DIR exists:", core_dir.exists())

    context.update({
        "PROJECT_ROOT": project_root,
        "PCA_DIR": pca_dir,
        "CORE_DIR": core_dir,
    })
    return context
