"""
test_pca_reducer_real_data_smoke.py – Real_Data_Smoke_Test
================================================================
Feature: pca-reducer-wrapper

Covers:
- Task 9.1: the second mandatory Parity_Gate fixture. Runs PCAReducer
  against the actual scaled data files under data/processed/splits/
  (train_scaled.csv, val_scaled.csv, test_scaled.csv), comparing legacy
  inline sklearn.decomposition.PCA output against PCAReducer output for
  numerical equivalence within rtol=1e-10, atol=1e-12.

Skips rather than fails when those files are absent from the execution
environment (Requirement 6.6).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for _p in (PROJECT_ROOT, SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from reduction.pca import PCAReducer  # noqa: E402

SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
TRAIN_SCALED_PATH = SPLITS_DIR / "train_scaled.csv"
VAL_SCALED_PATH = SPLITS_DIR / "val_scaled.csv"
TEST_SCALED_PATH = SPLITS_DIR / "test_scaled.csv"


def _real_scaled_splits_present() -> bool:
    return TRAIN_SCALED_PATH.exists() and VAL_SCALED_PATH.exists() and TEST_SCALED_PATH.exists()


@pytest.mark.skipif(
    not _real_scaled_splits_present(),
    reason=(
        "data/processed/splits/{train,val,test}_scaled.csv not present in this "
        "execution environment; Real_Data_Smoke_Test skips rather than fails "
        "(Requirement 6.6)."
    ),
)
def test_pca_reducer_matches_legacy_pca_on_real_scaled_data():
    train_scaled = pd.read_csv(TRAIN_SCALED_PATH, index_col=0, parse_dates=True)
    val_scaled = pd.read_csv(VAL_SCALED_PATH, index_col=0, parse_dates=True)
    test_scaled = pd.read_csv(TEST_SCALED_PATH, index_col=0, parse_dates=True)

    X_train = train_scaled.values
    X_val = val_scaled.values
    X_test = test_scaled.values

    random_state = 42

    # Legacy inline sklearn.decomposition.PCA usage, mirroring
    # PCA_Full_Call_Site (n_components=None) as today's pca_full does.
    legacy_full = PCA(n_components=None, random_state=random_state)
    legacy_full.fit(X_train)
    legacy_ev = legacy_full.explained_variance_ratio_
    legacy_cev = np.cumsum(legacy_ev)
    threshold = 0.95
    k_optimal = int(np.argmax(legacy_cev >= threshold)) + 1

    legacy_final = PCA(n_components=k_optimal, random_state=random_state)
    legacy_final.fit(X_train)
    legacy_train_pca = legacy_final.transform(X_train)
    legacy_val_pca = legacy_final.transform(X_val)
    legacy_test_pca = legacy_final.transform(X_test)

    # Wrapped PCAReducer usage, mirroring the post-integration
    # PCA_Full_Call_Site / PCA_Final_Call_Site.
    reducer_full = PCAReducer(n_components=None, random_state=random_state)
    reducer_full.fit(X_train)
    reducer_ev = reducer_full.get_metadata()["explained_variance_ratio"]
    reducer_cev = np.cumsum(reducer_ev)

    reducer_final = PCAReducer(n_components=k_optimal, random_state=random_state)
    reducer_final.fit(X_train)
    reducer_train_pca = reducer_final.transform(X_train)
    reducer_val_pca = reducer_final.transform(X_val)
    reducer_test_pca = reducer_final.transform(X_test)
    reducer_meta = reducer_final.get_metadata()

    assert reducer_full is not reducer_final

    np.testing.assert_allclose(reducer_ev, legacy_ev, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(reducer_cev, legacy_cev, rtol=1e-10, atol=1e-12)

    np.testing.assert_allclose(reducer_train_pca, legacy_train_pca, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(reducer_val_pca, legacy_val_pca, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(reducer_test_pca, legacy_test_pca, rtol=1e-10, atol=1e-12)

    np.testing.assert_allclose(
        reducer_meta["components_"], legacy_final.components_, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        reducer_meta["explained_variance_ratio"],
        legacy_final.explained_variance_ratio_,
        rtol=1e-10,
        atol=1e-12,
    )
