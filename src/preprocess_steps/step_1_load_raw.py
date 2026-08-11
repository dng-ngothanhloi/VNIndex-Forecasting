from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_raw_data(raw_path: Path) -> pd.DataFrame:
    """Load raw market data from xlsx/xls/csv without altering records."""
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}")

    if raw_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(raw_path)
    elif raw_path.suffix.lower() == ".csv":
        df = pd.read_csv(raw_path, sep=None, engine="python", encoding="utf-8-sig")
    else:
        raise ValueError(f"Unsupported format: {raw_path.suffix}")

    df.columns = [str(c).strip() for c in df.columns]
    print(f"[LOAD] Raw shape: {df.shape} | Columns: {df.columns.tolist()}")
    return df
