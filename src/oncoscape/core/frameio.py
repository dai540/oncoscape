from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_frame(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path, index=False)
        return path
    except Exception:
        fallback = Path(f"{path}.csv")
        frame.to_csv(fallback, index=False)
        return fallback


def read_frame(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    fallback = Path(f"{path}.csv")
    if fallback.exists():
        return pd.read_csv(fallback)
    raise FileNotFoundError(f"could not read dataframe from {path} or fallback {fallback}")
