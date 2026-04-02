from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    name = path.name.lower()
    suffix = path.suffix.lower()
    compression = "gzip" if name.endswith(".gz") else None
    base_name = name[:-3] if name.endswith(".gz") else name
    if base_name.endswith(".csv"):
        return pd.read_csv(path, compression=compression)
    if base_name.endswith(".tsv") or base_name.endswith(".txt"):
        return pd.read_csv(path, sep="\t", compression=compression)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported table format: {path}")
