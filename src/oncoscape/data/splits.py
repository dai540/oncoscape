from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from oncoscape.core import ensure_directory


def _bucket(value: str, n_buckets: int = 100) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % n_buckets


def assign_deterministic_split(value: str) -> str:
    bucket = _bucket(value)
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def build_split_tables(slides: pd.DataFrame, out_dir: str | Path) -> dict[str, str]:
    out_dir = ensure_directory(out_dir)
    patient = slides[["patient_id"]].drop_duplicates().copy()
    patient["split"] = patient["patient_id"].astype(str).map(assign_deterministic_split)
    patient_path = out_dir / "patient_splits.csv"
    patient.to_csv(patient_path, index=False)

    source = slides[["source_type", "platform"]].drop_duplicates().copy()
    source["source_key"] = source["source_type"].astype(str) + "::" + source["platform"].astype(str)
    source["split"] = source["source_key"].astype(str).map(assign_deterministic_split)
    source_path = out_dir / "source_splits.csv"
    source.to_csv(source_path, index=False)

    return {
        "patient_splits_path": str(patient_path.resolve()),
        "source_splits_path": str(source_path.resolve()),
    }
