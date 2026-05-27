from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


class SchemaValidationError(ValueError):
    pass


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)


def validate_required_columns(
    df: pd.DataFrame,
    required: Sequence[str],
    *,
    file_path: Path | str,
) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaValidationError(
            f"{file_path}: missing required columns: {missing}"
        )


def load_pair_parquet(
    pair: str,
    source_dir: Path,
    *,
    suffix: str = "",
    year_range: str = "2015_2025",
    required_columns: Sequence[str] | None = None,
    parse_est: bool = False,
    sort: bool = True,
) -> pd.DataFrame:
    name = f"{pair}_{year_range}_{suffix}.parquet" if suffix else f"{pair}_{year_range}.parquet"
    path = source_dir / name

    if not path.exists():
        raise FileNotFoundError(f"Missing parquet: {path}")

    df = pd.read_parquet(path)

    if required_columns is not None:
        validate_required_columns(df, required_columns, file_path=path)

    if df.empty:
        raise SchemaValidationError(f"{path}: parquet is empty")

    df = df.copy()
    if parse_est and "timestamp_est" in df.columns:
        df["timestamp_est"] = pd.to_datetime(df["timestamp_est"], errors="coerce")
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        if sort:
            df = df.sort_values("timestamp_utc").reset_index(drop=True)

    return df
