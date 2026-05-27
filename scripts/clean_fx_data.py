from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd

from config.constants import MIN_BARS_PER_DAY, PAIRS
from scripts._common import ensure_dir, save_csv, load_pair_parquet

DATA_DIR = PROJECT_DIR / "data"
PARQUET_DIR = DATA_DIR / "parquet"

CLEAN_ROOT_DIR = DATA_DIR / "processed"
CLEANED_DIR = CLEAN_ROOT_DIR / "cleaned"
CLEAN_REPORTS_DIR = CLEAN_ROOT_DIR / "reports"

CORE_COLUMNS = [
    "timestamp_est",
    "timestamp_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "pair",
    "year",
    "month",
    "session",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean FX parquet datasets using deterministic structural rules."
    )
    parser.add_argument(
        "--min-obs-day",
        type=int,
        default=MIN_BARS_PER_DAY,
        help="Drop all rows from days with fewer than this many observations.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute outputs even if they already exist.",
    )
    return parser.parse_args()


def load_canonical_pair(pair: str) -> pd.DataFrame:
    return load_pair_parquet(
        pair,
        PARQUET_DIR,
        required_columns=CORE_COLUMNS,
        parse_est=True,
        sort=False,
    )


def invalid_ohlc_mask(df: pd.DataFrame) -> pd.Series:
    return (
        (df["high"] < df["low"]) |
        (df["open"] > df["high"]) |
        (df["open"] < df["low"]) |
        (df["close"] > df["high"]) |
        (df["close"] < df["low"])
    )


def compute_daily_counts(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    temp["date"] = temp["timestamp_utc"].dt.date
    out = (
        temp.groupby("date", as_index=False)
        .size()
        .rename(columns={"size": "observations"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    return out


def build_cleaning_summary(
    pair: str,
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    dropped_days_df: pd.DataFrame,
    min_obs_day: int,
) -> pd.DataFrame:
    before_rows = len(before_df)
    after_rows = len(after_df)

    missing_ohlc_before = int(before_df[["open", "high", "low", "close"]].isna().any(axis=1).sum())
    duplicate_ts_before = int(before_df["timestamp_utc"].duplicated().sum())
    invalid_ohlc_before = int(invalid_ohlc_mask(before_df).sum())
    all_zero_volume = bool((before_df["volume"] == 0).all())

    dropped_days = len(dropped_days_df)
    dropped_rows_low_coverage = int(dropped_days_df["observations"].sum()) if not dropped_days_df.empty else 0

    summary = {
        "pair": pair,
        "rows_before": before_rows,
        "rows_after": after_rows,
        "rows_removed_total": before_rows - after_rows,
        "pct_rows_removed_total": (before_rows - after_rows) / before_rows if before_rows else 0.0,
        "missing_ohlc_rows_before": missing_ohlc_before,
        "duplicate_timestamps_before": duplicate_ts_before,
        "invalid_ohlc_rows_before": invalid_ohlc_before,
        "days_dropped_low_coverage": dropped_days,
        "rows_removed_low_coverage_days": dropped_rows_low_coverage,
        "min_obs_day_threshold": min_obs_day,
        "volume_all_zero": all_zero_volume,
        "start_utc_before": before_df["timestamp_utc"].min(),
        "end_utc_before": before_df["timestamp_utc"].max(),
        "start_utc_after": after_df["timestamp_utc"].min() if not after_df.empty else pd.NaT,
        "end_utc_after": after_df["timestamp_utc"].max() if not after_df.empty else pd.NaT,
    }
    return pd.DataFrame([summary])


def clean_pair(df: pd.DataFrame, min_obs_day: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply structural cleaning rules in order."""
    out = df.copy()

    out = out.sort_values("timestamp_utc").reset_index(drop=True)
    out = out.dropna(subset=["timestamp_utc", "open", "high", "low", "close"]).copy()
    out = out.drop_duplicates(subset=["timestamp_utc"], keep="first").copy()
    out = out.loc[~invalid_ohlc_mask(out)].copy()

    daily_counts = compute_daily_counts(out)
    dropped_days_df = daily_counts.loc[daily_counts["observations"] < min_obs_day].copy()

    if not dropped_days_df.empty:
        bad_days = set(dropped_days_df["date"].tolist())
        out["date"] = out["timestamp_utc"].dt.date
        out = out.loc[~out["date"].isin(bad_days)].copy()
        out = out.drop(columns=["date"])

    out = out.sort_values("timestamp_utc").reset_index(drop=True)
    return out, dropped_days_df


def process_pair(pair: str, min_obs_day: int, force: bool) -> None:
    cleaned_path = CLEANED_DIR / f"{pair}_2015_2025_clean.parquet"
    report_dir = ensure_dir(CLEAN_REPORTS_DIR / pair)
    summary_path = report_dir / "cleaning_summary.csv"
    dropped_days_path = report_dir / "dropped_days.csv"

    if cleaned_path.exists() and summary_path.exists() and dropped_days_path.exists() and not force:
        return

    before_df = load_canonical_pair(pair)
    after_df, dropped_days_df = clean_pair(before_df, min_obs_day=min_obs_day)
    summary_df = build_cleaning_summary(
        pair=pair,
        before_df=before_df,
        after_df=after_df,
        dropped_days_df=dropped_days_df,
        min_obs_day=min_obs_day,
    )

    ensure_dir(CLEANED_DIR)
    after_df.to_parquet(cleaned_path, index=False)
    save_csv(summary_df, summary_path)
    save_csv(dropped_days_df, dropped_days_path)


def main() -> None:
    args = parse_args()

    ensure_dir(CLEANED_DIR)
    ensure_dir(CLEAN_REPORTS_DIR)

    for pair in PAIRS:
        process_pair(pair, min_obs_day=args.min_obs_day, force=args.force)

    print("Cleaning complete.")
    print(f"Processed pairs: {', '.join(PAIRS)}")
    print(f"Cleaned datasets saved in: {CLEANED_DIR}")
    print(f"Cleaning reports saved in: {CLEAN_REPORTS_DIR}")


if __name__ == "__main__":
    main()
