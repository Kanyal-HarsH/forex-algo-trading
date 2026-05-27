from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd

from config.constants import HORIZON_PRIMARY, HORIZON_SECONDARY, PAIRS
from scripts._common import ensure_dir, save_csv, load_pair_parquet

FEATURE_ROOT_DIR = PROJECT_DIR / "features"
FEATURE_PAIR_DIR = FEATURE_ROOT_DIR / "pair"

LABEL_ROOT_DIR = PROJECT_DIR / "labels"
LABEL_PAIR_DIR = LABEL_ROOT_DIR / "pair"
LABEL_REPORTS_DIR = LABEL_ROOT_DIR / "reports"

REQUIRED_COLUMNS = [
    "timestamp_utc",
    "close",
    "pair",
    "session",
]

DEFAULT_THRESHOLD_PRIMARY = 0.0005
DEFAULT_THRESHOLD_SECONDARY = 0.0010


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build labeled FX datasets from feature-engineered pair data."
    )
    parser.add_argument(
        "--horizon-primary",
        type=int,
        default=HORIZON_PRIMARY,
        help="Primary forward return horizon in rows/minutes.",
    )
    parser.add_argument(
        "--horizon-secondary",
        type=int,
        default=HORIZON_SECONDARY,
        help="Secondary forward return horizon in rows/minutes.",
    )
    parser.add_argument(
        "--threshold-primary",
        type=float,
        default=DEFAULT_THRESHOLD_PRIMARY,
        help="Absolute return threshold for the primary 3-class label.",
    )
    parser.add_argument(
        "--threshold-secondary",
        type=float,
        default=DEFAULT_THRESHOLD_SECONDARY,
        help="Absolute return threshold for the secondary 3-class label.",
    )
    parser.add_argument(
        "--keep-tail",
        action="store_true",
        help="Keep unlabeled tail rows created by forward shifts.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute outputs even if they already exist.",
    )
    return parser.parse_args()


def load_feature_pair(pair: str) -> pd.DataFrame:
    return load_pair_parquet(
        pair,
        FEATURE_PAIR_DIR,
        suffix="features",
        required_columns=REQUIRED_COLUMNS,
    )


def compute_future_return(df: pd.DataFrame, horizon: int) -> pd.Series:
    return df["close"].shift(-horizon) / df["close"] - 1.0


def make_3class_label(future_ret: pd.Series, threshold: float) -> pd.Series:
    """Map forward returns to -1 (short), 0 (flat), 1 (long)."""
    label = pd.Series(0, index=future_ret.index, dtype="int8")
    label = label.mask(future_ret > threshold, 1)
    label = label.mask(future_ret < -threshold, -1)
    label = label.where(~future_ret.isna(), pd.NA)
    return label.astype("Int8")


def add_label_columns(
    df: pd.DataFrame,
    horizon_primary: int,
    horizon_secondary: int,
    threshold_primary: float,
    threshold_secondary: float,
) -> pd.DataFrame:
    out = df.copy()

    future_ret_primary_col = f"future_ret_{horizon_primary}"
    future_ret_secondary_col = f"future_ret_{horizon_secondary}"

    label_primary_col = f"label_3class_{horizon_primary}"
    label_secondary_col = f"label_3class_{horizon_secondary}"

    out[future_ret_primary_col] = compute_future_return(out, horizon_primary)
    out[future_ret_secondary_col] = compute_future_return(out, horizon_secondary)

    out[label_primary_col] = make_3class_label(out[future_ret_primary_col], threshold_primary)
    out[label_secondary_col] = make_3class_label(out[future_ret_secondary_col], threshold_secondary)

    out["label"] = out[label_primary_col]
    return out


def build_label_summary(
    pair: str,
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    horizon_primary: int,
    horizon_secondary: int,
    threshold_primary: float,
    threshold_secondary: float,
) -> pd.DataFrame:
    """Build a one-row record of label counts and parameters used."""
    label_primary_col = f"label_3class_{horizon_primary}"
    label_secondary_col = f"label_3class_{horizon_secondary}"

    primary_counts = df_after[label_primary_col].value_counts(dropna=False).to_dict()
    secondary_counts = df_after[label_secondary_col].value_counts(dropna=False).to_dict()

    summary = {
        "pair": pair,
        "rows_before": len(df_before),
        "rows_after": len(df_after),
        "rows_removed_tail": len(df_before) - len(df_after),
        "horizon_primary": horizon_primary,
        "horizon_secondary": horizon_secondary,
        "threshold_primary": threshold_primary,
        "threshold_secondary": threshold_secondary,
        "primary_long_count": int(primary_counts.get(1, 0)),
        "primary_no_trade_count": int(primary_counts.get(0, 0)),
        "primary_short_count": int(primary_counts.get(-1, 0)),
        "primary_nan_count": int(primary_counts.get(pd.NA, 0)) if pd.NA in primary_counts else 0,
        "secondary_long_count": int(secondary_counts.get(1, 0)),
        "secondary_no_trade_count": int(secondary_counts.get(0, 0)),
        "secondary_short_count": int(secondary_counts.get(-1, 0)),
        "secondary_nan_count": int(secondary_counts.get(pd.NA, 0)) if pd.NA in secondary_counts else 0,
    }
    return pd.DataFrame([summary])


def build_label_distribution_table(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    label_col = f"label_3class_{horizon}"
    out = (
        df[label_col]
        .value_counts(dropna=False)
        .rename_axis("label")
        .reset_index(name="count")
    )
    out["horizon"] = horizon
    return out[["horizon", "label", "count"]]


def build_session_label_distribution(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    label_col = f"label_3class_{horizon}"
    out = (
        df.groupby(["session", label_col], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .rename(columns={label_col: "label"})
    )
    out["horizon"] = horizon
    return out[["horizon", "session", "label", "count"]]


def process_pair(
    pair: str,
    horizon_primary: int,
    horizon_secondary: int,
    threshold_primary: float,
    threshold_secondary: float,
    keep_tail: bool,
    force: bool,
) -> None:
    output_path = LABEL_PAIR_DIR / f"{pair}_2015_2025_labeled.parquet"
    report_dir = ensure_dir(LABEL_REPORTS_DIR / pair)

    summary_path = report_dir / "label_summary.csv"
    dist_path = report_dir / "label_distribution.csv"
    session_dist_path = report_dir / "label_distribution_by_session.csv"

    if output_path.exists() and summary_path.exists() and dist_path.exists() and session_dist_path.exists() and not force:
        return

    before_df = load_feature_pair(pair)

    labeled_df = add_label_columns(
        df=before_df,
        horizon_primary=horizon_primary,
        horizon_secondary=horizon_secondary,
        threshold_primary=threshold_primary,
        threshold_secondary=threshold_secondary,
    )

    if not keep_tail:
        primary_col = f"label_3class_{horizon_primary}"
        secondary_col = f"label_3class_{horizon_secondary}"
        labeled_df = labeled_df.dropna(subset=[primary_col, secondary_col]).reset_index(drop=True)

    summary_df = build_label_summary(
        pair=pair,
        df_before=before_df,
        df_after=labeled_df,
        horizon_primary=horizon_primary,
        horizon_secondary=horizon_secondary,
        threshold_primary=threshold_primary,
        threshold_secondary=threshold_secondary,
    )

    dist_primary = build_label_distribution_table(labeled_df, horizon_primary)
    dist_secondary = build_label_distribution_table(labeled_df, horizon_secondary)
    dist_df = pd.concat([dist_primary, dist_secondary], ignore_index=True)

    session_dist_primary = build_session_label_distribution(labeled_df, horizon_primary)
    session_dist_secondary = build_session_label_distribution(labeled_df, horizon_secondary)
    session_dist_df = pd.concat([session_dist_primary, session_dist_secondary], ignore_index=True)

    ensure_dir(LABEL_PAIR_DIR)
    labeled_df.to_parquet(output_path, index=False)
    save_csv(summary_df, summary_path)
    save_csv(dist_df, dist_path)
    save_csv(session_dist_df, session_dist_path)


def main() -> None:
    args = parse_args()

    if args.horizon_primary <= 0 or args.horizon_secondary <= 0:
        raise ValueError("Horizons must be positive integers.")

    if args.threshold_primary < 0 or args.threshold_secondary < 0:
        raise ValueError("Thresholds must be non-negative.")

    ensure_dir(LABEL_PAIR_DIR)
    ensure_dir(LABEL_REPORTS_DIR)

    for pair in PAIRS:
        process_pair(
            pair=pair,
            horizon_primary=args.horizon_primary,
            horizon_secondary=args.horizon_secondary,
            threshold_primary=args.threshold_primary,
            threshold_secondary=args.threshold_secondary,
            keep_tail=args.keep_tail,
            force=args.force,
        )

    print("Label engineering complete.")
    print(f"Processed pairs: {', '.join(PAIRS)}")
    print(f"Labeled datasets saved in: {LABEL_PAIR_DIR}")
    print(f"Label reports saved in: {LABEL_REPORTS_DIR}")


if __name__ == "__main__":
    main()
