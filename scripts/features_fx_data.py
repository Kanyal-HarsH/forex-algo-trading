from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config.constants import VOL_HIGH_REGIME_PERCENTILE, VOL_REGIME_WINDOW

PROJECT_DIR = Path(__file__).resolve().parent.parent

CLEAN_ROOT_DIR = PROJECT_DIR / "data" / "processed"
CLEANED_DIR = CLEAN_ROOT_DIR / "cleaned"

FEATURE_ROOT_DIR = PROJECT_DIR / "features"
FEATURE_PAIR_DIR = FEATURE_ROOT_DIR / "pair"
FEATURE_REPORTS_DIR = FEATURE_ROOT_DIR / "reports"

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]

BASE_COLUMNS = [
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

RET_WINDOWS = [1, 5, 15]
VOL_WINDOWS = [10, 30, 60]
MA_WINDOWS = [10, 30, 60, 120]
MOM_WINDOWS = [5, 15, 30]
RANGE_MA_WINDOWS = [10, 30]

OVERLAP_START_HOUR = 13
OVERLAP_END_HOUR = 16


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build feature-engineered FX datasets from cleaned pair data."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute outputs even if they already exist.",
    )
    parser.add_argument(
        "--drop-warmup",
        action="store_true",
        help="Drop rows with missing rolling-feature values.",
    )
    return parser.parse_args()


def load_clean_pair(pair: str) -> pd.DataFrame:
    """Load and validate one cleaned pair parquet."""
    path = CLEANED_DIR / f"{pair}_2015_2025_clean.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing cleaned parquet: {path}")

    df = pd.read_parquet(path)
    missing = [col for col in BASE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{pair}: missing required columns: {missing}")

    df = df.copy()
    df["timestamp_est"] = pd.to_datetime(df["timestamp_est"], errors="coerce")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["timestamp_utc"].dt.hour
    out["day_of_week"] = out["timestamp_utc"].dt.dayofweek
    out["is_overlap_session"] = (
        (out["hour"] >= OVERLAP_START_HOUR) & (out["hour"] < OVERLAP_END_HOUR)
    ).astype("int8")
    return out


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_ret_1"] = np.log(out["close"]).diff()

    for window in RET_WINDOWS:
        out[f"ret_{window}"] = out["close"].pct_change(window)

    out["abs_ret_1"] = out["ret_1"].abs()
    return out


def add_range_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["range"] = out["high"] - out["low"]
    out["range_pct"] = np.where(out["close"] != 0, out["range"] / out["close"], np.nan)

    for window in RANGE_MA_WINDOWS:
        out[f"range_ma_{window}"] = out["range"].rolling(window=window, min_periods=window).mean()

    return out


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for window in VOL_WINDOWS:
        out[f"rv_{window}"] = out["ret_1"].rolling(window=window, min_periods=window).std()

    out["rv_ratio_10_60"] = np.where(out["rv_60"] != 0, out["rv_10"] / out["rv_60"], np.nan)
    return out


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for window in MA_WINDOWS:
        out[f"sma_{window}"] = out["close"].rolling(window=window, min_periods=window).mean()

    out["price_to_sma_30"] = np.where(out["sma_30"] != 0, out["close"] / out["sma_30"] - 1.0, np.nan)
    out["price_to_sma_60"] = np.where(out["sma_60"] != 0, out["close"] / out["sma_60"] - 1.0, np.nan)

    for window in MOM_WINDOWS:
        out[f"mom_{window}"] = out["close"] / out["close"].shift(window) - 1.0

    return out


def add_volatility_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    vol_col = f"rv_{VOL_REGIME_WINDOW}"
    threshold_col = f"{vol_col}_q{int(VOL_HIGH_REGIME_PERCENTILE * 100)}"

    out[threshold_col] = (
        out[vol_col]
        .expanding(min_periods=VOL_REGIME_WINDOW)
        .quantile(VOL_HIGH_REGIME_PERCENTILE)
    )
    out["volatility_regime_high"] = (out[vol_col] > out[threshold_col]).astype("int8")
    out["volatility_regime_code"] = out["volatility_regime_high"]

    return out


def build_cross_pair_return_map(cleaned_frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Build per-pair return signals for cross-pair feature joining."""
    cross_maps: dict[str, pd.DataFrame] = {}

    for pair, df in cleaned_frames.items():
        temp = df[["timestamp_utc", "close"]].copy().sort_values("timestamp_utc").reset_index(drop=True)
        temp["ret_1"] = temp["close"].pct_change(1)
        temp["ret_5"] = temp["close"].pct_change(5)

        cross_maps[pair] = temp[["timestamp_utc", "ret_1", "ret_5"]].rename(
            columns={
                "ret_1": f"{pair}_ret_1_xpair",
                "ret_5": f"{pair}_ret_5_xpair",
            }
        )

    return cross_maps


def add_cross_pair_features(
    pair: str,
    df: pd.DataFrame,
    cross_maps: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    out = df.copy()

    for other_pair, other_df in cross_maps.items():
        if other_pair == pair:
            continue
        out = out.merge(other_df, on="timestamp_utc", how="left")

    return out


def build_feature_summary(pair: str, before_rows: int, after_rows: int, df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in BASE_COLUMNS]
    summary = {
        "pair": pair,
        "rows_before_feature_drop": before_rows,
        "rows_after_feature_drop": after_rows,
        "rows_removed_feature_drop": before_rows - after_rows,
        "num_total_columns": len(df.columns),
        "num_feature_columns": len(feature_cols),
    }
    return pd.DataFrame([summary])


def build_pair_features(
    pair: str,
    df: pd.DataFrame,
    cross_maps: dict[str, pd.DataFrame],
    drop_warmup: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the full feature set for one pair."""
    out = df.copy()

    out = add_time_features(out)
    out = add_return_features(out)
    out = add_range_features(out)
    out = add_volatility_features(out)
    out = add_trend_features(out)
    out = add_volatility_regime_features(out)
    out = add_cross_pair_features(pair, out, cross_maps)

    before_drop = len(out)

    if drop_warmup:
        required_feature_cols = [
            "ret_1", "ret_5", "ret_15",
            "rv_10", "rv_30", "rv_60",
            "sma_10", "sma_30", "sma_60", "sma_120",
            "mom_5", "mom_15", "mom_30",
            "range_ma_10", "range_ma_30",
            "volatility_regime_high",
        ]
        out = out.dropna(subset=required_feature_cols).reset_index(drop=True)

    summary_df = build_feature_summary(pair, before_drop, len(out), out)
    return out, summary_df


def process_pair(
    pair: str,
    cleaned_frames: dict[str, pd.DataFrame],
    cross_maps: dict[str, pd.DataFrame],
    drop_warmup: bool,
    force: bool,
) -> None:
    output_path = FEATURE_PAIR_DIR / f"{pair}_2015_2025_features.parquet"
    report_dir = ensure_dir(FEATURE_REPORTS_DIR / pair)
    summary_path = report_dir / "feature_summary.csv"

    if output_path.exists() and summary_path.exists() and not force:
        return

    feature_df, summary_df = build_pair_features(
        pair=pair,
        df=cleaned_frames[pair],
        cross_maps=cross_maps,
        drop_warmup=drop_warmup,
    )

    ensure_dir(FEATURE_PAIR_DIR)
    feature_df.to_parquet(output_path, index=False)
    save_csv(summary_df, summary_path)


def main() -> None:
    args = parse_args()

    ensure_dir(FEATURE_PAIR_DIR)
    ensure_dir(FEATURE_REPORTS_DIR)

    cleaned_frames = {pair: load_clean_pair(pair) for pair in PAIRS}
    cross_maps = build_cross_pair_return_map(cleaned_frames)

    for pair in PAIRS:
        process_pair(
            pair=pair,
            cleaned_frames=cleaned_frames,
            cross_maps=cross_maps,
            drop_warmup=args.drop_warmup,
            force=args.force,
        )

    print("Feature engineering complete.")
    print(f"Processed pairs: {', '.join(PAIRS)}")
    print(f"Feature datasets saved in: {FEATURE_PAIR_DIR}")
    print(f"Feature reports saved in: {FEATURE_REPORTS_DIR}")


if __name__ == "__main__":
    main()
