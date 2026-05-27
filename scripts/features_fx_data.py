from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd

from config.constants import (
    VOL_HIGH_REGIME_PERCENTILE,
    VOL_REGIME_WINDOW,
    PAIRS,
    PAIR_SPREAD_PIPS,
    PAIR_PIP_SIZES,
    RET_WINDOWS,
    VOL_WINDOWS,
    MA_WINDOWS,
    MOM_WINDOWS,
    RANGE_MA_WINDOWS,
)
from scripts._common import ensure_dir, save_csv, load_pair_parquet

PROJECT_DIR = Path(__file__).resolve().parent.parent

CLEAN_ROOT_DIR = PROJECT_DIR / "data" / "processed"
CLEANED_DIR = CLEAN_ROOT_DIR / "cleaned"

FEATURE_ROOT_DIR = PROJECT_DIR / "features"
FEATURE_PAIR_DIR = FEATURE_ROOT_DIR / "pair"
FEATURE_REPORTS_DIR = FEATURE_ROOT_DIR / "reports"

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

OVERLAP_START_HOUR = 13
OVERLAP_END_HOUR = 16


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
    return load_pair_parquet(
        pair,
        CLEANED_DIR,
        suffix="clean",
        required_columns=BASE_COLUMNS,
        parse_est=True,
    )


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["timestamp_utc"].dt.hour
    out["day_of_week"] = out["timestamp_utc"].dt.dayofweek
    out["is_overlap_session"] = (
        (out["hour"] >= OVERLAP_START_HOUR) & (out["hour"] < OVERLAP_END_HOUR)
    ).astype("int8")
    return out


def add_f1_extended(df: pd.DataFrame) -> pd.DataFrame:
    """F1 extensions: cyclical time, session one-hot, month-end flag.

    Requires 'hour' (from add_time_features) and 'session' (from cleaned data).
    Session column values expected: 'Asia', 'London', 'Overlap', 'New_York'.
    """
    out = df.copy()
    out["minute_of_day"] = out["timestamp_utc"].dt.hour * 60 + out["timestamp_utc"].dt.minute
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["session_asia"]    = (out["session"] == "Asia").astype("int8")
    out["session_london"]  = (out["session"] == "London").astype("int8")
    out["session_overlap"] = (out["session"] == "Overlap").astype("int8")
    out["session_ny"]      = (out["session"] == "New_York").astype("int8")
    out["is_month_end"]    = out["timestamp_utc"].dt.is_month_end.astype("int8")
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


def add_f3_extended(df: pd.DataFrame) -> pd.DataFrame:
    """F3 extension: same-minute previous-day log bar range.

    Groups bars by clock time (HH:MM) and shifts within each group by 1 day.
    Handles missing bars (weekend/holiday gaps) correctly - only shifts within
    the same time-of-day group, not by a fixed 1440-bar offset.
    First occurrence of each HH:MM group will be NaN; fill with 0 at training time.
    """
    out = df.copy()
    bar_logrange = np.log(out["high"] / out["low"]).replace([np.inf, -np.inf], np.nan)
    time_key = out["timestamp_utc"].dt.strftime("%H:%M")
    out["same_minute_prev_day_logrange"] = (
        bar_logrange.groupby(time_key).shift(1).values
    )
    return out


def add_f5_extended(df: pd.DataFrame) -> pd.DataFrame:
    """F5 extension: RSI-14 using Wilder EWM (matches RSIMeanReversion._rsi formula)."""
    out = df.copy()
    period = 14
    delta = out["close"].diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
    return out


def add_spread_features(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """F6: spread / liquidity proxies.

    histdata's feed has no real bid/ask spread, so we use a fixed per-pair
    spread from PAIR_SPREAD_PIPS plus bar-shape proxies for liquidity.
    """
    out = df.copy()

    pip_size  = PAIR_PIP_SIZES.get(pair, 0.0001)
    spread_pp = PAIR_SPREAD_PIPS.get(pair, 1.0)

    out["spread_pips"] = spread_pp
    out["bar_range_pips"] = (out["high"] - out["low"]) / pip_size
    out["bar_range_to_spread_ratio"] = np.where(
        spread_pp != 0, out["bar_range_pips"] / spread_pp, np.nan
    )

    if "volume" in out.columns:
        vol_mean = out["volume"].rolling(window=60, min_periods=60).mean()
        vol_std  = out["volume"].rolling(window=60, min_periods=60).std()
        out["volume_zscore_60"] = np.where(
            vol_std != 0, (out["volume"] - vol_mean) / vol_std, np.nan
        )
    else:
        out["volume_zscore_60"] = np.nan

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
    out = add_f1_extended(out)
    out = add_return_features(out)
    out = add_range_features(out)
    out = add_volatility_features(out)
    out = add_f3_extended(out)
    out = add_trend_features(out)
    out = add_f5_extended(out)
    out = add_volatility_regime_features(out)
    out = add_spread_features(out, pair)
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
            "hour_sin", "hour_cos",
            "session_asia", "session_london", "session_ny", "session_overlap",
            "rsi_14",
            # same_minute_prev_day_logrange excluded - NaN for first ~1440 bars per pair
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
        print(f"  [SKIP] {pair}: already exists (use --force to recompute)")
        return

    t0 = time.time()
    print(f"  [START] {pair} ...")
    feature_df, summary_df = build_pair_features(
        pair=pair,
        df=cleaned_frames[pair],
        cross_maps=cross_maps,
        drop_warmup=drop_warmup,
    )

    ensure_dir(FEATURE_PAIR_DIR)
    feature_df.to_parquet(output_path, index=False)
    save_csv(summary_df, summary_path)
    elapsed = time.time() - t0
    print(f"  [DONE]  {pair}: {len(feature_df):,} rows, {len(feature_df.columns)} cols - {elapsed:.1f}s")


def main() -> None:
    args = parse_args()

    ensure_dir(FEATURE_PAIR_DIR)
    ensure_dir(FEATURE_REPORTS_DIR)

    print(f"Loading cleaned parquets for {len(PAIRS)} pairs ...")
    t_load = time.time()
    cleaned_frames = {pair: load_clean_pair(pair) for pair in PAIRS}
    print(f"Loaded in {time.time() - t_load:.1f}s. Building cross-pair return maps ...")
    cross_maps = build_cross_pair_return_map(cleaned_frames)
    print(f"Cross-pair maps ready. Processing {len(PAIRS)} pairs in parallel ...\n")

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=len(PAIRS)) as pool:
        futures = {
            pool.submit(
                process_pair,
                pair=pair,
                cleaned_frames=cleaned_frames,
                cross_maps=cross_maps,
                drop_warmup=args.drop_warmup,
                force=args.force,
            ): pair
            for pair in PAIRS
        }
        for fut in as_completed(futures):
            pair = futures[fut]
            exc = fut.exception()
            if exc:
                print(f"  [ERROR] {pair}: {exc}")

    total = time.time() - t_start
    print(f"\nFeature engineering complete in {total:.1f}s.")
    print(f"Processed pairs: {', '.join(PAIRS)}")
    print(f"Feature datasets saved in: {FEATURE_PAIR_DIR}")
    print(f"Feature reports saved in: {FEATURE_REPORTS_DIR}")


if __name__ == "__main__":
    main()
