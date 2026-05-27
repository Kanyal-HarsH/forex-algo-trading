from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd
from sklearn.preprocessing import StandardScaler

from config.constants import (
    PURGE_ROWS,
    HORIZON_SECONDARY,
    DEFAULT_N_FOLDS,
    PAIRS,
    DATASETS_DIR as SPLIT_ROOT_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    FOLDS_DIR,
    LR_FEATURES,
    SCALERS_DIR,
    SESSION_FILTER_MAP,
    SESSION_TRAIN_DIRS,
)
from scripts._common import ensure_dir, save_csv, load_pair_parquet

assert PURGE_ROWS >= HORIZON_SECONDARY, (
    f"PURGE_ROWS ({PURGE_ROWS}) must be >= HORIZON_SECONDARY ({HORIZON_SECONDARY}) "
    "to prevent label leakage across split boundaries."
)

PROJECT_DIR = Path(__file__).resolve().parent.parent

LABEL_ROOT_DIR = PROJECT_DIR / "labels"
LABEL_PAIR_DIR = LABEL_ROOT_DIR / "pair"

REPORTS_DIR    = SPLIT_ROOT_DIR / "reports"

REQUIRED_COLUMNS = [
    "timestamp_utc",
    "pair",
    "session",
    "close",
    "label",
]

DEFAULT_TRAIN_END   = "2021-12-31 23:59:59+00:00"
DEFAULT_VAL_END     = "2023-12-31 23:59:59+00:00"
FOLD_FIRST_VAL_YEAR = 2019




def parse_timestamp(ts_str: str) -> pd.Timestamp:
    ts = pd.Timestamp(ts_str)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def ts_year_end(year: int) -> pd.Timestamp:
    return parse_timestamp(f"{year}-12-31 23:59:59+00:00")


def ts_year_start(year: int) -> pd.Timestamp:
    return parse_timestamp(f"{year}-01-01 00:00:00+00:00")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split labeled FX datasets into:\n"
            "  1. Fixed train / val / test sets  (datasets/train|val|test/)\n"
            "  2. Walk-forward expanding folds   (datasets/folds/fold_N/)\n"
            "Existing outputs are skipped unless --force is passed."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--train-end",   type=str, default=DEFAULT_TRAIN_END,
                        help="End of the fixed training window.")
    parser.add_argument("--val-end",     type=str, default=DEFAULT_VAL_END,
                        help="End of the fixed validation window.")
    parser.add_argument("--purge-rows",  type=int, default=PURGE_ROWS,
                        help="Rows purged from every train tail before the next window.")
    parser.add_argument("--n-folds",     type=int, default=DEFAULT_N_FOLDS,
                        help="Number of walk-forward folds (default 5).")
    parser.add_argument("--force",       action="store_true",
                        help="Recompute and overwrite ALL outputs.")
    parser.add_argument("--force-folds", action="store_true",
                        help="Recompute fold outputs only (leaves fixed split untouched).")
    parser.add_argument("--force-fixed", action="store_true",
                        help="Recompute fixed split outputs only (leaves folds untouched).")
    parser.add_argument("--skip-scaler", action="store_true",
                        help="Skip fitting and saving the StandardScaler.")
    return parser.parse_args()



def load_labeled_pair(pair: str) -> pd.DataFrame:
    return load_pair_parquet(
        pair,
        LABEL_PAIR_DIR,
        suffix="labeled",
        required_columns=REQUIRED_COLUMNS,
    )


def apply_purge(df: pd.DataFrame, purge_rows: int) -> pd.DataFrame:
    """Drop the last `purge_rows` from a DataFrame slice."""
    if len(df) <= purge_rows:
        return df.copy()
    return df.iloc[: len(df) - purge_rows].copy()


def slice_window(
    df: pd.DataFrame,
    start: pd.Timestamp | None,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Return rows where timestamp_utc is within."""
    mask = df["timestamp_utc"] <= end
    if start is not None:
        mask &= df["timestamp_utc"] > start
    return df.loc[mask].copy()



def label_distribution(df: pd.DataFrame, prefix: str) -> dict:
    """Compute label counts and class percentages for one split."""
    total   = len(df)
    vc      = df["label"].value_counts(dropna=True)
    long_n  = int(vc.get(1,  0))
    flat_n  = int(vc.get(0,  0))
    short_n = int(vc.get(-1, 0))
    na_n    = int(df["label"].isna().sum())
    signal_n = long_n + short_n

    return {
        f"{prefix}_rows":       total,
        f"{prefix}_long_n":     long_n,
        f"{prefix}_flat_n":     flat_n,
        f"{prefix}_short_n":    short_n,
        f"{prefix}_na_n":       na_n,
        f"{prefix}_signal_pct": round(signal_n / total * 100, 2) if total else 0.0,
        f"{prefix}_long_pct":   round(long_n   / total * 100, 2) if total else 0.0,
        f"{prefix}_flat_pct":   round(flat_n   / total * 100, 2) if total else 0.0,
        f"{prefix}_short_pct":  round(short_n  / total * 100, 2) if total else 0.0,
        f"{prefix}_start":      df["timestamp_utc"].min() if total else pd.NaT,
        f"{prefix}_end":        df["timestamp_utc"].max() if total else pd.NaT,
    }


def session_distribution(df: pd.DataFrame, prefix: str) -> dict:
    """Compute session row percentages for one split."""
    total = len(df)
    vc = df["session"].value_counts(normalize=True) if total else pd.Series(dtype=float)
    return {
        f"{prefix}_asia_pct":    round(float(vc.get("Asia",     0)) * 100, 2),
        f"{prefix}_london_pct":  round(float(vc.get("London",   0)) * 100, 2),
        f"{prefix}_overlap_pct": round(float(vc.get("Overlap",  0)) * 100, 2),
        f"{prefix}_ny_pct":      round(float(vc.get("New_York", 0)) * 100, 2),
    }


def _fixed_outputs_exist(pair: str) -> bool:
    return (
        (TRAIN_DIR   / f"{pair}_train.parquet").exists()
        and (VAL_DIR / f"{pair}_val.parquet").exists()
        and (TEST_DIR / f"{pair}_test.parquet").exists()
        and (REPORTS_DIR / f"{pair}_split_summary.csv").exists()
    )


def process_fixed_split(
    pair: str,
    full_df: pd.DataFrame,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    purge_rows: int,
    force: bool,
    skip_scaler: bool = False,
) -> pd.DataFrame:
    """Produce the fixed train / val / test parquets for one pair."""
    summary_path = REPORTS_DIR / f"{pair}_split_summary.csv"

    if _fixed_outputs_exist(pair) and not force:
        print(f"  [fixed] {pair}: all outputs exist - skipping (use --force to recompute)")
        return pd.read_csv(summary_path)

    raw_train = slice_window(full_df, start=None,      end=train_end)
    raw_val   = slice_window(full_df, start=train_end, end=val_end)
    test_df   = slice_window(full_df, start=val_end,   end=full_df["timestamp_utc"].max())

    train_df = apply_purge(raw_train, purge_rows).reset_index(drop=True)
    val_df   = apply_purge(raw_val,   purge_rows).reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if split.empty:
            raise ValueError(
                f"{pair}: fixed {name} split is empty - check date boundaries."
            )

    ensure_dir(TRAIN_DIR)
    ensure_dir(VAL_DIR)
    ensure_dir(TEST_DIR)
    ensure_dir(REPORTS_DIR)

    train_df.to_parquet(TRAIN_DIR  / f"{pair}_train.parquet", index=False)
    val_df.to_parquet(  VAL_DIR    / f"{pair}_val.parquet",   index=False)
    test_df.to_parquet( TEST_DIR   / f"{pair}_test.parquet",  index=False)

    if not skip_scaler:
        fit_and_save_scaler(
            pair=pair,
            train_df=train_df,
            feature_cols=list(LR_FEATURES),
            scalers_dir=SCALERS_DIR,
        )

    record = {
        "pair":                 pair,
        "total_rows":           len(full_df),
        "configured_train_end": train_end,
        "configured_val_end":   val_end,
        "purge_rows":           purge_rows,
    }
    record.update(label_distribution(train_df, "train"))
    record.update(label_distribution(val_df,   "val"))
    record.update(label_distribution(test_df,  "test"))
    record.update(session_distribution(train_df, "train"))
    record.update(session_distribution(test_df,  "test"))
    summary_df = pd.DataFrame([record])
    save_csv(summary_df, summary_path)

    print(
        f"  [fixed] {pair}: train={len(train_df):,}  "
        f"val={len(val_df):,}  test={len(test_df):,}  "
        f"purge={purge_rows}"
    )
    return summary_df


def _fold_boundaries(n_folds: int, first_val_year: int) -> list[dict]:
    """Build boundary dicts for each fold."""
    folds = []
    for k in range(n_folds):
        val_year   = first_val_year + k
        train_year = val_year - 1
        folds.append({
            "fold":      k,
            "train_end": ts_year_end(train_year),
            "val_start": ts_year_start(val_year),
            "val_end":   ts_year_end(val_year),
        })
    return folds


def _fold_outputs_exist(pair: str, fold: int) -> bool:
    fold_dir = FOLDS_DIR / f"fold_{fold}"
    return (
        (fold_dir / f"{pair}_train.parquet").exists()
        and (fold_dir / f"{pair}_val.parquet").exists()
    )


def process_folds(
    pair: str,
    full_df: pd.DataFrame,
    n_folds: int,
    purge_rows: int,
    force: bool,
) -> list[dict]:
    """Produce walk-forward fold parquets for one pair."""
    boundaries   = _fold_boundaries(n_folds, FOLD_FIRST_VAL_YEAR)
    fold_summaries = []

    for b in boundaries:
        fold      = b["fold"]
        train_end = b["train_end"]
        val_start = b["val_start"]
        val_end   = b["val_end"]
        fold_dir  = FOLDS_DIR / f"fold_{fold}"

        if _fold_outputs_exist(pair, fold) and not force:
            existing_train = pd.read_parquet(
                fold_dir / f"{pair}_train.parquet",
                columns=["timestamp_utc", "label", "session"],
            )
            existing_val = pd.read_parquet(
                fold_dir / f"{pair}_val.parquet",
                columns=["timestamp_utc", "label", "session"],
            )
            print(
                f"  [fold {fold}] {pair}: outputs exist - skipping "
                f"(train={len(existing_train):,}  val={len(existing_val):,})"
            )
            row = {
                "pair": pair, "fold": fold,
                "train_end": train_end, "val_start": val_start,
                "val_end": val_end, "purge_rows": purge_rows,
            }
            row.update(label_distribution(existing_train, "train"))
            row.update(label_distribution(existing_val,   "val"))
            fold_summaries.append(row)
            continue

        raw_train = slice_window(full_df, start=None, end=train_end)
        raw_val   = slice_window(
            full_df,
            start=val_start - pd.Timedelta(seconds=1),
            end=val_end,
        )

        train_df = apply_purge(raw_train, purge_rows).reset_index(drop=True)
        val_df   = raw_val.reset_index(drop=True)

        if train_df.empty:
            raise ValueError(f"{pair} fold {fold}: train slice is empty.")
        if val_df.empty:
            raise ValueError(f"{pair} fold {fold}: val slice is empty.")

        ensure_dir(fold_dir)
        train_df.to_parquet(fold_dir / f"{pair}_train.parquet", index=False)
        val_df.to_parquet(  fold_dir / f"{pair}_val.parquet",   index=False)

        row = {
            "pair": pair, "fold": fold,
            "train_end": train_end, "val_start": val_start,
            "val_end": val_end, "purge_rows": purge_rows,
        }
        row.update(label_distribution(train_df, "train"))
        row.update(label_distribution(val_df,   "val"))
        fold_summaries.append(row)

        print(
            f"  [fold {fold}] {pair}: "
            f"train <={train_end.year}  ({len(train_df):,} rows)  "
            f"val {val_start.year}  ({len(val_df):,} rows)  "
            f"purge={purge_rows}"
        )

    return fold_summaries

def fit_and_save_scaler(
    pair: str,
    train_df: pd.DataFrame,
    feature_cols: list[str],
    scalers_dir: Path,
) -> StandardScaler:
    """Fit StandardScaler on train_df[feature_cols] and save to disk.

    Saves {"scaler": scaler, "feature_cols": available} so callers can
    always reconstruct the exact column order used at fit time.
    Columns missing from train_df are skipped with a warning.
    """
    available = [c for c in feature_cols if c in train_df.columns]
    missing = [c for c in feature_cols if c not in train_df.columns]
    if missing:
        print(
            f"  [scaler] {pair}: WARNING - {len(missing)} feature cols not in data "
            f"(run features_fx_data.py --force first): {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    X = train_df[available].fillna(0).values
    scaler = StandardScaler()
    scaler.fit(X)

    ensure_dir(scalers_dir)
    scaler_path = scalers_dir / f"{pair}_scaler.pkl"
    with open(scaler_path, "wb") as fh:
        pickle.dump({"scaler": scaler, "feature_cols": available}, fh)

    print(f"  [scaler] {pair}: fitted on {len(available)} features -> {scaler_path}")
    return scaler


def process_session_splits(
    pair: str,
    train_df: pd.DataFrame,
    session_filter_map: dict[str, list[str]],
    session_train_dirs: dict[str, Path],
    force: bool,
) -> None:
    """Write session-filtered training parquets for each non-global session.

    Session models use the same global scaler - no per-session refitting.
    """
    for session_name, session_values in session_filter_map.items():
        out_dir  = session_train_dirs[session_name]
        out_path = out_dir / f"{pair}_train.parquet"

        if out_path.exists() and not force:
            print(f"  [session-split] {pair}/{session_name}: exists - skip")
            continue

        mask = train_df["session"].isin(session_values)
        session_df = train_df.loc[mask].reset_index(drop=True)

        if session_df.empty:
            print(
                f"  [session-split] {pair}/{session_name}: WARNING - empty after "
                f"filter (session_values={session_values})"
            )
            continue

        ensure_dir(out_dir)
        session_df.to_parquet(out_path, index=False)
        print(
            f"  [session-split] {pair}/{session_name}: "
            f"{len(session_df):,} rows ({mask.mean() * 100:.1f}% of train) -> {out_path}"
        )


def main() -> None:
    args = parse_args()

    train_end  = parse_timestamp(args.train_end)
    val_end    = parse_timestamp(args.val_end)
    purge_rows = args.purge_rows
    n_folds    = args.n_folds

    force_fixed = args.force or args.force_fixed
    force_folds = args.force or args.force_folds

    if train_end >= val_end:
        raise ValueError("--train-end must be earlier than --val-end.")

    ensure_dir(TRAIN_DIR)
    ensure_dir(VAL_DIR)
    ensure_dir(TEST_DIR)
    ensure_dir(FOLDS_DIR)
    ensure_dir(REPORTS_DIR)

    fixed_summaries: list[pd.DataFrame] = []
    fold_rows:       list[dict]         = []

    for pair in PAIRS:
        print(f"\n{'-'*60}")
        print(f"Processing: {pair}")
        print(f"{'-'*60}")

        full_df = load_labeled_pair(pair)
        print(
            f"  Loaded {len(full_df):,} rows  "
            f"({full_df['timestamp_utc'].min().date()} -> "
            f"{full_df['timestamp_utc'].max().date()})"
        )

        summary = process_fixed_split(
            pair=pair, full_df=full_df,
            train_end=train_end, val_end=val_end,
            purge_rows=purge_rows, force=force_fixed,
            skip_scaler=args.skip_scaler,
        )
        fixed_summaries.append(summary)

        # Session-filtered training parquets for session-conditional models
        train_parquet = TRAIN_DIR / f"{pair}_train.parquet"
        if train_parquet.exists():
            train_for_session = pd.read_parquet(train_parquet)
            process_session_splits(
                pair=pair,
                train_df=train_for_session,
                session_filter_map=SESSION_FILTER_MAP,
                session_train_dirs=SESSION_TRAIN_DIRS,
                force=force_fixed,
            )

        pair_fold_rows = process_folds(
            pair=pair, full_df=full_df,
            n_folds=n_folds, purge_rows=purge_rows,
            force=force_folds,
        )
        fold_rows.extend(pair_fold_rows)

    fixed_manifest = pd.concat(fixed_summaries, ignore_index=True)
    save_csv(fixed_manifest, REPORTS_DIR / "split_manifest.csv")

    fold_manifest = pd.DataFrame(fold_rows)
    save_csv(fold_manifest, REPORTS_DIR / "fold_manifest.csv")

    print(f"\n{'='*60}")
    print("SPLIT COMPLETE")
    print(f"{'='*60}")
    print(f"Fixed split  ->  {TRAIN_DIR.parent}")
    print(f"Folds        ->  {FOLDS_DIR}")
    print(f"Reports      ->  {REPORTS_DIR}")

    print("\n-- Fixed split manifest --")
    preview_cols = [
        "pair", "total_rows",
        "train_rows", "val_rows", "test_rows",
        "train_signal_pct", "val_signal_pct", "test_signal_pct",
        "purge_rows",
    ]
    print(
        fixed_manifest[[c for c in preview_cols if c in fixed_manifest.columns]]
        .to_string(index=False)
    )

    if not fold_manifest.empty:
        print("\n-- Fold manifest (first 10 rows) --")
        fold_preview = [
            "pair", "fold",
            "train_rows", "train_end",
            "val_rows",   "val_start",
            "train_signal_pct", "val_signal_pct",
        ]
        print(
            fold_manifest[[c for c in fold_preview if c in fold_manifest.columns]]
            .head(10).to_string(index=False)
        )

    print("\nRun with --force        to recompute everything.")
    print("Run with --force-folds  to recompute folds only.")
    print("Run with --force-fixed  to recompute fixed split only.")


if __name__ == "__main__":
    main()
