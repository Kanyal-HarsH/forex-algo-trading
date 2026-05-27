from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from backtest.engine         import run_backtest, run_wf_folds
from backtest.report_generator import generate_report
from backtest.strategies     import STRATEGY_REGISTRY, get_strategy
from config.constants import (
    PAIRS,
    CLEANED_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    fold_parquet_path,
)

ALL_PAIRS      = list(PAIRS)
ALL_STRATEGIES = list(STRATEGY_REGISTRY.keys())

NAMED_SPLITS = ["full", "train", "val", "test"] + [f"fold_{i}" for i in range(5)]

_SESSION_HOURS: dict[str, tuple[int, int]] = {
    "london":  (7,  16),
    "ny":      (13, 22),
    "asia":    (23,  8),
    "overlap": (13, 16),
}


_TTY   = sys.stdout.isatty()
GREEN  = "\033[92m" if _TTY else ""
YELLOW = "\033[93m" if _TTY else ""
RED    = "\033[91m" if _TTY else ""
CYAN   = "\033[96m" if _TTY else ""
RESET  = "\033[0m"  if _TTY else ""


def _split_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--pair", nargs="+", default=["all"], metavar="PAIR",
        help="One or more pairs, or 'all'.\nExample: --pair EURUSD GBPUSD",
    )
    p.add_argument(
        "--strategy", nargs="+", default=["all"], metavar="STRATEGY",
        help="One or more strategy names, or 'all'.\nExample: --strategy MACrossover_f20_s50_EMA",
    )
    p.add_argument(
        "--split", choices=NAMED_SPLITS, default="full", metavar="SPLIT",
        help=(
            "Data split to run on.\n"
            "  full   -- entire cleaned history (DEFAULT)\n"
            "            loads from data/processed/cleaned/{pair}_2015_2025_clean.parquet\n"
            "  train  -- training partition\n"
            "  val    -- validation partition\n"
            "  test   -- held-out test partition\n"
            "  fold_N -- walk-forward fold N (0-4), reads from datasets/folds/fold_N/\n"
            "\n"
            "Rule-based strategies have no fittable parameters so 'full' is always correct.\n"
            "Use train/val/test only when evaluating ML-based strategies."
        ),
    )
    p.add_argument(
        "--from", dest="date_from", type=str, default=None, metavar="DATETIME",
        help=(
            "Start of the analysis window (inclusive).\n"
            "Accepts ISO 8601: '2022-01-01' or '2022-01-01T00:00:00'.\n"
            "Applied AFTER loading the split parquet."
        ),
    )
    p.add_argument(
        "--to", dest="date_to", type=str, default=None, metavar="DATETIME",
        help=(
            "End of the analysis window (inclusive).\n"
            "Accepts ISO 8601: '2022-12-31' or '2022-12-31T23:59:00'."
        ),
    )
    p.add_argument(
        "--folds", type=int, default=0, metavar="FOLDS",
        help="Walk-forward folds (uses train split). 0 = single full-period run.",
    )
    return p


def _cost_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--capital", type=float, default=10_000.0, metavar="CAPITAL",
        help="Starting capital in USD. Default: 10000.",
    )
    p.add_argument(
        "--spread", type=float, default=None, metavar="SPREAD",
        help="Override spread in pips for all pairs. Uses per-pair table if omitted.",
    )
    p.add_argument(
        "--tp-pips", type=float, default=None, metavar="TP_PIPS",
        help="Take-profit distance in pips.",
    )
    p.add_argument(
        "--sl-pips", type=float, default=None, metavar="SL_PIPS",
        help="Stop-loss distance in pips.",
    )
    p.add_argument(
        "--max-hold", type=int, default=None, metavar="BARS",
        help=(
            "Maximum bars to hold a position.\n"
            "With 1-min data: 60 = 1 h, 240 = 4 h."
        ),
    )
    return p


def _filter_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--session", choices=list(_SESSION_HOURS.keys()), default=None, metavar="SESSION",
        help=(
            "Only enter trades inside this session window (UTC).\n"
            "london 07-16  |  ny 13-22  |  asia 23-08  |  overlap 13-16"
        ),
    )
    p.add_argument(
        "--entry-time", type=str, default=None, metavar="HH:MM",
        help="Only enter trades at or after this UTC time each day. Example: 09:00",
    )
    p.add_argument(
        "--resample", type=str, default=None, metavar="FREQ",
        help="Resample bars before running. Any pandas offset string: 1H 4H 15min 1D.",
    )
    p.add_argument(
        "--direction", choices=["long_short", "long_only", "short_only"],
        default="long_short", metavar="MODE",
        help=(
            "Which signal directions to trade.\n"
            "long_short -- both longs and shorts (default)\n"
            "long_only  -- suppress short (-1) signals\n"
            "short_only -- suppress long  (+1) signals"
        ),
    )
    return p


def _output_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--mode", choices=["research", "simulation"], default=None, metavar="MODE",
        help=(
            "Override the run mode label written to the HTML report.\n"
            "If omitted, inferred automatically:\n"
            "  simulation -- when --from is provided\n"
            "  research   -- otherwise"
        ),
    )
    p.add_argument(
        "--out", type=Path, default=None, metavar="OUTPUT_PATH",
        help="Custom output .html path. Auto-named if omitted.",
    )
    p.add_argument(
        "--no-browser", action="store_true",
        help="Write report without opening it in the browser.",
    )
    return p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FX backtests and generate an HTML report.",
        formatter_class=argparse.RawTextHelpFormatter,
        parents=[_split_parser(), _cost_parser(), _filter_parser(), _output_parser()],
    )
    return parser.parse_args()


def resolve_pairs(raw: list[str]) -> list[str]:
    if raw == ["all"]:
        return ALL_PAIRS
    invalid = [p for p in raw if p not in ALL_PAIRS]
    if invalid:
        raise ValueError(f"Unknown pairs: {invalid}. Supported: {ALL_PAIRS}")
    return raw


_ML_STRATEGY_NAMES = {
    f"{m}_{s}"
    for m in ("LR", "LSTM")
    for s in ("global", "london", "ny", "asia")
}


def resolve_strategies(raw: list[str]) -> list[str]:
    if raw == ["all"]:
        return ALL_STRATEGIES
    invalid = [s for s in raw if s not in STRATEGY_REGISTRY and s not in _ML_STRATEGY_NAMES]
    if invalid:
        raise ValueError(
            f"Unknown strategies: {invalid}.\n"
            f"Rule-based: {ALL_STRATEGIES}\n"
            f"ML: {sorted(_ML_STRATEGY_NAMES)}"
        )
    return raw


def resolve_split_path(split: str, pair: str) -> Path:
    """Return the parquet path for a given split name and currency pair.

    fold_N splits are resolved to datasets/folds/fold_N/{pair}_train.parquet
    - the year-aligned, purge-gapped files written by scripts/split_fx_data.py.
    This ensures walk-forward evaluation uses proper calendar boundaries and no
    label leakage, rather than a raw index-slice of the monolithic train parquet.
    """
    if split == "full":
        return CLEANED_DIR / f"{pair}_2015_2025_clean.parquet"
    if split == "train":
        return TRAIN_DIR / f"{pair}_train.parquet"
    if split.startswith("fold_"):
        fold_idx = int(split.split("_")[1])
        return fold_parquet_path(pair, fold_idx, "train")
    if split == "val":
        return VAL_DIR   / f"{pair}_val.parquet"
    if split == "test":
        return TEST_DIR  / f"{pair}_test.parquet"
    raise ValueError(f"Unrecognised split: {split!r}")


def load_split_data(
    pair:      str,
    split:     str,
    date_from: str | None = None,
    date_to:   str | None = None,
) -> pd.DataFrame:
    path = resolve_split_path(split, pair)
    if not path.exists():
        if split.startswith("fold_"):
            hint = (
                f"Run scripts/split_fx_data.py --force-folds to generate "
                f"datasets/folds/ parquets."
            )
        elif split == "full":
            hint = "Run scripts/clean_fx_data.py first to generate data/processed/cleaned/ parquets."
        else:
            hint = "Run scripts/split_fx_data.py first to generate datasets/train|val|test/ parquets."
        raise FileNotFoundError(f"Parquet not found: {path}\n{hint}")

    df = pd.read_parquet(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    # NOTE: no index-slicing for fold_N - the fold parquet already contains
    # only the correct year-aligned, purge-gapped rows for that fold.

    actual_start = df["timestamp_utc"].iloc[0].strftime("%Y-%m-%d")
    actual_end   = df["timestamp_utc"].iloc[-1].strftime("%Y-%m-%d")

    if date_from:
        ts_from = pd.Timestamp(date_from, tz="UTC")
        df = df[df["timestamp_utc"] >= ts_from].reset_index(drop=True)
    if date_to:
        ts_to = pd.Timestamp(date_to, tz="UTC")
        if re.match(r"^\d{4}-\d{2}-\d{2}$", date_to.strip()):
            ts_to = ts_to + pd.Timedelta(hours=23, minutes=59, seconds=59)
        df = df[df["timestamp_utc"] <= ts_to].reset_index(drop=True)

    if df.empty:
        raise ValueError(
            f"No data for {pair} / {split} after applying "
            f"--from {date_from}  --to {date_to}.\n"
            f"  Available range for this split: {actual_start} -> {actual_end}\n"
            f"  Tip: rule-based strategies should use --split full."
        )

    return df


def _sharpe_colour(v: float) -> str:
    if v >= 0.5:
        return GREEN
    if v >= 0.0:
        return YELLOW
    return RED


def print_banner(
    pairs:      list[str],
    strategies: list[str],
    args:       argparse.Namespace,
    total_runs: int,
    mode:       str,
) -> None:
    cap_str      = f"${args.capital:,.2f}"
    max_hold_str = f"{args.max_hold} bars" if args.max_hold else "none"
    session_str  = args.session if args.session else "all hours"
    entry_str    = f"{args.entry_time} UTC" if args.entry_time else "any bar"
    resample_str = args.resample if args.resample else "native 1-min"
    spread_str   = (
        f"{args.spread} pips (override)" if args.spread is not None
        else "table default"
    )
    tp_str       = f"{args.tp_pips} pips" if args.tp_pips  is not None else "none"
    sl_str       = f"{args.sl_pips} pips" if args.sl_pips  is not None else "none"
    from_str     = args.date_from if args.date_from else "split start"
    to_str       = args.date_to   if args.date_to   else "split end"
    mode_colour  = CYAN if mode == "simulation" else YELLOW

    print()
    print(SEP2)
    print(f"  {CYAN}FXAlgo Backtest Runner{RESET}")
    print(SEP2)
    print(f"  Pairs        : {', '.join(pairs)}")
    print(f"  Strategies   : {', '.join(strategies)}")
    split_note = "  <- full history (data/processed/cleaned/)" if args.split == "full" else ""
    print(f"  Split        : {args.split.upper()}{split_note}")
    print(f"  Date from    : {from_str}")
    print(f"  Date to      : {to_str}")
    print(f"  Folds        : {args.folds if args.folds > 0 else 'single run (no CV)'}")
    print(f"  Mode         : {mode_colour}{mode.upper()}{RESET}")
    print(f"  Direction    : {args.direction}")
    print(f"  Capital      : {cap_str}")
    print(f"  Max Hold     : {max_hold_str}")
    print(f"  Session      : {session_str}")
    print(f"  Entry Time   : {entry_str}")
    print(f"  Resample     : {resample_str}")
    print(f"  Spread       : {spread_str}")
    print(f"  TP           : {tp_str}")
    print(f"  SL           : {sl_str}")
    print(f"  Total runs   : {total_runs}")
    print(SEP2)
    print()


SEP  = "-" * 115
SEP2 = "=" * 115


def print_header() -> None:
    print(SEP)
    print(
        f"  {'Pair':<10} {'Strategy':<40} {'Sharpe':>9} {'Return':>9} "
        f"{'MaxDD':>8} {'WinRate':>8} {'Trades':>8} {'Sortino':>9} {'Calmar':>9}"
    )
    print(SEP)


def print_run(pair: str, strat_name: str, r) -> None:
    colour = _sharpe_colour(r.net_sharpe)
    print(
        f"  {pair:<10} {strat_name:<40} "
        f"{colour}{r.net_sharpe:+.4f}{RESET} {r.total_return * 100:+8.2f}% "
        f"{r.max_drawdown * 100:7.2f}% {r.win_rate * 100:7.1f}% "
        f"{r.n_trades:>8,} {r.sortino:+9.4f} {r.calmar:+9.4f}"
    )


def print_summary(results: list) -> None:
    sharpes = [r.net_sharpe   for r in results]
    returns = [r.total_return for r in results]
    dds     = [r.max_drawdown for r in results]
    best_i  = int(np.argmax(sharpes))
    worst_i = int(np.argmin(sharpes))
    pos_runs = sum(1 for s in sharpes if s >= 0)

    print()
    print(SEP2)
    print(f"  {CYAN}SUMMARY{RESET}  ({len(results)} run{'s' if len(results) != 1 else ''})")
    print(SEP2)
    print(f"  Avg Net Sharpe    : {_sharpe_colour(np.mean(sharpes))}{np.mean(sharpes):+.4f}{RESET}")
    print(f"  Avg Return        : {np.mean(returns) * 100:+.2f}%")
    print(f"  Avg Max Drawdown  : {np.mean(dds) * 100:.2f}%")
    print(f"  Positive runs     : {pos_runs} / {len(results)}")
    print(
        f"  Best run          : {results[best_i].pair}  {results[best_i].strategy}  "
        f"-> Sharpe {sharpes[best_i]:+.4f}  Return {returns[best_i] * 100:+.2f}%"
    )
    print(
        f"  Worst run         : {results[worst_i].pair}  {results[worst_i].strategy}  "
        f"-> Sharpe {sharpes[worst_i]:+.4f}  Return {returns[worst_i] * 100:+.2f}%"
    )
    print(SEP2)
    print()


def _preflight_fold_paths(pairs: list[str], n_folds: int) -> None:
    """Fail fast if any (pair, k) fold parquet is missing before launching backtests."""
    missing = []
    for pair in pairs:
        for k in range(n_folds):
            p = fold_parquet_path(pair, k, "train")
            if not p.exists():
                missing.append(str(p))
    if missing:
        msg = "\n  ".join(missing)
        raise FileNotFoundError(
            f"Pre-flight check failed: {len(missing)} fold parquet(s) missing:\n  {msg}\n"
            f"Run scripts/split_fx_data.py --force-folds to generate them."
        )


def main() -> None:
    args       = parse_args()
    pairs      = resolve_pairs(args.pair)
    strategies = resolve_strategies(args.strategy)
    total_runs = len(pairs) * len(strategies)

    if args.folds > 0:
        _preflight_fold_paths(pairs, args.folds)

    # mode inferred from --from, not from capital
    if args.mode is not None:
        mode = args.mode
    else:
        mode = "simulation" if args.date_from is not None else "research"

    print_banner(pairs, strategies, args, total_runs, mode)
    print_header()

    results   = []
    completed = 0

    for pair in pairs:

        if args.folds > 0:
            for strat_name in strategies:
                fold_results = run_wf_folds(
                    pair            = pair,
                    strategy        = strat_name,
                    n_folds         = args.folds,
                    spread_pips     = args.spread,
                    tp_pips         = args.tp_pips,
                    sl_pips         = args.sl_pips,
                    capital_initial = args.capital,
                    max_hold_bars   = args.max_hold,
                    session         = args.session,
                    entry_time      = args.entry_time,
                    resample        = args.resample,
                    direction_mode  = args.direction,
                    mode            = mode,
                )
                results.extend(fold_results)
                completed += 1
                print_run(pair, strat_name, fold_results[-1])
                sys.stdout.flush()

        else:
            df_split = load_split_data(
                pair      = pair,
                split     = args.split,
                date_from = args.date_from,
                date_to   = args.date_to,
            )

            prices = df_split["close"].reset_index(drop=True)
            timestamps = (
                df_split["timestamp_utc"]
                .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                .tolist()
                if "timestamp_utc" in df_split.columns else []
            )

            for strat_name in strategies:
                strat   = get_strategy(strat_name, pair=pair)
                signals = strat.generate_signals(
                    df_split.reset_index(drop=True)
                ).reset_index(drop=True)

                r = run_backtest(
                    signals         = signals,
                    prices          = prices,
                    pair            = pair,
                    strategy        = strat.name,
                    split           = args.split,
                    spread_pips     = args.spread,
                    tp_pips         = args.tp_pips,
                    sl_pips         = args.sl_pips,
                    capital_initial = args.capital,
                    max_hold_bars   = args.max_hold,
                    timestamps      = timestamps,
                    session         = args.session,
                    entry_time      = args.entry_time,
                    resample        = args.resample,
                    direction_mode  = args.direction,
                    mode            = mode,
                    df_full         = df_split,
                )
                results.append(r)
                completed += 1
                print_run(pair, strat.name, r)
                sys.stdout.flush()

    print_summary(results)

    print("  Generating report...")
    out = generate_report(
        results      = results,
        out_path     = args.out,
        open_browser = not args.no_browser,
    )
    print(f"  Done. {out.name}")
    print()


if __name__ == "__main__":
    main()
