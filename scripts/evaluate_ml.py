"""
evaluate_ml.py - Unified ML strategy evaluation with Diebold-Mariano significance test.

Usage:
    python scripts/evaluate_ml.py [--pair EURUSD|all] [--spreads "0.5,1.0,2.0"] [--output-dir output]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from backtest.engine import run_backtest
from backtest.strategies import get_strategy
from config.constants import (
    MODELS_DIR,
    PAIRS,
    SESSION_ALIASES,
    parse_model_code,
    PAIR_SPREAD_PIPS,
    SESSION_FILTER_MAP,
    TEST_DIR,
)

ML_STRATEGIES = [
    "LR_global",
    "LR_london",
    "LR_ny",
    "LR_asia",
    "LSTM_global",
    "LSTM_london",
    "LSTM_ny",
    "LSTM_asia",
]

_SESSION_OF_STRATEGY: dict[str, str | None] = {
    "LR_global":   None,
    "LR_london":   "london",
    "LR_ny":       "ny",
    "LR_asia":     "asia",
    "LSTM_global": None,
    "LSTM_london": "london",
    "LSTM_ny":     "ny",
    "LSTM_asia":   "asia",
}

_SESSION_FILTER_COLS: dict[str, list[str]] = {
    "london": SESSION_FILTER_MAP["london"],
    "ny":     SESSION_FILTER_MAP["ny"],
    "asia":   SESSION_FILTER_MAP["asia"],
}


def _model_path(strategy_name: str, pair: str) -> Path:
    parts = strategy_name.split("_", 1)
    model_type = parts[0].lower()
    session    = parts[1].lower() if len(parts) > 1 else "global"
    if session == "global":
        subdir = MODELS_DIR / "global"
    else:
        subdir = MODELS_DIR / "session" / session
    ext = "pkl" if model_type == "lr" else "pt"
    fname = f"{pair}_logreg_model.{ext}" if model_type == "lr" else f"{pair}_lstm_model.{ext}"
    return subdir / fname


def _strategy_available(strategy_name: str, pair: str) -> bool:
    return _model_path(strategy_name, pair).exists()


def _filter_signals_by_session(
    signals: pd.Series,
    df: pd.DataFrame,
    session_values: list[str],
) -> pd.Series:
    """Zero out signals for bars not matching the given session values."""
    if "session" not in df.columns:
        warnings.warn("DataFrame missing 'session' column - session filter skipped.")
        return signals
    mask = df["session"].isin(session_values).values
    return signals.where(mask, other=0)


def evaluate_strategy(
    pair: str,
    strategy_name: str,
    test_df: pd.DataFrame,
    spread_multiplier: float = 1.0,
) -> dict:
    """Run a single ML strategy on test_df and return a metrics dict."""
    strat = get_strategy(strategy_name, pair=pair)
    signals = strat.generate_signals(test_df)

    session_key = _SESSION_OF_STRATEGY.get(strategy_name)
    if session_key is not None:
        signals = _filter_signals_by_session(
            signals, test_df, _SESSION_FILTER_COLS[session_key]
        )

    spread_pips = PAIR_SPREAD_PIPS[pair] * spread_multiplier
    result = run_backtest(
        signals=signals,
        prices=test_df["close"],
        pair=pair,
        strategy=strategy_name,
        split="test",
        spread_pips=spread_pips,
        timestamps=list(test_df.get("timestamp_utc", pd.Series())),
    )

    return {
        "pair":              pair,
        "strategy":          strategy_name,
        "spread_multiplier": spread_multiplier,
        "net_sharpe":        result.net_sharpe,
        "sortino":           result.sortino,
        "max_drawdown":      result.max_drawdown,
        "calmar":            result.calmar,
        "win_rate":          result.win_rate,
        "n_trades":          result.n_trades,
        "total_return":      result.total_return,
    }


def _bar_returns_from_signals(signals: pd.Series, prices: pd.Series) -> np.ndarray:
    """Approximate bar-level P&L for DM test (no spread, directional only)."""
    price_arr = prices.values.astype(float)
    sig_arr   = signals.values.astype(int)
    pos       = pd.Series(sig_arr).replace(0, np.nan).ffill().fillna(0).astype(int)
    pos       = pos.shift(1).fillna(0).astype(int).values
    ret       = np.diff(price_arr, prepend=price_arr[0])
    return pos * ret


def diebold_mariano_test(
    lstm_bar_rets: np.ndarray,
    lr_bar_rets:   np.ndarray,
) -> dict:
    """Harvey-Ley-Newbold modified Diebold-Mariano test (two-sided)."""
    from scipy.stats import t as t_dist

    n = min(len(lstm_bar_rets), len(lr_bar_rets))
    d = lstm_bar_rets[:n] - lr_bar_rets[:n]
    mean_d = np.mean(d)
    # long-run variance via Newey-West (0 lags for simplicity - sufficient for 1-step ahead)
    var_d = np.var(d, ddof=1) / n
    if var_d == 0:
        return {"dm_stat": 0.0, "p_value": 1.0, "n": n}
    dm_stat  = mean_d / np.sqrt(var_d)
    # HLN correction: adjust df
    p_value  = 2.0 * t_dist.sf(abs(dm_stat), df=n - 1)
    return {"dm_stat": round(float(dm_stat), 4), "p_value": round(float(p_value), 4), "n": n}


def _load_test_df(pair: str) -> pd.DataFrame | None:
    path = TEST_DIR / f"{pair}_test.parquet"
    if not path.exists():
        warnings.warn(f"Test parquet not found: {path}")
        return None
    return pd.read_parquet(path)


def _build_strategy_filter(
    model_codes: list[str],
) -> tuple[set[str] | None, set[str] | None]:
    """Parse --model shortcodes into (strategy_filter, pair_filter) sets.

    Accepts either 2-part codes (lr-gl -> strategy filter only)
    or 3-part codes (eurusd-lr-gl -> both pair and strategy filter).
    Returns (None, None) when no codes given (= no filtering).
    """
    strategy_filter: set[str] = set()
    pair_filter: set[str] = set()

    for code in model_codes:
        parts = code.lower().split("-")
        if len(parts) == 3:
            pair, model_raw, session = parse_model_code(code)
            pair_filter.add(pair)
            strategy_filter.add(f"{model_raw.upper()}_{session}")
        elif len(parts) == 2:
            model_raw, session_alias = parts
            session = SESSION_ALIASES.get(session_alias, session_alias)
            strategy_filter.add(f"{model_raw.upper()}_{session}")
        else:
            raise ValueError(
                f"Bad --model code {code!r}. Use lr-gl (strategy only) or eurusd-lr-gl (pair+strategy)."
            )

    return (strategy_filter or None), (pair_filter or None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Unified ML strategy evaluation.\n\n"
            "Filter examples:\n"
            "  --model lr-gl lstm-gl      only global models, all pairs\n"
            "  --model eurusd-lr-gl       only EURUSD global LR\n"
            "  --model lr-gl lr-ldn       LR global + LR london, all pairs\n\n"
            "Session aliases: gl=global  ldn=london  ny=ny  as=asia"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pair",       nargs="+", default=["all"])
    parser.add_argument("--spreads",    default="0.5,1.0,2.0")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument(
        "--model", nargs="+", default=None, metavar="CODE",
        help="Shortcode(s) to filter e.g. lr-gl lstm-ldn eurusd-lr-gl",
    )
    args = parser.parse_args()

    pairs = list(PAIRS) if "all" in args.pair else args.pair
    spread_mults = [float(x) for x in args.spreads.split(",")]
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    strategy_filter, pair_filter = (None, None)
    if args.model:
        strategy_filter, pair_filter = _build_strategy_filter(args.model)

    if pair_filter:
        pairs = [p for p in pairs if p in pair_filter]

    rows: list[dict] = []
    dm_rows: list[dict] = []

    for pair in pairs:
        test_df = _load_test_df(pair)
        if test_df is None:
            print(f"[SKIP] {pair}: no test data")
            continue

        print(f"\n{'-'*60}")
        print(f"  {pair}")
        print(f"{'-'*60}")

        # collect bar returns for DM test
        pair_bar_rets: dict[str, np.ndarray] = {}

        for strategy_name in ML_STRATEGIES:
            if strategy_filter and strategy_name not in strategy_filter:
                continue
            if not _strategy_available(strategy_name, pair):
                print(f"  [SKIP] {strategy_name}: model not found at {_model_path(strategy_name, pair)}")
                continue

            for spread_mult in spread_mults:
                try:
                    row = evaluate_strategy(pair, strategy_name, test_df, spread_mult)
                    rows.append(row)
                    print(
                        f"  {strategy_name:<14}  spread*{spread_mult:.1f}  "
                        f"Sharpe={row['net_sharpe']:+.3f}  "
                        f"Sortino={row['sortino']:+.3f}  "
                        f"MaxDD={row['max_drawdown']:.3f}  "
                        f"WinRate={row['win_rate']:.2%}  "
                        f"Trades={row['n_trades']}"
                    )
                except Exception as exc:
                    warnings.warn(f"evaluate_strategy failed for {pair}/{strategy_name}: {exc}")

            # collect bar-returns at 1x spread for DM test
            if spread_mults and 1.0 in spread_mults:
                try:
                    strat   = get_strategy(strategy_name, pair=pair)
                    signals = strat.generate_signals(test_df)
                    session_key = _SESSION_OF_STRATEGY.get(strategy_name)
                    if session_key is not None:
                        signals = _filter_signals_by_session(
                            signals, test_df, _SESSION_FILTER_COLS[session_key]
                        )
                    pair_bar_rets[strategy_name] = _bar_returns_from_signals(
                        signals, test_df["close"]
                    )
                except Exception as exc:
                    warnings.warn(f"bar-return collection failed for {pair}/{strategy_name}: {exc}")

        # DM test: LSTM_global vs LR_global
        if "LSTM_global" in pair_bar_rets and "LR_global" in pair_bar_rets:
            dm = diebold_mariano_test(
                pair_bar_rets["LSTM_global"], pair_bar_rets["LR_global"]
            )
            dm_rows.append({"pair": pair, **dm})
            sign = ">" if dm["dm_stat"] > 0 else "<"
            print(
                f"\n  DM test (LSTM_global {sign} LR_global): "
                f"stat={dm['dm_stat']:+.4f}  p={dm['p_value']:.4f}  n={dm['n']}"
            )

    if not rows:
        print("\nNo results to save - no models found or no test data available.")
        return

    comparison_df = pd.DataFrame(rows)
    csv_path  = output_dir / "ml_comparison.csv"
    html_path = output_dir / "ml_comparison.html"

    comparison_df.to_csv(csv_path, index=False)

    html_body = comparison_df.to_html(index=False, float_format=lambda x: f"{x:.4f}")
    dm_df     = pd.DataFrame(dm_rows) if dm_rows else None
    dm_html   = dm_df.to_html(index=False, float_format=lambda x: f"{x:.4f}") if dm_df is not None else ""

    html_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ML Strategy Comparison</title>
<style>
  body  {{ font-family: monospace; margin: 2rem; }}
  table {{ border-collapse: collapse; margin-bottom: 2rem; }}
  th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: right; }}
  th {{ background: #eee; }}
</style>
</head>
<body>
<h1>ML Strategy Comparison</h1>
{html_body}
<h2>Diebold-Mariano Test (LSTM_global vs LR_global)</h2>
{dm_html}
</body>
</html>
"""
    html_path.write_text(html_page, encoding="utf-8")

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {html_path}")

    if dm_rows:
        print("\nDiebold-Mariano summary:")
        print(pd.DataFrame(dm_rows).to_string(index=False))


if __name__ == "__main__":
    main()
