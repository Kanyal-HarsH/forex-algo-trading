from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from backtest.strategies import STRATEGY_REGISTRY, get_strategy
from config.constants import (
    ANNUALISATION_FACTOR,
    PROFIT_FACTOR_CAP,
    ROLLING_SHARPE_WINDOW,
    PAIRS,
    PAIR_SPREAD_PIPS,
    PAIR_PIP_SIZES,
    fold_parquet_path,
)

_PIP = PAIR_PIP_SIZES
_DEFAULT_SPREAD: dict[str, float] = dict(PAIR_SPREAD_PIPS)

_SESSION_HOURS: dict[str, tuple[int, int]] = {
    "london":  (7,  16),
    "ny":      (13, 22),
    "asia":    (23,  8),
    "overlap": (13, 16),
}


def _in_session(hour: pd.Series, session: str) -> pd.Series:
    if session not in _SESSION_HOURS:
        valid = list(_SESSION_HOURS.keys())
        raise ValueError(f"Unknown session '{session}'. Valid options: {valid}")
    start, end = _SESSION_HOURS[session]
    if start < end:
        return (hour >= start) & (hour < end)
    return (hour >= start) | (hour < end)


@dataclass
class BacktestResult:
    pair:           str
    strategy:       str
    split:          str
    mode:           str
    direction_mode: str
    fold_index:     int

    equity:         List[float] = field(default_factory=list)
    equity_dollars: List[float] = field(default_factory=list)
    rolling_sharpe: List[float] = field(default_factory=list)
    timestamps:     List[str]   = field(default_factory=list)

    signal_dist:    dict        = field(default_factory=dict)

    net_sharpe:      float = 0.0
    gross_sharpe:    float = 0.0
    total_return:    float = 0.0
    max_drawdown:    float = 0.0
    calmar:          float = 0.0
    win_rate:        float = 0.0
    profit_factor:   float = 0.0
    avg_trade_bars:  float = 0.0
    turnover:        float = 0.0
    sortino:         float = 0.0
    n_trades:        int   = 0
    capital_initial: float = 10_000.0
    capital_final:   float = 0.0

    trade_log: List[dict] = field(default_factory=list)

    @property
    def metrics(self) -> dict:
        return {
            "net_sharpe":      self.net_sharpe,
            "gross_sharpe":    self.gross_sharpe,
            "total_return":    self.total_return,
            "max_drawdown":    self.max_drawdown,
            "calmar":          self.calmar,
            "win_rate":        self.win_rate,
            "profit_factor":   self.profit_factor,
            "avg_trade_bars":  self.avg_trade_bars,
            "turnover":        self.turnover,
            "sortino":         self.sortino,
            "n_trades":        self.n_trades,
            "capital_initial": self.capital_initial,
            "capital_final":   self.capital_final,
        }


def _sharpe(returns: np.ndarray, periods_per_year: int = ANNUALISATION_FACTOR) -> float:
    if len(returns) < 2:
        return 0.0
    std = returns.std()
    if std == 0:
        return 0.0
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def _sortino(returns: np.ndarray, periods_per_year: int = ANNUALISATION_FACTOR) -> float:
    downside = returns[returns < 0]
    if len(downside) < 2:
        return 0.0
    dd_std = downside.std()
    if dd_std == 0:
        return 0.0
    return float(returns.mean() / dd_std * np.sqrt(periods_per_year))


def _max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min())


def _rolling_sharpe(returns: np.ndarray, window: int = ROLLING_SHARPE_WINDOW) -> list[float]:
    out = []
    for i in range(len(returns)):
        w = returns[max(0, i - window + 1): i + 1]
        out.append(_sharpe(w) if len(w) >= 2 else 0.0)
    return out


def _resample_df(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    df = df.copy()
    df = df.set_index("timestamp_utc")
    agg = {
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }
    existing = {k: v for k, v in agg.items() if k in df.columns}
    df = df.resample(freq).agg(existing).dropna(subset=["close"])
    df = df.reset_index()
    return df


def _simulate_loop(
    price_arr:        np.ndarray,
    pos_arr:          np.ndarray,
    spread:           float,
    tp_dist:          Optional[float],
    sl_dist:          Optional[float],
    max_hold_bars:    Optional[int],
    capital_initial:  float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], float]:
    n = len(price_arr)
    equity_curve = np.ones(n)
    dollar_curve = np.full(n, capital_initial)
    bar_returns  = np.zeros(n)

    trade_log: list[dict] = []
    open_trade: Optional[dict] = None
    capital = capital_initial

    _equity_base = capital_initial if capital_initial != 0.0 else 1.0

    for i in range(1, n):
        cur_pos = pos_arr[i]
        price   = price_arr[i]
        prev_p  = price_arr[i - 1]

        if open_trade is not None:
            d           = open_trade["direction"]
            ep          = open_trade["entry_price"]
            bars_held   = i - open_trade["entry_bar"]
            exit_reason = None

            if tp_dist and d == 1  and price >= ep + tp_dist:
                exit_reason = "TP"
            elif tp_dist and d == -1 and price <= ep - tp_dist:
                exit_reason = "TP"
            elif sl_dist and d == 1  and price <= ep - sl_dist:
                exit_reason = "SL"
            elif sl_dist and d == -1 and price >= ep + sl_dist:
                exit_reason = "SL"
            elif max_hold_bars and bars_held >= max_hold_bars:
                exit_reason = "MAX_HOLD"
            elif cur_pos != d and cur_pos != 0:
                exit_reason = "SIGNAL_FLIP"

            if exit_reason:
                raw_ret     = d * (price - ep) / ep
                net_ret     = raw_ret - spread / ep
                pnl_dollars = net_ret * capital

                capital += pnl_dollars
                bar_returns[i]   = net_ret
                equity_curve[i]  = capital / _equity_base
                dollar_curve[i]  = capital

                trade_log.append({
                    "entry_bar":   open_trade["entry_bar"],
                    "exit_bar":    i,
                    "bars_held":   bars_held,
                    "direction":   d,
                    "entry_price": round(ep, 6),
                    "exit_price":  round(price, 6),
                    "pnl_pct":     round(net_ret * 100, 6),
                    "pnl_dollars": round(pnl_dollars, 4),
                    "exit_reason": exit_reason,
                })
                open_trade = None
            else:
                raw_ret          = d * (price - prev_p) / prev_p
                bar_returns[i]   = raw_ret
                equity_curve[i]  = capital / _equity_base
                dollar_curve[i]  = capital

        else:
            equity_curve[i] = equity_curve[i - 1]
            dollar_curve[i] = dollar_curve[i - 1]

        if open_trade is None and cur_pos != 0:
            open_trade = {
                "entry_bar":   i,
                "entry_price": price,
                "direction":   cur_pos,
            }

    if open_trade is not None:
        i         = n - 1
        d         = open_trade["direction"]
        ep        = open_trade["entry_price"]
        price     = price_arr[i]
        bars_held = i - open_trade["entry_bar"]
        raw_ret   = d * (price - ep) / ep
        net_ret   = raw_ret - spread / ep
        pnl_dollars = net_ret * capital
        capital  += pnl_dollars
        trade_log.append({
            "entry_bar":   open_trade["entry_bar"],
            "exit_bar":    i,
            "bars_held":   bars_held,
            "direction":   d,
            "entry_price": round(ep, 6),
            "exit_price":  round(price, 6),
            "pnl_pct":     round(net_ret * 100, 6),
            "pnl_dollars": round(pnl_dollars, 4),
            "exit_reason": "END_OF_DATA",
        })

    return equity_curve, dollar_curve, bar_returns, trade_log, capital


def _compute_metrics(
    equity:      np.ndarray,
    bar_returns: np.ndarray,
    trade_log:   list[dict],
    pos_arr:     np.ndarray,
) -> dict:
    total_return = float(equity[-1] - 1.0)
    mdd          = _max_drawdown(equity)
    calmar_v     = (total_return / abs(mdd)) if mdd < 0 else 0.0

    # Trade-level Sharpe/Sortino - consistent with total_return/capital_final.
    # bar_returns stores MTM returns during holding PLUS the full entry->exit
    # return on the closing bar, double-counting the price path.  This makes
    # bar-level Sharpe unreliable for TP/SL strategies (positive Sharpe while
    # the account loses money).  Trade returns are the ground truth.
    if trade_log:
        trade_rets = np.array([t["pnl_pct"] / 100.0 for t in trade_log])
        avg_hold   = float(np.mean([t["bars_held"] for t in trade_log]))
        # Annualise by trade frequency: bars_per_year / avg_bars_per_trade
        tpy        = int(max(1.0, ANNUALISATION_FACTOR / max(avg_hold, 1.0)))
        net_sh     = _sharpe(trade_rets,   periods_per_year=tpy)
        sortino_v  = _sortino(trade_rets,  periods_per_year=tpy)
        pos_rets   = trade_rets[trade_rets > 0]
        gross_sh   = _sharpe(pos_rets, periods_per_year=tpy) if len(pos_rets) > 1 else 0.0
    else:
        net_sh    = 0.0
        sortino_v = 0.0
        gross_sh  = 0.0

    wins     = [t for t in trade_log if t["pnl_pct"] >= 0]
    losses   = [t for t in trade_log if t["pnl_pct"] < 0]
    n_trades = len(trade_log)
    win_rate = len(wins) / n_trades if n_trades else 0.0

    gross_profit = sum(t["pnl_pct"] for t in wins)
    gross_loss   = abs(sum(t["pnl_pct"] for t in losses))
    pf = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    pf = min(pf, PROFIT_FACTOR_CAP)

    avg_bars = float(np.mean([t["bars_held"] for t in trade_log])) if trade_log else 0.0
    turnover = float((pos_arr != 0).mean())

    return {
        "total_return":   total_return,
        "max_drawdown":   mdd,
        "net_sharpe":     net_sh,
        "sortino":        sortino_v,
        "gross_sharpe":   gross_sh,
        "calmar":         calmar_v,
        "win_rate":       win_rate,
        "profit_factor":  pf,
        "avg_trade_bars": avg_bars,
        "turnover":       turnover,
        "n_trades":       n_trades,
    }


def run_backtest(
    signals:         pd.Series,
    prices:          pd.Series,
    pair:            str,
    strategy:        str,
    split:           str              = "val",
    spread_pips:     Optional[float]  = None,
    tp_pips:         Optional[float]  = None,
    sl_pips:         Optional[float]  = None,
    capital_initial: float            = 10_000.0,
    max_hold_bars:   Optional[int]    = None,
    timestamps:      Optional[list]   = None,
    session:         Optional[str]    = None,
    entry_time:      Optional[str]    = None,
    resample:        Optional[str]    = None,
    direction_mode:  str              = "long_short",
    mode:            str              = "research",
    fold_index:      int              = 0,
    df_full:         Optional[pd.DataFrame] = None,
) -> BacktestResult:
    pip = _PIP.get(pair, 0.0001)
    spread = (spread_pips if spread_pips is not None
              else _DEFAULT_SPREAD.get(pair, 1.0)) * pip

    prices = prices.reset_index(drop=True)
    signals = signals.reset_index(drop=True)

    if direction_mode == "long_only":
        signals = signals.where(signals >= 0, 0)
    elif direction_mode == "short_only":
        signals = signals.where(signals <= 0, 0)

    if (session or entry_time) and df_full is not None:
        ts = pd.to_datetime(df_full["timestamp_utc"], utc=True)
        mask = pd.Series(True, index=signals.index)
        if session:
            mask &= _in_session(ts.dt.hour.values, session)
        if entry_time:
            hh, mm = map(int, entry_time.split(":"))
            mask &= (ts.dt.hour > hh) | ((ts.dt.hour == hh) & (ts.dt.minute >= mm))
        signals = signals.where(mask.values, 0)

    pos = signals.replace(0, np.nan).ffill().fillna(0).astype(int)
    pos = pos.shift(1).fillna(0).astype(int)

    price_arr = prices.values.astype(float)
    pos_arr   = pos.values.astype(int)

    tp_dist = tp_pips * pip if tp_pips is not None else None
    sl_dist = sl_pips * pip if sl_pips is not None else None

    equity_curve, dollar_curve, bar_returns, trade_log, capital = _simulate_loop(
        price_arr=price_arr,
        pos_arr=pos_arr,
        spread=spread,
        tp_dist=tp_dist,
        sl_dist=sl_dist,
        max_hold_bars=max_hold_bars,
        capital_initial=capital_initial,
    )

    m = _compute_metrics(equity_curve, bar_returns, trade_log, pos_arr)

    sig_counts  = signals.value_counts().to_dict()
    signal_dist = {
        "Long":  int(sig_counts.get(1, 0)),
        "Short": int(sig_counts.get(-1, 0)),
        "Flat":  int(sig_counts.get(0, 0)),
    }

    roll_sh = _rolling_sharpe(bar_returns, window=ROLLING_SHARPE_WINDOW)
    ts_list = list(timestamps) if timestamps else []

    return BacktestResult(
        pair            = pair,
        strategy        = strategy,
        split           = split,
        mode            = mode,
        direction_mode  = direction_mode,
        fold_index      = fold_index,
        equity          = [round(float(v), 8) for v in equity_curve],
        equity_dollars  = [round(float(v), 4) for v in dollar_curve],
        rolling_sharpe  = [round(float(v), 6) for v in roll_sh],
        timestamps      = ts_list,
        signal_dist     = signal_dist,
        net_sharpe      = round(m["net_sharpe"],     6),
        gross_sharpe    = round(m["gross_sharpe"],   6),
        total_return    = round(m["total_return"],   8),
        max_drawdown    = round(m["max_drawdown"],   8),
        calmar          = round(m["calmar"],         6),
        win_rate        = round(m["win_rate"],       6),
        profit_factor   = round(m["profit_factor"],  6),
        avg_trade_bars  = round(m["avg_trade_bars"], 4),
        turnover        = round(m["turnover"],       6),
        sortino         = round(m["sortino"],        6),
        n_trades        = m["n_trades"],
        capital_initial = capital_initial,
        capital_final   = round(capital, 4),
        trade_log       = trade_log,
    )


def run_wf_folds(
    pair:            str,
    strategy:        str,
    n_folds:         int             = 5,
    split:           str             = "train",
    spread_pips:     Optional[float] = None,
    tp_pips:         Optional[float] = None,
    sl_pips:         Optional[float] = None,
    capital_initial: float           = 10_000.0,
    max_hold_bars:   Optional[int]   = None,
    session:         Optional[str]   = None,
    entry_time:      Optional[str]   = None,
    resample:        Optional[str]   = None,
    direction_mode:  str             = "long_short",
    mode:            str             = "research",
) -> list[BacktestResult]:
    results = []
    strat   = get_strategy(strategy, pair=pair)

    for k in range(n_folds):
        fold_path = fold_parquet_path(pair, k, "train")
        if not fold_path.exists():
            raise FileNotFoundError(
                f"Fold parquet not found: {fold_path}\n"
                f"Run scripts/split_fx_data.py --force-folds to generate "
                f"datasets/folds/fold_{k}/ before calling run_wf_folds."
            )

        fold_df = pd.read_parquet(fold_path)
        fold_df["timestamp_utc"] = pd.to_datetime(fold_df["timestamp_utc"], utc=True)
        fold_df = fold_df.sort_values("timestamp_utc").reset_index(drop=True)

        if resample:
            fold_df = _resample_df(fold_df, resample)

        signals    = strat.generate_signals(fold_df).reset_index(drop=True)
        prices     = fold_df["close"].reset_index(drop=True)
        timestamps = (
            fold_df["timestamp_utc"]
            .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            .tolist()
        )

        r = run_backtest(
            signals         = signals,
            prices          = prices,
            pair            = pair,
            strategy        = strat.name,
            split           = f"fold_{k}",
            spread_pips     = spread_pips,
            tp_pips         = tp_pips,
            sl_pips         = sl_pips,
            capital_initial = capital_initial,
            max_hold_bars   = max_hold_bars,
            timestamps      = timestamps,
            session         = session,
            entry_time      = entry_time,
            direction_mode  = direction_mode,
            mode            = mode,
            fold_index      = k,
            df_full         = fold_df,
        )
        results.append(r)

    return results
