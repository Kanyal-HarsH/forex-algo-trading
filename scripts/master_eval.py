#!/usr/bin/env python3
"""
master_eval.py - Unified master evaluation.

Part A: Rule-based strategies through T1 (per-pair survival) -> T2 (session/direction)
        -> T3 (TP/SL grid) -> T4 (walk-forward folds).
Part B: All ML models evaluated on all cross-session combinations (test split).
Part C: Unified rankings, DM tests, transfer matrices, feature importance.

Usage:
  python scripts/master_eval.py                           # full run, all pairs
  python scripts/master_eval.py --rule-based-only         # skip ML section
  python scripts/master_eval.py --ml-only                 # skip T1-T4
  python scripts/master_eval.py --pairs eurusd gbpusd     # subset
  python scripts/master_eval.py --ml-split val            # ML uses val instead of test
  python scripts/master_eval.py --workers 6
  python scripts/master_eval.py --out output/run1.txt
  python scripts/master_eval.py --t1-min-sharpe -0.2
  python scripts/master_eval.py --spreads 0.5 1.0 2.0
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import math
import os
import pickle
import time
import traceback
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing import freeze_support
from statistics import mean, pstdev
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

from backtest.engine import run_backtest, run_wf_folds
from backtest.strategies import STRATEGY_REGISTRY, get_strategy
from backtest.run_backtest import load_split_data
from config.constants import (
    PAIRS, PAIR_SPREAD_PIPS,
    SESSION_FILTER_MAP, SESSION_NAMES,
    MODELS_DIR, TEST_DIR, VAL_DIR,
    TP_SL_GRID as _CONFIG_TP_SL_GRID,
    TEST_START, TEST_END,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR  = PROJECT_DIR / "output" / "master_eval"

# --- Constants

ALL_STRATEGIES: list[str] = [
    s for s in STRATEGY_REGISTRY.keys() if not s.startswith(("LR_", "LSTM_"))
]
ML_STRATEGIES: list[str] = [
    "LR_global", "LR_london", "LR_ny", "LR_asia",
    "LSTM_global", "LSTM_london", "LSTM_ny", "LSTM_asia",
]
ALL_PAIRS:     list[str] = list(PAIRS)
RB_SESSIONS:   list[Optional[str]] = [None, "london", "ny"]
ALL_DIRECTIONS: list[str] = ["long_short", "long_only", "short_only"]
TP_SL_COMBOS:  list[tuple[int, int]] = [(int(tp), int(sl)) for tp, sl in _CONFIG_TP_SL_GRID]
EVAL_SESSIONS: list[str] = list(SESSION_NAMES)   # global, london, ny, asia
SPREAD_MULTS:  list[float] = [0.5, 1.0, 2.0]
MIN_TRADES     = 10
MAX_DD_GATE    = -0.95

STRATEGY_FAMILY: dict[str, str] = {}
for _s in ALL_STRATEGIES:
    if "MACrossover" in _s or "MACD" in _s:
        STRATEGY_FAMILY[_s] = "trend"
    elif "Momentum" in _s:
        STRATEGY_FAMILY[_s] = "momentum"
    elif "BB" in _s or "RSI" in _s:
        STRATEGY_FAMILY[_s] = "mean_reversion"
    elif "Donchian" in _s:
        STRATEGY_FAMILY[_s] = "breakout"
    else:
        STRATEGY_FAMILY[_s] = "other"

SESSION_EVAL_FILTER: dict[str, list[str]] = {
    "global": [],
    "london": SESSION_FILTER_MAP.get("london", ["London", "Overlap"]),
    "ny":     SESSION_FILTER_MAP.get("ny",     ["New_York", "Overlap"]),
    "asia":   SESSION_FILTER_MAP.get("asia",   ["Asia"]),
}

# --- Data classes

@dataclass
class RunSpec:
    tier:       str
    pair:       str
    strategy:   str
    split:      str
    session:    Optional[str]   = None
    direction:  str             = "long_short"
    tp_pips:    Optional[float] = None
    sl_pips:    Optional[float] = None
    folds:      int             = 0
    capital:    float           = 10_000.0
    spread:     Optional[float] = None
    max_hold:   Optional[int]   = None
    mode:       str             = "research"
    eval_session: Optional[str] = None
    spread_mult:  float         = 1.0


@dataclass
class RunOutcome:
    ok:            bool
    status:        str
    tier:          str
    pair:          str
    strategy:      str
    split:         str
    session:       Optional[str]
    direction:     str
    tp_pips:       Optional[float]
    sl_pips:       Optional[float]
    folds:         int
    net_sharpe:    float = 0.0
    gross_sharpe:  float = 0.0
    total_return:  float = 0.0
    max_drawdown:  float = 0.0
    calmar:        float = 0.0
    win_rate:      float = 0.0
    profit_factor: float = 0.0
    avg_trade_bars: float = 0.0
    turnover:      float = 0.0
    sortino:       float = 0.0
    n_trades:      int   = 0
    capital_final: float = 0.0
    signal_long:   int   = 0
    signal_short:  int   = 0
    signal_flat:   int   = 0
    fold_mean_sharpe:    float = 0.0
    fold_std_sharpe:     float = 0.0
    negative_fold_count: int   = 0
    composite_score: float = 0.0
    stability_score: float = 0.0
    grade:           str   = "F"
    eval_session:    Optional[str] = None
    spread_mult:     float = 1.0
    model_type:      Optional[str] = None
    train_session:   Optional[str] = None
    warning_flags:   list[str] = None
    notes:           str   = ""
    error:           str   = ""

    def __post_init__(self):
        if self.warning_flags is None:
            self.warning_flags = []


# --- Scoring

def _composite_score(net_sharpe: float, sortino: float, calmar: float,
                     max_drawdown: float, n_trades: int) -> float:
    """4-component composite score (0-100). No fold stability term."""
    if n_trades < MIN_TRADES or max_drawdown < MAX_DD_GATE:
        return 0.0
    s  = min(max(net_sharpe, 0.0), 5.0) / 5.0 * 35.0
    so = min(max(sortino,    0.0), 5.0) / 5.0 * 25.0
    ca = min(max(calmar,     0.0), 3.0) / 3.0 * 25.0
    dd = max(0.0, (1.0 + max_drawdown)) * 100.0 * 0.15
    return round(s + so + ca + dd, 2)


def _stability_score(fold_mean: float, fold_std: float) -> float:
    return round(fold_mean - 0.5 * fold_std, 4)


def _assign_grade(score: float) -> str:
    if score >= 80: return "A"
    if score >= 60: return "B"
    if score >= 40: return "C"
    if score >= 20: return "D"
    return "F"


def _score(r: RunOutcome) -> RunOutcome:
    r.composite_score = _composite_score(
        r.net_sharpe, r.sortino, r.calmar, r.max_drawdown, r.n_trades)
    r.grade = _assign_grade(r.composite_score)
    if r.folds > 0:
        r.stability_score = _stability_score(r.fold_mean_sharpe, r.fold_std_sharpe)
    return r


# --- Helpers

def _mean(v: list[float]) -> float:
    return mean(v) if v else 0.0


def _std(v: list[float]) -> float:
    return pstdev(v) if len(v) > 1 else 0.0


def _fmt(x: Any) -> str:
    if x is None: return "-"
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x): return "nan"
        return f"{x:.4f}"
    return str(x)


def _pct(x: float) -> str:
    if math.isnan(x): return "  nan%"
    return f"{x * 100:+.2f}%"


def _dur(s: float) -> str:
    s = max(0, int(s))
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# --- Baselines

def _bah_return(pair: str, split: str) -> float:
    try:
        df = load_split_data(pair, split)
        p  = df["close"].values
        return float((p[-1] - p[0]) / p[0]) if len(p) > 1 else 0.0
    except Exception:
        return float("nan")


# --- Bar returns for DM test

def _bar_rets(signals: pd.Series, prices: pd.Series) -> np.ndarray:
    sig = signals.values[:-1].astype(float)
    ret = np.diff(prices.values) / np.where(prices.values[:-1] == 0, 1, prices.values[:-1])
    return sig * ret


# --- DM test (HLN-corrected)

def _dm_test(a: np.ndarray, b: np.ndarray) -> dict:
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if n < 10:
        return {"dm_stat": float("nan"), "p_value": 1.0, "n": n, "sig": False}
    d    = a - b
    dbar = np.mean(d)
    Vd   = np.var(d, ddof=1) / n
    if Vd <= 0:
        return {"dm_stat": float("nan"), "p_value": 1.0, "n": n, "sig": False}
    dm = dbar / math.sqrt(Vd)
    pv = float(2 * stats.t.sf(abs(dm), df=n - 1))
    return {"dm_stat": round(dm, 4), "p_value": round(pv, 4), "n": n, "sig": pv < 0.05}


# --- ML helpers

def _ml_model_path(strategy_name: str, pair: str) -> Path:
    model_type, session = strategy_name.split("_", 1)
    is_lr = model_type.upper() == "LR"
    ext   = "pkl" if is_lr else "pt"
    fname = f"{pair}_logreg_model.{ext}" if is_lr else f"{pair}_lstm_model.{ext}"
    if session == "global":
        return MODELS_DIR / "global" / fname
    return MODELS_DIR / "session" / session / fname


def _ml_available(strategy_name: str, pair: str) -> bool:
    return _ml_model_path(strategy_name, pair).exists()


def _filter_signals(signals: pd.Series, df: pd.DataFrame, eval_session: str) -> pd.Series:
    vals = SESSION_EVAL_FILTER.get(eval_session, [])
    if not vals:
        return signals
    if "session" not in df.columns:
        return signals
    out = signals.copy()
    out[~df["session"].isin(vals)] = 0
    return out


# --- Rule-based: worker

def _outcome_base(spec: RunSpec, **kw: Any) -> RunOutcome:
    base = dict(tier=spec.tier, pair=spec.pair, strategy=spec.strategy,
                split=spec.split, session=spec.session, direction=spec.direction,
                tp_pips=spec.tp_pips, sl_pips=spec.sl_pips, folds=spec.folds)
    base.update(kw)
    return RunOutcome(**base)


def _validate(spec: RunSpec) -> list[str]:
    flags: list[str] = []
    spread = spec.spread if spec.spread is not None else PAIR_SPREAD_PIPS.get(spec.pair, 1.0)
    if (spec.tp_pips is None) != (spec.sl_pips is None):
        flags.append("PARTIAL_TP_SL")
    if spec.tp_pips is not None and spec.tp_pips <= spread:
        flags.append("TP_LE_SPREAD")
    if spec.sl_pips is not None and spec.sl_pips <= spread:
        flags.append("SL_LE_SPREAD")
    return flags


def _exec_folds(spec: RunSpec, flags: list[str]) -> RunOutcome:
    folds = run_wf_folds(
        pair=spec.pair, strategy=spec.strategy, n_folds=spec.folds,
        split="train", spread_pips=spec.spread,
        tp_pips=spec.tp_pips, sl_pips=spec.sl_pips,
        capital_initial=spec.capital, max_hold_bars=spec.max_hold,
        session=spec.session, direction_mode=spec.direction, mode=spec.mode,
    )
    sh  = [f.net_sharpe for f in folds]
    neg = sum(1 for s in sh if s < 0)
    last = folds[-1]
    r = _outcome_base(
        spec, ok=True, status="OK",
        net_sharpe=_mean(sh),
        gross_sharpe=last.gross_sharpe,
        total_return=_mean([f.total_return  for f in folds]),
        max_drawdown=_mean([f.max_drawdown  for f in folds]),
        calmar      =_mean([f.calmar        for f in folds]),
        win_rate    =_mean([f.win_rate      for f in folds]),
        profit_factor=_mean([f.profit_factor for f in folds]),
        avg_trade_bars=_mean([f.avg_trade_bars for f in folds]),
        turnover    =_mean([f.turnover      for f in folds]),
        sortino     =_mean([f.sortino       for f in folds]),
        n_trades    =sum(f.n_trades         for f in folds),
        capital_final=last.capital_final,
        signal_long =last.signal_dist.get("Long",  0),
        signal_short=last.signal_dist.get("Short", 0),
        signal_flat =last.signal_dist.get("Flat",  0),
        fold_mean_sharpe=_mean(sh),
        fold_std_sharpe =_std(sh),
        negative_fold_count=neg,
        warning_flags=flags, notes="WF_FOLDS",
    )
    return _score(r)


def _exec_single(spec: RunSpec, df: pd.DataFrame, flags: list[str]) -> RunOutcome:
    strat  = get_strategy(spec.strategy)
    warmup = strat.warmup_bars
    if len(df) <= warmup:
        return _outcome_base(spec, ok=False, status="SKIP_WARMUP_GT_DATA",
                             warning_flags=flags + ["WARMUP_GT_DATA"],
                             notes=f"rows={len(df)} warmup={warmup}")
    df2  = df.reset_index(drop=True)
    sigs = strat.generate_signals(df2).reset_index(drop=True)
    if sigs.abs().sum() == 0:
        return _outcome_base(spec, ok=False, status="ZERO_SIGNALS",
                             warning_flags=flags + ["ZERO_SIGNALS"])
    ts = (df2["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
          if "timestamp_utc" in df2.columns else [])
    bt = run_backtest(
        signals=sigs, prices=df2["close"],
        pair=spec.pair, strategy=strat.name, split=spec.split,
        spread_pips=spec.spread, tp_pips=spec.tp_pips, sl_pips=spec.sl_pips,
        capital_initial=spec.capital, max_hold_bars=spec.max_hold,
        timestamps=ts, session=spec.session,
        direction_mode=spec.direction, mode=spec.mode, df_full=df2,
    )
    if bt.n_trades == 0: flags.append("ZERO_TRADES")
    if bt.profit_factor >= 998: flags.append("PF_CAPPED")
    r = _outcome_base(
        spec, ok=True,
        status="ZERO_TRADES" if bt.n_trades == 0 else "OK",
        net_sharpe=bt.net_sharpe, gross_sharpe=bt.gross_sharpe,
        total_return=bt.total_return, max_drawdown=bt.max_drawdown,
        calmar=bt.calmar, win_rate=bt.win_rate, profit_factor=bt.profit_factor,
        avg_trade_bars=bt.avg_trade_bars, turnover=bt.turnover, sortino=bt.sortino,
        n_trades=bt.n_trades, capital_final=bt.capital_final,
        signal_long=bt.signal_dist.get("Long", 0),
        signal_short=bt.signal_dist.get("Short", 0),
        signal_flat=bt.signal_dist.get("Flat", 0),
        warning_flags=flags,
    )
    return _score(r)


def _single_run(spec: RunSpec) -> RunOutcome:
    """Worker - runs in a subprocess. Must be top-level for pickling."""
    try:
        flags = _validate(spec)
        if spec.folds > 0:
            return _exec_folds(spec, flags)
        df = load_split_data(spec.pair, spec.split)
        if "timestamp_utc" in df.columns:
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
            df = df.sort_values("timestamp_utc").drop_duplicates("timestamp_utc").reset_index(drop=True)
        return _exec_single(spec, df, flags)
    except FileNotFoundError as e:
        return _outcome_base(spec, ok=False, status="MISSING_FILE", error=str(e))
    except Exception as e:
        return _outcome_base(spec, ok=False, status="ENGINE_ERROR",
                             error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


# --- Parallel runner

def _run_parallel(specs: list[RunSpec], workers: int, label: str) -> list[RunOutcome]:
    if not specs:
        print(f"  [{label}] No specs.")
        return []
    results: list[RunOutcome] = []
    total = len(specs)
    t0    = time.perf_counter()
    last  = t0
    ctr:  Counter[str] = Counter()
    print(f"\n  [{label}] {total} runs | workers={workers}")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fmap = {ex.submit(_single_run, s): s for s in specs}
        for idx, fut in enumerate(as_completed(fmap), 1):
            spec = fmap[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = _outcome_base(spec, ok=False, status="FUTURE_ERROR",
                                    error=f"{type(e).__name__}: {e}")
            results.append(res)
            ctr[res.status] += 1
            now = time.perf_counter()
            if idx == total or idx == 1 or (now - last) >= 10:
                eta = (now - t0) / idx * (total - idx)
                top = ", ".join(f"{k}:{v}" for k, v in ctr.most_common(3))
                print(f"  [{label}] {idx}/{total} ({idx/total*100:.1f}%) ETA {_dur(eta)} | {top}")
                last = now
    print(f"  [{label}] Done {_dur(time.perf_counter() - t0)}")
    return results


# --- T1: per-pair survival

def _select_t1_survivors(results: list[RunOutcome], min_sharpe: float) -> dict[str, list[str]]:
    """Returns {pair: [strategy, ...]} - top 40% per pair by val net_sharpe."""
    survivors: dict[str, list[str]] = {}
    for pair in ALL_PAIRS:
        scores: dict[str, float] = {}
        for strat in ALL_STRATEGIES:
            rows = [r for r in results if r.pair == pair and r.strategy == strat
                    and r.split == "val" and r.ok and r.status in ("OK", "ZERO_TRADES")]
            if rows:
                scores[strat] = _mean([r.net_sharpe for r in rows])
        if not scores:
            continue
        qualified = {s: v for s, v in scores.items() if v > min_sharpe}
        ranked    = sorted(qualified.items(), key=lambda kv: kv[1], reverse=True)
        keep_n    = max(1, int(len(ranked) * 0.4))
        top       = [s for s, _ in ranked[:keep_n]]
        # Family diversity floor: at least 1 per family regardless of score
        family_map = [
            ("trend",          [s for s in ALL_STRATEGIES if "MACrossover" in s or "MACD" in s]),
            ("momentum",       [s for s in ALL_STRATEGIES if "Momentum" in s]),
            ("mean_reversion", [s for s in ALL_STRATEGIES if "BB" in s or "RSI" in s]),
            ("breakout",       [s for s in ALL_STRATEGIES if "Donchian" in s]),
        ]
        for _fam, fstrats in family_map:
            if not any(s in top for s in fstrats):
                cands = [(s, scores[s]) for s in fstrats if s in scores]
                if cands:
                    best = max(cands, key=lambda x: x[1])[0]
                    if best not in top:
                        top.append(best)
        if top:
            survivors[pair] = top
    return survivors


def _build_t1_specs(pairs: list[str]) -> list[RunSpec]:
    return [RunSpec(tier="T1", pair=p, strategy=s, split="val")
            for s in ALL_STRATEGIES for p in pairs]


def _build_t2_specs(survivors: dict[str, list[str]]) -> list[RunSpec]:
    specs = []
    for pair, strats in survivors.items():
        for strat in strats:
            for sess in RB_SESSIONS:
                for dirn in ALL_DIRECTIONS:
                    specs.append(RunSpec(tier="T2", pair=pair, strategy=strat,
                                        split="val", session=sess, direction=dirn))
    return specs


def _select_t2_best(results: list[RunOutcome], survivors: dict[str, list[str]]) -> list[RunSpec]:
    out = []
    for pair, strats in survivors.items():
        for strat in strats:
            rows = [r for r in results if r.pair == pair and r.strategy == strat and r.ok]
            if not rows:
                continue
            best = max(rows, key=lambda r: r.composite_score)
            out.append(RunSpec(tier="T3", pair=pair, strategy=strat, split="val",
                               session=best.session, direction=best.direction))
    return out


def _build_t3_specs(t2_best: list[RunSpec]) -> list[RunSpec]:
    specs = []
    for base in t2_best:
        for tp, sl in TP_SL_COMBOS:
            specs.append(RunSpec(tier="T3", pair=base.pair, strategy=base.strategy,
                                split="val", session=base.session, direction=base.direction,
                                tp_pips=tp, sl_pips=sl))
    return specs


def _select_t3_best(results: list[RunOutcome], t2_best: list[RunSpec],
                    t2_results: list[RunOutcome]) -> list[RunSpec]:
    """Pick best TP/SL combo per (pair, strategy), but only if it beats the T2
    no-TP/SL baseline; otherwise carry through with no TP/SL."""
    # Index T2 baselines (no TP/SL) by (pair, strategy, session, direction)
    t2_score: dict[tuple, float] = {}
    for r in t2_results:
        if not r.ok:
            continue
        key = (r.pair, r.strategy, r.session, r.direction)
        t2_score[key] = max(t2_score.get(key, -1e9), r.composite_score)

    out = []
    for spec in t2_best:
        rows = [r for r in results if r.pair == spec.pair and r.strategy == spec.strategy
                and r.session == spec.session and r.direction == spec.direction and r.ok]
        baseline = t2_score.get((spec.pair, spec.strategy, spec.session, spec.direction), -1e9)
        if not rows or max(r.composite_score for r in rows) <= baseline:
            # No TP/SL combo improved - carry through with no TP/SL
            out.append(RunSpec(tier="T4", pair=spec.pair, strategy=spec.strategy,
                               split="train", session=spec.session, direction=spec.direction,
                               folds=5))
            continue
        best = max(rows, key=lambda r: r.composite_score)
        out.append(RunSpec(tier="T4", pair=best.pair, strategy=best.strategy,
                           split="train", session=best.session, direction=best.direction,
                           tp_pips=best.tp_pips, sl_pips=best.sl_pips, folds=5))
    return out


def _final_rank_t4(results: list[RunOutcome]) -> list[RunOutcome]:
    good = [r for r in results if r.ok and r.status == "OK"
            and r.negative_fold_count <= 2
            and r.n_trades >= MIN_TRADES]
    return sorted(good, key=lambda r: r.stability_score, reverse=True)


def _build_t5_specs(t4_finals: list[RunOutcome],
                    date_from: Optional[str], date_to: Optional[str]) -> list[RunSpec]:
    """Re-run T4 survivors on test split - final out-of-sample evaluation."""
    specs = []
    for r in t4_finals:
        specs.append(RunSpec(
            tier="T5", pair=r.pair, strategy=r.strategy,
            split="test", session=r.session, direction=r.direction,
            tp_pips=r.tp_pips, sl_pips=r.sl_pips, folds=0,
            # date_from/date_to are handled by load_split_data caller via spec notes
        ))
    return specs


def _run_t5(t4_finals: list[RunOutcome], workers: int,
            date_from: Optional[str], date_to: Optional[str]) -> list[RunOutcome]:
    """Run T4 survivors on test split with optional date window."""
    if not t4_finals:
        return []
    results: list[RunOutcome] = []
    total = len(t4_finals)
    t0 = time.perf_counter()
    print(f"\n  [T5] {total} test-split runs | date={date_from or 'start'}->{date_to or 'end'}")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fmap = {}
        for r in t4_finals:
            spec = RunSpec(
                tier="T5", pair=r.pair, strategy=r.strategy,
                split="test", session=r.session, direction=r.direction,
                tp_pips=r.tp_pips, sl_pips=r.sl_pips, folds=0,
            )
            fmap[ex.submit(_single_run_dated, spec, date_from, date_to)] = spec
        for idx, fut in enumerate(as_completed(fmap), 1):
            spec = fmap[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = _outcome_base(spec, ok=False, status="FUTURE_ERROR",
                                    error=f"{type(e).__name__}: {e}")
            results.append(res)
            if idx == total or idx == 1 or idx % max(1, total // 5) == 0:
                print(f"  [T5] {idx}/{total} ({idx/total*100:.1f}%)")
    print(f"  [T5] Done {_dur(time.perf_counter() - t0)}")
    return results


def _single_run_dated(spec: RunSpec, date_from: Optional[str],
                      date_to: Optional[str]) -> RunOutcome:
    """Like _single_run but applies a date window after loading the split."""
    try:
        flags = _validate(spec)
        df = load_split_data(spec.pair, spec.split)
        if "timestamp_utc" in df.columns:
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
            df = df.sort_values("timestamp_utc").drop_duplicates("timestamp_utc").reset_index(drop=True)
            if date_from:
                df = df[df["timestamp_utc"] >= pd.Timestamp(date_from, tz="UTC")]
            if date_to:
                df = df[df["timestamp_utc"] <= pd.Timestamp(date_to, tz="UTC")]
            df = df.reset_index(drop=True)
        if len(df) == 0:
            return _outcome_base(spec, ok=False, status="NO_DATA_IN_WINDOW",
                                 notes=f"date_from={date_from} date_to={date_to}")
        return _exec_single(spec, df, flags)
    except FileNotFoundError as e:
        return _outcome_base(spec, ok=False, status="MISSING_FILE", error=str(e))
    except Exception as e:
        return _outcome_base(spec, ok=False, status="ENGINE_ERROR",
                             error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


# --- ML evaluation

def _gen_ml_signals(pair: str, strat_name: str,
                    df: pd.DataFrame) -> tuple[str, str, Optional[pd.Series], Optional[np.ndarray]]:
    if not _ml_available(strat_name, pair):
        return pair, strat_name, None, None
    try:
        strat = get_strategy(strat_name, pair=pair)
        sigs  = strat.generate_signals(df.reset_index(drop=True)).reset_index(drop=True)
        prices = df["close"].reset_index(drop=True).values
        return pair, strat_name, sigs, prices
    except Exception as e:
        warnings.warn(f"Signal gen failed {strat_name} {pair}: {e}")
        return pair, strat_name, None, None


def _run_ml_bt(pair: str, strat_name: str, sigs: pd.Series, prices: np.ndarray,
               df: pd.DataFrame, eval_sess: str, spread_mult: float) -> RunOutcome:
    model_type, train_sess = strat_name.split("_", 1)
    spread   = PAIR_SPREAD_PIPS.get(pair, 1.0) * spread_mult
    filtered = _filter_signals(sigs, df, eval_sess)
    prices_s = pd.Series(prices, index=sigs.index)
    ts = (df["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
          if "timestamp_utc" in df.columns else [])
    try:
        bt = run_backtest(
            signals=filtered, prices=prices_s,
            pair=pair, strategy=strat_name, split="test",
            spread_pips=spread, capital_initial=10_000.0,
            timestamps=ts, df_full=df,
        )
        r = RunOutcome(
            ok=True, status="OK",
            tier="ML", pair=pair, strategy=strat_name, split="test",
            session=None, direction="long_short", tp_pips=None, sl_pips=None, folds=0,
            net_sharpe=bt.net_sharpe, gross_sharpe=bt.gross_sharpe,
            total_return=bt.total_return, max_drawdown=bt.max_drawdown,
            calmar=bt.calmar, win_rate=bt.win_rate, profit_factor=bt.profit_factor,
            avg_trade_bars=bt.avg_trade_bars, turnover=bt.turnover, sortino=bt.sortino,
            n_trades=bt.n_trades, capital_final=bt.capital_final,
            signal_long=bt.signal_dist.get("Long", 0),
            signal_short=bt.signal_dist.get("Short", 0),
            signal_flat=bt.signal_dist.get("Flat", 0),
            eval_session=eval_sess, spread_mult=spread_mult,
            model_type=model_type.lower(), train_session=train_sess,
        )
        return _score(r)
    except Exception as e:
        return RunOutcome(
            ok=False, status="ENGINE_ERROR",
            tier="ML", pair=pair, strategy=strat_name, split="test",
            session=None, direction="long_short", tp_pips=None, sl_pips=None, folds=0,
            eval_session=eval_sess, spread_mult=spread_mult,
            model_type=model_type.lower(), train_session=train_sess,
            error=str(e),
        )


def _run_ml_evaluation(pairs: list[str], spread_mults: list[float],
                       eval_sessions: list[str], ml_split: str,
                       workers: int,
                       date_from: Optional[str] = None,
                       date_to:   Optional[str] = None) -> list[RunOutcome]:
    # Load data
    dfs: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        try:
            df = load_split_data(pair, ml_split)
            if "timestamp_utc" in df.columns:
                df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
                df = df.sort_values("timestamp_utc").drop_duplicates("timestamp_utc").reset_index(drop=True)
                if date_from:
                    df = df[df["timestamp_utc"] >= pd.Timestamp(date_from, tz="UTC")]
                if date_to:
                    df = df[df["timestamp_utc"] <= pd.Timestamp(date_to, tz="UTC")]
                df = df.reset_index(drop=True)
            if len(df) == 0:
                warnings.warn(f"No data for {pair} after date filter [{date_from} - {date_to}]")
                continue
            dfs[pair] = df
        except Exception as e:
            warnings.warn(f"Cannot load {ml_split} for {pair}: {e}")

    # Phase 1: generate signals (cached)
    tasks = [(p, s) for p in pairs for s in ML_STRATEGIES if p in dfs and _ml_available(s, p)]
    print(f"\n  [ML-SIGNALS] {len(tasks)} models available")
    cache: dict[tuple[str, str], tuple[pd.Series, np.ndarray]] = {}
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(tasks)))) as ex:
        futs = {ex.submit(_gen_ml_signals, p, s, dfs[p]): (p, s) for p, s in tasks}
        for fut in as_completed(futs):
            p, s, sigs, prices = fut.result()
            if sigs is not None:
                cache[(p, s)] = (sigs, prices)
    print(f"  [ML-SIGNALS] {len(cache)} cached")

    # Phase 2: backtests
    bt_args = [(p, s, sigs, prices, dfs[p], es, sm)
               for (p, s), (sigs, prices) in cache.items()
               for es in eval_sessions for sm in spread_mults]
    print(f"\n  [ML-BACKTEST] {len(bt_args)} backtests")
    results: list[RunOutcome] = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(workers * 2, os.cpu_count() or 4)) as ex:
        futs2 = [ex.submit(_run_ml_bt, *a) for a in bt_args]
        for idx, fut in enumerate(as_completed(futs2), 1):
            results.append(fut.result())
            if idx % max(1, len(bt_args) // 10) == 0 or idx == len(bt_args):
                print(f"  [ML-BACKTEST] {idx}/{len(bt_args)} ({idx/len(bt_args)*100:.1f}%)")
    print(f"  [ML-BACKTEST] Done {_dur(time.perf_counter() - t0)}")
    return results


# --- Analytics

def _transfer_matrix(ml_results: list[RunOutcome], model_type: str, pair: str) -> pd.DataFrame:
    """4x4 DataFrame (train_session x eval_session). NaN for missing cells."""
    mat = pd.DataFrame(np.nan, index=EVAL_SESSIONS, columns=EVAL_SESSIONS, dtype=float)
    mat.index.name = "train_session"
    mat.columns.name = "eval_session"
    for r in ml_results:
        if (r.pair == pair and r.model_type == model_type
                and r.ok and abs(r.spread_mult - 1.0) < 0.01
                and r.train_session in EVAL_SESSIONS and r.eval_session in EVAL_SESSIONS):
            mat.loc[r.train_session, r.eval_session] = r.net_sharpe
    return mat


def _session_generalisability(ml_results: list[RunOutcome]) -> pd.DataFrame:
    gen: dict[str, list[float]] = {s: [] for s in EVAL_SESSIONS}
    for r in ml_results:
        if (r.ok and r.train_session and r.eval_session
                and abs(r.spread_mult - 1.0) < 0.01
                and r.train_session != r.eval_session):
            gen[r.train_session].append(r.net_sharpe)
    rows = []
    for sess, vals in gen.items():
        rows.append({"train_session": sess,
                     "avg_off_diag_sharpe": round(float(np.nanmean(vals)), 4) if vals else float("nan"),
                     "n": len(vals)})
    return pd.DataFrame(rows)


def _lr_feature_importance(pair: str, session: str) -> Optional[dict]:
    path = _ml_model_path(f"LR_{session}", pair)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        # LR models are saved as raw sklearn objects, not dicts.
        model        = bundle.get("model") if isinstance(bundle, dict) else bundle
        feature_cols = bundle.get("feature_cols", []) if isinstance(bundle, dict) else []
        # Fall back to the scaler file which always carries feature_cols.
        if not feature_cols:
            scaler_path = PROJECT_DIR / "scalers" / f"{pair}_scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, "rb") as f:
                    sc = pickle.load(f)
                if isinstance(sc, dict):
                    feature_cols = sc.get("feature_cols", [])
        if not hasattr(model, "coef_") or not feature_cols:
            return None
        imp = np.abs(model.coef_).mean(axis=0)
        if len(feature_cols) != len(imp):
            return None
        top_idx = np.argsort(imp)[::-1][:5]
        return {"pair": pair, "session": session,
                "features":   [feature_cols[i] for i in top_idx],
                "importance": [round(float(imp[i]), 6) for i in top_idx]}
    except Exception:
        return None


def _cost_breakeven(ml_results: list[RunOutcome]) -> list[dict]:
    rows = []
    for pair in ALL_PAIRS:
        for strat in ML_STRATEGIES:
            for es in EVAL_SESSIONS:
                pts = sorted(
                    [(r.spread_mult, r.net_sharpe) for r in ml_results
                     if r.pair == pair and r.strategy == strat
                     and r.eval_session == es and r.ok],
                    key=lambda x: x[0])
                if len(pts) < 2:
                    continue
                be: Any = None
                for i in range(len(pts) - 1):
                    x1, y1 = pts[i]; x2, y2 = pts[i + 1]
                    if y1 >= 0 >= y2 or y2 >= 0 >= y1:
                        be = round(x1 + (0 - y1) * (x2 - x1) / ((y2 - y1) or 1e-9), 2)
                        break
                if be is None:
                    be = ">2.0" if all(y >= 0 for _, y in pts) else "<0.5"
                s1x = next((y for x, y in pts if abs(x - 1.0) < 0.01), float("nan"))
                rows.append({"pair": pair, "strategy": strat, "eval_session": es,
                             "break_even_mult": be, "sharpe_at_1x": s1x})
    return rows


def _run_dm_tests(ml_results: list[RunOutcome], rb_results: list[RunOutcome],
                  pairs: list[str], val_dfs: dict[str, pd.DataFrame],
                  test_dfs: dict[str, pd.DataFrame]) -> list[dict]:
    """Four DM comparison types per pair (Fix 2)."""
    rows = []

    for pair in pairs:
        # 1. Best rule-based vs buy-and-hold (val split)
        rb_val = [r for r in rb_results if r.pair == pair and r.ok and r.split == "val"
                  and r.n_trades >= MIN_TRADES]
        if rb_val and pair in val_dfs:
            best_rb = max(rb_val, key=lambda r: r.composite_score)
            df_v    = val_dfs[pair]
            try:
                strat   = get_strategy(best_rb.strategy)
                sigs    = strat.generate_signals(df_v.reset_index(drop=True)).reset_index(drop=True)
                rb_rets  = _bar_rets(sigs, df_v["close"].reset_index(drop=True))
                bah_rets = np.diff(df_v["close"].values) / np.maximum(df_v["close"].values[:-1], 1e-9)
                dm = _dm_test(rb_rets, bah_rets)
                rows.append({"pair": pair, "comparison": "best_RB_vs_BAH",
                             "strategy_a": best_rb.strategy, "strategy_b": "buy_and_hold", **dm})
            except Exception:
                pass

        # 2. Best ML vs buy-and-hold (test split, 1x spread)
        ml_1x = [r for r in ml_results if r.pair == pair and r.ok
                 and abs(r.spread_mult - 1.0) < 0.01 and r.n_trades >= MIN_TRADES]
        if ml_1x and pair in test_dfs:
            best_ml = max(ml_1x, key=lambda r: r.composite_score)
            df_t    = test_dfs[pair]
            try:
                strat    = get_strategy(best_ml.strategy, pair=pair)
                sigs     = strat.generate_signals(df_t.reset_index(drop=True)).reset_index(drop=True)
                filtered = _filter_signals(sigs, df_t, best_ml.eval_session or "global")
                ml_rets  = _bar_rets(filtered, df_t["close"].reset_index(drop=True))
                bah_t    = np.diff(df_t["close"].values) / np.maximum(df_t["close"].values[:-1], 1e-9)
                dm = _dm_test(ml_rets, bah_t)
                rows.append({"pair": pair, "comparison": "best_ML_vs_BAH",
                             "strategy_a": best_ml.strategy, "strategy_b": "buy_and_hold", **dm})
            except Exception:
                pass

        # 3. In-domain ML vs out-of-domain ML per session (test split, 1x spread)
        for sess in ["london", "ny", "asia"]:
            in_dom = [r for r in ml_results if r.pair == pair and r.ok
                      and r.train_session == sess and r.eval_session == sess
                      and abs(r.spread_mult - 1.0) < 0.01]
            out_dom = [r for r in ml_results if r.pair == pair and r.ok
                       and r.train_session == sess and r.eval_session != sess
                       and abs(r.spread_mult - 1.0) < 0.01]
            if not in_dom or not out_dom or pair not in test_dfs:
                continue
            df_t = test_dfs[pair]
            try:
                def _get_rets(outcome: RunOutcome) -> np.ndarray:
                    strat = get_strategy(outcome.strategy, pair=pair)
                    sigs  = strat.generate_signals(df_t.reset_index(drop=True)).reset_index(drop=True)
                    filt  = _filter_signals(sigs, df_t, outcome.eval_session or "global")
                    return _bar_rets(filt, df_t["close"].reset_index(drop=True))

                id_rets  = _get_rets(in_dom[0])
                od_stack = [_get_rets(r) for r in out_dom]
                n_min    = min(len(id_rets), min(len(x) for x in od_stack))
                od_mean  = np.mean(np.vstack([x[:n_min] for x in od_stack]), axis=0)
                dm = _dm_test(id_rets[:n_min], od_mean)
                rows.append({"pair": pair, "comparison": f"in_domain_{sess}_vs_transfer",
                             "strategy_a": f"LR/LSTM_{sess}_in_domain",
                             "strategy_b": f"LR/LSTM_{sess}_out_of_domain", **dm})
            except Exception:
                pass

        # 4a. RB champion vs runner-up - both on val (apples-to-apples)
        if len(rb_val) >= 2 and pair in val_dfs:
            try:
                rb_sorted = sorted(rb_val, key=lambda r: r.composite_score, reverse=True)
                if rb_sorted[0].strategy != rb_sorted[1].strategy:
                    df_v = val_dfs[pair]
                    s1   = get_strategy(rb_sorted[0].strategy)
                    s2   = get_strategy(rb_sorted[1].strategy)
                    sg1  = s1.generate_signals(df_v.reset_index(drop=True)).reset_index(drop=True)
                    sg2  = s2.generate_signals(df_v.reset_index(drop=True)).reset_index(drop=True)
                    ret1 = _bar_rets(sg1, df_v["close"].reset_index(drop=True))
                    ret2 = _bar_rets(sg2, df_v["close"].reset_index(drop=True))
                    dm = _dm_test(ret1, ret2)
                    rows.append({"pair": pair, "comparison": "RB_champion_vs_runner_up",
                                 "strategy_a": rb_sorted[0].strategy,
                                 "strategy_b": rb_sorted[1].strategy, **dm})
            except Exception:
                pass

        # 4b. ML champion vs runner-up - both on test (apples-to-apples)
        if len(ml_1x) >= 2 and pair in test_dfs:
            try:
                ml_sorted = sorted(ml_1x, key=lambda r: r.composite_score, reverse=True)
                # Pick top 2 with distinct (strategy, eval_session) combos
                seen = set()
                picks = []
                for r in ml_sorted:
                    key = (r.strategy, r.eval_session)
                    if key in seen:
                        continue
                    seen.add(key)
                    picks.append(r)
                    if len(picks) == 2:
                        break
                if len(picks) == 2:
                    df_t = test_dfs[pair]
                    s1   = get_strategy(picks[0].strategy, pair=pair)
                    s2   = get_strategy(picks[1].strategy, pair=pair)
                    sg1  = _filter_signals(
                        s1.generate_signals(df_t.reset_index(drop=True)).reset_index(drop=True),
                        df_t, picks[0].eval_session or "global")
                    sg2  = _filter_signals(
                        s2.generate_signals(df_t.reset_index(drop=True)).reset_index(drop=True),
                        df_t, picks[1].eval_session or "global")
                    ret1 = _bar_rets(sg1, df_t["close"].reset_index(drop=True))
                    ret2 = _bar_rets(sg2, df_t["close"].reset_index(drop=True))
                    dm = _dm_test(ret1, ret2)
                    rows.append({"pair": pair, "comparison": "ML_champion_vs_runner_up",
                                 "strategy_a": f"{picks[0].strategy}/eval={picks[0].eval_session}",
                                 "strategy_b": f"{picks[1].strategy}/eval={picks[1].eval_session}",
                                 **dm})
            except Exception:
                pass

    return rows


# --- CSV output

def _to_df(results: list[RunOutcome]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "tier": r.tier, "pair": r.pair, "strategy": r.strategy,
            "split": r.split, "session": r.session, "direction": r.direction,
            "tp_pips": r.tp_pips, "sl_pips": r.sl_pips, "folds": r.folds,
            "eval_session": r.eval_session, "spread_mult": r.spread_mult,
            "model_type": r.model_type, "train_session": r.train_session,
            "ok": r.ok, "status": r.status,
            "net_sharpe": r.net_sharpe, "gross_sharpe": r.gross_sharpe,
            "total_return": r.total_return, "max_drawdown": r.max_drawdown,
            "calmar": r.calmar, "win_rate": r.win_rate,
            "profit_factor": r.profit_factor, "avg_trade_bars": r.avg_trade_bars,
            "turnover": r.turnover, "sortino": r.sortino,
            "n_trades": r.n_trades, "capital_final": r.capital_final,
            "signal_long": r.signal_long, "signal_short": r.signal_short,
            "signal_flat": r.signal_flat,
            "fold_mean_sharpe": r.fold_mean_sharpe, "fold_std_sharpe": r.fold_std_sharpe,
            "negative_fold_count": r.negative_fold_count,
            "composite_score": r.composite_score, "stability_score": r.stability_score,
            "grade": r.grade,
            "warning_flags": ",".join(r.warning_flags) if r.warning_flags else "",
            "notes": r.notes, "error": r.error[:200] if r.error else "",
        })
    return pd.DataFrame(rows)


def _save_csvs(out_dir: Path, rb: list[RunOutcome], ml: list[RunOutcome],
               tr_lr: dict, tr_lstm: dict, gen_df: pd.DataFrame,
               fi: list[dict], dm: list[dict], be: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rb_df  = _to_df(rb)
    ml_df  = _to_df(ml)
    all_df = pd.concat([rb_df, ml_df], ignore_index=True)

    rb_df.to_csv(out_dir / "results_rule_based.csv", index=False)
    ml_df.to_csv(out_dir / "results_ml.csv", index=False)
    all_df.sort_values("composite_score", ascending=False).to_csv(
        out_dir / "results_all.csv", index=False)

    # Best/worst per pair
    bw = []
    for pair in ALL_PAIRS:
        ok = all_df[(all_df["pair"] == pair) & all_df["ok"] & (all_df["n_trades"] >= MIN_TRADES)]
        if ok.empty:
            continue
        bw.append({"pair": pair, "type": "best",  **ok.loc[ok["composite_score"].idxmax()].to_dict()})
        bw.append({"pair": pair, "type": "worst", **ok.loc[ok["net_sharpe"].idxmin()].to_dict()})
    if bw:
        pd.DataFrame(bw).to_csv(out_dir / "best_worst_per_pair.csv", index=False)

    for pair, mat in tr_lr.items():
        mat.to_csv(out_dir / f"transfer_matrix_lr_{pair}.csv")
    for pair, mat in tr_lstm.items():
        mat.to_csv(out_dir / f"transfer_matrix_lstm_{pair}.csv")

    if not gen_df.empty:
        gen_df.to_csv(out_dir / "session_generalisability.csv", index=False)
    if fi:
        pd.DataFrame(fi).to_csv(out_dir / "lr_feature_importance.csv", index=False)
    if dm:
        pd.DataFrame(dm).to_csv(out_dir / "dm_test_results.csv", index=False)
    if be:
        pd.DataFrame(be).to_csv(out_dir / "cost_breakeven.csv", index=False)


# --- Text report

_S1 = "=" * 80
_S2 = "-" * 80


def _write_report(path: Path, run_id: str, rb: list[RunOutcome], ml: list[RunOutcome],
                  t5: list[RunOutcome],
                  survivors: dict, t4_finals: list[RunOutcome],
                  tr_lr: dict, tr_lstm: dict, gen_df: pd.DataFrame,
                  fi: list[dict], dm: list[dict], be: list[dict],
                  bvals: dict, btest: dict,
                  date_from: Optional[str], date_to: Optional[str]) -> None:
    L: list[str] = []

    def w(*lines: str) -> None:
        L.extend(lines)

    w(_S1, "FX ALGO MASTER REPORT",
      f"generated_utc : {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
      f"run_id        : {run_id}", _S1, "")

    w("SCORING SYSTEM", _S2,
      "  Composite Score (0-100) - same formula for ALL tiers and ML:",
      "    35%  Net Sharpe   (0 if negative, capped 5.0)",
      "    25%  Sortino      (0 if negative, capped 5.0)",
      "    25%  Calmar       (0 if negative, capped 3.0)",
      "    15%  Drawdown Safety = max(0, (1+max_drawdown) x 100)",
      "  Min gates: n_trades >= 10  AND  max_drawdown > -0.95  (else score=0)",
      "",
      "  Stability Score = fold_mean_sharpe - 0.5 x fold_std_sharpe  [T4 only]",
      "  T4 ranking key + filter (reject negative_folds > 2).",
      "  NOT in composite_score.",
      "  Grade: A=80+  B=60-79  C=40-59  D=20-39  F<20", "")

    w("BASELINES", _S2,
      "  Rule-based baselines: val split.  ML baselines: test split.",
      f"  {'Pair':<8}  {'BaH-val':>10}  {'BaH-test':>10}",
      "  " + "-" * 32)
    for pair in ALL_PAIRS:
        w(f"  {pair:<8}  {_pct(bvals.get(pair, float('nan'))):>10}  {_pct(btest.get(pair, float('nan'))):>10}")
    w("")

    # --- PART A
    w(_S1, "PART A: RULE-BASED STRATEGIES", _S1, "")

    t1r = [r for r in rb if r.tier == "T1"]
    t2r = [r for r in rb if r.tier == "T2"]
    t3r = [r for r in rb if r.tier == "T3"]
    t4r = [r for r in rb if r.tier == "T4"]

    w("T1 SURVIVOR SUMMARY  (per-pair survival, top 40% by val net_sharpe + family floor)", _S2)
    total_surv = sum(len(v) for v in survivors.values())
    w(f"  T1 runs: {len(t1r)}  OK: {sum(1 for r in t1r if r.ok)}",
      f"  Surviving (pair, strategy) tuples: {total_surv}", "")
    for pair in ALL_PAIRS:
        strats = survivors.get(pair, [])
        w(f"  {pair:<8}  {len(strats)} survivors: {', '.join(strats) or 'none'}")
    w("")

    w(f"T2 SESSION+DIRECTION: {len(t2r)} runs  OK: {sum(1 for r in t2r if r.ok)}")
    w(f"T3 TP/SL GRID:        {len(t3r)} runs  OK: {sum(1 for r in t3r if r.ok)}")
    w(f"T4 WALK-FORWARD:      {len(t4r)} runs  OK: {sum(1 for r in t4r if r.ok)}", "")

    w("T4 FINAL CONFIGS  (stability_score, filter: negative_folds <= 2)", _S2)
    w(f"  {'#':<4} {'Gr':<3} {'Score':>6} {'Pair':<8} {'Strategy':<28} "
      f"{'Sess':<8} {'Dir':<12} {'TP':>5} {'SL':>5} "
      f"{'Sharpe':>7} {'Sortino':>7} {'MaxDD':>7} {'Trades':>7} {'Stab':>8}",
      "  " + "-" * 120)
    for i, r in enumerate(t4_finals, 1):
        w(f"  {i:<4} {r.grade:<3} {r.composite_score:>6.1f} {r.pair:<8} {r.strategy:<28} "
          f"{(r.session or 'all'):<8} {r.direction:<12} "
          f"{(str(int(r.tp_pips)) if r.tp_pips else '-'):>5} "
          f"{(str(int(r.sl_pips)) if r.sl_pips else '-'):>5} "
          f"{r.net_sharpe:>7.4f} {r.sortino:>7.4f} {r.max_drawdown:>7.4f} "
          f"{r.n_trades:>7} {r.stability_score:>8.4f}")
    w("")

    test_window = f"{date_from or 'test-start'} -> {date_to or 'test-end'}"
    w(f"T5 FINAL TEST EVALUATION  (T4 survivors re-run on test split: {test_window})", _S2)
    if not t5:
        w("  (skipped - no T4 survivors or --ml-only mode)", "")
    else:
        t5_ok = [r for r in t5 if r.ok and r.n_trades >= MIN_TRADES]
        w(f"  {'#':<4} {'Gr':<3} {'Score':>6} {'Pair':<8} {'Strategy':<28} "
          f"{'Sess':<8} {'Dir':<12} {'TP':>5} {'SL':>5} "
          f"{'Sharpe':>7} {'Sortino':>7} {'MaxDD':>7} {'Trades':>7}",
          "  " + "-" * 115)
        t5_sorted = sorted(t5_ok, key=lambda r: r.composite_score, reverse=True)
        for i, r in enumerate(t5_sorted, 1):
            w(f"  {i:<4} {r.grade:<3} {r.composite_score:>6.1f} {r.pair:<8} {r.strategy:<28} "
              f"{(r.session or 'all'):<8} {r.direction:<12} "
              f"{(str(int(r.tp_pips)) if r.tp_pips else '-'):>5} "
              f"{(str(int(r.sl_pips)) if r.sl_pips else '-'):>5} "
              f"{r.net_sharpe:>7.4f} {r.sortino:>7.4f} {r.max_drawdown:>7.4f} "
              f"{r.n_trades:>7}")
        w("")
        bah_test_rb = btest  # reuse test-split BaH computed for ML
        w(f"  {'Pair':<8}  {'BaH-test':>10}  {'Best RB T5 Sharpe':>18}  {'vs BaH':>8}")
        w("  " + "-" * 50)
        for pair in ALL_PAIRS:
            prows = [r for r in t5_ok if r.pair == pair]
            bah = bah_test_rb.get(pair, float("nan"))
            if not prows:
                w(f"  {pair:<8}  {_pct(bah):>10}  {'no result':>18}  {'':>8}")
                continue
            best = max(prows, key=lambda r: r.composite_score)
            vs = best.net_sharpe - bah if not math.isnan(bah) else float("nan")
            w(f"  {pair:<8}  {_pct(bah):>10}  {best.net_sharpe:>+18.4f}  "
              f"{('+' if vs >= 0 else '') + f'{vs:.4f}':>8}")
        w("")

    w("BEST/WORST RULE-BASED PER PAIR  (vs val-split buy-and-hold)", _S2)
    rb_ok = [r for r in rb if r.ok and r.n_trades >= MIN_TRADES and r.status == "OK"
             and r.tier not in ("T5",)]
    for pair in ALL_PAIRS:
        prows = [r for r in rb_ok if r.pair == pair]
        bah   = bvals.get(pair, float("nan"))
        w(f"  {pair}  BaH-val: {_pct(bah)}")
        if not prows:
            w("    no valid results", "")
            continue
        best  = max(prows, key=lambda r: r.composite_score)
        worst = min(prows, key=lambda r: r.net_sharpe)
        for lbl, r in [("BEST ", best), ("WORST", worst)]:
            w(f"    {lbl}  {r.strategy:<28} sess={r.session or 'all':<8} dir={r.direction:<12} "
              f"sharpe={r.net_sharpe:+.4f}  sortino={r.sortino:+.4f}  "
              f"dd={r.max_drawdown:+.4f}  trades={r.n_trades}  score={r.composite_score:.1f} {r.grade}  tier={r.tier}")
        w("")

    w("STRATEGY FAMILY PERFORMANCE", _S2,
      f"  {'Family':<18}  {'AvgSharpe':>10}  {'AvgSortino':>11}  {'BestPair':<9}  {'WorstPair':<9}",
      "  " + "-" * 65)
    for fam in sorted(set(STRATEGY_FAMILY.values()) | {"ml_lr", "ml_lstm"}):
        if fam.startswith("ml_"):
            mt = fam.split("_")[1]
            frows = [r for r in ml if r.ok and r.model_type == mt
                     and abs(r.spread_mult - 1.0) < 0.01]
        else:
            frows = [r for r in rb_ok if STRATEGY_FAMILY.get(r.strategy) == fam]
        if not frows:
            continue
        avg_sh = _mean([r.net_sharpe for r in frows])
        avg_so = _mean([r.sortino    for r in frows])
        by_pair = {}
        for r in frows:
            by_pair.setdefault(r.pair, []).append(r.net_sharpe)
        pavg = {p: _mean(v) for p, v in by_pair.items()}
        bp   = max(pavg, key=pavg.get) if pavg else "-"
        wp   = min(pavg, key=pavg.get) if pavg else "-"
        w(f"  {fam:<18}  {avg_sh:>10.4f}  {avg_so:>11.4f}  {bp:<9}  {wp:<9}")
    w("")

    # --- PART B
    w(_S1, "PART B: ML STRATEGIES  (cross-session transfer)", _S1, "")

    w("AVAILABLE MODELS  (checkmark = file found, dash = not trained)", _S2)
    hdr = f"  {'Pair':<8}" + "".join(f"  {s:<20}" for s in ML_STRATEGIES)
    w(hdr, "  " + "-" * (8 + 22 * len(ML_STRATEGIES)))
    for pair in ALL_PAIRS:
        row = f"  {pair:<8}"
        for s in ML_STRATEGIES:
            row += f"  {'yes' if _ml_available(s, pair) else '---':<20}"
        w(row)
    w("")

    for lbl, matrices in [("LR", tr_lr), ("LSTM", tr_lstm)]:
        w(f"CROSS-SESSION TRANSFER MATRIX - {lbl}  (net_sharpe @ 1x spread)", _S2,
          "  Rows=train_session  Cols=eval_session  nan=model not trained", "")
        for pair in ALL_PAIRS:
            mat = matrices.get(pair)
            if mat is None or mat.isna().all().all():
                w(f"  {pair}: no data")
                continue
            w(f"  {pair}")
            w("  " + f"{'':>16}" + "".join(f"  {c:>12}" for c in mat.columns))
            for ts in mat.index:
                row = f"  {ts:>16}"
                for es in mat.columns:
                    v = mat.loc[ts, es]
                    row += f"  {('---' if math.isnan(v) else f'{v:+.4f}'):>12}"
                w(row)
            w("")

    w("SESSION GENERALISABILITY SCORE  (avg off-diagonal net_sharpe @ 1x spread)", _S2,
      "  High -> transfers well.  Low -> overfits to training session.", "")
    if not gen_df.empty:
        w(f"  {'Train Session':<16}  {'Avg Off-Diag Sharpe':>20}  {'N':>6}")
        w("  " + "-" * 46)
        for _, row in gen_df.iterrows():
            w(f"  {row['train_session']:<16}  {row['avg_off_diag_sharpe']:>20.4f}  {int(row['n']):>6}")
    w("")

    w("IN-DOMAIN vs TRANSFER GAP  (diag_sharpe - avg_off_diag_sharpe)", _S2,
      "  Large gap -> overfitting.  Small gap -> robust model.", "")
    w(f"  {'Pair':<8}  {'Type':<6}  {'Session':<8}  {'In-Dom':>8}  {'Avg-OD':>8}  {'Gap':>8}")
    w("  " + "-" * 56)
    for pair in ALL_PAIRS:
        for mt, matrices in [("lr", tr_lr), ("lstm", tr_lstm)]:
            mat = matrices.get(pair)
            if mat is None:
                continue
            for sess in EVAL_SESSIONS:
                if sess not in mat.index or sess not in mat.columns:
                    continue
                in_d  = mat.loc[sess, sess]
                offs  = [mat.loc[sess, c] for c in mat.columns
                         if c != sess and not math.isnan(mat.loc[sess, c])]
                avg_o = float(np.mean(offs)) if offs else float("nan")
                gap   = (in_d - avg_o) if not (math.isnan(in_d) or math.isnan(avg_o)) else float("nan")
                w(f"  {pair:<8}  {mt:<6}  {sess:<8}  "
                  f"{('---' if math.isnan(in_d) else f'{in_d:+.4f}'):>8}  "
                  f"{('---' if math.isnan(avg_o) else f'{avg_o:+.4f}'):>8}  "
                  f"{('---' if math.isnan(gap) else f'{gap:+.4f}'):>8}")
    w("")

    w("LR FEATURE IMPORTANCE  (top-5 per model)", _S2)
    for row in fi:
        feats = ", ".join(f"{f}({v:.4f})" for f, v in zip(row["features"], row["importance"]))
        w(f"  {row['pair']:<8} {row['session']:<8}  {feats}")
    w("")

    w("DIEBOLD-MARIANO TEST  (HLN-corrected, two-sided, * = p<0.05)", _S2,
      "  Comparisons:",
      "   1. best_RB_vs_BAH         -> does rule-based beat passive investing? (val split)",
      "   2. best_ML_vs_BAH         -> does ML beat passive investing? (test split)",
      "   3. in_domain_X_vs_transfer -> does in-session training beat out-of-session? (test)",
      "   4. champion_vs_runner_up   -> is the top strategy significantly better?", "")
    w(f"  {'Pair':<8}  {'Comparison':<38}  {'DM-stat':>8}  {'p-value':>8}  {'N':>7}  Sig")
    w("  " + "-" * 80)
    for row in dm:
        sig  = "*" if row.get("sig") else " "
        dm_s = f"{row['dm_stat']:>8.4f}" if not math.isnan(float(row.get("dm_stat") or "nan")) else "     nan"
        pv_s = f"{row['p_value']:>8.4f}" if row.get("p_value") is not None else "     nan"
        w(f"  {row['pair']:<8}  {row['comparison']:<38}  {dm_s}  {pv_s}  {row['n']:>7}  {sig}")
    w("")

    w("COST BREAK-EVEN  (spread multiplier where net_sharpe = 0)", _S2,
      "  >2.0 = profitable at all tested spreads.  <0.5 = never profitable.", "")
    if not be:
        w("  (no data - requires at least 2 spread multipliers; re-run without --spreads or with 3 values)", "")
    else:
        w(f"  {'Pair':<8}  {'Strategy':<22}  {'EvalSess':<10}  {'BreakEven':>10}  {'Sharpe@1x':>10}")
        w("  " + "-" * 68)
        for row in sorted(be, key=lambda r: (r["pair"], r["strategy"], r["eval_session"])):
            s1x = row["sharpe_at_1x"]
            s1x_s = f"{s1x:>10.4f}" if not (isinstance(s1x, float) and math.isnan(s1x)) else f"{'nan':>10}"
            w(f"  {row['pair']:<8}  {row['strategy']:<22}  {row['eval_session']:<10}  "
              f"{str(row['break_even_mult']):>10}  {s1x_s}")
        w("")

    w("BEST/WORST ML CONFIG PER PAIR  (vs test-split buy-and-hold, 1x spread)", _S2)
    ml_ok = [r for r in ml if r.ok and r.n_trades >= MIN_TRADES and abs(r.spread_mult - 1.0) < 0.01]
    for pair in ALL_PAIRS:
        prows = [r for r in ml_ok if r.pair == pair]
        bah   = btest.get(pair, float("nan"))
        w(f"  {pair}  BaH-test: {_pct(bah)}")
        if not prows:
            w("    no ML models trained", "")
            continue
        best  = max(prows, key=lambda r: r.composite_score)
        worst = min(prows, key=lambda r: r.net_sharpe)
        for lbl, r in [("BEST ", best), ("WORST", worst)]:
            w(f"    {lbl}  {r.strategy:<22} train={r.train_session:<8} eval={r.eval_session:<8} "
              f"sharpe={r.net_sharpe:+.4f}  sortino={r.sortino:+.4f}  "
              f"dd={r.max_drawdown:+.4f}  trades={r.n_trades}  score={r.composite_score:.1f} {r.grade}")
        w("")

    # --- PART C
    w(_S1, "PART C: UNIFIED RANKINGS  (rule-based + ML combined)", _S1, "")

    all_ok = sorted([r for r in (rb + ml) if r.ok and r.n_trades >= MIN_TRADES
                     and r.composite_score > 0],
                    key=lambda r: r.composite_score, reverse=True)

    w("UNIFIED BEST/WORST PER PAIR", _S2)
    for pair in ALL_PAIRS:
        prows = [r for r in all_ok if r.pair == pair]
        w(f"  {pair}")
        if not prows:
            w("    no valid configs", "")
            continue
        best  = prows[0]
        worst = min(prows, key=lambda r: r.net_sharpe)
        for lbl, r in [("BEST ", best), ("WORST", worst)]:
            tag = (f"ML:{r.train_session}->{r.eval_session}" if r.tier == "ML"
                   else f"RB:{r.tier}")
            w(f"    {lbl}  {r.strategy:<26} [{tag}]  "
              f"sharpe={r.net_sharpe:+.4f}  score={r.composite_score:.1f}  {r.grade}")
        w("")

    top_n = min(50, len(all_ok))
    w(f"FULL RANKED TABLE  (top {top_n} of {len(all_ok)} by composite_score)", _S2)
    w(f"  {'#':<4} {'Gr':<3} {'Score':>6} {'Pair':<8} {'Type':<4} {'Strategy':<26} "
      f"{'Sharpe':>7} {'Sortino':>7} {'Calmar':>7} {'MaxDD':>7} {'Trades':>7}",
      "  " + "-" * 95)
    for i, r in enumerate(all_ok[:top_n], 1):
        mt = "ML" if r.tier == "ML" else "RB"
        w(f"  {i:<4} {r.grade:<3} {r.composite_score:>6.1f} {r.pair:<8} {mt:<4} {r.strategy:<26} "
          f"{r.net_sharpe:>7.4f} {r.sortino:>7.4f} {r.calmar:>7.4f} "
          f"{r.max_drawdown:>7.4f} {r.n_trades:>7}")
    w("")

    w("FINAL TEST COMMANDS  (top T4 survivors)", _S2)
    for r in t4_finals[:20]:
        cmd = (f"python backtest/run_backtest.py --pair {r.pair} --strategy {r.strategy} "
               f"--split test --direction {r.direction}")
        if r.session:
            cmd += f" --session {r.session}"
        if r.tp_pips:
            cmd += f" --tp-pips {int(r.tp_pips)} --sl-pips {int(r.sl_pips)}"
        cmd += " --no-browser"
        w(f"  {cmd}")
    w("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"\n  Report: {path}")


# --- Main

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified master evaluation: rule-based T1-T4 + ML cross-session.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pairs",           nargs="+", default=["all"])
    parser.add_argument("--workers",         type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--t1-min-sharpe",   type=float, default=-0.2)
    parser.add_argument("--spreads",         nargs="+", type=float, default=SPREAD_MULTS)
    parser.add_argument("--ml-split",        default="test", choices=["test", "val"])
    parser.add_argument("--rule-based-only", action="store_true")
    parser.add_argument("--ml-only",         action="store_true")
    parser.add_argument("--eval-year",        type=int, default=None,
                        help="Restrict BOTH T5 and ML evaluation to a single calendar year "
                             f"within the test split ({TEST_START[:4]}-{TEST_END[:4]}). "
                             "E.g. --eval-year 2024")
    parser.add_argument("--from",            dest="date_from", default=None,
                        help="Override eval window start (ISO date, e.g. 2024-06-01). "
                             "Applies to T5 and ML. Defaults to TEST_START from config.")
    parser.add_argument("--to",              dest="date_to",   default=None,
                        help="Override eval window end (ISO date, e.g. 2024-12-31). "
                             "Applies to T5 and ML. Defaults to TEST_END from config.")
    parser.add_argument("--out",             type=str, default=None)
    parser.add_argument("--output-dir",      type=str, default=None,
                        help="Output directory. Default: output/master_eval/run_<RUN_ID> "
                             "(each run is preserved in its own subfolder).")
    args = parser.parse_args()

    # Resolve evaluation window - --eval-year takes precedence over --from/--to.
    # Both T5 (rule-based final) and ML always share the same window for fair comparison.
    # ML is restricted to the test split; rule-based can technically use any period but
    # T5 always loads split="test" so the window is already constrained.
    test_start_str = TEST_START if hasattr(TEST_START, "startswith") else str(TEST_START)
    test_end_str   = TEST_END   if hasattr(TEST_END,   "startswith") else str(TEST_END)

    if args.eval_year is not None:
        y = args.eval_year
        ts_year = int(test_start_str[:4])
        te_year = int(test_end_str[:4])
        if not (ts_year <= y <= te_year):
            parser.error(
                f"--eval-year {y} is outside the test split "
                f"({test_start_str[:4]}-{test_end_str[:4]}). "
                f"ML models were NOT evaluated on data before {test_start_str}."
            )
        args.date_from = f"{y}-01-01"
        args.date_to   = f"{y}-12-31"
    else:
        # Default to full test split when no date args given - automatic, no hardcoding needed
        if args.date_from is None:
            args.date_from = test_start_str
        if args.date_to is None:
            args.date_to = test_end_str

    run_id   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        out_dir = OUTPUT_DIR / f"run_{run_id}"
    else:
        out_dir = Path(args.output_dir)
    out_path = Path(args.out) if args.out else out_dir / "master_report.txt"
    pairs    = ALL_PAIRS if "all" in args.pairs else [p.upper() for p in args.pairs]

    print(f"\n{'='*80}")
    print(f"FX ALGO MASTER EVAL  |  run_id={run_id}")
    print(f"{'='*80}")
    print(f"  Pairs   : {', '.join(pairs)}")
    print(f"  Workers : {args.workers}")
    print(f"  T1 min  : {args.t1_min_sharpe}")
    print(f"  Spreads : {args.spreads}")
    print(f"  ML split: {args.ml_split}")
    print(f"  Mode    : {'ML-only' if args.ml_only else 'RB-only' if args.rule_based_only else 'Full'}")
    print(f"  Eval win: {args.date_from} -> {args.date_to}  (T5 rule-based + ML, same window)")
    print()

    rb:        list[RunOutcome] = []
    ml:        list[RunOutcome] = []
    t5:        list[RunOutcome] = []
    survivors: dict[str, list[str]] = {}
    t4_finals: list[RunOutcome] = []
    val_dfs:   dict[str, pd.DataFrame] = {}
    test_dfs:  dict[str, pd.DataFrame] = {}

    # Part A
    if not args.ml_only:
        print(f"\n{'='*80}\nPART A: RULE-BASED\n{'='*80}")
        t1_results = _run_parallel(_build_t1_specs(pairs), args.workers, "T1")
        rb.extend(t1_results)

        survivors = _select_t1_survivors(t1_results, args.t1_min_sharpe)
        n_surv = sum(len(v) for v in survivors.values())
        print(f"\n  T1: {n_surv} (pair,strategy) tuples survive")

        t2_results = _run_parallel(_build_t2_specs(survivors), args.workers, "T2")
        rb.extend(t2_results)
        t2_best = _select_t2_best(t2_results, survivors)

        t3_results = _run_parallel(_build_t3_specs(t2_best), args.workers, "T3")
        rb.extend(t3_results)
        t3_best = _select_t3_best(t3_results, t2_best, t2_results)

        t4_results = _run_parallel(t3_best, args.workers, "T4")
        rb.extend(t4_results)
        t4_finals = _final_rank_t4(t4_results)
        print(f"\n  T4: {len(t4_finals)} configs pass stability filter")

        # T5: re-run T4 survivors on test split (final out-of-sample evaluation)
        t5 = _run_t5(t4_finals, args.workers, args.date_from, args.date_to)
        rb.extend(t5)

        # Load val DFs for DM test
        for pair in pairs:
            try:
                df = load_split_data(pair, "val")
                if "timestamp_utc" in df.columns:
                    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
                    df = df.sort_values("timestamp_utc").drop_duplicates("timestamp_utc").reset_index(drop=True)
                val_dfs[pair] = df
            except Exception:
                pass

    # Part B
    if not args.rule_based_only:
        print(f"\n{'='*80}\nPART B: ML\n{'='*80}")
        ml = _run_ml_evaluation(pairs, args.spreads, EVAL_SESSIONS, args.ml_split, args.workers,
                                date_from=args.date_from, date_to=args.date_to)
        print(f"\n  ML: {len(ml)} backtest results")

        # Load test DFs for DM test (same date filter)
        for pair in pairs:
            try:
                df = load_split_data(pair, args.ml_split)
                if "timestamp_utc" in df.columns:
                    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
                    df = df.sort_values("timestamp_utc").drop_duplicates("timestamp_utc").reset_index(drop=True)
                    if args.date_from:
                        df = df[df["timestamp_utc"] >= pd.Timestamp(args.date_from, tz="UTC")]
                    if args.date_to:
                        df = df[df["timestamp_utc"] <= pd.Timestamp(args.date_to, tz="UTC")]
                    df = df.reset_index(drop=True)
                test_dfs[pair] = df
            except Exception:
                pass

    # Analytics
    print("\n  Computing analytics...")
    bvals = {p: _bah_return(p, "val")  for p in pairs}
    btest = {p: _bah_return(p, "test") for p in pairs}

    tr_lr:   dict[str, pd.DataFrame] = {}
    tr_lstm: dict[str, pd.DataFrame] = {}
    gen_df  = pd.DataFrame()
    fi:     list[dict] = []
    dm_rows: list[dict] = []
    be:     list[dict] = []

    if ml:
        for pair in pairs:
            tr_lr[pair]   = _transfer_matrix(ml, "lr",   pair)
            tr_lstm[pair] = _transfer_matrix(ml, "lstm", pair)
        gen_df = _session_generalisability(ml)
        for pair in pairs:
            for sess in EVAL_SESSIONS:
                row = _lr_feature_importance(pair, sess)
                if row:
                    fi.append(row)
        be = _cost_breakeven(ml)

    dm_rows = _run_dm_tests(ml, rb, pairs, val_dfs, test_dfs)

    # Output
    print("\n  Writing outputs...")
    _write_report(out_path, run_id, rb, ml, t5, survivors, t4_finals,
                  tr_lr, tr_lstm, gen_df, fi, dm_rows, be, bvals, btest,
                  args.date_from, args.date_to)
    _save_csvs(out_dir, rb, ml, tr_lr, tr_lstm, gen_df, fi, dm_rows, be)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "latest_run.txt").write_text(
        str(out_dir.resolve()) + "\n", encoding="utf-8"
    )

    print(f"\n{'='*80}")
    print(f"DONE  |  run_id={run_id}")
    print(f"Report : {out_path}")
    print(f"CSVs   : {out_dir}")
    print(f"Latest : {OUTPUT_DIR / 'latest_run.txt'}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    freeze_support()
    main()
