#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import math
import os
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean, pstdev
from typing import Any, Optional

import pandas as pd

from backtest.engine import run_backtest, run_wf_folds
from backtest.strategies import STRATEGY_REGISTRY, get_strategy
from backtest.run_backtest import load_split_data
from config.constants import PAIRS, PAIR_SPREAD_PIPS, TP_SL_GRID as _CONFIG_TP_SL_GRID

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

ALL_STRATEGIES = list(STRATEGY_REGISTRY.keys())

ALL_PAIRS = list(PAIRS)

ALL_SESSIONS = [None, "london", "ny", "asia", "overlap"]
ALL_DIRECTIONS = ["long_short", "long_only", "short_only"]
TP_SL_GRID: list[tuple[int | None, int | None]] = [(None, None)] + [
    (int(tp), int(sl)) for tp, sl in _CONFIG_TP_SL_GRID
]


@dataclass
class RunSpec:
    tier: str
    pair: str
    strategy: str
    split: str
    session: Optional[str] = None
    direction: str = "long_short"
    tp_pips: Optional[float] = None
    sl_pips: Optional[float] = None
    folds: int = 0
    capital: float = 10000.0
    spread: Optional[float] = None
    max_hold: Optional[int] = None
    resample: Optional[str] = None
    entry_time: Optional[str] = None
    mode: str = "research"
    ml_family: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass
class RunOutcome:
    ok: bool
    status: str
    tier: str
    pair: str
    strategy: str
    split: str
    session: Optional[str]
    direction: str
    tp_pips: Optional[float]
    sl_pips: Optional[float]
    folds: int
    net_sharpe: float = 0.0
    gross_sharpe: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_bars: float = 0.0
    turnover: float = 0.0
    sortino: float = 0.0
    n_trades: int = 0
    capital_final: float = 0.0
    signal_long: int = 0
    signal_short: int = 0
    signal_flat: int = 0
    fold_mean_sharpe: float = 0.0
    fold_std_sharpe: float = 0.0
    negative_fold_count: int = 0
    warning_flags: list[str] = None
    notes: str = ""
    error: str = ""

    def __post_init__(self):
        if self.warning_flags is None:
            self.warning_flags = []


class FutureModelAdapter:
    """
    Placeholder interface so ML models can slot into the same runner later.

    Expected future API:
    - fit(train_df)
    - predict_signals(df) -> pd.Series in {-1, 0, 1}
    """
    name: str = "ML_PLACEHOLDER"

    def fit(self, train_df: pd.DataFrame) -> None:
        raise NotImplementedError

    def predict_signals(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def safe_std(values: list[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def fmt(x: Any) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return "nan"
        return f"{x:.6f}"
    return str(x)


def fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def validate_run_spec(spec: RunSpec) -> list[str]:
    flags: list[str] = []

    if spec.capital <= 0:
        raise ValueError("capital must be > 0")

    if spec.session is not None and spec.session not in ALL_SESSIONS:
        raise ValueError(f"invalid session: {spec.session}")

    if spec.direction not in ALL_DIRECTIONS:
        raise ValueError(f"invalid direction: {spec.direction}")

    if spec.tp_pips is not None and spec.tp_pips <= 0:
        raise ValueError("tp_pips must be > 0 when provided")

    if spec.sl_pips is not None and spec.sl_pips <= 0:
        raise ValueError("sl_pips must be > 0 when provided")

    if (spec.tp_pips is None) != (spec.sl_pips is None):
        flags.append("PARTIAL_TP_SL")

    spread = spec.spread if spec.spread is not None else PAIR_SPREAD_PIPS.get(spec.pair, 1.0)

    if spec.tp_pips is not None and spec.tp_pips <= spread:
        flags.append("TP_LE_SPREAD")

    if spec.sl_pips is not None and spec.sl_pips <= spread:
        flags.append("SL_LE_SPREAD")

    return flags


def _outcome_from_spec(spec: RunSpec, **overrides: Any) -> RunOutcome:
    """Build a RunOutcome inheriting all RunSpec descriptor fields, then apply overrides."""
    base = dict(
        tier=spec.tier,
        pair=spec.pair,
        strategy=spec.strategy,
        split=spec.split,
        session=spec.session,
        direction=spec.direction,
        tp_pips=spec.tp_pips,
        sl_pips=spec.sl_pips,
        folds=spec.folds,
    )
    base.update(overrides)
    return RunOutcome(**base)


def _execute_wf_folds(spec: RunSpec, warn_flags: list[str]) -> RunOutcome:
    fold_results = run_wf_folds(
        pair=spec.pair,
        strategy=spec.strategy,
        n_folds=spec.folds,
        split="train",
        spread_pips=spec.spread,
        tp_pips=spec.tp_pips,
        sl_pips=spec.sl_pips,
        capital_initial=spec.capital,
        max_hold_bars=spec.max_hold,
        session=spec.session,
        entry_time=spec.entry_time,
        resample=spec.resample,
        direction_mode=spec.direction,
        mode=spec.mode,
    )
    sharpes = [r.net_sharpe for r in fold_results]
    neg_folds = sum(1 for s in sharpes if s < 0)
    last = fold_results[-1]

    return _outcome_from_spec(
        spec,
        ok=True,
        status="OK",
        net_sharpe=safe_mean(sharpes),
        gross_sharpe=last.gross_sharpe,
        total_return=safe_mean([r.total_return for r in fold_results]),
        max_drawdown=safe_mean([r.max_drawdown for r in fold_results]),
        calmar=safe_mean([r.calmar for r in fold_results]),
        win_rate=safe_mean([r.win_rate for r in fold_results]),
        profit_factor=safe_mean([r.profit_factor for r in fold_results]),
        avg_trade_bars=safe_mean([r.avg_trade_bars for r in fold_results]),
        turnover=safe_mean([r.turnover for r in fold_results]),
        sortino=safe_mean([r.sortino for r in fold_results]),
        n_trades=sum(r.n_trades for r in fold_results),
        capital_final=last.capital_final,
        signal_long=last.signal_dist.get("Long", 0),
        signal_short=last.signal_dist.get("Short", 0),
        signal_flat=last.signal_dist.get("Flat", 0),
        fold_mean_sharpe=safe_mean(sharpes),
        fold_std_sharpe=safe_std(sharpes),
        negative_fold_count=neg_folds,
        warning_flags=warn_flags,
        notes="WF_FOLDS",
    )


def _prepare_split_data(spec: RunSpec) -> pd.DataFrame:
    df = load_split_data(spec.pair, spec.split)
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df = (
            df.sort_values("timestamp_utc")
            .drop_duplicates(subset=["timestamp_utc"])
            .reset_index(drop=True)
        )
    return df


def _execute_run(spec: RunSpec, df: pd.DataFrame, warn_flags: list[str]) -> RunOutcome:
    warmup = get_strategy(spec.strategy).warmup_bars
    if len(df) <= warmup:
        return _outcome_from_spec(
            spec,
            ok=False,
            status="SKIP_WARMUP_GT_DATA",
            warning_flags=warn_flags + ["WARMUP_GT_DATA"],
            notes=f"rows={len(df)} warmup={warmup}",
        )

    strat = get_strategy(spec.strategy)
    signals = strat.generate_signals(df.reset_index(drop=True)).reset_index(drop=True)
    prices = df["close"].reset_index(drop=True)

    timestamps = (
        df["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
        if "timestamp_utc" in df.columns
        else []
    )

    if signals.abs().sum() == 0:
        return _outcome_from_spec(
            spec,
            ok=False,
            status="ZERO_SIGNALS",
            warning_flags=warn_flags + ["ZERO_SIGNALS"],
            notes="all emitted signals are flat",
        )

    r = run_backtest(
        signals=signals,
        prices=prices,
        pair=spec.pair,
        strategy=strat.name,
        split=spec.split,
        spread_pips=spec.spread,
        tp_pips=spec.tp_pips,
        sl_pips=spec.sl_pips,
        capital_initial=spec.capital,
        max_hold_bars=spec.max_hold,
        timestamps=timestamps,
        session=spec.session,
        entry_time=spec.entry_time,
        resample=spec.resample,
        direction_mode=spec.direction,
        mode=spec.mode,
        df_full=df,
    )

    status = "OK"
    if r.n_trades == 0:
        status = "ZERO_TRADES"
        warn_flags.append("ZERO_TRADES")
    if r.profit_factor >= 998:
        warn_flags.append("PF_CAPPED")
    if r.signal_dist.get("Long", 0) == 0 and r.signal_dist.get("Short", 0) == 0:
        warn_flags.append("NO_DIRECTIONAL_SIGNALS")

    return _outcome_from_spec(
        spec,
        ok=True,
        status=status,
        net_sharpe=r.net_sharpe,
        gross_sharpe=r.gross_sharpe,
        total_return=r.total_return,
        max_drawdown=r.max_drawdown,
        calmar=r.calmar,
        win_rate=r.win_rate,
        profit_factor=r.profit_factor,
        avg_trade_bars=r.avg_trade_bars,
        turnover=r.turnover,
        sortino=r.sortino,
        n_trades=r.n_trades,
        capital_final=r.capital_final,
        signal_long=r.signal_dist.get("Long", 0),
        signal_short=r.signal_dist.get("Short", 0),
        signal_flat=r.signal_dist.get("Flat", 0),
        warning_flags=warn_flags,
        notes="",
    )


def _single_run(spec: RunSpec) -> RunOutcome:
    try:
        warn_flags = validate_run_spec(spec)
        if spec.folds > 0:
            return _execute_wf_folds(spec, warn_flags)
        df = _prepare_split_data(spec)
        return _execute_run(spec, df, warn_flags)

    except FileNotFoundError as e:
        return _outcome_from_spec(spec, ok=False, status="MISSING_FILE", error=str(e))

    except Exception as e:
        return _outcome_from_spec(
            spec,
            ok=False,
            status="ENGINE_ERROR",
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )


def run_parallel(specs: list[RunSpec], max_workers: int, tier_name: str = "RUN", progress_interval: int = 10) -> list[RunOutcome]:
    results: list[RunOutcome] = []
    total = len(specs)
    start_time = time.perf_counter()
    last_print_time = start_time
    status_counter: Counter[str] = Counter()

    print()
    print("=" * 110)
    print(f"{tier_name} | Starting {total} runs with {max_workers} workers")
    print("=" * 110)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {ex.submit(_single_run, spec): spec for spec in specs}

        for idx, fut in enumerate(as_completed(fut_map), start=1):
            spec = fut_map[fut]

            try:
                result = fut.result()
            except Exception as e:
                result = RunOutcome(
                    ok=False,
                    status="FUTURE_ERROR",
                    tier=spec.tier,
                    pair=spec.pair,
                    strategy=spec.strategy,
                    split=spec.split,
                    session=spec.session,
                    direction=spec.direction,
                    tp_pips=spec.tp_pips,
                    sl_pips=spec.sl_pips,
                    folds=spec.folds,
                    error=f"future exception: {type(e).__name__}: {e}",
                )

            results.append(result)
            status_counter[result.status] += 1

            now = time.perf_counter()
            elapsed = now - start_time
            avg_per_run = elapsed / idx if idx else 0.0
            remaining_runs = total - idx
            eta_seconds = avg_per_run * remaining_runs
            pct = (idx / total) * 100

            should_print = (
                idx == 1
                or idx == total
                or idx % max(1, total // 20) == 0
                or (now - last_print_time) >= progress_interval
            )

            if should_print:
                top_status = ", ".join(
                    f"{k}:{v}" for k, v in status_counter.most_common(5)
                ) or "none"

                print(
                    f"[{tier_name}] "
                    f"{idx}/{total} ({pct:6.2f}%) | "
                    f"Elapsed: {fmt_seconds(elapsed)} | "
                    f"ETA: {fmt_seconds(eta_seconds)} | "
                    f"Avg/run: {avg_per_run:.2f}s | "
                    f"Statuses: {top_status}"
                )
                last_print_time = now

    total_elapsed = time.perf_counter() - start_time
    print(f"[{tier_name}] Done in {fmt_seconds(total_elapsed)}")
    return results


def summarize_by_strategy(results: list[RunOutcome]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for s in ALL_STRATEGIES:
        rows = [r for r in results if r.strategy == s and r.ok]
        out[s] = {
            "runs": len(rows),
            "mean_sharpe": safe_mean([r.net_sharpe for r in rows]),
            "mean_return": safe_mean([r.total_return for r in rows]),
            "mean_dd": safe_mean([r.max_drawdown for r in rows]),
            "mean_trades": safe_mean([float(r.n_trades) for r in rows]),
        }
    return out


def select_t1_survivors(results: list[RunOutcome], min_val_sharpe: float, keep_top_n: int) -> list[str]:
    val_rows = [r for r in results if r.split == "val" and r.ok]
    per_strat = {}

    for s in ALL_STRATEGIES:
        rows = [r for r in val_rows if r.strategy == s]
        if not rows:
            continue
        sh = safe_mean([r.net_sharpe for r in rows])
        if sh > min_val_sharpe:
            per_strat[s] = sh

    ranked = sorted(per_strat.items(), key=lambda kv: kv[1], reverse=True)
    return [k for k, _ in ranked[:keep_top_n]]


def select_t2_best_configs(results: list[RunOutcome]) -> list[RunSpec]:
    best_specs = []
    grouped: dict[tuple[str, str], list[RunOutcome]] = {}

    for r in results:
        if not r.ok:
            continue
        grouped.setdefault((r.strategy, r.pair), []).append(r)

    for (strategy, pair), rows in grouped.items():
        best = max(rows, key=lambda x: x.net_sharpe)
        best_specs.append(
            RunSpec(
                tier="T3",
                pair=pair,
                strategy=strategy,
                split="val",
                session=best.session,
                direction=best.direction,
            )
        )
    return best_specs


def select_t3_best_configs(results: list[RunOutcome], top_pairs_per_strategy: int) -> list[RunSpec]:
    by_strategy: dict[str, list[RunOutcome]] = {}

    for r in results:
        if r.ok:
            by_strategy.setdefault(r.strategy, []).append(r)

    out: list[RunSpec] = []

    for strategy, rows in by_strategy.items():
        ranked = sorted(rows, key=lambda x: x.net_sharpe, reverse=True)
        chosen_pairs = set()

        for row in ranked:
            if row.pair in chosen_pairs:
                continue

            out.append(
                RunSpec(
                    tier="T4",
                    pair=row.pair,
                    strategy=row.strategy,
                    split="train",
                    session=row.session,
                    direction=row.direction,
                    tp_pips=row.tp_pips,
                    sl_pips=row.sl_pips,
                    folds=5,
                )
            )
            chosen_pairs.add(row.pair)

            if len(chosen_pairs) >= top_pairs_per_strategy:
                break

    return out


def final_rank(results: list[RunOutcome], top_n: int) -> list[RunOutcome]:
    good = [r for r in results if r.ok and r.status == "OK"]
    stable = [r for r in good if r.negative_fold_count <= 2 and r.fold_std_sharpe <= 1.5]
    ranked = sorted(stable, key=lambda x: (x.fold_mean_sharpe, -x.fold_std_sharpe), reverse=True)
    return ranked[:top_n]


def write_txt_report(path: Path, config: dict[str, Any], tiers: dict[str, list[RunOutcome]], finals: list[RunOutcome]) -> None:
    lines: list[str] = []

    lines.append("FX ALGO TEST MASTER PLAN")
    lines.append("=" * 120)
    lines.append(f"generated_utc: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    lines.append(f"config: {json.dumps(config, sort_keys=True)}")
    lines.append("")

    lines.append("ARCHITECTURE NOTES")
    lines.append("- Tiered pruning plan designed for current rule-based strategies and future ML strategies.")
    lines.append("- Future ML models should expose fit(train_df) and predict_signals(df)->{-1,0,1}.")
    lines.append("- Test split is intentionally excluded from selection tiers and reserved for final locked configs.")
    lines.append("")

    for tier_name in ["T1", "T2", "T3", "T4"]:
        rows = tiers.get(tier_name, [])
        lines.append(f"{tier_name} RESULTS")
        lines.append("-" * 120)
        lines.append(f"runs: {len(rows)}")
        ok_count = sum(1 for r in rows if r.ok)
        err_count = len(rows) - ok_count
        lines.append(f"ok_runs: {ok_count}")
        lines.append(f"failed_or_skipped: {err_count}")
        lines.append("")
        lines.append(
            "pair | strategy | split | session | direction | tp | sl | status | net_sharpe | total_return | max_dd | trades | fold_mean | fold_std | flags | notes"
        )

        for r in sorted(rows, key=lambda x: (x.strategy, x.pair, x.split)):
            lines.append(
                f"{r.pair} | {r.strategy} | {r.split} | {r.session or '-'} | {r.direction} | "
                f"{fmt(r.tp_pips)} | {fmt(r.sl_pips)} | {r.status} | {fmt(r.net_sharpe)} | "
                f"{fmt(r.total_return)} | {fmt(r.max_drawdown)} | {r.n_trades} | "
                f"{fmt(r.fold_mean_sharpe)} | {fmt(r.fold_std_sharpe)} | "
                f"{','.join(r.warning_flags) if r.warning_flags else '-'} | {r.notes or '-'}"
            )
        lines.append("")

        lines.append(f"{tier_name} STRATEGY SUMMARY")
        summary = summarize_by_strategy(rows)
        lines.append("strategy | runs | mean_sharpe | mean_return | mean_dd | mean_trades")

        for strat, vals in sorted(summary.items(), key=lambda kv: kv[1]["mean_sharpe"], reverse=True):
            lines.append(
                f"{strat} | {vals['runs']} | {vals['mean_sharpe']:.6f} | {vals['mean_return']:.6f} | "
                f"{vals['mean_dd']:.6f} | {vals['mean_trades']:.2f}"
            )
        lines.append("")

    lines.append("FINAL LOCKED CONFIGS")
    lines.append("-" * 120)
    lines.append("rank | pair | strategy | session | direction | tp | sl | mean_fold_sharpe | fold_std | negative_folds")

    for i, r in enumerate(finals, 1):
        lines.append(
            f"{i} | {r.pair} | {r.strategy} | {r.session or '-'} | {r.direction} | "
            f"{fmt(r.tp_pips)} | {fmt(r.sl_pips)} | {fmt(r.fold_mean_sharpe)} | "
            f"{fmt(r.fold_std_sharpe)} | {r.negative_fold_count}"
        )
    lines.append("")

    lines.append("FINAL TEST COMMANDS")
    lines.append("-" * 120)
    for r in finals:
        cmd = (
            f"python backtest/run_backtest.py --pair {r.pair} --strategy {r.strategy} --split test "
            f"--direction {r.direction}"
        )
        if r.session:
            cmd += f" --session {r.session}"
        if r.tp_pips is not None:
            cmd += f" --tp-pips {r.tp_pips} --sl-pips {r.sl_pips}"
        cmd += " --no-browser"
        lines.append(cmd)
    lines.append("")

    lines.append("EDGE CASE POLICY")
    lines.append("- ZERO_SIGNALS: recorded, never promoted.")
    lines.append("- ZERO_TRADES: recorded, usually eliminated unless explicitly retained for diagnostics.")
    lines.append("- TP_LE_SPREAD / SL_LE_SPREAD: retained but flagged as economically suspect.")
    lines.append("- MISSING_FILE / ENGINE_ERROR / FUTURE_ERROR: logged, not fatal to full plan execution.")
    lines.append("- Fold instability rule: reject if >2 negative folds or fold std sharpe > 1.5.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def build_t1_specs(use_full: bool) -> list[RunSpec]:
    splits = ["train", "val"] if not use_full else ["train", "full"]
    specs = []

    for strategy in ALL_STRATEGIES:
        for pair in ALL_PAIRS:
            for split in splits:
                specs.append(
                    RunSpec(
                        tier="T1",
                        pair=pair,
                        strategy=strategy,
                        split=split,
                    )
                )
    return specs


def build_t2_specs(strategies: list[str]) -> list[RunSpec]:
    specs = []

    for strategy in strategies:
        for pair in ALL_PAIRS:
            for session in [s for s in ALL_SESSIONS if s is not None]:
                for direction in ALL_DIRECTIONS:
                    specs.append(
                        RunSpec(
                            tier="T2",
                            pair=pair,
                            strategy=strategy,
                            split="val",
                            session=session,
                            direction=direction,
                        )
                    )
    return specs


def build_t3_specs(best_t2_specs: list[RunSpec]) -> list[RunSpec]:
    specs = []

    for base in best_t2_specs:
        for tp, sl in [x for x in TP_SL_GRID if x != (None, None)]:
            specs.append(
                RunSpec(
                    tier="T3",
                    pair=base.pair,
                    strategy=base.strategy,
                    split="val",
                    session=base.session,
                    direction=base.direction,
                    tp_pips=tp,
                    sl_pips=sl,
                )
            )
    return specs


def build_t4_specs(best_t3_specs: list[RunSpec]) -> list[RunSpec]:
    specs = []

    for base in best_t3_specs:
        specs.append(
            RunSpec(
                tier="T4",
                pair=base.pair,
                strategy=base.strategy,
                split="train",
                session=base.session,
                direction=base.direction,
                tp_pips=base.tp_pips,
                sl_pips=base.sl_pips,
                folds=5,
            )
        )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiered FX master test runner with progress updates and room for future ML models.")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--t1-min-sharpe", type=float, default=0.0)
    parser.add_argument("--t1-keep-top", type=int, default=8)
    parser.add_argument("--t3-top-pairs-per-strategy", type=int, default=3)
    parser.add_argument("--final-top-n", type=int, default=10)
    parser.add_argument("--use-full-in-t1", action="store_true", help="Use train+full in T1 instead of train+val.")
    parser.add_argument("--progress-interval", type=int, default=10, help="Seconds between progress updates.")
    parser.add_argument("--out", type=str, default=str(OUTPUT_DIR / "fx_master_test_plan_results.txt"))
    args = parser.parse_args()

    config = {
        "workers": args.workers,
        "t1_min_sharpe": args.t1_min_sharpe,
        "t1_keep_top": args.t1_keep_top,
        "t3_top_pairs_per_strategy": args.t3_top_pairs_per_strategy,
        "final_top_n": args.final_top_n,
        "use_full_in_t1": args.use_full_in_t1,
        "progress_interval": args.progress_interval,
        "strategy_count": len(ALL_STRATEGIES),
        "pair_count": len(ALL_PAIRS),
        "sessions": ALL_SESSIONS,
        "directions": ALL_DIRECTIONS,
        "tp_sl_grid": TP_SL_GRID,
        "ml_extension_ready": True,
    }

    tiers: dict[str, list[RunOutcome]] = {}

    print("\n" + "=" * 110)
    print("TIER 1: Broad strategy screening")
    print("=" * 110)
    t1_specs = build_t1_specs(args.use_full_in_t1)
    tiers["T1"] = run_parallel(t1_specs, args.workers, "T1", args.progress_interval)

    survivors = select_t1_survivors(tiers["T1"], args.t1_min_sharpe, args.t1_keep_top)
    print(f"[T1] Survivors ({len(survivors)}): {', '.join(survivors) if survivors else 'none'}")

    print("\n" + "=" * 110)
    print("TIER 2: Session and direction optimization")
    print("=" * 110)
    t2_specs = build_t2_specs(survivors)
    tiers["T2"] = run_parallel(t2_specs, args.workers, "T2", args.progress_interval)

    best_t2_specs = select_t2_best_configs(tiers["T2"])

    print("\n" + "=" * 110)
    print("TIER 3: TP/SL optimization")
    print("=" * 110)
    t3_specs = build_t3_specs(best_t2_specs)
    tiers["T3"] = run_parallel(t3_specs, args.workers, "T3", args.progress_interval)

    best_t3_specs = select_t3_best_configs(tiers["T3"], args.t3_top_pairs_per_strategy)

    print("\n" + "=" * 110)
    print("TIER 4: Walk-forward validation")
    print("=" * 110)
    t4_specs = build_t4_specs(best_t3_specs)
    tiers["T4"] = run_parallel(t4_specs, args.workers, "T4", args.progress_interval)

    finals = final_rank(tiers["T4"], args.final_top_n)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_txt_report(out_path, config, tiers, finals)

    print()
    print("=" * 110)
    print("FINAL SUMMARY")
    print("=" * 110)
    print(f"Output report : {out_path}")
    print(f"Final configs : {len(finals)}")
    if finals:
        print("Top configs:")
        for i, r in enumerate(finals[:10], 1):
            print(
                f"  {i:>2}. {r.pair:<7} | {r.strategy:<30} | "
                f"session={r.session or '-':<7} | direction={r.direction:<10} | "
                f"tp={r.tp_pips if r.tp_pips is not None else '-':<5} | "
                f"sl={r.sl_pips if r.sl_pips is not None else '-':<5} | "
                f"fold_mean_sharpe={r.fold_mean_sharpe:+.4f} | "
                f"fold_std={r.fold_std_sharpe:.4f}"
            )
    print("=" * 110)
    print(f"Wrote: {out_path}")
    print()


if __name__ == "__main__":
    main()
