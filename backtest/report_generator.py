from __future__ import annotations

import json
import math
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing   import List, Optional

import numpy as np

from backtest.engine import BacktestResult

TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "report.html"
DEFAULT_OUT   = Path(__file__).resolve().parent / "reports"


class _NumpyEncoder(json.JSONEncoder):
    """Converts numpy scalars and arrays to plain Python types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

    def iterencode(self, o, _one_shot=False):
        o = _sanitise(o)
        return super().iterencode(o, _one_shot=_one_shot)


def _sanitise(obj):
    """Replace nan/inf floats with None."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise(v) for v in obj]
    return obj


def _serialise_result(r: BacktestResult) -> dict:
    trade_log = [
        {**t, "pair": r.pair, "strategy": r.strategy}
        for t in r.trade_log
    ]

    return {
        "pair":           r.pair,
        "strategy":       r.strategy,
        "split":          r.split,
        "mode":           r.mode,
        "direction_mode": r.direction_mode,
        "fold_index":     r.fold_index,
        "equity":         r.equity,
        "equity_dollars": r.equity_dollars,
        "rolling_sharpe": r.rolling_sharpe,
        "timestamps":     r.timestamps,
        "signal_dist":    r.signal_dist,
        "metrics":        r.metrics,
        "trade_log":      trade_log,
    }


def _build_meta(results: List[BacktestResult], title: str) -> dict:
    pairs      = sorted({r.pair     for r in results})
    strategies = sorted({r.strategy for r in results})
    splits     = sorted({r.split    for r in results})

    mode           = results[0].mode           if results else "research"
    direction_mode = results[0].direction_mode if results else "long_short"

    cap_initial    = results[0].capital_initial if results else None
    cap_final_vals = [r.capital_final for r in results if r.capital_final > 0]
    cap_final_avg  = (
        round(sum(cap_final_vals) / len(cap_final_vals), 2)
        if cap_final_vals else None
    )

    return {
        "title":             title,
        "generated":         datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "pairs":             pairs,
        "strategies":        strategies,
        "splits":            splits,
        "mode":              mode,
        "direction_mode":    direction_mode,
        "capital_initial":   cap_initial,
        "capital_final_avg": cap_final_avg,
    }


def generate_report(
    results:      List[BacktestResult],
    out_path:     Optional[Path] = None,
    title:        str            = "FXAlgo Report",
    open_browser: bool           = True,
) -> Path:
    """Renders all BacktestResults into a self-contained HTML report."""
    if not results:
        raise ValueError("results list is empty -- nothing to report.")

    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(
            f"Report template missing: {TEMPLATE_PATH}\n"
            f"Expected at backtest/templates/report.html"
        )

    if out_path is None:
        DEFAULT_OUT.mkdir(parents=True, exist_ok=True)
        stamp    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        pairs_s  = "_".join(sorted({r.pair     for r in results}))[:40]
        strats_s = "_".join(sorted({r.strategy for r in results}))[:40]
        out_path = DEFAULT_OUT / f"report_{pairs_s}_{strats_s}_{stamp}.html"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "meta":    _build_meta(results, title),
        "results": [_serialise_result(r) for r in results],
    }

    data_json = json.dumps(data, cls=_NumpyEncoder)

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    rendered = template.replace(
        "__DATA_JSON__",
        data_json,
    ).replace(
        "__TITLE__",
        title,
    )

    out_path.write_text(rendered, encoding="utf-8")

    if open_browser:
        webbrowser.open(out_path.resolve().as_uri())

    return out_path
