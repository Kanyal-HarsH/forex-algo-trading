from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

TRADING_DAYS_PER_YEAR = int(os.getenv("TRADING_DAYS_PER_YEAR", "252"))

BARS_PER_TRADING_DAY = int(os.getenv("BARS_PER_TRADING_DAY", "390"))

ANNUALISATION_FACTOR = TRADING_DAYS_PER_YEAR * BARS_PER_TRADING_DAY

MIN_BARS_PER_DAY = int(os.getenv("MIN_BARS_PER_DAY", "1200"))

MIN_BARS_FOR_SHARPE = 2

ROLLING_SHARPE_WINDOW = int(os.getenv("ROLLING_SHARPE_WINDOW", "390"))

PROFIT_FACTOR_CAP = 999.0

DEFAULT_CAPITAL = float(os.getenv("DEFAULT_CAPITAL", "10000.0"))

DEFAULT_N_FOLDS = int(os.getenv("DEFAULT_N_FOLDS", "5"))

PURGE_ROWS = int(os.getenv("PURGE_ROWS", "15"))

HORIZON_PRIMARY = int(os.getenv("HORIZON_PRIMARY", "5"))

HORIZON_SECONDARY = int(os.getenv("HORIZON_SECONDARY", "15"))

VOL_REGIME_WINDOW = int(os.getenv("VOL_REGIME_WINDOW", "30"))

VOL_HIGH_REGIME_PERCENTILE = float(os.getenv("VOL_HIGH_REGIME_PERCENTILE", "0.80"))

DOWNLOAD_RETRY_COUNT = 3

DOWNLOAD_RETRY_WAIT_SECS = 5

MIN_ZIP_BYTES = 100

PAIR_SPREAD_PIPS: dict[str, float] = {
    "EURUSD": 0.6,
    "GBPUSD": 0.8,
    "USDJPY": 0.7,
    "USDCHF": 1.0,
    "AUDUSD": 0.8,
    "USDCAD": 1.0,
    "NZDUSD": 1.4,
}

PAIRS: tuple[str, ...] = tuple(PAIR_SPREAD_PIPS.keys())

PAIR_PIP_SIZES: dict[str, float] = {p: (0.01 if "JPY" in p else 0.0001) for p in PAIRS}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
FOLDS_DIR    = DATASETS_DIR / "folds"
TRAIN_DIR    = DATASETS_DIR / "train"
VAL_DIR      = DATASETS_DIR / "val"
TEST_DIR     = DATASETS_DIR / "test"
CLEANED_DIR  = PROJECT_ROOT / "data" / "processed" / "cleaned"


def fold_parquet_path(pair: str, k: int, split_kind: str = "train") -> Path:
    return FOLDS_DIR / f"fold_{k}" / f"{pair}_{split_kind}.parquet"


def _env_int_tuple(env_var: str, default: str) -> tuple[int, ...]:
    raw = os.getenv(env_var, default)
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


RET_WINDOWS       = _env_int_tuple("RET_WINDOWS",       "1,5,15")
VOL_WINDOWS       = _env_int_tuple("VOL_WINDOWS",       "10,30,60")
MA_WINDOWS        = _env_int_tuple("MA_WINDOWS",        "10,30,60,120")
MOM_WINDOWS       = _env_int_tuple("MOM_WINDOWS",       "5,15,30")
RANGE_MA_WINDOWS  = _env_int_tuple("RANGE_MA_WINDOWS",  "10,30")


def _env_tp_sl_grid(env_var: str, default: str) -> tuple[tuple[int, int], ...]:
    raw = os.getenv(env_var, default)
    out: list[tuple[int, int]] = []
    for token in raw.split(";"):
        token = token.strip()
        if not token:
            continue
        tp, sl = token.split(",")
        out.append((int(tp.strip()), int(sl.strip())))
    return tuple(out)


TP_SL_GRID: tuple[tuple[int, int], ...] = _env_tp_sl_grid(
    "TP_SL_GRID", "10,5;15,7;20,10;30,15;50,20;100,40"
)

TRAIN_END  = "2021-12-31"
VAL_START  = "2022-01-01"
VAL_END    = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END   = "2025-12-31"

# --- ML directories
SCALERS_DIR = PROJECT_ROOT / "scalers"
MODELS_DIR  = PROJECT_ROOT / "models"

# --- LR feature list (frozen - do not modify)
LR_FEATURES: list[str] = [
    "hour_sin", "hour_cos",
    "session_asia", "session_london", "session_ny", "session_overlap",
    "ret_1", "ret_5", "ret_15", "abs_ret_1", "range_pct",
    "rv_10", "rv_30", "rv_60", "volatility_regime_high",
    "price_to_sma_30", "mom_5", "mom_15",
]

# --- LSTM feature lists (frozen - do not modify)
LSTM_SHORT_FEATURES: list[str] = ["ret_1", "ret_5", "abs_ret_1", "range_pct", "rv_10"]
LSTM_LONG_FEATURES:  list[str] = ["rv_10", "rv_30", "rv_60", "rv_ratio_10_60"]
# same_minute_prev_day_logrange added conditionally in train_model.py when available
LSTM_SESSION_FEATURES: list[str] = [
    "session_asia", "session_london", "session_ny", "session_overlap"
]
LSTM_SHORT_SEQ: int = 15
LSTM_LONG_SEQ:  int = 60

# --- LR regularisation sweep
LR_C_VALUES: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0)

# --- Session names
SESSION_NAMES: tuple[str, ...] = ("global", "london", "ny", "asia")

# --- Session filter map: maps session name -> list of "session" column values
# "Overlap" rows (UTC 13-16) included in both london and ny training data.
SESSION_FILTER_MAP: dict[str, list[str]] = {
    "london": ["London", "Overlap"],
    "ny":     ["New_York", "Overlap"],
    "asia":   ["Asia"],
}

# --- Session-specific train directories
SESSION_TRAIN_DIRS: dict[str, Path] = {
    session: DATASETS_DIR / f"train_{session}"
    for session in SESSION_NAMES
    if session != "global"
}

# --- Model shortcode aliases
SESSION_ALIASES: dict[str, str] = {
    "gl":  "global",
    "ldn": "london",
    "ny":  "ny",
    "as":  "asia",
}


def parse_model_code(code: str) -> tuple[str, str, str]:
    """Parse 'eurusd-lr-gl' -> (pair='EURUSD', model_type='lr', session='global')."""
    parts = code.lower().split("-")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid model code {code!r}. Expected {{pair}}-{{model}}-{{session}}, e.g. eurusd-lr-gl"
        )
    pair_raw, model_raw, session_raw = parts
    pair = pair_raw.upper()
    if pair not in PAIRS:
        raise ValueError(f"Unknown pair {pair!r} in code {code!r}. Valid: {list(PAIRS)}")
    if model_raw not in ("lr", "lstm"):
        raise ValueError(f"Unknown model {model_raw!r} in code {code!r}. Valid: lr, lstm")
    session = SESSION_ALIASES.get(session_raw)
    if session is None:
        raise ValueError(
            f"Unknown session alias {session_raw!r} in code {code!r}. Valid: {list(SESSION_ALIASES)}"
        )
    return pair, model_raw, session
