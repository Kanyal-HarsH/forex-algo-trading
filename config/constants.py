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
    "NZDUSD": 1.2,
}

TRAIN_END  = "2021-12-31"
VAL_START  = "2022-01-01"
VAL_END    = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END   = "2025-12-31"
