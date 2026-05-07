from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from config.constants import (
    LSTM_LONG_FEATURES,
    LSTM_LONG_SEQ,
    LSTM_SESSION_FEATURES,
    LSTM_SHORT_FEATURES,
    LSTM_SHORT_SEQ,
    MODELS_DIR,
    SCALERS_DIR,
)


class BaseStrategy(ABC):

    name: str = ""

    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        ...

    @property
    @abstractmethod
    def warmup_bars(self) -> int:
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    @staticmethod
    def _apply_cooldown(signals: pd.Series, cooldown_bars: int) -> pd.Series:
        """Suppress re-entry for `cooldown_bars` bars after each signal event."""
        if cooldown_bars <= 0:
            return signals
        arr = signals.to_numpy().copy()
        remaining = 0
        for i in range(len(arr)):
            if remaining > 0:
                arr[i] = 0
                remaining -= 1
            elif arr[i] != 0:
                remaining = cooldown_bars
        return pd.Series(arr, index=signals.index, dtype=signals.dtype)


class MACrossover(BaseStrategy):

    name = "MACrossover"

    def __init__(
        self,
        fast: int = 20,
        slow: int = 50,
        ma_type: str = "ema",
        cooldown_bars: int = 0,
    ) -> None:
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be less than slow ({slow}).")
        if ma_type not in ("ema", "sma"):
            raise ValueError("ma_type must be 'ema' or 'sma'.")
        self.fast          = fast
        self.slow          = slow
        self.ma_type       = ma_type
        self.cooldown_bars = cooldown_bars
        self.name          = f"MACrossover_f{fast}_s{slow}_{ma_type.upper()}"
        if cooldown_bars > 0:
            self.name += f"_cd{cooldown_bars}"

    @property
    def warmup_bars(self) -> int:
        return self.slow

    def _ma(self, series: pd.Series, period: int) -> pd.Series:
        if self.ma_type == "ema":
            return series.ewm(span=period, adjust=False).mean()
        return series.rolling(period).mean()

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        close      = prices["close"]
        fast_ma    = self._ma(close, self.fast)
        slow_ma    = self._ma(close, self.slow)
        fast_above = fast_ma > slow_ma

        crossed_up   = fast_above  & ~fast_above.shift(1).fillna(False)
        crossed_down = ~fast_above &  fast_above.shift(1).fillna(True)

        signals = pd.Series(0, index=prices.index, dtype=int)
        signals[crossed_up]   =  1
        signals[crossed_down] = -1
        signals.iloc[: self.slow - 1] = 0
        return self._apply_cooldown(signals, self.cooldown_bars)


class MomentumStrategy(BaseStrategy):

    name = "Momentum"

    def __init__(self, lookback: int = 60, cooldown_bars: int = 0) -> None:
        if lookback < 2:
            raise ValueError("lookback must be >= 2.")
        self.lookback      = lookback
        self.cooldown_bars = cooldown_bars
        self.name          = f"Momentum_lb{lookback}"
        if cooldown_bars > 0:
            self.name += f"_cd{cooldown_bars}"

    @property
    def warmup_bars(self) -> int:
        return self.lookback

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        close        = prices["close"]
        rolling_high = close.shift(1).rolling(self.lookback).max()
        rolling_low  = close.shift(1).rolling(self.lookback).min()

        broke_up   = (close > rolling_high) & close.shift(1).le(rolling_high)
        broke_down = (close < rolling_low)  & close.shift(1).ge(rolling_low)

        signals = pd.Series(0, index=prices.index, dtype=int)
        signals[broke_up]   =  1
        signals[broke_down] = -1
        signals.iloc[: self.lookback] = 0
        return self._apply_cooldown(signals, self.cooldown_bars)


class DonchianBreakout(BaseStrategy):

    name = "Donchian"

    def __init__(self, period: int = 20, cooldown_bars: int = 0) -> None:
        if period < 2:
            raise ValueError("period must be >= 2.")
        self.period        = period
        self.cooldown_bars = cooldown_bars
        self.name          = f"Donchian_p{period}"
        if cooldown_bars > 0:
            self.name += f"_cd{cooldown_bars}"

    @property
    def warmup_bars(self) -> int:
        return self.period

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        high = prices["high"] if "high" in prices.columns else prices["close"]
        low  = prices["low"]  if "low"  in prices.columns else prices["close"]

        upper = high.shift(1).rolling(self.period).max()
        lower = low.shift(1).rolling(self.period).min()

        broke_up   = prices["close"] > upper
        broke_down = prices["close"] < lower

        prev_broke_up   = broke_up.shift(1).fillna(False)
        prev_broke_down = broke_down.shift(1).fillna(False)

        signals = pd.Series(0, index=prices.index, dtype=int)
        signals[broke_up   & ~prev_broke_up]   =  1
        signals[broke_down & ~prev_broke_down] = -1
        signals.iloc[: self.period] = 0
        return self._apply_cooldown(signals, self.cooldown_bars)


class RSIMeanReversion(BaseStrategy):

    name = "RSI"

    def __init__(
        self,
        period:        int   = 14,
        oversold:      float = 30.0,
        overbought:    float = 70.0,
        cooldown_bars: int   = 0,
    ) -> None:
        if period < 2:
            raise ValueError("period must be >= 2.")
        if oversold >= overbought:
            raise ValueError("oversold must be less than overbought.")
        self.period        = period
        self.oversold      = oversold
        self.overbought    = overbought
        self.cooldown_bars = cooldown_bars
        self.name          = f"RSI_p{period}_os{int(oversold)}_ob{int(overbought)}"
        if cooldown_bars > 0:
            self.name += f"_cd{cooldown_bars}"

    @property
    def warmup_bars(self) -> int:
        return self.period

    def _rsi(self, close: pd.Series) -> pd.Series:
        delta    = close.diff()
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1.0 / self.period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / self.period, adjust=False).mean()
        rs       = avg_gain / avg_loss.replace(0, np.nan)
        rsi      = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        rsi = self._rsi(prices["close"])

        was_oversold   = rsi.shift(1) <= self.oversold
        was_overbought = rsi.shift(1) >= self.overbought
        now_above_os   = rsi > self.oversold
        now_below_ob   = rsi < self.overbought

        crossed_up   = was_oversold   & now_above_os
        crossed_down = was_overbought & now_below_ob

        signals = pd.Series(0, index=prices.index, dtype=int)
        signals[crossed_up]   =  1
        signals[crossed_down] = -1
        signals.iloc[: self.period] = 0
        return self._apply_cooldown(signals, self.cooldown_bars)


class BollingerBreakout(BaseStrategy):

    name = "BB"

    def __init__(
        self,
        period:        int   = 20,
        std_dev:       float = 2.0,
        cooldown_bars: int   = 0,
    ) -> None:
        if period < 2:
            raise ValueError("period must be >= 2.")
        if std_dev <= 0:
            raise ValueError("std_dev must be positive.")
        self.period        = period
        self.std_dev       = std_dev
        self.cooldown_bars = cooldown_bars
        self.name          = f"BB_p{period}_std{str(std_dev).replace('.', '_')}"
        if cooldown_bars > 0:
            self.name += f"_cd{cooldown_bars}"

    @property
    def warmup_bars(self) -> int:
        return self.period

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        mid   = close.rolling(self.period).mean()
        sigma = close.rolling(self.period).std()
        upper = mid + self.std_dev * sigma
        lower = mid - self.std_dev * sigma

        above_upper = close > upper
        below_lower = close < lower

        broke_up   = above_upper & ~above_upper.shift(1).fillna(False)
        broke_down = below_lower & ~below_lower.shift(1).fillna(False)

        signals = pd.Series(0, index=prices.index, dtype=int)
        signals[broke_up]   =  1
        signals[broke_down] = -1
        signals.iloc[: self.period] = 0
        return self._apply_cooldown(signals, self.cooldown_bars)


class MACDSignalCross(BaseStrategy):

    name = "MACD"

    def __init__(
        self,
        fast:          int = 12,
        slow:          int = 26,
        signal_period: int = 9,
        cooldown_bars: int = 0,
    ) -> None:
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be less than slow ({slow}).")
        if signal_period < 1:
            raise ValueError("signal_period must be >= 1.")
        self.fast          = fast
        self.slow          = slow
        self.signal_period = signal_period
        self.cooldown_bars = cooldown_bars
        self.name          = f"MACD_f{fast}_s{slow}_sig{signal_period}"
        if cooldown_bars > 0:
            self.name += f"_cd{cooldown_bars}"

    @property
    def warmup_bars(self) -> int:
        return self.slow + self.signal_period - 1

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        close       = prices["close"]
        ema_fast    = close.ewm(span=self.fast,   adjust=False).mean()
        ema_slow    = close.ewm(span=self.slow,   adjust=False).mean()
        macd_line   = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()

        macd_above = macd_line > signal_line

        crossed_up   = macd_above  & ~macd_above.shift(1).fillna(False)
        crossed_down = ~macd_above &  macd_above.shift(1).fillna(True)

        warmup = self.slow + self.signal_period - 1

        signals = pd.Series(0, index=prices.index, dtype=int)
        signals[crossed_up]   =  1
        signals[crossed_down] = -1
        signals.iloc[: warmup] = 0
        return self._apply_cooldown(signals, self.cooldown_bars)


STRATEGY_REGISTRY: dict[str, BaseStrategy] = {
    # Trend - MA Crossover (3)
    # f20/s50: classic swing scale, 1h cooldown prevents re-entry within same move
    # f50/s200: meaningful daily-scale trend (50-min fast, ~3.3h slow), shorter cooldown
    "MACrossover_f20_s50_EMA":   MACrossover(20,  50,  "ema", cooldown_bars=60),
    "MACrossover_f50_s200_EMA":  MACrossover(50,  200, "ema", cooldown_bars=30),
    "MACrossover_f20_s50_SMA":   MACrossover(20,  50,  "sma", cooldown_bars=60),

    # Breakout - Momentum (2)
    "Momentum_lb60":             MomentumStrategy(lookback=60,  cooldown_bars=60),
    "Momentum_lb120":            MomentumStrategy(lookback=120, cooldown_bars=30),

    # Breakout - Donchian (2)
    "Donchian_p20":              DonchianBreakout(period=20,  cooldown_bars=60),
    "Donchian_p55":              DonchianBreakout(period=55,  cooldown_bars=30),

    # Mean Reversion - RSI (2)
    # os30/ob70 + 1h cooldown: standard thresholds, cooldown prevents churn
    # os20/ob80: extreme thresholds fire rarely; shorter cooldown sufficient
    "RSI_p14_os30_ob70":         RSIMeanReversion(14, 30, 70, cooldown_bars=60),
    "RSI_p14_os20_ob80":         RSIMeanReversion(14, 20, 80, cooldown_bars=30),

    # Mean Reversion - Bollinger (2)
    # p20/2std + 1h cooldown: tight bands need cooldown to avoid false-breakout churn
    # p60/2std: 1-hour BB already fires slowly; shorter cooldown sufficient
    "BB_p20_std2_0":             BollingerBreakout(20, 2.0, cooldown_bars=60),
    "BB_p60_std2_0":             BollingerBreakout(60, 2.0, cooldown_bars=30),

    # MACD (2) - scaled to economically meaningful timeframes on 1-min data
    # f26/s65: 26-min fast / ~1h slow - medium-scale trend confirmation
    # f78/s195: ~1.3h fast / ~3.3h slow - slow trend filter, very few signals
    "MACD_f26_s65_sig9":         MACDSignalCross(26,  65,  9, cooldown_bars=60),
    "MACD_f78_s195_sig13":       MACDSignalCross(78, 195, 13, cooldown_bars=30),
}


class MLStrategy(BaseStrategy):
    """Adapter wrapping a fitted sklearn or PyTorch model as a backtest strategy."""

    def __init__(
        self,
        model: object,
        scaler: object,
        feature_cols: list[str],
        model_name: str,
        model_type: str = "sklearn",
        sequence_len: int | None = None,
        long_feature_cols: list[str] | None = None,
        session_feature_cols: list[str] | None = None,
    ) -> None:
        self.model               = model
        self.scaler              = scaler
        self.feature_cols        = feature_cols
        self.model_name          = model_name
        self.model_type          = model_type
        self.sequence_len        = sequence_len
        self.long_feature_cols   = long_feature_cols or []
        self.session_feature_cols = session_feature_cols or []
        self.name                = model_name

    @property
    def warmup_bars(self) -> int:
        return self.sequence_len or 0

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"MLStrategy '{self.name}' requires features missing from df: {missing}. "
                f"Re-run scripts/split_fx_data.py to regenerate splits with full feature set."
            )
        # The LSTM long branch reads features that are NOT in the LR scaler's
        # feature_cols (e.g. rv_ratio_10_60, same_minute_prev_day_logrange)
        # directly from df. Silently substituting zeros for those columns made
        # earlier replication runs produce a degenerate all-FLAT signal with no
        # error at all, which is hard to diagnose. Validate up-front instead.
        if self.model_type == "lstm" and self.long_feature_cols:
            scaler_cols = set(self.feature_cols)
            df_cols     = set(df.columns)
            missing_long = [
                c for c in self.long_feature_cols
                if c not in scaler_cols and c not in df_cols
            ]
            if missing_long:
                raise KeyError(
                    f"MLStrategy '{self.name}' (LSTM) needs long-branch features "
                    f"that are missing from both the scaler and the dataframe: "
                    f"{missing_long}. This usually means the test/val parquet was "
                    f"copied from an older clone without the F3 extended features. "
                    f"Re-run scripts/features_fx_data.py and scripts/split_fx_data.py "
                    f"to regenerate the dataset with the current feature schema."
                )
        X = df[self.feature_cols].fillna(0).values
        X_scaled = self.scaler.transform(X)

        if self.model_type == "sklearn":
            preds = self.model.predict(X_scaled)
        elif self.model_type == "lstm":
            preds = self._lstm_predict(X_scaled, df.index, df)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type!r}")

        signals = pd.Series(preds, index=df.index, dtype=int).clip(-1, 1)
        if self.warmup_bars > 0:
            signals.iloc[: self.warmup_bars] = 0
        return signals

    def _lstm_predict(
        self,
        X_scaled: np.ndarray,
        index: "pd.Index",
        df: pd.DataFrame,
    ) -> np.ndarray:
        """Batched LSTM inference using vectorised sliding short + long windows."""
        import torch

        short_seq = LSTM_SHORT_SEQ
        long_seq  = LSTM_LONG_SEQ
        n         = len(X_scaled)
        preds     = np.zeros(n, dtype=int)

        if n < long_seq:
            return preds

        feat_idx  = {f: i for i, f in enumerate(self.feature_cols)}
        short_idx = [feat_idx[f] for f in LSTM_SHORT_FEATURES if f in feat_idx]
        sess_cols = [c for c in self.session_feature_cols if c in df.columns]

        # Long-branch array: use scaler-scaled value if available, else raw df.
        # Mirrors scripts/train_model.py:_scale_cols.
        long_arr = np.zeros((n, len(self.long_feature_cols)), dtype=np.float32)
        for j, c in enumerate(self.long_feature_cols):
            if c in feat_idx:
                long_arr[:, j] = X_scaled[:, feat_idx[c]]
            elif c in df.columns:
                long_arr[:, j] = df[c].fillna(0).values.astype(np.float32)

        sess_arr  = (df[sess_cols].fillna(0).values.astype(np.float32)
                     if sess_cols else np.zeros((n, 0), dtype=np.float32))
        short_arr = X_scaled[:, short_idx].astype(np.float32, copy=False)

        # Build sliding windows once. A "sample" is one prediction; the i-th
        # sample's bar is (long_seq - 1 + i) so the long window has enough
        # history. Total samples: n - long_seq + 1.
        n_samples = n - long_seq + 1

        # short windows: (n_samples, short_seq, n_short_feats)
        short_view = np.lib.stride_tricks.sliding_window_view(
            short_arr, window_shape=short_seq, axis=0
        )  # -> (n - short_seq + 1, n_feats, short_seq)
        short_windows = np.ascontiguousarray(
            short_view[long_seq - short_seq:].transpose(0, 2, 1)
        )

        # long windows: (n_samples, long_seq, n_long_feats)
        long_view = np.lib.stride_tricks.sliding_window_view(
            long_arr, window_shape=long_seq, axis=0
        )  # -> (n - long_seq + 1, n_feats, long_seq)
        long_windows = np.ascontiguousarray(long_view.transpose(0, 2, 1))

        # session inputs: take session features AT each prediction bar
        sess_inputs = np.ascontiguousarray(sess_arr[long_seq - 1:])

        batch_size = 4096
        self.model.eval()
        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                s_t = torch.from_numpy(short_windows[start:end])
                l_t = torch.from_numpy(long_windows[start:end])
                z_t = torch.from_numpy(sess_inputs[start:end])
                logits = self.model(s_t, l_t, z_t)
                cls = logits.argmax(dim=1).cpu().numpy().astype(np.int64) - 1
                preds[long_seq - 1 + start : long_seq - 1 + end] = cls

        return preds


class FutureModelAdapter(BaseStrategy):
    """Placeholder for ML models not yet trained."""

    name = "FutureModelAdapter"

    @property
    def warmup_bars(self) -> int:
        return 0

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=df.index, dtype=int)


def get_ml_strategy(pair: str, strategy_name: str) -> MLStrategy:
    """Load a trained ML model + scaler and return a configured MLStrategy.

    strategy_name format: 'LR_global', 'LR_london', 'LSTM_global', etc.
    """
    parts = strategy_name.split("_")
    if len(parts) != 2:
        raise ValueError(
            f"ML strategy name must be '<MODEL_TYPE>_<SESSION>' "
            f"(e.g. 'LR_global', 'LSTM_london'). Got: {strategy_name!r}"
        )
    model_type_str, session = parts[0].upper(), parts[1].lower()

    if model_type_str not in ("LR", "LSTM"):
        raise ValueError(f"Unknown ML model type: {model_type_str!r}")
    if session not in ("global", "london", "ny", "asia"):
        raise ValueError(f"Unknown session: {session!r}")

    model_dir = (
        MODELS_DIR / "global"
        if session == "global"
        else MODELS_DIR / "session" / session
    )

    scaler_path = SCALERS_DIR / f"{pair}_scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found: {scaler_path}\n"
            f"Run: python scripts/split_fx_data.py --force"
        )
    with open(scaler_path, "rb") as fh:
        scaler_obj = pickle.load(fh)
    scaler      = scaler_obj["scaler"]
    scaler_cols = scaler_obj["feature_cols"]

    if model_type_str == "LR":
        model_path = model_dir / f"{pair}_logreg_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"LR model not found: {model_path}\n"
                f"Run: python scripts/train_model.py --pair {pair} "
                f"--model-type lr --session {session}"
            )
        with open(model_path, "rb") as fh:
            model = pickle.load(fh)
        return MLStrategy(
            model=model,
            scaler=scaler,
            feature_cols=scaler_cols,
            model_name=strategy_name,
            model_type="sklearn",
        )

    # LSTM
    import torch
    from scripts.train_model import FXMultiScaleLSTM

    if FXMultiScaleLSTM is None:
        raise ImportError(
            "PyTorch is not installed. Install with: pip install torch"
        )

    model_path = model_dir / f"{pair}_lstm_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"LSTM model not found: {model_path}\n"
            f"Run: python scripts/train_model.py --pair {pair} "
            f"--model-type lstm --session {session}"
        )

    checkpoint     = torch.load(model_path, map_location="cpu")
    long_feat_cols = checkpoint.get("long_feature_cols", LSTM_LONG_FEATURES)

    lstm_model = FXMultiScaleLSTM(
        short_input_size=checkpoint.get("short_input_size", len(LSTM_SHORT_FEATURES)),
        long_input_size=checkpoint.get("long_input_size",  len(long_feat_cols)),
        session_size=checkpoint.get("session_size",        len(LSTM_SESSION_FEATURES)),
    )
    state_dict = checkpoint["model_state_dict"]
    # torch.compile() saves keys with "_orig_mod." prefix - strip it for plain loading
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    lstm_model.load_state_dict(state_dict)
    lstm_model.eval()

    return MLStrategy(
        model=lstm_model,
        scaler=scaler,
        feature_cols=scaler_cols,
        model_name=strategy_name,
        model_type="lstm",
        sequence_len=LSTM_LONG_SEQ,
        long_feature_cols=long_feat_cols,
        session_feature_cols=LSTM_SESSION_FEATURES,
    )


def get_strategy(name: str, pair: str | None = None) -> BaseStrategy:
    """Return a strategy by name.

    For ML strategy names (starting with 'LR_' or 'LSTM_'), pair is required
    to load the correct model files. Rule-based strategies ignore pair.
    """
    if name.startswith(("LR_", "LSTM_")):
        if pair is None:
            raise ValueError(
                f"ML strategy {name!r} requires pair= to load model files. "
                f"Example: get_strategy('LR_global', pair='EURUSD')"
            )
        return get_ml_strategy(pair=pair, strategy_name=name)

    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{name}'.\n"
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    return STRATEGY_REGISTRY[name]


if __name__ == "__main__":
    n = 2000
    np.random.seed(0)

    close  = pd.Series(np.exp(np.random.randn(n).cumsum() * 0.001 + 7.0))
    high   = close * (1 + np.abs(np.random.randn(n)) * 0.0005)
    low    = close * (1 - np.abs(np.random.randn(n)) * 0.0005)
    prices = pd.DataFrame({"close": close, "high": high, "low": low})

    # Mean-reversion families fire more events due to threshold crossings.
    # Crossover families (MA, MACD, Momentum, Donchian) are sparse by design.
    MEAN_REVERSION_STRATEGIES = {
        "RSI_p14_os30_ob70",
        "RSI_p14_os20_ob80",
        "BB_p20_std2_0",
        "BB_p60_std2_0",
    }

    print("Running strategy smoke tests...")

    for key, strat in STRATEGY_REGISTRY.items():
        sigs = strat.generate_signals(prices)

        assert sigs.isin([-1, 0, 1]).all(), f"{key}: invalid signal values"
        assert len(sigs) == n,              f"{key}: length mismatch"
        assert not sigs.isna().any(),       f"{key}: NaN in signals"

        counts   = sigs.value_counts().to_dict()
        n_long   = counts.get(1,  0)
        n_short  = counts.get(-1, 0)
        n_flat   = counts.get(0,  0)
        flat_pct = n_flat / n * 100

        print(
            f"  {key:<40} "
            f"Long: {n_long:>4}  Short: {n_short:>4}  "
            f"Flat: {n_flat:>5} ({flat_pct:.1f}%)  "
            f"Events: {n_long + n_short}  "
            f"Warmup: {strat.warmup_bars}"
        )

        min_flat = 50.0 if key in MEAN_REVERSION_STRATEGIES else 80.0
        assert flat_pct > min_flat, (
            f"{key}: only {flat_pct:.1f}% flat bars "
            f"(threshold {min_flat:.0f}%) -- "
            f"likely outputting positions not crossover events"
        )

    print("All strategy smoke tests passed.")
