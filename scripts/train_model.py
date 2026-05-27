from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from config.constants import (
    ANNUALISATION_FACTOR,
    LR_C_VALUES,
    LR_FEATURES,
    LSTM_LONG_FEATURES,
    LSTM_LONG_SEQ,
    LSTM_SESSION_FEATURES,
    LSTM_SHORT_FEATURES,
    LSTM_SHORT_SEQ,
    MODELS_DIR,
    PAIRS,
    SCALERS_DIR,
    SESSION_FILTER_MAP,
    SESSION_NAMES,
    SESSION_TRAIN_DIRS,
    TRAIN_DIR,
    VAL_DIR,
    parse_model_code,
)
from scripts._common import ensure_dir

# --- PyTorch: optional dependency
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    class FXMultiScaleLSTM(nn.Module):
        """Two-branch LSTM with short-term and long-term branches plus session injection."""

        def __init__(
            self,
            short_input_size: int = 5,
            long_input_size: int = 4,
            session_size: int = 4,
            hidden_size: int = 32,
            n_classes: int = 3,
        ) -> None:
            super().__init__()
            self.short_lstm = nn.LSTM(
                short_input_size, hidden_size, num_layers=1, batch_first=True
            )
            self.long_lstm = nn.LSTM(
                long_input_size, hidden_size, num_layers=1, batch_first=True
            )
            merged_size = hidden_size * 2 + session_size  # 32+32+4 = 68
            self.head = nn.Sequential(
                nn.Linear(merged_size, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, n_classes),
                nn.LogSoftmax(dim=1),
            )

        def forward(
            self,
            short_seq: "torch.Tensor",  # (batch, 15, short_input_size)
            long_seq: "torch.Tensor",   # (batch, 60, long_input_size)
            session: "torch.Tensor",    # (batch, session_size)
        ) -> "torch.Tensor":
            _, (h_short, _) = self.short_lstm(short_seq)
            _, (h_long, _) = self.long_lstm(long_seq)
            merged = torch.cat(
                [h_short.squeeze(0), h_long.squeeze(0), session], dim=1
            )
            return self.head(merged)

    class FXSequenceDataset(Dataset):
        """Yields (short_seq, long_seq, session_feats, label) tuples.

        Sequences are built within a single DataFrame - they never span the
        train/val boundary because train and val come from separate parquets.
        """

        def __init__(
            self,
            X_short: np.ndarray,
            X_long: np.ndarray,
            X_sess: np.ndarray,
            y: np.ndarray,
            short_seq: int = LSTM_SHORT_SEQ,
            long_seq: int = LSTM_LONG_SEQ,
            max_bar_idx: int | None = None,
        ) -> None:
            self.X_short = X_short
            self.X_long = X_long
            self.X_sess = X_sess
            self.y = y
            self.short_seq = short_seq
            self.long_seq = long_seq
            end = max_bar_idx if max_bar_idx is not None else len(y) - 1
            self.valid_idx = list(range(long_seq - 1, end + 1))

        def __len__(self) -> int:
            return len(self.valid_idx)

        def __getitem__(self, idx: int):
            i = self.valid_idx[idx]
            short_window = self.X_short[i - self.short_seq + 1 : i + 1]
            long_window = self.X_long[i - self.long_seq + 1 : i + 1]
            sess = self.X_sess[i]
            label = self.y[i]
            return (
                torch.tensor(short_window, dtype=torch.float32),
                torch.tensor(long_window, dtype=torch.float32),
                torch.tensor(sess, dtype=torch.float32),
                torch.tensor(label, dtype=torch.long),
            )

    _TORCH_AVAILABLE = True

except ImportError:
    FXMultiScaleLSTM = None  # type: ignore[assignment,misc]
    FXSequenceDataset = None  # type: ignore[assignment,misc]
    _TORCH_AVAILABLE = False


# --- Argument parsing

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train LR or LSTM model for a given pair and session.\n\n"
            "Shortcode form (preferred):\n"
            "  python train_model.py eurusd-lr-gl --c-sweep\n\n"
            "Verbose form (still supported):\n"
            "  python train_model.py --pair EURUSD --model-type lr --session global --c-sweep\n\n"
            "Session aliases: gl=global  ldn=london  ny=ny  as=asia"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "code", nargs="?", default=None, metavar="CODE",
        help="Shortcode e.g. eurusd-lr-gl. Overrides --pair/--model-type/--session when given.",
    )
    parser.add_argument(
        "--pair", default=None, choices=list(PAIRS),
        help="Currency pair to train on.",
    )
    parser.add_argument(
        "--model-type", default=None, choices=["lr", "lstm"],
        dest="model_type",
        help="Model type: logistic regression or multi-scale LSTM.",
    )
    parser.add_argument(
        "--session", default="global", choices=list(SESSION_NAMES),
        help="Training session filter (default: global - all sessions).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing model files.",
    )
    parser.add_argument(
        "--c-sweep", action="store_true",
        help="(LR only) Sweep C regularisation values and pick best on val Sharpe.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2048, dest="batch_size",
        help="(LSTM only) Mini-batch size. Reduce to 512 if GPU runs out of memory (default: 2048).",
    )
    parser.add_argument(
        "--no-amp", action="store_true", dest="no_amp",
        help="(LSTM only) Disable automatic mixed precision. Use if you see NaN losses.",
    )
    return parser.parse_args()


# --- Data loading

def load_scaler(pair: str) -> tuple[StandardScaler, list[str]]:
    path = SCALERS_DIR / f"{pair}_scaler.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Scaler not found: {path}\n"
            f"Run: python scripts/split_fx_data.py --force"
        )
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    return obj["scaler"], obj["feature_cols"]


def load_data(pair: str, session: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train + val DataFrames for a pair/session combination.

    Non-global sessions load from SESSION_TRAIN_DIRS (subset of global train).
    Val data is always the full val set regardless of session.
    """
    if session == "global":
        train_path = TRAIN_DIR / f"{pair}_train.parquet"
    else:
        train_path = SESSION_TRAIN_DIRS[session] / f"{pair}_train.parquet"

    val_path = VAL_DIR / f"{pair}_val.parquet"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Train parquet not found: {train_path}\n"
            f"Run: python scripts/split_fx_data.py --force"
        )
    if not val_path.exists():
        raise FileNotFoundError(
            f"Val parquet not found: {val_path}\n"
            f"Run: python scripts/split_fx_data.py --force"
        )

    return pd.read_parquet(train_path), pd.read_parquet(val_path)


# --- LR helpers

def _signal_sharpe(preds: np.ndarray, close: pd.Series) -> float:
    """Approximate gross Sharpe from signals * next-bar returns (no spread).

    Used only for C-sweep selection - not the same as backtest Sharpe.
    """
    ret_1 = close.reset_index(drop=True).pct_change().shift(-1).fillna(0).values
    signal_ret = preds * ret_1
    std = signal_ret.std()
    if std == 0:
        return 0.0
    return float(signal_ret.mean() / std * np.sqrt(ANNUALISATION_FACTOR))


def _model_dir(session: str) -> Path:
    if session == "global":
        return MODELS_DIR / "global"
    return MODELS_DIR / "session" / session


# --- LR training

def train_lr(pair: str, session: str, c_sweep: bool, force: bool) -> None:
    model_dir  = _model_dir(session)
    model_path = model_dir / f"{pair}_logreg_model.pkl"

    if model_path.exists() and not force:
        print(f"[LR] {pair}/{session}: model exists - skip (use --force to overwrite)")
        return

    train_df, val_df = load_data(pair, session)
    scaler, scaler_cols = load_scaler(pair)

    feature_cols = [
        f for f in LR_FEATURES
        if f in train_df.columns and f in val_df.columns and f in scaler_cols
    ]
    missing = [f for f in LR_FEATURES if f not in feature_cols]
    if missing:
        print(f"  WARNING: {len(missing)} LR features missing from data: {missing}")

    label_col = "label"
    train_labeled = train_df.dropna(subset=[label_col]).reset_index(drop=True)
    val_labeled   = val_df.dropna(subset=[label_col]).reset_index(drop=True)

    X_train = train_labeled[feature_cols].fillna(0).values
    y_train = train_labeled[label_col].astype(int).values
    X_val   = val_labeled[feature_cols].fillna(0).values
    y_val   = val_labeled[label_col].astype(int).values

    # Scale using column positions from the scaler
    col_idx = {c: i for i, c in enumerate(scaler_cols)}
    X_full_train = train_labeled[scaler_cols].fillna(0).values
    X_full_val   = val_labeled[scaler_cols].fillna(0).values
    X_train_scaled = scaler.transform(X_full_train)[:, [col_idx[c] for c in feature_cols]]
    X_val_scaled   = scaler.transform(X_full_val)[:,   [col_idx[c] for c in feature_cols]]

    best_c = LR_C_VALUES[-1]
    best_sharpe = -np.inf

    if c_sweep:
        print(f"[LR] {pair}/{session}: C sweep over {LR_C_VALUES}")
        for c_val in LR_C_VALUES:
            lr = LogisticRegression(
                C=c_val, solver="lbfgs",
                max_iter=1000, class_weight="balanced", random_state=42,
            )
            lr.fit(X_train_scaled, y_train)
            val_preds  = lr.predict(X_val_scaled)
            val_sharpe = _signal_sharpe(val_preds, val_labeled["close"])
            val_acc    = accuracy_score(y_val, val_preds)
            val_f1     = f1_score(y_val, val_preds, average="macro", zero_division=0)
            print(
                f"    C={c_val:<8}  acc={val_acc:.4f}  "
                f"macro_f1={val_f1:.4f}  signal_sharpe={val_sharpe:+.4f}"
            )
            if val_sharpe > best_sharpe:
                best_sharpe = val_sharpe
                best_c = c_val
        print(f"  Best C={best_c}  (val signal Sharpe={best_sharpe:+.4f})")

    final_lr = LogisticRegression(
        C=best_c, solver="lbfgs",
        max_iter=1000, class_weight="balanced", random_state=42,
    )
    final_lr.fit(X_train_scaled, y_train)

    val_preds  = final_lr.predict(X_val_scaled)
    val_acc    = accuracy_score(y_val, val_preds)
    val_f1     = f1_score(y_val, val_preds, average="macro", zero_division=0)
    val_dist   = dict(zip(*np.unique(val_preds, return_counts=True)))
    val_sharpe = _signal_sharpe(val_preds, val_labeled["close"])

    print(f"[LR] {pair}/{session}: FINAL  C={best_c}")
    print(f"  Val accuracy         : {val_acc:.4f}")
    print(f"  Val macro F1         : {val_f1:.4f}")
    print(f"  Val signal dist      : {val_dist}")
    print(f"  Val signal Sharpe    : {val_sharpe:+.4f}  (approx, no spread)")

    ensure_dir(model_dir)
    with open(model_path, "wb") as fh:
        pickle.dump(final_lr, fh)
    print(f"  Saved: {model_path}")


# --- LSTM training

def train_lstm(pair: str, session: str, force: bool, batch_size: int = 2048, use_amp: bool = True) -> None:
    if not _TORCH_AVAILABLE or FXMultiScaleLSTM is None:
        raise ImportError(
            "PyTorch is required for LSTM training.\n"
            "Install with: pip install torch"
        )

    model_dir  = _model_dir(session)
    model_path = model_dir / f"{pair}_lstm_model.pt"

    if model_path.exists() and not force:
        print(f"[LSTM] {pair}/{session}: model exists - skip (use --force to overwrite)")
        return

    import torch  # already imported above if available

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    amp_enabled = use_amp and device.type == "cuda"
    print(f"[LSTM] Using device: {device}  AMP: {amp_enabled}")

    train_df, val_df = load_data(pair, session)
    scaler, scaler_cols = load_scaler(pair)

    # Determine long-branch feature set (optionally include same_minute_prev_day_logrange)
    long_feat_cols = list(LSTM_LONG_FEATURES)
    optional_col = "same_minute_prev_day_logrange"
    if (
        optional_col in train_df.columns
        and train_df[optional_col].notna().mean() > 0.5
    ):
        long_feat_cols.append(optional_col)
        print(f"  Using {optional_col} in long branch ({len(long_feat_cols)} features)")

    label_col = "label"
    train_df = train_df.dropna(subset=[label_col]).reset_index(drop=True)
    val_df   = val_df.dropna(subset=[label_col]).reset_index(drop=True)

    # Remap labels: -1->0, 0->1, 1->2
    y_train = (train_df[label_col].astype(int) + 1).values
    y_val   = (val_df[label_col].astype(int) + 1).values

    col_idx = {c: i for i, c in enumerate(scaler_cols)}

    def _scale_cols(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
        """Scale columns using global scaler; fall back to raw values if unknown."""
        out_arr = np.zeros((len(df), len(cols)), dtype=np.float32)
        X_full = df[scaler_cols].fillna(0).values if scaler_cols else None
        X_scaled_full = scaler.transform(X_full) if X_full is not None else None
        for j, c in enumerate(cols):
            if c in col_idx and X_scaled_full is not None:
                out_arr[:, j] = X_scaled_full[:, col_idx[c]]
            elif c in df.columns:
                out_arr[:, j] = df[c].fillna(0).values.astype(np.float32)
        return out_arr

    def _get_raw_cols(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
        out_arr = np.zeros((len(df), len(cols)), dtype=np.float32)
        for j, c in enumerate(cols):
            if c in df.columns:
                out_arr[:, j] = df[c].fillna(0).values.astype(np.float32)
        return out_arr

    X_short_train = _scale_cols(train_df, LSTM_SHORT_FEATURES)
    X_long_train  = _scale_cols(train_df, long_feat_cols)
    X_sess_train  = _get_raw_cols(train_df, LSTM_SESSION_FEATURES)

    X_short_val = _scale_cols(val_df, LSTM_SHORT_FEATURES)
    X_long_val  = _scale_cols(val_df, long_feat_cols)
    X_sess_val  = _get_raw_cols(val_df, LSTM_SESSION_FEATURES)

    # Class weights to handle label imbalance (FLAT dominates)
    class_counts  = np.bincount(y_train, minlength=3).astype(float)
    class_weights = len(y_train) / (3.0 * class_counts + 1e-8)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    train_dataset = FXSequenceDataset(
        X_short_train, X_long_train, X_sess_train, y_train,
        short_seq=LSTM_SHORT_SEQ, long_seq=LSTM_LONG_SEQ,
        max_bar_idx=len(train_df) - 1,
    )
    val_dataset = FXSequenceDataset(
        X_short_val, X_long_val, X_sess_val, y_val,
        short_seq=LSTM_SHORT_SEQ, long_seq=LSTM_LONG_SEQ,
        max_bar_idx=len(val_df) - 1,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda"), persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda"), persistent_workers=True,
    )

    model = FXMultiScaleLSTM(
        short_input_size=len(LSTM_SHORT_FEATURES),
        long_input_size=len(long_feat_cols),
        session_size=len(LSTM_SESSION_FEATURES),
    ).to(device)
    try:
        model = torch.compile(model)
        print("[LSTM] torch.compile enabled")
    except Exception:
        pass

    criterion = torch.nn.NLLLoss(weight=weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # torch.amp.GradScaler was introduced in PyTorch 2.3; fall back to
    # torch.cuda.amp.GradScaler for 2.2.x (still available in 2.3+ as well).
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_val_loss = float("inf")
    best_state: dict | None = None
    patience   = 10
    no_improve = 0

    print(
        f"[LSTM] {pair}/{session}: training on {len(train_dataset):,} sequences, "
        f"validating on {len(val_dataset):,}"
    )

    for epoch in range(1, 51):
        model.train()
        train_loss = 0.0
        for short_b, long_b, sess_b, labels_b in train_loader:
            short_b, long_b, sess_b, labels_b = (
                short_b.to(device), long_b.to(device), sess_b.to(device), labels_b.to(device)
            )
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(short_b, long_b, sess_b)
                loss   = criterion(logits, labels_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(labels_b)
        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for short_b, long_b, sess_b, labels_b in val_loader:
                short_b, long_b, sess_b, labels_b = (
                    short_b.to(device), long_b.to(device), sess_b.to(device), labels_b.to(device)
                )
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    logits    = model(short_b, long_b, sess_b)
                    val_loss += criterion(logits, labels_b).item() * len(labels_b)
        val_loss /= len(val_dataset)

        print(
            f"  epoch {epoch:>3}: train_loss={train_loss:.6f}  val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}  (patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.cpu()  # save as CPU checkpoint so it loads on any machine

    ensure_dir(model_dir)
    torch.save(
        {
            "model_state_dict":  model.state_dict(),
            "long_feature_cols": long_feat_cols,
            "short_input_size":  len(LSTM_SHORT_FEATURES),
            "long_input_size":   len(long_feat_cols),
            "session_size":      len(LSTM_SESSION_FEATURES),
            "pair":              pair,
            "session":           session,
            "best_val_loss":     best_val_loss,
        },
        model_path,
    )
    print(f"[LSTM] Saved: {model_path}  (best_val_loss={best_val_loss:.6f})")


# --- Entry point

def main() -> None:
    args = parse_args()

    if args.code is not None:
        args.pair, args.model_type, args.session = parse_model_code(args.code)
    else:
        if args.pair is None or args.model_type is None:
            import sys as _sys
            print(
                "error: provide a shortcode (e.g. eurusd-lr-gl) "
                "or both --pair and --model-type.",
                file=_sys.stderr,
            )
            _sys.exit(2)

    if args.model_type == "lr":
        train_lr(
            pair=args.pair,
            session=args.session,
            c_sweep=args.c_sweep,
            force=args.force,
        )
    else:
        train_lstm(
            pair=args.pair,
            session=args.session,
            force=args.force,
            batch_size=args.batch_size,
            use_amp=not args.no_amp,
        )


if __name__ == "__main__":
    main()
