"""
train_all.py - Automated training runner for all LR and/or LSTM models.

Trains every pair x session combination, catches failures, and prints a
pass/fail summary table at the end.

Usage:
    python scripts/train_all.py                         # LR only, all pairs
    python scripts/train_all.py --model-type lstm       # LSTM only, all pairs
    python scripts/train_all.py --model-type all        # LR + LSTM, all pairs
    python scripts/train_all.py --pairs eurusd gbpusd   # subset of pairs
    python scripts/train_all.py --force                 # overwrite existing models
    python scripts/train_all.py --no-c-sweep            # skip LR regularisation sweep
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config.constants import MODELS_DIR, PAIRS, SESSION_NAMES

_TTY   = sys.stdout.isatty()
GREEN  = "\033[92m" if _TTY else ""
RED    = "\033[91m" if _TTY else ""
YELLOW = "\033[93m" if _TTY else ""
CYAN   = "\033[96m" if _TTY else ""
BOLD   = "\033[1m"  if _TTY else ""
RESET  = "\033[0m"  if _TTY else ""

SESSIONS = list(SESSION_NAMES)  # global, london, ny, asia

SESSION_ALIASES = {"global": "gl", "london": "ldn", "ny": "ny", "asia": "as"}


def _model_path(pair: str, model_type: str, session: str) -> Path:
    subdir = MODELS_DIR / "global" if session == "global" else MODELS_DIR / "session" / session
    ext  = "pkl" if model_type == "lr" else "pt"
    name = f"{pair}_logreg_model.{ext}" if model_type == "lr" else f"{pair}_lstm_model.{ext}"
    return subdir / name


def _shortcode(pair: str, model_type: str, session: str) -> str:
    return f"{pair.lower()}-{model_type}-{SESSION_ALIASES[session]}"


def run_one(
    pair: str,
    model_type: str,
    session: str,
    force: bool,
    c_sweep: bool,
    batch_size: int = 2048,
    no_amp: bool = False,
) -> tuple[bool, float, str]:
    """Run train_model.py for one combination. Returns (success, elapsed_s, error_msg)."""
    code = _shortcode(pair, model_type, session)
    cmd  = [sys.executable, str(PROJECT_DIR / "scripts" / "train_model.py"), code]
    if force:
        cmd.append("--force")
    if c_sweep and model_type == "lr" and session == "global":
        cmd.append("--c-sweep")
    if model_type == "lstm":
        if batch_size != 2048:
            cmd += ["--batch-size", str(batch_size)]
        if no_amp:
            cmd.append("--no-amp")

    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        elapsed = time.time() - t0
        if result.returncode != 0:
            return False, elapsed, f"exit code {result.returncode}"
        model_file = _model_path(pair, model_type, session)
        if not model_file.exists():
            return False, elapsed, f"model file missing: {model_file}"
        return True, elapsed, ""
    except Exception as exc:
        return False, time.time() - t0, str(exc)


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train all LR and/or LSTM models for all pairs and sessions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-type", choices=["lr", "lstm", "all"], default="lr",
        help="Which model type(s) to train (default: lr).",
    )
    parser.add_argument(
        "--pairs", nargs="+", default=["all"], metavar="PAIR",
        help="Pairs to train, or 'all' (default). e.g. --pairs eurusd gbpusd",
    )
    parser.add_argument("--force",      action="store_true", help="Overwrite existing models.")
    parser.add_argument("--no-c-sweep", action="store_true", help="Skip LR regularisation sweep.")
    parser.add_argument(
        "--batch-size", type=int, default=2048, dest="batch_size",
        help="(LSTM only) Mini-batch size. Use 512 if GPU runs out of memory (default: 2048).",
    )
    parser.add_argument(
        "--no-amp", action="store_true", dest="no_amp",
        help="(LSTM only) Disable automatic mixed precision.",
    )
    args = parser.parse_args()

    pairs = [p.upper() for p in args.pairs] if "all" not in args.pairs else list(PAIRS)
    model_types = ["lr", "lstm"] if args.model_type == "all" else [args.model_type]
    c_sweep = not args.no_c_sweep

    invalid = [p for p in pairs if p not in PAIRS]
    if invalid:
        print(f"Unknown pairs: {invalid}. Valid: {list(PAIRS)}", file=sys.stderr)
        sys.exit(1)

    # Build full job list
    jobs: list[tuple[str, str, str]] = []
    for model_type in model_types:
        for pair in pairs:
            for session in SESSIONS:
                jobs.append((pair, model_type, session))

    total  = len(jobs)
    passed = 0
    failed = 0
    results: list[dict] = []

    print(f"\n{BOLD}Training {total} model(s): {', '.join(model_types)} x {len(pairs)} pair(s) x {len(SESSIONS)} sessions{RESET}\n")

    for i, (pair, model_type, session) in enumerate(jobs, 1):
        code = _shortcode(pair, model_type, session)
        already = _model_path(pair, model_type, session).exists()
        if already and not args.force:
            print(f"  [{i:>2}/{total}] {YELLOW}SKIP{RESET}  {code:<20}  (exists - use --force to retrain)")
            results.append({"code": code, "status": "skip", "elapsed": 0.0, "error": ""})
            passed += 1
            continue

        print(f"  [{i:>2}/{total}] {CYAN}RUN {RESET}  {code:<20} ...", end="", flush=True)
        ok, elapsed, err = run_one(pair, model_type, session, args.force, c_sweep, args.batch_size, args.no_amp)

        if ok:
            passed += 1
            print(f"\r  [{i:>2}/{total}] {GREEN}PASS{RESET}  {code:<20}  {_fmt_time(elapsed)}")
            results.append({"code": code, "status": "pass", "elapsed": elapsed, "error": ""})
        else:
            failed += 1
            print(f"\r  [{i:>2}/{total}] {RED}FAIL{RESET}  {code:<20}  {_fmt_time(elapsed)}  {err}")
            results.append({"code": code, "status": "fail", "elapsed": elapsed, "error": err})

    # Summary
    print(f"\n{'-'*55}")
    print(f"{BOLD}Summary{RESET}")
    print(f"{'-'*55}")
    total_time = sum(r["elapsed"] for r in results)
    print(f"  Total:   {total}")
    print(f"  {GREEN}Passed:  {passed}{RESET}")
    if failed:
        print(f"  {RED}Failed:  {failed}{RESET}")
        print(f"\n{RED}Failed models:{RESET}")
        for r in results:
            if r["status"] == "fail":
                print(f"  {r['code']:<22}  {r['error']}")
    else:
        print(f"  {GREEN}All models trained successfully.{RESET}")
    print(f"\n  Total training time: {_fmt_time(total_time)}")
    print(f"{'-'*55}\n")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
