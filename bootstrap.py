#!/usr/bin/env python3
"""One-step setup for forex-algo-trading.

Common invocations:

    python bootstrap.py --doctor          Diagnostic report only; no installs.
    python bootstrap.py --minimal         Env only; skip pipeline and training.
    python bootstrap.py --yes             Full unattended setup end to end.
    python bootstrap.py --resume          Continue from the last failed stage.

Flag groups:

  Setup variant
    --yes, -y           Accept all interactive prompts.
    --minimal           Skip pipeline and training (env only).
    --with-pdf          Also install playwright + chromium for PDF export.

  Environment
    --cpu               Force CPU torch wheel.
    --gpu               Force CUDA (cu121) torch wheel.
    --rebuild-venv      Delete and recreate venv even if present.
    --offline           Refuse network calls; fail early if any are needed.
    --no-tests          Skip pytest verification.

  Resume / re-run
    --resume            Continue from the last unfinished stage.
    --force-stage NAME  Re-run a specific stage by name. Repeatable. Stage
                        names: download clean features labels split train.

  Diagnostics
    --doctor            Print Python / pip / OS / disk / CUDA / network and exit.
    --log PATH          Write bootstrap log to PATH (default ./bootstrap.log).

After bootstrap completes, activate the environment:

    source venv/bin/activate              macOS / Linux
    venv\\Scripts\\activate                 Windows PowerShell

And run the master evaluation:

    python scripts/master_eval.py --eval-year 2024 --spreads 1.0
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import socket
import subprocess
import sys
import textwrap
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
VENV_DIR = PROJECT_DIR / "venv"
ENV_FILE = PROJECT_DIR / ".env"
ENV_EXAMPLE = PROJECT_DIR / ".env.example"
STATE_FILE = PROJECT_DIR / ".bootstrap_state.json"
LOG_FILE_DEFAULT = PROJECT_DIR / "bootstrap.log"

REQUIREMENTS_CORE = PROJECT_DIR / "requirements-core.txt"
REQUIREMENTS_EXTRAS = PROJECT_DIR / "requirements-extras.txt"
REQUIREMENTS_LOCK = PROJECT_DIR / "requirements.lock.txt"

PY_MIN = (3, 10)
PY_MAX = (3, 13)
MIN_DISK_GB = 60

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
SESSIONS_NON_GLOBAL = ["london", "ny", "asia"]

CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu121"

PIPELINE_STAGES = [
    ("download", "scripts/download_fx_data.py",
        "Stage 1: raw download. Skipped automatically if data/parquet/*.parquet ship in-tree."),
    ("clean",    "scripts/clean_fx_data.py",
        "Stage 2: clean (5-10 min)."),
    ("features", "scripts/features_fx_data.py",
        "Stage 3: features (20-60 min, the largest stage)."),
    ("labels",   "scripts/labels_fx_data.py",
        "Stage 4: labels (5-10 min)."),
    ("split",    "scripts/split_fx_data.py",
        "Stage 5: train/val/test splits, folds, scalers (10-15 min)."),
]
PIPELINE_NAMES = [name for name, _, _ in PIPELINE_STAGES]
ALL_STAGE_NAMES = PIPELINE_NAMES + ["train"]

_TTY = sys.stdout.isatty()
GREEN  = "\033[92m" if _TTY else ""
RED    = "\033[91m" if _TTY else ""
YELLOW = "\033[93m" if _TTY else ""
CYAN   = "\033[96m" if _TTY else ""
BOLD   = "\033[1m"  if _TTY else ""
DIM    = "\033[2m"  if _TTY else ""
RESET  = "\033[0m"  if _TTY else ""

_LOG_FH = None


# ----------------------------------------------------------------------
# Logging and console output
# ----------------------------------------------------------------------

def _log_open(path: Path) -> None:
    global _LOG_FH
    _LOG_FH = path.open("a", encoding="utf-8")
    _LOG_FH.write(f"\n--- bootstrap start {datetime.now(timezone.utc).isoformat()} ---\n")
    _LOG_FH.flush()


def _log(msg: str) -> None:
    if _LOG_FH is not None:
        _LOG_FH.write(msg + "\n")
        _LOG_FH.flush()


def _emit(prefix: str, msg: str, color: str = "") -> None:
    line = f"  {prefix:<6}{msg}"
    print(f"{color}{line}{RESET}" if color else line)
    _log(line)


def _ok(msg: str) -> None:   _emit("ok",   msg, GREEN)
def _info(msg: str) -> None: _emit("..",   msg)
def _warn(msg: str) -> None: _emit("warn", msg, YELLOW)
def _err(msg: str) -> None:  _emit("!!",   msg, RED)
def _skip(msg: str) -> None: _emit("skip", msg, DIM)


def _print_header(title: str) -> None:
    bar = "=" * 78
    print()
    print(f"{BOLD}{bar}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{bar}{RESET}")
    _log(f"\n{bar}\n  {title}\n{bar}")


def _print_step(n: int, total: int, title: str) -> None:
    print()
    print(f"{BOLD}[{n}/{total}] {title}{RESET}")
    print("-" * 78)
    _log(f"\n[{n}/{total}] {title}\n" + "-" * 78)


def _confirm(question: str, default_yes: bool = False, assume_yes: bool = False) -> bool:
    if assume_yes:
        _log(f"AUTO-YES: {question}")
        return True
    suffix = " [Y/n] " if default_yes else " [y/N] "
    while True:
        try:
            answer = input(question + suffix).strip().lower()
        except EOFError:
            return default_yes
        if not answer:
            return default_yes
        if answer in {"y", "yes"}: return True
        if answer in {"n", "no"}:  return False


# ----------------------------------------------------------------------
# Subprocess
# ----------------------------------------------------------------------

def _venv_python() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _run(cmd: list[str], *, cwd: Path | None = None, check: bool = False,
         capture: bool = False) -> tuple[int, str]:
    """Run a subprocess. Stream by default; capture if asked."""
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"    {DIM}$ {cmd_str}{RESET}")
    _log(f"$ {cmd_str}")
    if capture:
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None,
                              text=True, capture_output=True)
        _log(proc.stdout)
        if proc.stderr:
            _log(f"[stderr]\n{proc.stderr}")
        if check and proc.returncode != 0:
            _err(f"command failed (exit {proc.returncode}): {cmd_str}")
            sys.exit(proc.returncode)
        return proc.returncode, proc.stdout
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if check and proc.returncode != 0:
        _err(f"command failed (exit {proc.returncode}): {cmd_str}")
        sys.exit(proc.returncode)
    return proc.returncode, ""


# ----------------------------------------------------------------------
# Bootstrap state machine
# ----------------------------------------------------------------------

def _state_load() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            _warn(f"State file {STATE_FILE.name} is corrupted; starting fresh.")
    return {"version": 1, "stages": {}}


def _state_save(state: dict) -> None:
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    if "started_at" not in state:
        state["started_at"] = state["last_updated"]
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _state_set(state: dict, stage: str, status: str) -> None:
    state["stages"][stage] = status
    _state_save(state)


# ----------------------------------------------------------------------
# Preflight checks
# ----------------------------------------------------------------------

def _check_python() -> tuple[bool, str]:
    major, minor = sys.version_info[:2]
    label = f"Python {major}.{minor}.{sys.version_info.micro}"
    if (major, minor) < PY_MIN:
        return False, (f"{label} is too old. Need {PY_MIN[0]}.{PY_MIN[1]} or newer; "
                       f"install from python.org or pyenv and re-run.")
    if (major, minor) > PY_MAX:
        return False, (f"{label} is newer than tested. Need <= "
                       f"{PY_MAX[0]}.{PY_MAX[1]}. Some wheels may not exist yet; "
                       f"install Python {PY_MAX[0]}.{PY_MAX[1]} or earlier.")
    return True, f"{label} on {platform.system()} {platform.release()}"


def _check_pip() -> tuple[bool, str]:
    """Verify pip is reachable; try ensurepip if not."""
    code, _ = _run([sys.executable, "-m", "pip", "--version"], capture=True)
    if code == 0:
        return True, "pip is available"
    _warn("pip not found; attempting ensurepip --upgrade")
    code, _ = _run([sys.executable, "-m", "ensurepip", "--upgrade"], capture=True)
    if code != 0:
        return False, ("pip is unavailable and ensurepip failed. Reinstall Python "
                       "with the 'pip' option enabled, or run "
                       "`python -m ensurepip --upgrade` manually.")
    return True, "pip installed via ensurepip"


def _check_git() -> tuple[bool, str]:
    if not shutil.which("git"):
        return False, "git not on PATH (needed only for cloning; ignore on local checkouts)"
    code, out = _run(["git", "--version"], capture=True)
    return code == 0, out.strip() or "git unknown"


def _check_disk(path: Path = PROJECT_DIR) -> tuple[bool, str]:
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024 ** 3)
    msg = f"{free_gb:.1f} GB free on the project partition"
    if free_gb < MIN_DISK_GB:
        return False, msg + f" (recommend >= {MIN_DISK_GB} GB before running the pipeline)"
    return True, msg


def _check_network(host: str = "www.histdata.com", timeout: float = 3.0) -> tuple[bool, str]:
    try:
        socket.gethostbyname(host)
    except OSError as exc:
        return False, f"DNS lookup failed for {host}: {exc}"
    try:
        with urllib.request.urlopen(f"https://{host}", timeout=timeout) as resp:
            return True, f"{host} reachable (HTTP {resp.status})"
    except Exception as exc:
        return False, f"{host} not reachable: {exc.__class__.__name__}"


def _detect_cuda() -> tuple[str, str]:
    """Return (variant, detail). variant is one of 'cu121', 'cpu', 'mps'."""
    if platform.system() == "Darwin":
        return "cpu", "macOS detected; using CPU torch (MPS works at runtime)"
    if shutil.which("nvidia-smi"):
        code, out = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture=True)
        if code == 0 and out.strip():
            return "cu121", f"GPU(s) detected: {out.strip().splitlines()[0]}"
    return "cpu", "no NVIDIA GPU detected; using CPU torch"


def _stage_already_done(stage: str) -> bool:
    """Idempotency: return True if the stage's outputs are already on disk."""
    if stage == "download":
        files = list((PROJECT_DIR / "data" / "parquet").glob("*.parquet"))
        return len(files) >= len(PAIRS)
    if stage == "clean":
        d = PROJECT_DIR / "data" / "processed" / "cleaned"
        return d.exists() and len(list(d.glob("*_clean.parquet"))) >= len(PAIRS)
    if stage == "features":
        d = PROJECT_DIR / "features" / "pair"
        return d.exists() and len(list(d.glob("*.parquet"))) >= len(PAIRS)
    if stage == "labels":
        d = PROJECT_DIR / "labels" / "pair"
        return d.exists() and len(list(d.glob("*.parquet"))) >= len(PAIRS)
    if stage == "split":
        test_d = PROJECT_DIR / "datasets" / "test"
        scal_d = PROJECT_DIR / "scalers"
        return (test_d.exists()
                and len(list(test_d.glob("*.parquet"))) >= len(PAIRS)
                and len(list(scal_d.glob("*_scaler.pkl"))) >= len(PAIRS))
    if stage == "train":
        for pair in PAIRS:
            for path in (PROJECT_DIR / "models" / "global" / f"{pair}_logreg_model.pkl",
                         PROJECT_DIR / "models" / "global" / f"{pair}_lstm_model.pt"):
                if not path.exists():
                    return False
            for sess in SESSIONS_NON_GLOBAL:
                for path in (PROJECT_DIR / "models" / "session" / sess / f"{pair}_logreg_model.pkl",
                             PROJECT_DIR / "models" / "session" / sess / f"{pair}_lstm_model.pt"):
                    if not path.exists():
                        return False
        return True
    return False


# ----------------------------------------------------------------------
# Steps
# ----------------------------------------------------------------------

def step_doctor() -> int:
    _print_header("Diagnostic report")
    rows = []
    ok_py, msg_py = _check_python();         rows.append(("Python", ok_py, msg_py))
    ok_pip, msg_pip = _check_pip();          rows.append(("pip",    ok_pip, msg_pip))
    ok_git, msg_git = _check_git();          rows.append(("git",    ok_git, msg_git))
    ok_disk, msg_disk = _check_disk();       rows.append(("disk",   ok_disk, msg_disk))
    ok_net, msg_net = _check_network();      rows.append(("net",    ok_net, msg_net))
    cuda_variant, msg_cuda = _detect_cuda(); rows.append(("torch",  True, f"{cuda_variant}: {msg_cuda}"))
    print()
    for label, ok, msg in rows:
        glyph = f"{GREEN}OK  {RESET}" if ok else f"{YELLOW}WARN{RESET}"
        print(f"  {glyph} {label:<8} {msg}")
    print()
    print(f"  Project root: {PROJECT_DIR}")
    print(f"  venv path:    {VENV_DIR}  (exists: {VENV_DIR.exists()})")
    for stage in ALL_STAGE_NAMES:
        done = _stage_already_done(stage)
        glyph = f"{GREEN}done   {RESET}" if done else f"{DIM}pending{RESET}"
        print(f"  stage {stage:<10} {glyph}")
    print()
    return 0


def step_preflight(args: argparse.Namespace) -> None:
    _print_step(1, 10, "Preflight checks")
    fatal = False
    ok, msg = _check_python();   (_ok if ok else _err)(f"Python: {msg}");   fatal |= not ok
    ok, msg = _check_pip();      (_ok if ok else _err)(f"pip:    {msg}");   fatal |= not ok
    ok, msg = _check_disk();     (_ok if ok else _warn)(f"disk:   {msg}")
    ok, msg = _check_git();      (_ok if ok else _warn)(f"git:    {msg}")
    if not args.offline:
        ok, msg = _check_network(); (_ok if ok else _warn)(f"net:    {msg}")
    if fatal:
        _err("Preflight has a blocking error. Resolve and re-run.")
        sys.exit(1)


def step_venv(args: argparse.Namespace) -> None:
    _print_step(2, 10, "Virtual environment")
    if VENV_DIR.exists() and not args.rebuild_venv:
        _ok(f"Reusing existing venv at {VENV_DIR}")
    else:
        if VENV_DIR.exists():
            _info(f"Removing existing venv at {VENV_DIR}")
            shutil.rmtree(VENV_DIR)
        _info(f"Creating venv at {VENV_DIR}")
        _run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
        if not _venv_python().exists():
            _err(f"venv interpreter missing at {_venv_python()}")
            sys.exit(1)
        _ok(f"venv ready: {VENV_DIR}")
    _info("Upgrading pip, setuptools, wheel")
    _run([str(_venv_python()), "-m", "pip", "install", "--upgrade",
          "pip", "setuptools", "wheel"], check=True)
    _ok("pip toolchain upgraded")


def step_install_torch(args: argparse.Namespace) -> str:
    _print_step(3, 10, "Install torch (wheel selection)")
    if args.offline:
        _warn("--offline set; skipping torch install. Existing torch in venv will be used.")
        return "offline"
    if args.cpu:
        variant, detail = "cpu", "forced via --cpu"
    elif args.gpu:
        variant, detail = "cu121", "forced via --gpu"
    else:
        variant, detail = _detect_cuda()
    _info(f"torch variant: {variant} ({detail})")
    cmd = [str(_venv_python()), "-m", "pip", "install", "--upgrade", "torch"]
    if variant == "cu121":
        cmd += ["--index-url", CUDA_INDEX_URL]
    _run(cmd, check=True)
    _ok(f"torch installed ({variant})")
    return variant


def step_install_deps(args: argparse.Namespace) -> None:
    _print_step(4, 10, "Install core + extras")
    if not REQUIREMENTS_CORE.exists():
        _err(f"{REQUIREMENTS_CORE.name} missing"); sys.exit(1)
    _run([str(_venv_python()), "-m", "pip", "install", "-r", str(REQUIREMENTS_CORE)],
         check=True)
    _ok(f"installed {REQUIREMENTS_CORE.name}")
    if REQUIREMENTS_EXTRAS.exists():
        _run([str(_venv_python()), "-m", "pip", "install", "-r", str(REQUIREMENTS_EXTRAS)],
             check=True)
        _ok(f"installed {REQUIREMENTS_EXTRAS.name}")
    if args.with_pdf:
        _info("Installing playwright + chromium for PDF export")
        _run([str(_venv_python()), "-m", "pip", "install", "playwright~=1.45"], check=True)
        _run([str(_venv_python()), "-m", "playwright", "install", "chromium"], check=True)
        _ok("playwright + chromium ready")
    _info("Writing requirements.lock.txt (pip freeze)")
    code, out = _run([str(_venv_python()), "-m", "pip", "freeze"], capture=True)
    if code == 0 and out:
        REQUIREMENTS_LOCK.write_text(out, encoding="utf-8")
        _ok(f"lockfile written: {REQUIREMENTS_LOCK.name}")


def step_env(args: argparse.Namespace) -> str:
    _print_step(5, 10, "Configuration (.env)")
    if ENV_FILE.exists():
        _ok(f"{ENV_FILE.name} already present (not touched)")
        return "kept"
    if not ENV_EXAMPLE.exists():
        _warn(f"{ENV_EXAMPLE.name} missing; cannot bootstrap .env")
        return "missing-example"
    shutil.copy(ENV_EXAMPLE, ENV_FILE)
    _ok(f"created {ENV_FILE.name} from {ENV_EXAMPLE.name}")
    return "created"


def step_tests(args: argparse.Namespace) -> bool:
    _print_step(6, 10, "Verify with pytest")
    if args.no_tests:
        _skip("--no-tests set; skipping verification")
        return True
    code, _ = _run([str(_venv_python()), "-m", "pytest", "tests/", "-q"],
                   cwd=PROJECT_DIR)
    if code != 0:
        _err("pytest reported failures. Environment installed; verification did not pass.")
        return False
    _ok("pytest green")
    return True


def step_pipeline(args: argparse.Namespace, state: dict) -> None:
    _print_step(7, 10, "Data pipeline (stages 1-5)")
    if args.minimal:
        _skip("--minimal set; skipping pipeline")
        return
    if not _confirm("Run the data pipeline now? (90 min - 3 h)", assume_yes=args.yes):
        _skip("user declined")
        return
    for name, script, desc in PIPELINE_STAGES:
        if name in args.force_stage:
            _info(f"--force-stage {name}: re-running even if outputs exist")
        elif args.resume and state["stages"].get(name) == "ok":
            _skip(f"{name}: marked ok in resume state"); continue
        elif _stage_already_done(name):
            _skip(f"{name}: outputs already on disk")
            _state_set(state, name, "ok"); continue
        _info(f"Running {name}: {desc}")
        _state_set(state, name, "running")
        code, _ = _run([str(_venv_python()), script], cwd=PROJECT_DIR)
        if code != 0:
            _state_set(state, name, "failed")
            _err(f"Stage {name} failed (exit {code}).")
            _err(f"Re-run with: python bootstrap.py --resume   (or fix and re-run: python {script})")
            sys.exit(code)
        _state_set(state, name, "ok")
        _ok(f"{name} complete")


def step_train(args: argparse.Namespace, state: dict) -> None:
    _print_step(8, 10, "Train model grid")
    if args.minimal:
        _skip("--minimal set; skipping training")
        return
    if "train" in args.force_stage:
        _info("--force-stage train: re-running full grid")
    elif args.resume and state["stages"].get("train") == "ok":
        _skip("train: marked ok in resume state"); return
    elif _stage_already_done("train"):
        _skip("train: all 56 checkpoints present")
        _state_set(state, "train", "ok"); return
    print()
    print(textwrap.dedent("""\
          Training fits 28 LR + 28 LSTM cells across 7 pairs and 4 sessions.
          LR cells under 5 min each; LSTM cells 40-90 min each. Total runtime
          is 8-14 hours unattended on CPU; faster on GPU.
        """))
    if not _confirm("Train the full grid now?", assume_yes=args.yes):
        _skip("user declined"); return
    _state_set(state, "train", "running")
    code, _ = _run([str(_venv_python()), "scripts/train_all.py", "--model-type", "all"],
                   cwd=PROJECT_DIR)
    if code != 0:
        _state_set(state, "train", "failed")
        _err(f"Training failed (exit {code}). Re-run with: python bootstrap.py --resume")
        sys.exit(code)
    _state_set(state, "train", "ok")
    _ok("model grid complete")


def step_summary(results: dict) -> None:
    _print_header("Bootstrap summary")
    print()
    print(f"  {'Step':<28} {'Result':<14} {'Time':>8}")
    print(f"  {'-' * 28} {'-' * 14} {'-' * 8}")
    for name, (result, secs) in results.items():
        color = GREEN if result == "ok" else (DIM if result == "skip" else RED)
        t = f"{secs:>7.1f}s" if secs is not None else "       -"
        print(f"  {name:<28} {color}{result:<14}{RESET} {t}")
    print()
    activate = ("venv\\Scripts\\activate" if platform.system() == "Windows"
                else "source venv/bin/activate")
    print("  Activate the venv in your shell:")
    print(f"      {activate}")
    print()
    print("  Run the master evaluation on 2024:")
    print("      python scripts/master_eval.py --eval-year 2024 --spreads 1.0")
    print()
    print("  Single backtest sanity check:")
    print("      python backtest/run_backtest.py --pair EURUSD \\")
    print("          --strategy RSI_p14_os30_ob70 --split full --capital 10000 --no-browser")
    print()


# ----------------------------------------------------------------------
# Argument parsing and main
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-step bootstrap for forex-algo-trading.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Common invocations:
              python bootstrap.py --doctor      diagnostic report only
              python bootstrap.py --minimal     env only, no pipeline or training
              python bootstrap.py --yes         unattended full setup
              python bootstrap.py --resume      continue from last failed stage
        """),
    )
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Accept all prompts.")
    parser.add_argument("--minimal", action="store_true",
                        help="Skip the data pipeline and the training grid.")
    parser.add_argument("--with-pdf", action="store_true",
                        help="Also install playwright + chromium for PDF export.")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU torch wheel.")
    parser.add_argument("--gpu", action="store_true",
                        help="Force CUDA (cu121) torch wheel.")
    parser.add_argument("--rebuild-venv", action="store_true",
                        help="Delete and recreate the venv.")
    parser.add_argument("--offline", action="store_true",
                        help="Skip network operations.")
    parser.add_argument("--no-tests", action="store_true",
                        help="Skip the pytest verification step.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from the last unfinished stage.")
    parser.add_argument("--force-stage", action="append", default=[],
                        metavar="NAME", choices=ALL_STAGE_NAMES,
                        help=f"Re-run one stage by name. Repeatable. "
                             f"Names: {', '.join(ALL_STAGE_NAMES)}.")
    parser.add_argument("--doctor", action="store_true",
                        help="Print diagnostic report and exit.")
    parser.add_argument("--log", default=str(LOG_FILE_DEFAULT),
                        metavar="PATH", help="Log file path.")
    args = parser.parse_args()
    if args.cpu and args.gpu:
        parser.error("--cpu and --gpu are mutually exclusive")
    return args


def main() -> None:
    args = parse_args()
    _log_open(Path(args.log))

    if args.doctor:
        sys.exit(step_doctor())

    _print_header("forex-algo-trading bootstrap")
    print()
    print(f"  Project root : {PROJECT_DIR}")
    print(f"  venv path    : {VENV_DIR}")
    print(f"  Log file     : {args.log}")
    print(f"  Platform     : {platform.system()} {platform.release()}")
    if args.resume:
        print(f"  Resume mode  : reading {STATE_FILE.name}")
    print()

    state = _state_load()
    results: dict[str, tuple[str, float | None]] = {}

    def time_step(name: str, fn, *fn_args) -> None:
        t0 = time.time()
        try:
            fn(*fn_args)
            results[name] = ("ok", time.time() - t0)
            _state_set(state, name, "ok")
        except SystemExit:
            results[name] = ("FAIL", time.time() - t0)
            _state_save(state)
            raise

    time_step("preflight",   step_preflight, args)
    time_step("venv",        step_venv, args)
    time_step("torch wheel", step_install_torch, args)
    time_step("deps",        step_install_deps, args)
    time_step("env",         step_env, args)
    t0 = time.time()
    tests_passed = step_tests(args)
    results["tests"] = ("ok" if tests_passed else "FAIL", time.time() - t0)
    _state_set(state, "tests", "ok" if tests_passed else "failed")
    t0 = time.time()
    step_pipeline(args, state)
    results["pipeline"] = ("skip" if args.minimal else "ok", time.time() - t0)
    t0 = time.time()
    step_train(args, state)
    results["training"] = ("skip" if args.minimal else "ok", time.time() - t0)

    step_summary(results)


if __name__ == "__main__":
    main()
