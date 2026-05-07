#!/usr/bin/env python3
"""One-step setup for forex-algo-trading.

Run this from the repository root after cloning:

    python bootstrap.py

The script handles every step needed to get from a fresh clone to a working
research environment:

    1. Verifies the Python version is 3.11 or 3.12 (3.13 is rejected because
       the pinned torch==2.6.0 wheels are not yet published for 3.13).
    2. Creates a virtual environment at ./venv if one is not already present.
    3. Upgrades pip, setuptools, and wheel inside the venv.
    4. Installs every direct dependency from requirements.txt.
    5. Verifies the install by running the pytest suite.
    6. Optionally runs the data pipeline stages 1 to 5 (downloads, cleans,
       computes features, builds labels, writes splits and scalers).
    7. Optionally trains every model cell in the LR x LSTM grid.

Steps 6 and 7 prompt before running because they take hours. Use --yes to
accept all prompts and run the full setup unattended. Use --no-pipeline or
--no-train to skip the long-running steps explicitly.

Examples:

    # Interactive setup (recommended on first run):
    python bootstrap.py

    # Unattended full setup, accept all prompts:
    python bootstrap.py --yes

    # Set up the environment only, skip the pipeline and training:
    python bootstrap.py --no-pipeline --no-train

    # Pipeline only, skip training:
    python bootstrap.py --no-train

After bootstrap completes, activate the environment in your shell with:

    source venv/bin/activate          # macOS / Linux
    venv\\Scripts\\activate            # Windows

And run the master evaluation:

    python scripts/master_eval.py --eval-year 2024 --spreads 1.0
"""
from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
VENV_DIR = PROJECT_DIR / "venv"
REQUIREMENTS = PROJECT_DIR / "requirements.txt"

MIN_PY_MAJOR = 3
MIN_PY_MINOR = 11
# Upper bound is set by the pinned torch wheel availability. torch==2.6.0 has
# wheels for cp311 and cp312; cp313 wheels are not yet published. Bumping
# this to 13 (or removing the cap) requires either upgrading the torch pin or
# accepting that users on 3.13 will need to build torch from source.
MAX_PY_MINOR = 12

PIPELINE_STAGES = [
    ("download", "scripts/download_fx_data.py", "10 to 20 min, network-bound"),
    ("clean",    "scripts/clean_fx_data.py",    "5 to 10 min"),
    ("features", "scripts/features_fx_data.py", "45 to 90 min, CPU-bound"),
    ("labels",   "scripts/labels_fx_data.py",   "5 to 10 min"),
    ("split",    "scripts/split_fx_data.py",    "10 to 15 min"),
]


# ----- output helpers


def _print_header(title: str) -> None:
    line = "=" * 78
    print()
    print(line)
    print(f"  {title}")
    print(line)


def _print_step(n: int, total: int, title: str) -> None:
    print()
    print(f"[{n}/{total}] {title}")
    print("-" * 78)


def _ok(msg: str) -> None:
    print(f"  ok  {msg}")


def _info(msg: str) -> None:
    print(f"  ..  {msg}")


def _err(msg: str) -> None:
    print(f"  !!  {msg}")


def _confirm(question: str, default_yes: bool = False, assume_yes: bool = False) -> bool:
    if assume_yes:
        return True
    suffix = " [Y/n] " if default_yes else " [y/N] "
    while True:
        try:
            answer = input(question + suffix).strip().lower()
        except EOFError:
            return default_yes
        if not answer:
            return default_yes
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False


# ----- venv helpers


def _venv_python() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _venv_pip() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "pip.exe"
    return VENV_DIR / "bin" / "pip"


def _run(cmd: list[str | Path], **kwargs) -> int:
    """Run a subprocess, stream output, return the exit code."""
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"    $ {cmd_str}")
    result = subprocess.run(cmd, **kwargs)
    return result.returncode


# ----- step implementations


def step_check_python() -> None:
    _print_step(1, 7, "Verify Python version")
    major, minor = sys.version_info[:2]
    _info(f"Detected Python {major}.{minor}.{sys.version_info.micro} on {platform.system()}")
    if (major, minor) < (MIN_PY_MAJOR, MIN_PY_MINOR):
        _err(
            f"Python {MIN_PY_MAJOR}.{MIN_PY_MINOR}+ is required. "
            f"Found {major}.{minor}. "
            f"Install a newer Python from https://www.python.org/downloads/ and re-run."
        )
        sys.exit(1)
    if major == MIN_PY_MAJOR and minor > MAX_PY_MINOR:
        _err(
            f"Python {MIN_PY_MAJOR}.{minor} is too new for the pinned dependency set.\n"
            f"  The current pin requires Python {MIN_PY_MAJOR}.{MIN_PY_MINOR} or "
            f"{MIN_PY_MAJOR}.{MAX_PY_MINOR}, because torch==2.6.0 does not yet ship "
            f"wheels for cp{MIN_PY_MAJOR}{minor}.\n"
            f"  Install Python {MIN_PY_MAJOR}.{MAX_PY_MINOR} from "
            f"https://www.python.org/downloads/ and re-run, or set up a fresh venv "
            f"with a 3.{MAX_PY_MINOR} interpreter (e.g. via pyenv or conda)."
        )
        sys.exit(1)
    _ok(f"Python {major}.{minor} is in the supported range "
        f"({MIN_PY_MAJOR}.{MIN_PY_MINOR} to {MIN_PY_MAJOR}.{MAX_PY_MINOR})")


def step_create_venv(reuse: bool = True) -> None:
    _print_step(2, 7, "Create virtual environment")
    if VENV_DIR.exists():
        if reuse:
            _ok(f"Reusing existing venv at {VENV_DIR}")
            return
        _info(f"Removing stale venv at {VENV_DIR}")
        shutil.rmtree(VENV_DIR)
    _info(f"Creating venv at {VENV_DIR}")
    rc = _run([sys.executable, "-m", "venv", str(VENV_DIR)])
    if rc != 0:
        _err(f"venv creation failed with exit code {rc}")
        sys.exit(rc)
    if not _venv_python().exists():
        _err(f"venv created but interpreter not found at {_venv_python()}")
        sys.exit(1)
    _ok(f"venv ready: {VENV_DIR}")


def step_upgrade_pip() -> None:
    _print_step(3, 7, "Upgrade pip, setuptools, and wheel")
    rc = _run([str(_venv_python()), "-m", "pip", "install",
               "--upgrade", "pip", "setuptools", "wheel"])
    if rc != 0:
        _err("pip upgrade failed; you can try again or continue manually.")
        sys.exit(rc)
    _ok("pip, setuptools, wheel upgraded")


def step_install_requirements() -> None:
    _print_step(4, 7, "Install runtime dependencies")
    if not REQUIREMENTS.exists():
        _err(f"requirements.txt not found at {REQUIREMENTS}")
        sys.exit(1)
    _info(f"Installing from {REQUIREMENTS.name}")
    rc = _run([str(_venv_python()), "-m", "pip", "install",
               "-r", str(REQUIREMENTS)])
    if rc != 0:
        _err("Dependency install failed.")
        sys.exit(rc)
    _ok("All runtime dependencies installed")


def step_run_tests() -> None:
    _print_step(5, 7, "Verify install with pytest")
    rc = _run([str(_venv_python()), "-m", "pytest", "tests/", "-q"],
              cwd=str(PROJECT_DIR))
    if rc != 0:
        _err("Tests failed. The environment is installed but verification did not pass.")
        _err("Investigate before running the pipeline or training.")
        return
    _ok("All tests pass")


def step_run_pipeline(assume_yes: bool, skip: bool) -> None:
    _print_step(6, 7, "Run data pipeline (stages 1-5)")
    if skip:
        _info("Skipped via --no-pipeline.")
        return
    print()
    print(textwrap.dedent("""\
          The data pipeline downloads ~10 years of minute-level FX data for 7 pairs,
          cleans it, computes features, builds labels, and writes train/val/test
          splits. Total runtime is roughly 90 minutes to 3 hours on a recent laptop.
          The largest stage is feature computation (45 to 90 minutes).

          The pipeline is one-time: subsequent runs read from on-disk Parquets.
        """))
    if not _confirm("  Run the data pipeline now?", assume_yes=assume_yes):
        _info("Skipped. Run individual stages later with the commands in docs/SETUP.md.")
        return
    for i, (name, script, eta) in enumerate(PIPELINE_STAGES, start=1):
        print()
        _info(f"Stage {i}/5: {name} ({eta})")
        rc = _run([str(_venv_python()), script], cwd=str(PROJECT_DIR))
        if rc != 0:
            _err(f"Stage {name} failed with exit code {rc}.")
            _err(f"Fix the issue and re-run: python {script}")
            sys.exit(rc)
        _ok(f"Stage {name} complete")
    _ok("Data pipeline complete")


def step_train_models(assume_yes: bool, skip: bool) -> None:
    _print_step(7, 7, "Train model cells")
    if skip:
        _info("Skipped via --no-train.")
        return
    print()
    print(textwrap.dedent("""\
          Training fits every cell in the LR x LSTM grid: 7 pairs x 4 sessions x
          2 model types = 56 checkpoints. LR cells are fast (under 5 min each).
          LSTM cells are slow (40 to 90 min each). Total grid time is roughly
          8 to 14 hours unattended.

          Training is one-time per cell: re-runs only happen if features or
          labels change.
        """))
    if not _confirm("  Train the full model grid now?", assume_yes=assume_yes):
        _info("Skipped. Train individual cells later with scripts/train_model.py")
        _info("or the full grid with scripts/train_all.py.")
        return
    rc = _run([str(_venv_python()), "scripts/train_all.py"],
              cwd=str(PROJECT_DIR))
    if rc != 0:
        _err(f"Training failed with exit code {rc}.")
        sys.exit(rc)
    _ok("Model grid complete")


def print_next_steps() -> None:
    _print_header("Setup complete")
    venv_activate = (
        "venv\\Scripts\\activate" if platform.system() == "Windows"
        else "source venv/bin/activate"
    )
    print()
    print("  Activate the environment in your shell:")
    print(f"      {venv_activate}")
    print()
    print("  Run the master evaluation on 2024:")
    print("      python scripts/master_eval.py --eval-year 2024 --spreads 1.0")
    print()
    print("  Run a single backtest:")
    print("      python backtest/run_backtest.py --pair EURUSD --strategy RSI_p14_os30_ob70 --split full --no-browser")
    print()
    print("  Documentation:")
    print("      README.md         project overview")
    print("      docs/SETUP.md     full setup and CLI reference")
    print("      ARCHITECTURE.md   system internals and decision records")
    print("      docs/EXPERIMENTS.md   experimental framework and reproducibility")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-step bootstrap for forex-algo-trading.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Without flags the script runs interactively, prompting before the
            long-running pipeline and training steps.

            Use --yes to accept all prompts. Use --no-pipeline or --no-train
            to skip the long-running steps explicitly.
        """),
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Accept all prompts (run pipeline and training without asking).",
    )
    parser.add_argument(
        "--no-pipeline", action="store_true",
        help="Skip the data pipeline (stages 1-5).",
    )
    parser.add_argument(
        "--no-train", action="store_true",
        help="Skip model training.",
    )
    parser.add_argument(
        "--no-tests", action="store_true",
        help="Skip the pytest verification step.",
    )
    parser.add_argument(
        "--rebuild-venv", action="store_true",
        help="Delete and recreate the venv even if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    _print_header("forex-algo-trading bootstrap")
    print()
    print(f"  Project root : {PROJECT_DIR}")
    print(f"  venv path    : {VENV_DIR}")
    print(f"  Requirements : {REQUIREMENTS}")
    print(f"  Platform     : {platform.system()} {platform.release()}")
    print()

    step_check_python()
    step_create_venv(reuse=not args.rebuild_venv)
    step_upgrade_pip()
    step_install_requirements()

    if not args.no_tests:
        step_run_tests()

    step_run_pipeline(assume_yes=args.yes, skip=args.no_pipeline)
    step_train_models(assume_yes=args.yes, skip=args.no_train)

    print_next_steps()


if __name__ == "__main__":
    main()
