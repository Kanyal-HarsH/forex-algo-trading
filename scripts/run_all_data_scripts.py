"""
run_all_data_scripts.py

Orchestrates the full FX data preparation pipeline in order:

  Step 1 — Download  : Fetches raw FX data and saves to data/parquet
  Step 2 — Inspect   : Runs read-only diagnostics on raw data
  Step 3 — Clean     : Removes gaps, bad ticks, duplicates
  Step 4 — EDA       : Exploratory analysis, charts, distribution reports
  Step 5 — Features  : Engineers indicators and feature columns
  Step 6 — Labels    : Generates forward-return labels for ML targets
  Step 7 — Split     : Creates train / val / test dataset splits

After all steps complete, the system is ready for strategy backtesting.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Full pipeline (first-time setup):
    python scripts/run_all_data_scripts.py

  Resume from a specific step (e.g. step 3):
    python scripts/run_all_data_scripts.py --from-step 3

  Skip steps whose outputs already exist:
    python scripts/run_all_data_scripts.py --skip-existing

  Force re-run even if outputs already exist:
    python scripts/run_all_data_scripts.py --force

  Preview what would run without executing anything:
    python scripts/run_all_data_scripts.py --dry-run

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import subprocess
import sys
import os
import time
import argparse
from datetime import datetime
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Pipeline definition ───────────────────────────────────────────────────────

STEPS = [
    {
        "number": 1,
        "name": "Download",
        "description": "Fetching raw FX tick data from source",
        "module": "scripts.download_fx_data",
        "script": "download_fx_data.py",
        "validates": ["data/parquet"],
        "skip_if_exist": ["data/parquet"],
    },
    {
        "number": 2,
        "name": "Inspect",
        "description": "Running diagnostics on raw data (read-only)",
        "module": "scripts.inspect_fx_data",
        "script": "inspect_fx_data.py",
        "validates": [],
        "skip_if_exist": [],
    },
    {
        "number": 3,
        "name": "Clean",
        "description": "Cleaning gaps, duplicates, and bad ticks",
        "module": "scripts.clean_fx_data",
        "script": "clean_fx_data.py",
        "validates": ["data/processed/cleaned", "data/processed/reports"],
        "skip_if_exist": ["data/processed/cleaned"],
    },
    {
        "number": 4,
        "name": "EDA",
        "description": "Running exploratory data analysis",
        "module": "scripts.eda_fx_data",
        "script": "eda_fx_data.py",
        "validates": ["eda/raw_snapshot", "eda/samples", "eda/reports"],
        "skip_if_exist": ["eda/reports"],
    },
    {
        "number": 5,
        "name": "Features",
        "description": "Engineering indicators and feature columns",
        "module": "scripts.features_fx_data",
        "script": "features_fx_data.py",
        "validates": ["features/pair", "features/reports"],
        "skip_if_exist": ["features/pair"],
    },
    {
        "number": 6,
        "name": "Labels",
        "description": "Generating forward-return labels",
        "module": "scripts.labels_fx_data",
        "script": "labels_fx_data.py",
        "validates": ["labels/pair", "labels/reports"],
        "skip_if_exist": ["labels/pair"],
    },
    {
        "number": 7,
        "name": "Split",
        "description": "Creating train / val / test dataset splits",
        "module": "scripts.split_fx_data",
        "script": "split_fx_data.py",
        "validates": ["datasets/train", "datasets/val", "datasets/test", "datasets/reports"],
        "skip_if_exist": ["datasets/train", "datasets/val", "datasets/test"],
    },
]

TOTAL_STEPS = len(STEPS)

# Packages where pip name and import name differ
PIP_TO_IMPORT = {
    "scikit-learn":   "sklearn",
    "pyyaml":         "yaml",
    "python-dotenv":  "dotenv",
    "beautifulsoup4": "bs4",
}

# Packages that must NOT be auto-installed — platform-specific builds
MANUAL_INSTALL = {
    "torch": (
        "PyTorch requires a platform-specific install command.\n"
        "  Visit https://pytorch.org/get-started/locally/ for the right\n"
        "  command for your OS / CUDA version, then re-run this script."
    ),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str = "") -> None:
    print(msg)

def is_windows() -> bool:
    return sys.platform.startswith("win")

def dir_nonempty(path: Path) -> bool:
    return path.is_dir() and any(True for f in path.rglob("*") if f.is_file())

def can_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


# ── Dependency checking ───────────────────────────────────────────────────────

def parse_requirements() -> list:
    req_file = PROJECT_ROOT / "requirements.txt"
    if not req_file.exists():
        return []
    packages = []
    for raw in req_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pip_name = line.split("==")[0].split(">=")[0].split("#")[0].strip()
        if pip_name:
            import_name = PIP_TO_IMPORT.get(pip_name.lower(), pip_name.lower())
            packages.append((pip_name, import_name))
    return packages


def pip_install(args_str: str) -> bool:
    cmd = [sys.executable, "-m", "pip", "install"] + args_str.split() + ["--quiet"]
    log(f"  pip install {args_str} ...")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            log(r.stderr.strip())
            return False
        return True
    except Exception as exc:
        log(f"  pip error: {exc}")
        return False


def ensure_dependencies() -> bool:
    log("  Checking dependencies ...")
    packages = parse_requirements()
    if not packages:
        log("  requirements.txt not found — skipping dependency check.")
        return True

    req_file = PROJECT_ROOT / "requirements.txt"

    missing_auto   = [(p, i) for p, i in packages if p.lower() not in MANUAL_INSTALL and not can_import(i)]
    missing_manual = [(p, i) for p, i in packages if p.lower() in MANUAL_INSTALL and not can_import(i)]
    all_ok = True

    if missing_auto:
        log(f"  Missing: {', '.join(p for p, _ in missing_auto)}")
        log("  Installing from requirements.txt ...")
        if pip_install(f"-r {req_file}"):
            still = [p for p, i in missing_auto if not can_import(i)]
            if still:
                log(f"  ERROR: installed but still unimportable: {', '.join(still)}")
                log("  Run  pip install -r requirements.txt  manually.")
                all_ok = False
            else:
                log("  All packages installed successfully  ✓")
        else:
            all_ok = False
    else:
        log("  All dependencies present  ✓")

    for pip_name, _ in missing_manual:
        log(f"\n  ✗  '{pip_name}' is not installed.\n  {MANUAL_INSTALL[pip_name.lower()]}")
        all_ok = False

    return all_ok


# ── Pre-flight ────────────────────────────────────────────────────────────────

def run_preflight() -> bool:
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 11):
        log(
            f"  ERROR: Python 3.11+ required — you are running {major}.{minor}.\n"
            "  Download from https://www.python.org/downloads/"
        )
        return False
    log(f"  Python {major}.{minor}  ✓")
    return ensure_dependencies()


# ── Output validation ─────────────────────────────────────────────────────────

def validate_outputs(step: dict) -> bool:
    if not step["validates"]:
        return True
    problems = [
        f"    ✗  {'missing' if not (PROJECT_ROOT / p).exists() else 'empty'}:  {p}"
        for p in step["validates"]
        if not dir_nonempty(PROJECT_ROOT / p)
    ]
    if problems:
        log(f"\n  Output validation failed (step {step['number']}):\n" + "\n".join(problems))
        return False
    return True


# ── Skip logic ────────────────────────────────────────────────────────────────

def should_skip(step: dict) -> bool:
    if not step["skip_if_exist"]:
        return False
    if all(dir_nonempty(PROJECT_ROOT / p) for p in step["skip_if_exist"]):
        log(f"  ⏭  Step {step['number']}/{TOTAL_STEPS}: {step['name']} — outputs exist, skipping.")
        return True
    return False


# ── Command builder ───────────────────────────────────────────────────────────

def build_command(step: dict) -> list:
    if is_windows():
        return [sys.executable, "-m", step["module"]]
    return [sys.executable, str(PROJECT_ROOT / "scripts" / step["script"])]


# ── Step runner ───────────────────────────────────────────────────────────────

def run_step(step: dict) -> bool:
    n, name = step["number"], step["name"]
    cmd = build_command(step)

    log(f"\n  Step {n}/{TOTAL_STEPS}: {name}")
    log(f"  {step['description']} ...")

    start = time.time()
    try:
        env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            cwd=str(PROJECT_ROOT), env=env,
        )
    except FileNotFoundError:
        log(f"  ERROR: script not found: {cmd[-1]}")
        return False
    except Exception as exc:
        log(f"  ERROR: failed to launch step {n}: {exc}")
        return False

    elapsed = time.time() - start

    if result.returncode != 0:
        if result.stderr:
            for line in result.stderr.strip().splitlines()[-25:]:
                log(f"    {line}")
        log(f"\n  Step {n} ({name}) FAILED (exit {result.returncode}, {elapsed:.1f}s)")
        return False

    if not validate_outputs(step):
        return False

    log(f"  Step {n}/{TOTAL_STEPS}: {name} — complete  ✓  ({elapsed:.1f}s)")
    return True


# ── Completion banner ─────────────────────────────────────────────────────────

def print_complete() -> None:
    log("\n" + "═" * 56)
    log("  ✅  System initialised — all pipeline steps complete!")
    log("═" * 56 + "\n")

    guide_path = PROJECT_ROOT / "backtest" / "backtest_guide.txt"
    if guide_path.exists():
        content = guide_path.read_text(encoding="utf-8").strip()
        log(content if content else "  (backtest_guide.txt is empty)")
    else:
        log("  You can now test your strategies:")
        log("    python scripts/backtest.py --strategy <name>")
        log("    python scripts/backtest.py --list-strategies")
        log("\n  Datasets:  datasets/train/   datasets/val/   datasets/test/")
        log("  Tip: add your guide to  backtest/backtest_guide.txt")
    log("")


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        prog="run_all_data_scripts.py",
        description="FX Data Pipeline — see module docstring for full usage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--from-step", type=int, default=1, metavar="N",
                        help="Start from step N (1–7). Default: 1.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip steps whose outputs already exist.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run all steps even if outputs exist. Overrides --skip-existing.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print steps and commands without executing anything.")
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    log("\n" + "═" * 56)
    log("  FX Data Pipeline")
    log(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("═" * 56)
    log(f"  --from-step {args.from_step}  |  --skip-existing {args.skip_existing}"
        f"  |  --force {args.force}  |  --dry-run {args.dry_run}\n")

    if not (1 <= args.from_step <= TOTAL_STEPS):
        log(f"  ERROR: --from-step must be 1–{TOTAL_STEPS}, got {args.from_step}")
        sys.exit(1)

    if args.dry_run:
        log("  DRY RUN — nothing will be executed.\n")
        for step in STEPS:
            log(f"    Step {step['number']}: {step['name']:<12}  →  {' '.join(build_command(step))}")
        log("")
        return

    if args.from_step == 1:
        if not run_preflight():
            sys.exit(1)
        log("")
    else:
        log(f"  Resuming from step {args.from_step} — pre-flight skipped.\n")

    steps_to_run = [s for s in STEPS if s["number"] >= args.from_step]
    pipeline_start = time.time()
    executed = skipped = 0

    for step in steps_to_run:
        if args.skip_existing and not args.force and should_skip(step):
            skipped += 1
            continue

        if not run_step(step):
            log("\n" + "═" * 56)
            log(f"  Pipeline stopped at step {step['number']}: {step['name']}")
            log(f"  Resume:  python scripts/run_all_data_scripts.py --from-step {step['number']}")
            log("═" * 56)
            sys.exit(1)

        executed += 1

    mins, secs = divmod(int(time.time() - pipeline_start), 60)
    log("\n" + "─" * 56)
    log(f"  Steps run: {executed}   Skipped: {skipped}   Total time: {mins}m {secs}s")
    log("─" * 56)

    print_complete()


if __name__ == "__main__":
    main()