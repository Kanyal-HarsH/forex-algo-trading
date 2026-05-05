# Experimentation setup

This document describes how forex-algo-trading is set up as a research apparatus. The seven-stage pipeline is the experimental protocol. The locked splits, frozen feature lists, and deterministic master evaluation script are the reproducibility controls. The three research questions (RQ0, RQ1, RQ2) are the experiments themselves.


## Experimental framework

The experimental protocol is encoded as a sequence of seven scripts in the `scripts/` directory, executed in order: `download_fx_data.py` → `clean_fx_data.py` → `features_fx_data.py` → `labels_fx_data.py` → `split_fx_data.py` → `train_model.py` → `master_eval.py`. Each stage writes to a fixed location with a fixed schema, and each stage is its own script so that a failure in one stage can be repaired and re-run without invalidating the rest of the pipeline state.

Reproducibility is treated as a research question in its own right rather than an engineering concern. The platform makes a number of design choices specifically to honour reproducibility, sometimes at the cost of speed or convenience. The locked split dates are constants in `config/constants.py` rather than CLI flags. The LR feature list and both LSTM feature lists are frozen with explicit `do not modify` comments in source. Every model checkpoint is saved with the random seed and the input feature column list it was trained on. The scaler contract pairs each fitted `StandardScaler` with the column-order list it expects at inference. The pytest suite (twelve files) covers the kinds of regressions that erode reproducibility silently: wrong index alignment between features and labels, scaler fit on the wrong split, session masks that include or exclude the boundary minute inconsistently, fold-path resolution that drifts between row-index slicing and on-disk Parquet reads.

The master evaluation script is deterministic given a fixed configuration. Running `scripts/master_eval.py` with the same `--pairs`, `--eval-year`, `--spreads`, and on-disk model checkpoints produces diff-equal output CSVs and text reports across runs. This determinism is the operational form of RQ0 and is implicitly tested by every per-stage regression test in the suite.

Output artefacts land in a per-run subdirectory under `output/master_eval/run_<window>/`, where `<window>` is a slug derived from the eval window (`1day`, `1week`, `1month`, `6month`, `12month`, `24month`). A pointer file `output/master_eval/latest_run.txt` records the most recent subdirectory. Each subdirectory contains one text report (`master_report.txt`) plus a CSV suite of ten distinct families. `results_all.csv` combines rule-based and ML results sorted by composite score. `results_rule_based.csv` and `results_ml.csv` contain the per-class breakdowns. `best_worst_per_pair.csv` summarises per-pair extremes. `dm_test_results.csv` reports the four Diebold-Mariano comparisons per pair. `session_generalisability.csv` summarises the in-domain-versus-transfer pattern across training sessions. `lr_feature_importance.csv` lists the top-five features per (pair, session) cell. Per-pair transfer matrices (4 by 4 Sharpe matrices for LR and LSTM) live in `transfer_matrix_lr_{PAIR}.csv` and `transfer_matrix_lstm_{PAIR}.csv`. The remaining file, `cost_breakeven.csv`, reports the spread-multiplier breakpoint at which each strategy's net return falls to zero (populates only when at least two `--spreads` values are supplied).

A figure gallery regenerates from the same per-run directory. Run `python scripts/make_report_plots.py` to read `latest_run.txt` and write 33 publication-grade figures plus a `TABLES.md` index to `docs/assets/plots/run_<window>/`. The headline numbers from the most recent end-to-end run are summarised in [FINDINGS.md](FINDINGS.md).


## Experiment catalogue

| Experiment ID | Hypothesis | Variables | Controlled | Status |
|---------------|------------|-----------|------------|--------|
| RQ0 | Identical inputs (seeds, splits, code, environment) produce identical evaluation outputs | none (precondition) | seeds, splits, code, environment | answered yes; verified by per-stage regression tests and an empty `diff` on `results_all.csv` across re-runs |
| RQ1 | Session-conditional models achieve higher cost-adjusted Sharpe than a single global model | session conditioning (4 levels: global, london, ny, asia) | data, splits, costs, scoring, model class | answered: mixed and pair-specific (2 of 14 in-domain DM probes significant). Detail in [FINDINGS.md](FINDINGS.md) |
| RQ2 | A multi-scale LSTM with two branches outperforms Logistic Regression under identical evaluation conditions | model class (LR vs LSTM) | data, splits, costs, scoring, session conditioning | answered: no on aggregate (LR avg Sharpe -7.90, LSTM avg Sharpe -6.43, no DM champion-vs-runner-up below `p = 0.19`). Detail in [FINDINGS.md](FINDINGS.md) |

RQ0 is ongoing in the sense that every change to the codebase is implicitly an RQ0 test: if the change introduces non-determinism, the regression suite catches it. RQ1 and RQ2 each require comparing 28 cells against 28 cells (seven pairs by four sessions for each model class); the full grid is in place on disk and the master evaluation has been run end-to-end across six windows from one trading day up to the full 24-month test span.

The full ML grid is 56 trained checkpoints: 28 Logistic Regression plus 28 multi-scale LSTM. Inventory:

```bash
ls models/global/ ; ls models/session/london/ ; ls models/session/ny/ ; ls models/session/asia/
```

Each subdirectory contains seven `*_logreg_model.pkl` files and seven `*_lstm_model.pt` files. The master evaluation reports any missing cell as `NaN` rather than silently skipping it, so a partial-grid run remains interpretable.


## Running experiments

The simplest meaningful experiment is a single-year master evaluation across all pairs:

```bash
python scripts/master_eval.py --eval-year 2024 --spreads 1.0
```

Expected runtime: 35 to 50 minutes on a recent laptop. Output structure:

```
output/master_eval/
├── latest_run.txt                       pointer to the most recent run subdir
└── run_<window>/                        per-run output (e.g. run_1day, run_24month)
    ├── master_report.txt                definitive text report; read this first
    ├── results_all.csv                  all backtest rows, sorted by composite score
    ├── results_rule_based.csv           T1-to-T5 breakdown
    ├── results_ml.csv                   ML cross-session breakdown
    ├── best_worst_per_pair.csv          per-pair extremes
    ├── transfer_matrix_lr_EURUSD.csv    one per pair (LR), 7 files total
    ├── transfer_matrix_lstm_EURUSD.csv  one per pair (LSTM), 7 files total
    ├── session_generalisability.csv     avg off-diagonal Sharpe per training session
    ├── lr_feature_importance.csv        top-5 features per (pair, session) cell
    ├── dm_test_results.csv              4 DM comparisons per pair
    └── cost_breakeven.csv               populated only with >=2 spread multipliers
```

The full-window evaluation (the entire test split, 2024 to 2025, all pairs, all spreads) takes longer and is appropriate for the final research run:

```bash
python scripts/master_eval.py --spreads 0.5 1.0 2.0
```

Expected runtime: 90 to 150 minutes.

For a single-pair smoke test useful when iterating on a strategy implementation:

```bash
python scripts/master_eval.py --pairs EURUSD --eval-year 2024 --spreads 1.0
```

Expected runtime: 8 to 12 minutes.

For an ML-only run that skips the tiered rule-based path:

```bash
python scripts/master_eval.py --ml-only --eval-year 2024 --spreads 1.0
```

Expected runtime: 10 to 15 minutes.

For a custom evaluation window inside the test split:

```bash
python scripts/master_eval.py --from 2024-06-01 --to 2024-12-31 --spreads 1.0
```

The script validates that any custom date window lies inside the locked test span (`TEST_START` to `TEST_END` in `config/constants.py`) and refuses anything outside.

To run all six standard evaluation windows in a single orchestrator call:

```bash
python scripts/run_all_windows.py --workers 8
```

The orchestrator dispatches `master_eval.py` for the 1-day, 1-week, 1-month, 6-month, 12-month, and 24-month windows in sequence, each into its own `output/master_eval/run_<window>/` subdirectory. Subsequent calls regenerate the corresponding figure gallery via `python scripts/make_report_plots.py --eval-dir output/master_eval/run_<window> --out-dir docs/assets/plots/run_<window>`.


## Reproducibility checklist

Before committing or publishing a result, confirm each item below.

- Random seed fixed in `scripts/train_model.py` (default 42; same seed used for both LR and LSTM training)
- Locked split constants frozen in `config/constants.py` (`TRAIN_END=2021-12-31`, `VAL_START=2022-01-01`, `VAL_END=2023-12-31`, `TEST_START=2024-01-01`, `TEST_END=2025-12-31`)
- Frozen `LR_FEATURES` list (18 items, `do not modify` comment)
- Frozen `LSTM_SHORT_FEATURES` list (5 items)
- Frozen `LSTM_LONG_FEATURES` list (4 items, optionally extended with `same_minute_prev_day_logrange` when present in training data)
- Per-pair scaler saved as a dict with `scaler` and `feature_cols` keys
- Per-pair flat spread fixed in `PAIR_SPREAD_PIPS` constant
- pytest suite runs on every change (12 files)
- Master evaluation script is deterministic given fixed inputs
- Per-stage regression tests cover the scaler contract, fold path resolution, session mask boundaries, and label-feature alignment
- Environment overrides documented in `.env.example` at the repository root


## Diebold-Mariano test design

Significance testing is integrated into the master evaluation. Each evaluation pass runs four DM comparisons per pair.

| Comparison | Period | Domain |
|------------|--------|--------|
| Best rule-based vs Buy-and-hold | Validation split | Rule-based survival check |
| Best ML vs Buy-and-hold | Test split | ML headline result |
| In-domain session vs Transfer | Test split | Session-conditioning hypothesis (RQ1) |
| Champion vs runner-up (within-class) | Validation (RB) or Test (ML) | Within-class significance |

The within-class champion-versus-runner-up comparison is restricted within model class to avoid mixing scores from different time periods (validation versus test) and different model classes (rule-based on validation versus ML on test).

The DM test takes two return series and tests the null that their forecast loss differential has zero mean against the alternative that it does not. The loss differential is the per-bar net P&L difference, and a significant negative test statistic means the second strategy beats the first. The variance estimator is Newey-West HAC with a bar-frequency-appropriate lag, because consecutive minute bars are not independent and the naive variance would be much too small. The Harvey-Leybourne-Newbold small-sample correction is applied on top of the Newey-West variance to control the size of the test in finite samples; this matches the standard recommendation in the FX-evaluation literature for minute-level data and is what the master-evaluation script reports. The output is a p-value per comparison plus the raw mean differential and the test statistic. A p-value below 0.05 is treated as evidence of a real difference. Anything above is treated as inconclusive, not as evidence of equivalence.


## Walk-forward stability analysis

T4 of the rule-based path runs each surviving (pair, strategy) configuration against five contiguous, non-overlapping walk-forward folds inside the training span. For each fold, the per-fold Sharpe is recorded. The stability score is the mean Sharpe across folds minus half the standard deviation: `stability = mean(sharpe_folds) - 0.5 * std(sharpe_folds)`. This is a simple penalised-mean that prefers consistent performers over high-variance lottery tickets.

Two filters reject configurations from advancing to T5:

1. **Negative-fold filter.** More than two negative folds out of five → rejected. The filter rejects strategies that worked once and broke.
2. **Trade-count filter.** Fewer than ten trades on average per fold → rejected. The filter rejects strategies that are too sparse to evaluate meaningfully.

Survivors of T4 carry forward to T5 (the only tier that touches the locked test split).


## What changed between iterations

The platform has been through several internal iterations during development. The current evaluation pipeline is the third major revision.

The first iteration was a single-pass screen that ran every strategy at every parameter on every pair against every split. The result drowned in uninterpretable output and made it impossible to tell which lever caused which change.

The second iteration introduced a tiered structure but kept tuning and final evaluation in the same loop. The result let configurations cherry-pick parameter combinations that worked once, which is exactly the over-fitting failure mode the platform is built to prevent.

The third (and current) iteration enforces strict tier boundaries: T1 to T3 are tuning tiers and run only on the validation split; T4 is a stability check on training folds; T5 is the only tier that reads from the locked test split, and it touches the test split exactly once per evaluation cycle. The boundaries are enforced at the script level, not just in documentation.
