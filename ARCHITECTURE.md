# Architecture

This document is the long-form architecture reference for forex-algo-trading. It contains six diagrams covering the system shape, the master-evaluation flow, the strategy class hierarchy, the data flow, the evaluation tier state machine, and the dependency graph. The architecture decision records and the source-map table follow the diagrams. The lightweight summary lives in [README.md](README.md). This file is the place to look when investigating a specific component, planning a refactor, or reviewing a design choice.


## High-level system flowchart

```mermaid
flowchart TB
  subgraph DATA ["Data pipeline"]
    direction LR
    D1[Stage 1: Download<br/>histdata yearly CSVs]
    D2[Stage 2: Clean<br/>UTC, OHLC validate, Parquet]
    D3[Stage 3: Features<br/>6 families, ~70 cols]
    D4[Stage 4: Labels<br/>3-class, pair-specific FLAT band]
    D5[Stage 5: Splits<br/>train/val/test + 5 folds + scalers]
    D1 --> D2 --> D3 --> D4 --> D5
  end
  subgraph TRAIN ["Stage 6: Train"]
    direction LR
    T1[Logistic Regression<br/>18 features x 4 sessions x 7 pairs = 28 cells]
    T2[Multi-scale LSTM<br/>2 branches x 4 sessions x 7 pairs = 28 cells]
  end
  subgraph EVAL ["Stage 7: Evaluate"]
    direction LR
    E1[Rule-based path<br/>T1 to T5]
    E2[ML cross-session<br/>8 strategies x 7 pairs x 4 sessions x 3 spreads]
  end
  D5 --> TRAIN
  D5 --> EVAL
  TRAIN --> EVAL
  EVAL --> OUT[(output/master_eval/<br/>master_report.txt + 7 CSVs)]
  style DATA fill:#0f2027,stroke:#2c5364,color:#fff
  style TRAIN fill:#16213e,stroke:#f5a623,color:#fff
  style EVAL fill:#1b1b2f,stroke:#39ff14,color:#fff
```

Each pipeline stage writes to a fixed location with a fixed schema, and each stage is its own script. The isolation is deliberate: when a single stage fails, the offending stage can be re-run without invalidating downstream state. The stages also have stable boundaries that the test suite exercises directly. A regression in stage three is caught before it can corrupt stage four.


## Master evaluation sequence

The master evaluation orchestrates the rule-based and ML paths in a single pass.

```mermaid
sequenceDiagram
  autonumber
  actor User
  participant CLI as scripts/master_eval.py
  participant RB as Rule-based path
  participant ML as ML path
  participant Eng as backtest/engine.py
  participant Out as Output writer
  User->>CLI: python scripts/master_eval.py --eval-year 2024 --spreads 1.0
  CLI->>CLI: Resolve config, load PAIRS, load locked splits
  CLI->>RB: T1: screen 13 rule-based strategies x 7 pairs on val
  RB->>Eng: run_backtest(strategy, pair, val) per cell
  Eng-->>RB: BacktestResult (13 metrics)
  RB-->>CLI: top survivors per pair
  CLI->>RB: T2: sweep (session, direction) on survivors
  CLI->>RB: T3: sweep TP/SL grid on T2 survivors
  CLI->>RB: T4: 5-fold walk-forward stability
  RB-->>CLI: T4 survivors (filtered)
  CLI->>RB: T5: final on locked test
  RB-->>CLI: rule-based final scores
  CLI->>ML: 8 strategies x 7 pairs x 4 sessions x 3 spreads
  ML->>ML: Generate signals once per (pair, ml_strategy)
  ML->>Eng: run_backtest with cached signals
  Eng-->>ML: BacktestResult per (pair, strategy, session, spread)
  ML-->>CLI: ML scores + per-pair transfer matrices
  CLI->>CLI: Compute 4 DM tests per pair
  CLI->>Out: write master_report.txt + 7 CSVs
  Out-->>User: output/master_eval/
```

The signal-cache decision (step 13) saves substantial runtime. The ML signal series for a given (pair, ml_strategy) does not change with the evaluation session or the spread; only the cost-and-mask bookkeeping changes. Caching the signal once and reusing it across the twelve session-by-spread combinations turns a four-hour ML evaluation into a fifteen-minute one.


## Strategy class hierarchy

The strategy module exposes a single abstract base class with concrete subclasses for each rule-based family and adapter classes for the learned models. The backtest engine consumes strategies through the base interface only.

```mermaid
classDiagram
  class BaseStrategy {
    <<abstract>>
    +str name
    +int warmup_bars
    +generate_signals(df) Series
  }
  class MACrossover {
    +int fast
    +int slow
    +str ma_type
    +int cooldown_bars
  }
  class MomentumStrategy {
    +int lookback
    +int cooldown_bars
  }
  class DonchianBreakout {
    +int period
    +int cooldown_bars
  }
  class RSIMeanReversion {
    +int period
    +float oversold
    +float overbought
    +int cooldown_bars
  }
  class BollingerBreakout {
    +int period
    +float std_mult
    +int cooldown_bars
  }
  class MACDSignalCross {
    +int fast
    +int slow
    +int signal
    +int cooldown_bars
  }
  class LRStrategy {
    +str pair
    +str session
    +load_model() void
    +predict(df) Series
  }
  class LSTMStrategy {
    +str pair
    +str session
    +load_checkpoint() void
    +batched_predict(df) Series
  }
  class BacktestResult {
    <<dataclass>>
    +float net_sharpe
    +float gross_sharpe
    +float total_return
    +float max_drawdown
    +float calmar
    +float sortino
    +float win_rate
    +float profit_factor
    +float avg_trade_bars
    +float turnover
    +int n_trades
    +float capital_initial
    +float capital_final
  }
  BaseStrategy <|-- MACrossover
  BaseStrategy <|-- MomentumStrategy
  BaseStrategy <|-- DonchianBreakout
  BaseStrategy <|-- RSIMeanReversion
  BaseStrategy <|-- BollingerBreakout
  BaseStrategy <|-- MACDSignalCross
  BaseStrategy <|-- LRStrategy
  BaseStrategy <|-- LSTMStrategy
```

The complete `STRATEGY_REGISTRY` (in `backtest/strategies.py`) instantiates 13 named rule-based strategies across the six families, plus 8 ML strategies (4 LR conditions and 4 LSTM conditions).

| Family class | Named instances |
|--------------|-----------------|
| `MACrossover` | `MACrossover_f20_s50_EMA`, `MACrossover_f50_s200_EMA`, `MACrossover_f20_s50_SMA` |
| `MomentumStrategy` | `Momentum_lb60`, `Momentum_lb120` |
| `DonchianBreakout` | `Donchian_p20`, `Donchian_p55` |
| `RSIMeanReversion` | `RSI_p14_os30_ob70`, `RSI_p14_os20_ob80` |
| `BollingerBreakout` | `BB_p20_std2_0`, `BB_p60_std2_0` |
| `MACDSignalCross` | `MACD_f26_s65_sig9`, `MACD_f78_s195_sig13` |
| `LRStrategy` | `LR_global`, `LR_london`, `LR_ny`, `LR_asia` |
| `LSTMStrategy` | `LSTM_global`, `LSTM_london`, `LSTM_ny`, `LSTM_asia` |

Every strategy returns a signal series with the same index as the input price series and values in the set `{-1, 0, +1}` representing DOWN, FLAT, UP. The engine handles position management, cost application, and metric computation. Strategies do not see prices; they see features and emit signals.

The `BacktestResult` dataclass is the engine's only return type. Its fields populate the 13-metric set used by every tier of the master evaluation and by the standalone `run_backtest.py` CLI.


## Data flow

```mermaid
flowchart LR
  RAW[Raw histdata CSVs<br/>data/extracted/]
  CLEAN[Cleaned Parquet<br/>data/processed/cleaned/]
  FEATS[Features Parquet<br/>features/PAIR/<br/>~70 columns]
  LABS[Labels Parquet<br/>labels/PAIR/<br/>3-class]
  TRAIN[(datasets/train/<br/>2015 to 2021)]
  VAL[(datasets/val/<br/>2022 to 2023)]
  TEST[(datasets/test/<br/>2024 to 2025)]
  FOLDS[(datasets/folds/<br/>fold_0 to fold_4)]
  SCALER[scalers/PAIR_scaler.pkl<br/>scaler + feature_cols]
  LR[models/.../PAIR_logreg_model.pkl]
  LSTM[models/.../PAIR_lstm_model.pt]
  SIG[Signal series]
  ENG[backtest/engine.py]
  MET[13 metrics]

  RAW --> CLEAN --> FEATS --> LABS
  LABS --> TRAIN
  LABS --> VAL
  LABS --> TEST
  LABS --> FOLDS
  TRAIN --> SCALER
  SCALER --> LR
  SCALER --> LSTM
  TRAIN --> LR
  TRAIN --> LSTM
  LR --> SIG
  LSTM --> SIG
  SIG --> ENG --> MET
```

The split direction is one-way. Features and labels are computed on the full per-pair series, then sliced into train, validation, test, and folds on row indices, never on time-aware joins. This guarantees that feature definitions are identical across splits and that no leakage exists at the join layer.

The scaler contract is small but load-bearing. Each `scalers/{PAIR}_scaler.pkl` is a dict with two keys, `scaler` (a fitted `StandardScaler`) and `feature_cols` (the list of column names the scaler was fit against, in order). Every consumer reads both. A silent column reordering between training and inference would otherwise produce systematically wrong predictions. A regression test in `tests/test_ml_features.py` asserts that the contract is honoured end-to-end.


## Evaluation tier state machine

```mermaid
stateDiagram-v2
  [*] --> T1
  T1 --> T2 : top survivors per pair<br/>plus family-diversity floor
  T2 --> T3 : best (session, direction)<br/>per (pair, strategy)
  T3 --> T4 : best TP/SL combo<br/>only if it beats baseline
  T4 --> T5 : <=2 negative folds AND >=10 trades
  T4 --> Rejected : >2 negative folds OR <10 trades
  T5 --> [*] : final test result
  Rejected --> [*]
  note right of T1 : Validation split<br/>2022 to 2023
  note right of T2 : Validation split
  note right of T3 : Validation split
  note right of T4 : Train folds<br/>2015 to 2021
  note right of T5 : Test split (locked)<br/>2024 to 2025
```

The tier progression encodes the project's selection-versus-evaluation discipline. T1, T2, and T3 are tuning tiers and run only on the validation split. T4 is a stability check on the training folds. T5 is the only tier that touches the locked test split, and it touches the test split exactly once per evaluation cycle. There is no T6 and no further tuning beyond T5. Whatever T5 reports for a (pair, strategy) cell is the answer for that cell.


## Dependency graph

```mermaid
flowchart LR
  PROJ[forex-algo-trading]
  PROJ --> P1[pandas]
  PROJ --> P2[numpy]
  PROJ --> P3[pyarrow]
  PROJ --> P4[scikit-learn]
  PROJ --> P5[matplotlib]
  PROJ --> P6[seaborn]
  PROJ --> P7[pyyaml]
  PROJ --> P8[python-dotenv]
  PROJ --> P9[torch]
  PROJ --> P10[scipy]
  PROJ -.optional.-> P11[playwright]
  P9 -.large.-> SIZE[~2.5 GB install]
  style PROJ fill:#16213e,stroke:#f5a623,color:#fff
  style P9 fill:#1b1b2f,stroke:#39ff14,color:#fff
```

Ten direct production dependencies plus one optional dependency (`playwright`, used only by the optional PDF export of HTML backtest reports). The torch dependency dominates the install size. The LSTM is the only consumer.


## Architecture decision records

| Decision | Rationale | Alternatives considered | Why not the alternatives |
|----------|-----------|-------------------------|--------------------------|
| Tiered evaluation T1 to T5 | A single-pass screen produced uninterpretable results that drowned the signal in noise. The tiered structure separates concerns: screen, narrow, tune, stabilise, evaluate. | Single grand-search over all dimensions | The cross-product of strategy x session x direction x TP/SL x fold is too large to navigate without intermediate filters. |
| Flat per-pair pip spread | A flat spread is identical for every strategy in the system, which makes the comparison harder to abuse. The cost is the same for every strategy; differences are due to strategy behaviour, not modelling choices. | Stochastic spread with slippage and partial-fill simulation | More parameters introduce more degrees of freedom for the comparison to be subtly skewed by modelling choices rather than by strategy differences. |
| Locked split dates as constants | Comparability across runs requires that every strategy sees the same calendar windows. Hard-coded constants prevent silent slides during ablations. | Configurable splits via CLI | Any slide invalidates the comparability that the rest of the platform is built around. |
| 3-class label with pair-specific FLAT band | Sign-of-return labels caused class imbalance and pushed every model toward degenerate "always UP" or "always FLAT" solutions. The FLAT band is calibrated per-pair on the training distribution. | Sign-of-return; meta-labelling; triple-barrier | Sign-of-return collapses; meta-labelling and triple-barrier add complexity without solving the comparability question. |
| Two-branch LSTM | Minute-level FX has structure at multiple time scales. A single-branch model with one window length either underweights the long-horizon volatility regime or ignores short-horizon momentum. | Single-branch with 60-bar window; single-branch with 15-bar window | Each fails one of the two horizons. The two-branch architecture is, by construction, a compromise between those failure modes. |
| Session injection at the merge point | The session changes slowly relative to the per-bar features. Injecting the session at every time step is mostly redundant; injecting it as a dense conditioning vector on the merged recurrent representation is closer to the intent. | Inject session at every time step; learnable session embedding instead of one-hot | Per-step injection is wasteful; a learnable embedding adds parameters without a research justification. |
| Exclude XGBoost and LightGBM | Research scope. The framing of RQ2 is LR versus LSTM; a third learned model would dilute the answer rather than sharpen it. | Add boosted trees as a third class | Distracts from the pairwise framing. The question "does LSTM beat boosted trees" is interesting but is not the project's question. |
| 35 / 25 / 25 / 15 composite weights | Net Sharpe is the canonical risk-adjusted return metric and earns the largest weight. Sortino and Calmar add downside-and-drawdown information. Drawdown safety is implicit in Calmar plus the gates and earns the smallest weight. | Equal weights; Sharpe-only; expected utility | Equal weights underweight the most informative metric. Sharpe-only ignores drawdown information. Expected utility is opinionated about risk preferences. |
| Hard gates on n_trades < 10 and max_dd < -0.95 | Below those thresholds the strategy is either statistically meaningless or economically catastrophic. A fluke high Sharpe on three lucky trades should not appear at the top of the leaderboard. | Soft penalties via gradient | Soft penalties allow lottery-ticket strategies to ride to the top with enough other-metric strength. Hard gates are unambiguous. |
| Diebold-Mariano with Newey-West HAC | Consecutive bars are not independent. The naive variance estimator would be much too small and the test statistic would be inflated. | Naive DM; bootstrap | Naive DM is wrong here. Bootstrap is defensible but more expensive and gives a less canonical result. |
| Frozen feature lists | Feature schemas drift over time when not pinned. A drifted feature list breaks scaler contracts, breaks model checkpoints, and breaks reproducibility. The frozen list with a `do not modify` comment is enforced socially and tested mechanically. | Per-experiment feature configs | Convenient for individual experimentation, fatal for cross-experiment comparison. |
| Signal cache in master evaluation | The ML signal does not change with evaluation session or spread. Caching it once per (pair, strategy) and reusing across 12 session-by-spread combinations cuts ML evaluation runtime by 16x. | Recompute every combination | Redundant computation; the dominant cost is loading the model and producing predictions across 1.5M test bars. |


## Source map

| Folder | Purpose |
|--------|---------|
| `backtest/` | Backtest engine (`engine.py` with `run_backtest`, `run_wf_folds`, and `BacktestResult`), strategy implementations (`strategies.py`), per-strategy CLI (`run_backtest.py`), HTML report generator (`report_generator.py`), Jinja templates (`templates/`), generated reports (`reports/`, gitignored). |
| `scripts/` | Seven pipeline stage scripts (`download_fx_data.py`, `clean_fx_data.py`, `features_fx_data.py`, `labels_fx_data.py`, `split_fx_data.py`, `train_model.py`, `master_eval.py`), the multi-cell training driver (`train_all.py`), shared helpers (`_common.py`), exploratory data analysis scripts (`eda_fx_data.py`, `eda_split_readiness.py`, `inspect_fx_data.py`), the optional PDF exporter (`export_report_pdf.py`), and legacy runners kept for reference (`evaluate_ml.py`, `fx_master_test_runner.py`). |
| `config/` | Frozen runtime constants (`constants.py`: locked split dates, frozen feature lists, env-overridable runtime values, per-pair pip spreads, the TP/SL grid), root logger setup (`logging_setup.py`). |
| `tests/` | pytest test suite, twelve files covering engine loop, walk-forward fold paths, metric definitions, ML feature plumbing, ML strategy adapter behaviour, pair centralisation, position management, session masks, and walk-forward boundary handling. |
| `output/master_eval/` | Master evaluation artefacts. `master_report.txt` is the definitive text report. The seven CSVs (`results_all.csv`, `results_ml.csv`, `results_rule_based.csv`, `best_worst_per_pair.csv`, `dm_test_results.csv`, `session_generalisability.csv`, plus per-pair transfer matrices) are the structured outputs. |
| `models/global/` | Trained global-condition checkpoints. One pickle per pair for LR (`{PAIR}_logreg_model.pkl`), one PyTorch checkpoint per pair for LSTM (`{PAIR}_lstm_model.pt`). Directory shape ships via `.gitkeep`; the actual checkpoint files are gitignored. |
| `models/session/` | Trained session-conditional checkpoints. `london/`, `ny/`, and `asia/` subdirectories each contain per-pair LR pickles and per-pair LSTM checkpoints, where trained. |
| `scalers/` | One pickle per pair containing a fitted `StandardScaler` and the list of feature column names it was fit against. |
| `data/` | Raw extracted CSVs and cleaned per-pair Parquets. Gitignored, around 2.6 GB on disk. Regenerable from stages 1 and 2. |
| `features/` | Per-pair feature Parquets (one Parquet per pair, around seventy columns each). Gitignored, around 6.5 GB on disk. Regenerable from stage 3. |
| `labels/` | Per-pair label Parquets. Gitignored, around 6.9 GB on disk. Regenerable from stage 4. |
| `datasets/` | Train, validation, test, and walk-forward fold slices per pair. Gitignored, around 30 GB on disk. Regenerable from stage 5. |
| `docs/` | Documentation: setup, experiments, findings. |
| `docs/assets/` | Demo screenshots (`demo1.png` through `demo5.png`) and the sample HTML report (`sample_report.html`). |
| `eda/` | Exploratory data analysis outputs (e.g. `split_readiness/split_readiness.csv`). |


## Implementation notes

Several implementation details have non-obvious correctness implications and are documented here so that contributors do not stumble into them.

**Label remap for the LSTM.** PyTorch's `CrossEntropyLoss` requires non-negative integer class indices. The platform's raw labels use `{-1, 0, +1}` because that mapping makes the backtest engine's signal arithmetic clean (position times signed signal yields the right sign convention). At LSTM training time the labels are remapped: `-1 -> 0`, `0 -> 1`, `+1 -> 2`. At inference time the inverse is applied: `argmax==0 -> -1`, `argmax==1 -> 0`, `argmax==2 -> +1`. The remap appears in two places in the codebase, and a regression test asserts that the round-trip is correct.

**Batched LSTM inference.** The first version of LSTM inference iterated bar-by-bar, building a fresh `(15, 5)` short-window tensor and a fresh `(60, 4)` long-window tensor for each test bar. The result was correct and roughly two orders of magnitude too slow for the cross-session evaluation grid. The current version builds all sliding-window arrays up front using NumPy stride tricks, then forwards through the model in chunks of 4096 samples at a time. Throughput is around 31,000 bars per second on a CPU.

**Scaler contract.** Every consumer of a scaler reads both the `scaler` and the `feature_cols` keys. The features tensor at inference is assembled in the order specified by `feature_cols`, not by the natural column order of whatever DataFrame was passed in. This protects against silent column reorderings between training and inference. A regression test asserts that the contract is honoured.

**Fold parquet paths.** Walk-forward folds resolve to `datasets/folds/fold_N/` Parquet files, not to slices of the training Parquet by row index. Earlier versions of the code used row-index slicing as a fallback. The fallback was removed because it produced subtly different results than the canonical Parquet form and made fold-level reproducibility harder to audit. The path resolution is in `backtest/run_backtest.py::resolve_split_path`.

**Determinism in master_eval.** The master evaluation script is deterministic given a fixed configuration: same pair set, same evaluation window, same spread multipliers, same model checkpoints on disk. Back-to-back runs produce diff-equal outputs. This is a precondition for RQ0 and the test suite covers it indirectly through the per-stage regression tests.

**Session boundary inclusivity.** Session masks include the start hour and exclude the end hour (`hour >= start AND hour < end`), with the Asia session wrapping across midnight (`hour >= 23 OR hour < 8`). A regression test in `tests/test_session_filter.py` covers the boundary minutes explicitly.

**Cooldown bars in rule-based strategies.** Each rule-based strategy carries a `cooldown_bars` parameter that suppresses re-entry for N bars after a position closes. Without it, a fast crossover would whipsaw and the trade ledger would explode. The cooldown is enforced inside the strategy class, not in the engine.

**Per-pair pip sizing.** The cost model uses `PAIR_PIP_SIZES` to convert pips to price units: `0.0001` for non-JPY pairs and `0.01` for JPY pairs. The flat spread in `PAIR_SPREAD_PIPS` is multiplied by this pip size to give the per-trade cost in price terms.
