# Universal CMAPSS RUL Measurement Pipelines

This repository is now focused on **measurement only**: Remaining Useful Life (RUL) prediction and ANN pruning experiments on the NASA C-MAPSS benchmark. The older control-oriented FOPID component is no longer part of the active workflow.

The package path remains `src/measurement_control/` for backward compatibility, but the maintained pipelines are measurement-focused:
- a reusable PyTorch two-stage ANN + PSO workflow
- an experimental PyTorch ANN + PSO + MILP pruning workflow
- a lightweight legacy scikit-learn baseline in `main_universal.py`

## Active Scope

- RUL prediction on `FD001` through `FD004`
- architecture screening with PSO
- selective retuning of top candidates
- ANN pruning with MILP-based structure reduction
- timing, validation, and official-test reporting

## C-MAPSS Datasets

| Dataset | Train Traj. | Conditions | Fault Modes | Practical difficulty |
|---|---:|---:|---:|---|
| FD001 | 100 | 1 | 1 | Baseline: single regime, single fault |
| FD002 | 260 | 6 | 1 | Regime-shift difficulty |
| FD003 | 100 | 1 | 2 | Multi-fault difficulty |
| FD004 | 249 | 6 | 2 | Regime-shift plus multi-fault difficulty |

The project uses the local files under `data/CMAPSSData`:
- `train_FD00x.txt`
- `test_FD00x.txt`
- `RUL_FD00x.txt`
- `readme.txt`

Additional dataset details and descriptive statistics are documented in [docs/DATA_CARD.md](docs/DATA_CARD.md).

## Recommended Workflows

### 1. PyTorch Two-Stage ANN + PSO

- Module: `src/measurement_control/torch_rul_pso.py`
- Wrapper: `scripts/run_torch_pipeline.py`

This is the main reusable architecture-screening workflow:
1. Stage 1 uses PSO under a shared low-cost training budget to screen ANN structures.
2. Stage 2 fully retrains only the top-k candidates.
3. The final selected model is retrained on the full official training split.
4. The official NASA test split is evaluated once at the end.

Example:

```bash
python scripts/run_torch_pipeline.py --skip-install --data-root data/CMAPSSData
```

Useful controls:
- `--normalization-mode`
- `--min-hidden-layers` / `--max-hidden-layers`
- `--min-neurons` / `--max-neurons`
- `--activation-choices`
- `--n-particles` / `--n-iter`
- `--low-fidelity-epochs` / `--low-fidelity-patience`
- `--top-k`
- `--full-tuning-epochs` / `--full-tuning-patience`
- `--final-train-epochs`
- `--tuning-learning-rates` / `--tuning-batch-sizes` / `--tuning-weight-decays`
- `--complexity-penalty-weight`
- `--seed`

Current main-workflow defaults:
- `20` PSO particles
- `2` to `3` hidden layers
- `10` to `100` neurons per hidden layer
- `global_standard` normalization

Outputs:
- `outputs/torch_pytorch/torch_rul_summary.csv`
- `outputs/torch_pytorch/torch_two_stage_report_<DATASET>.json`
- `outputs/torch_pytorch/fig_torch_mlp_convergence_<DATASET>.png`
- `outputs/torch_pytorch/fig_torch_rul_prediction_<DATASET>.png`

### 2. Experimental PyTorch ANN + PSO + MILP Pruning

- Module: `src/measurement_control/torch_rul_pso_milp_pruning.py`
- Wrapper: `scripts/run_torch_milp_pruning_pipeline.py`

This branch extends the measurement workflow with pruning:
1. Stage 1 runs PSO to screen dense ANN structures.
2. The top-k candidates are converted into dense reference models.
3. A pruning optimization stage removes arcs while preserving dense-model behavior on a calibration subset.
4. The pruned candidates are fine-tuned and compared on validation data.
5. The final selected architecture is evaluated on the official NASA test split.

Important notes:
- This is still a measurement workflow. There is no control-stage optimization.
- Exact MILP pruning is implemented for `1`- and `2`-hidden-layer candidates.
- For `2` hidden layers, the default strategy is a reduced-neighborhood exact MILP seeded by an activation-aware local-search heuristic.
- The default pruning branch evaluates all four subsets: `FD001`, `FD002`, `FD003`, and `FD004`.
- The default `training_fraction=0.3` is a development compromise. Use `--training-fraction 1.0` for a full-data run.

Example:

```bash
python scripts/run_torch_milp_pruning_pipeline.py --skip-install --data-root data/CMAPSSData
```

Direct module execution also works:

```bash
python src/measurement_control/torch_rul_pso_milp_pruning.py --data-root data/CMAPSSData
```

Current pruning-workflow defaults:
- `50` PSO particles
- `1` to `2` hidden layers
- `10` to `100` neurons per hidden layer
- `training_fraction=0.3`
- `full_tuning_epochs=120`
- `full_tuning_patience=12`
- `pruning_finetune_epochs=80`
- `pruning_finetune_patience=10`

Outputs:
- `outputs/torch_pytorch_milp_pruning/torch_milp_pruning_summary.csv`
- `outputs/torch_pytorch_milp_pruning/torch_milp_pruning_before_after_summary.csv`
- `outputs/torch_pytorch_milp_pruning/torch_milp_pruning_before_after_summary.md`
- `outputs/torch_pytorch_milp_pruning/torch_milp_pruning_report_<DATASET>.json`
- `outputs/torch_pytorch_milp_pruning/torch_milp_pruning_candidates_<DATASET>.csv`
- `outputs/torch_pytorch_milp_pruning/torch_milp_pruning_histories_<DATASET>.csv`
- `outputs/torch_pytorch_milp_pruning/fig_torch_milp_pruning_candidates_<DATASET>.png`
- `outputs/torch_pytorch_milp_pruning/fig_torch_milp_pruning_histories_<DATASET>.png`
- dense and pruned official-test prediction plots

Additional pruning-model notes:
- [docs/PRUNING_OPTIMIZATION_MODEL.md](docs/PRUNING_OPTIMIZATION_MODEL.md)

### 3. Legacy scikit-learn Baseline

- Script: `main_universal.py`

This script is now a simple measurement-only baseline. It:
- loads `train_FD001` through `train_FD004`
- builds windowed RUL samples
- screens shallow MLP sizes with a lightweight PSO loop
- trains a final scikit-learn `MLPRegressor`
- generates a dashboard and timing plot

Run:

```bash
python main_universal.py
```

Outputs:
- `universal_dashboard.png`
- `universal_time_analysis.png`

## Split Protocol

- The official NASA benchmark split is preserved.
- `train_FD00x.txt` is used for development.
- `test_FD00x.txt` plus `RUL_FD00x.txt` are reserved for final benchmark evaluation.
- The PyTorch pipelines create an internal validation split only inside the official training set, at the **engine level**, not the window level.
- The official test set is touched only once for the final selected model.

## Normalization Protocol

- The recommended default is `global_standard`.
- A training-derived scaler is fit on development data and then applied consistently to validation and official-test data.
- This avoids the train/test mismatch introduced by per-engine min-max scaling on truncated test trajectories.
- `global_minmax` and `per_unit_minmax` remain available for controlled comparisons.

## Feature Scope

- `docs/DATA_CARD.md` documents the raw 26-column C-MAPSS schema.
- The legacy baseline uses a fixed literature-aligned 10-sensor subset for comparability.
- The reusable PyTorch pipeline keeps a measurement-focused feature space while handling the multi-regime subsets more carefully than the old baseline.

## Package Layout

- `src/measurement_control/`: reusable measurement pipelines
- `scripts/`: thin CLI wrappers
- `docs/`: dataset and pruning-method notes
- `data/CMAPSSData/`: local NASA benchmark files
- `outputs/`: generated experiment artifacts

## Paper Asset Regeneration

The paper figures and LaTeX table snippets for [docs/article.tex](docs/article.tex) can be regenerated from the pruning outputs with:

```bash
python scripts/generate_article_assets.py
```

This writes combined figures and table snippets into `docs/generated/` so the article uses reproducible assets rather than manually copied numbers or screenshots.

## Reference

A. Saxena, K. Goebel, D. Simon, and N. Eklund, “Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation,” Proceedings of PHM08, Denver, Colorado, October 2008.
