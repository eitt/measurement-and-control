# Universal CMAPSS RUL Prediction & Robust FOPID Control

This project implements a comprehensive framework for **Remaining Useful Life (RUL)** prediction across all four NASA CMAPSS jet engine datasets (`FD001` through `FD004`). It integrates a Particle Swarm Optimization (PSO) tuned Neural Network (MLP) for prognosis and a robust **Fractional Order PID (FOPID)** controller for system stability.

## Key Features
* ** Multi-Dataset Analysis:** Automatically loops through and processes `FD001`, `FD002`, `FD003`, and `FD004` to compare performance across varying operating conditions and fault modes.
* **PSO-Optimized Architectures:** Uses Particle Swarm Optimization to dynamically find the optimal hidden layer structure for the Neural Network based on the complexity of each dataset.
* **Robust FOPID Control:** Implements a fractional calculus controller with stability safeguards (handling complex number errors) to simulate engine control response based on RUL predictions.
* **Computational Benchmarking:** Tracks and visualizes the training vs. optimization time for each dataset.
* **Universal Dashboard:** Automatically generates a 2x2 summary dashboard comparing MSE/MAE, error clustering, and computational efficiency.

## Datasets (NASA CMAPSS)The project models and synthesizes the following datasets:

| Dataset | Train Traj. | Conditions | Fault Modes | Complexity |
| --- | --- | --- | --- | --- |
| **FD001** | 100 | 1 (Sea Level) | 1 (HPC) | Low |
| **FD002** | 260 | 6 (Mixed) | 1 (HPC) | Medium |
| **FD003** | 100 | 1 (Sea Level) | 2 (HPC, Fan) | Medium |
| **FD004** | 248 | 6 (Mixed) | 2 (HPC, Fan) | High |

## Data FormatInput files (`train_FD00x.txt`) are space-separated text files with 26 columns:

1. **Unit Number**
2. **Time (Cycles)**
3. **Op. Setting 1**
4. **Op. Setting 2**
5. **Op. Setting 3**
6. **Sensor Measurement 1**
...
7. **Sensor Measurement 21**

## Usage###1. PrerequisitesEnsure you have the required Python libraries installed:

```bash
pip install -r requirements.txt

```

### 2. SetupPlace the dataset files (`train_FD001.txt` ... `train_FD004.txt`) in the same directory as the script.

### 3. ExecutionRun the universal main script to process all datasets and generate reports:

```bash
python main_universal.py

```

### 4. OutputsThe script generates the following analysis files:

* `universal_dashboard.png`: A 2x2 Summary of Global Performance (MSE/MAE trends and Error Clustering).
* `universal_time_analysis.png`: Bar chart comparing computational cost across datasets.
* `universal_fopid_convergence.png`: Optimization curve for the control parameters.
* Console Logs: Real-time training metrics and architectural choices.

## Methodology
### 1. RUL Prediction (MLP + PSO)* **Preprocessing:** MinMax Scaling, Sequence generation (window size: 30), and feature selection.
* **Architecture Search:** A PSO algorithm explores the hyperparameter space (hidden neurons) to minimize validation MSE.
* **Training:** The best architecture found is retrained on the full training set.

### 2. Control System (FOPID)
* **Simulation:** Simulates a plant response using a Fractional Order PID controller: C(s) = K_p + K_i s^{-\lambda} + K_d s^{\mu}.
* **Robustness:** Mathematical safeguards (`abs()` and `sign()`) prevent complex number instability during fractional differentiation.

## Reference
A. Saxena, K. Goebel, D. Simon, and N. Eklund, “Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation”, in the Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.

## Organized Layout (New)
- `src/measurement_control/`: reusable pipeline code.
- `scripts/`: runnable entrypoint scripts.
- `docs/`: project documentation and organization notes.
- `outputs/`: generated artifacts from newer pipelines.

Detailed structure notes are available in `docs/PROJECT_STRUCTURE.md`.

## PyTorch Two-Stage Workflow
The reusable ANN + PSO pipeline now lives in:
- Module: `src/measurement_control/torch_rul_pso.py`
- Entrypoint: `scripts/run_torch_pipeline.py`

The workflow is intentionally split into two stages:
1. **Stage 1: low-fidelity PSO screening**
   - PSO searches ANN structure under a small common training budget.
   - Candidates are ranked by `validation_mse + lambda_complexity * complexity_penalty`.
   - This stage is for reducing the architecture search space, not for claiming a final global optimum.
2. **Stage 2: selective retuning**
   - Only the top-k Stage 1 structures are retrained with a larger budget.
   - Learning rate, batch size, weight decay, and activation can be tuned here.
   - The final model is selected from these retuned candidates.
3. **Final evaluation**
   - The official CMAPSS test split is evaluated only once for the final selected model.

### Split Protocol
- The official NASA benchmark split is preserved: `train_FD00x.txt` is used for model development and `test_FD00x.txt` plus `RUL_FD00x.txt` are used only for the final benchmark evaluation.
- Inside the official training split, the reusable PyTorch pipeline creates an internal validation partition at the **engine-unit level**, not at the rolling-window level.
- This means all windows from one engine stay together in either the search/train portion or the validation portion, which avoids leakage between highly overlapping windows from the same trajectory.
- After Stage 1 screening and Stage 2 retuning, the selected model is retrained on the full official training split and evaluated once on the official test split.
- This protocol is intended to stay comparable with papers that report results on the official C-MAPSS test split while still keeping a clean validation set for architecture search and tuning.

### Normalization Protocol
- The default normalization mode is `global_standard`.
- In this mode, a `StandardScaler` is fit on training data only and then applied consistently to validation and official test samples.
- For Stage 1 and Stage 2 model selection, the scaler is fit on the internal search/train engines only. For the final benchmark run, it is refit on the full official training split and then applied to the official test split.
- This choice keeps a common sensor reference frame across engines and avoids relying on per-engine minima and maxima from truncated test trajectories.
- The pipeline also supports `global_minmax` and `per_unit_minmax` through `--normalization-mode`, but `global_standard` is the recommended default because it is less sensitive to outliers than global min-max scaling and more deployment-consistent than per-unit min-max scaling.

### Run the full experiment
```bash
python scripts/run_torch_pipeline.py --skip-install --data-root data/CMAPSSData
```

If you want the wrapper to bootstrap dependencies first:
```bash
python scripts/run_torch_pipeline.py --data-root data/CMAPSSData
```

### Example reproducible screening + retuning run
```bash
python scripts/run_torch_pipeline.py --skip-install --data-root data/CMAPSSData --datasets FD001 --n-particles 20 --n-iter 5 --low-fidelity-epochs 15 --top-k 3 --full-tuning-epochs 100 --final-train-epochs 100 --complexity-penalty-weight 0.1
```

### Useful controls
- `--normalization-mode`
- `--min-hidden-layers` / `--max-hidden-layers`
- `--min-neurons` / `--max-neurons`
- `--activation-choices`
- `--n-particles` / `--n-iter` / `--pso-inertia` / `--pso-c1` / `--pso-c2`
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

### Outputs
Each dataset run writes:
- `outputs/torch_pytorch/torch_rul_summary.csv`
- `outputs/torch_pytorch/torch_two_stage_report_<DATASET>.json`
- `outputs/torch_pytorch/fig_torch_mlp_convergence_<DATASET>.png`
- `outputs/torch_pytorch/fig_torch_rul_prediction_<DATASET>.png`

## Experimental MILP Pruning Workflow
An alternate experimental pipeline is available for testing ANN pruning after Stage 1 architecture screening:
- Module: `src/measurement_control/torch_rul_pso_milp_pruning.py`
- Entrypoint: `scripts/run_torch_milp_pruning_pipeline.py`

This variant keeps the official NASA train/test split and uses:
- ReLU ANNs with `1` to `2` hidden layers and `10` to `100` neurons per layer
- a default `50`-particle PSO screen in this pruning branch
- Stage 1 PSO screening on the dense architecture
- Stage 2 cheap dense reference fits and MILP pruning for the top-k PSO candidates
- Stage 3 tuning of the pruned candidates only
- final selection by validation performance plus complexity penalty
- side-by-side dense vs pruned timing and official-test metrics for the selected architecture
- a default `--training-fraction 0.3` development mode so this experimental branch uses more signal than the old 10% fast screen without forcing a full-data run by default
- Stage 3 also stores per-epoch training loss and validation-MSE histories for the top-k pruned candidates
- by default the pruning runner evaluates all four CMAPSS subsets: `FD001`, `FD002`, `FD003`, and `FD004`

Important limitation:
- This is a tractable approximation of the pruning idea, not a full large-scale MIQP implementation. The pruning MILP uses a linear teacher-matching objective on a calibration subset so it can be solved with the local SciPy/HiGHS stack.
- The `training_fraction` subsamples windowed training, validation, and full-training sets only inside this pruning branch. The official NASA test split remains untouched.
- The current pruning defaults are intentionally less brittle on FD002 and FD003 than the earlier fast setup: `training_fraction=0.3`, `tuning_learning_rates=(1e-3, 5e-4, 3e-4)`, `tuning_weight_decays=(0.0, 1e-5, 1e-4)`, `full_tuning_epochs=120`, `full_tuning_patience=12`, `pruning_finetune_epochs=80`, and `pruning_finetune_patience=10`. For a final full-data run, set `--training-fraction 1.0`.
- The candidate-comparison artifacts now include a per-dataset CSV and a small plot summarizing top-k validation performance and candidate runtime.
- Exact MILP pruning is now implemented for both `1`- and `2`-hidden-layer candidates. If the solver reaches the time limit or does not return a solution, the branch falls back to a transparent global magnitude-pruning mask so the run can still complete end to end.
- To avoid benchmark leakage, the branch stores training and validation learning curves during Stage 3 and reserves the official test split for one final evaluation only.

Example run:
```bash
python scripts/run_torch_milp_pruning_pipeline.py --skip-install --data-root data/CMAPSSData --normalization-mode global_standard
```

Direct module-file execution also works now:
```bash
python src/measurement_control/torch_rul_pso_milp_pruning.py --data-root data/CMAPSSData
```

Outputs include:
- `outputs/torch_pytorch_milp_pruning/torch_milp_pruning_summary.csv`
- `outputs/torch_pytorch_milp_pruning/torch_milp_pruning_before_after_summary.csv`
- `outputs/torch_pytorch_milp_pruning/torch_milp_pruning_before_after_summary.md`
- `outputs/torch_pytorch_milp_pruning/torch_milp_pruning_report_<DATASET>.json`
- `outputs/torch_pytorch_milp_pruning/torch_milp_pruning_candidates_<DATASET>.csv`
- `outputs/torch_pytorch_milp_pruning/torch_milp_pruning_histories_<DATASET>.csv`
- `outputs/torch_pytorch_milp_pruning/fig_torch_milp_pruning_candidates_<DATASET>.png`
- `outputs/torch_pytorch_milp_pruning/fig_torch_milp_pruning_histories_<DATASET>.png`
- dense and pruned official-test prediction plots
