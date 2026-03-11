# Algorithm Comparison: `code.py` vs `measure_control_colab.ipynb`

## Overview
Both files implement two major algorithmic pipelines:
1. Remaining Useful Life (RUL) prediction using CMAPSS data + PSO-tuned MLP.
2. Fractional-Order PID (FOPID) controller tuning using PSO.

`code.py` is a fixed and more robust script version, while `measure_control_colab.ipynb` is an interactive notebook workflow.

## Shared Core Algorithms
- CMAPSS loader with 26-column schema.
- RUL target generation from maximum cycle per engine.
- Sequence/window construction for supervised learning.
- PSO-based search for MLP hidden-layer architecture.
- PSO-based search for FOPID parameters minimizing ISE.
- Visualization of optimization convergence and model/controller behavior.

## Key Differences

| Area | `code.py` | `measure_control_colab.ipynb` | Practical Impact |
|---|---|---|---|
| Execution style | Scripted (`main()`), file outputs (`savefig`) | Interactive notebook (`show`) | Script is better for reproducible runs; notebook is better for exploration. |
| MLP PSO bounds | `(10, 200)` | `(10, 100)` | Script can explore larger architectures. |
| MLP max iterations | 200 (PSO eval), 300 (final train) | 1000 (PSO eval and final train) | Notebook may train each candidate longer but slower. |
| Particle motion logic (MLP PSO) | Position update every particle step | Includes guard to skip zero-velocity particles | Notebook may skip some evaluations; script is simpler and more exhaustive per iteration. |
| Feature dropping logic | Fixed helper `drop_uninformative` list | Dataset-aware (`keep_settings`) for FD002/FD004 | Notebook preserves operating settings for multi-regime datasets. |
| FOPID numeric robustness | Uses sign-preserving fractional powers (`sign * abs^order`) + stability penalty | Uses absolute error and powers directly | Script explicitly avoids complex/unstable numeric behavior. |
| FOPID global best init | Initialized from first particle (fixed) | Starts as `None`, then updates inside loop | Script is safer against `NoneType` edge cases. |
| FOPID search bounds | Wider: `Kp up to 5`, `Ki/Kd up to 2`, `lambda/mu 0.1..1.9` | Narrower: `Kp up to 2`, `Ki/Kd up to 1`, `lambda 0.5..1.5`, `mu 0..1` | Script explores broader controller space; notebook is more conservative. |
| RUL full pipeline execution | Currently skipped in `main()` (commented intent) | Fully executed for FD001-FD004 | Notebook is complete for RUL experiments; script currently focuses on FOPID run. |

## Solvers Used
- MLP training solver: `MLPRegressor(..., solver='adam')` in both files.
- Hyperparameter solver (MLP architecture): PSO (`pso_optimize`) in both files.
- Controller parameter solver (FOPID): PSO (`pso_fopid`) in both files.
- Plant dynamics integration: explicit forward-Euler time stepping inside `simulate_fopid` and `simulate_fopid_response`.

## Time Complexity Analysis
### Notation
- `N`: number of rows in dataset.
- `U`: number of engines.
- `F`: number of selected features.
- `L`: sequence length (30).
- `S`: number of generated sequences, approximately `sum_u (n_u - L + 1)`.
- `d`: flattened input size to MLP, `d = L * F`.
- `h1, h2`: hidden layer sizes.
- `E`: MLP training iterations (`max_iter` upper bound).
- `P_mlp, I_mlp`: particles and iterations for PSO-MLP.
- `P_f, I_f`: particles and iterations for PSO-FOPID.
- `T`: simulation steps for FOPID, `T = duration/dt` (default `1000`).

### 1) Data Prep and Sequence Construction
- Loading + RUL computation: `O(N)`.
- Feature filtering/scaling: `O(N * F)`.
- Sequence generation + flattening: `O(S * L * F)`.
- Memory for sequence matrix: `O(S * L * F)`.

### 2) One MLP Training Run (Adam)
For a 2-hidden-layer MLP with 1 output:
- Per-epoch cost is approximately:
  `O(S_train * (d*h1 + h1*h2 + h2))`
- Full training:
  `O(E * S_train * (d*h1 + h1*h2 + h2))`

Validation prediction adds:
- `O(S_val * (d*h1 + h1*h2 + h2))`

### 3) PSO for MLP Architecture
Each PSO particle evaluation trains one MLP. So:
- `O(P_mlp * I_mlp * E * S_train * (d*h1 + h1*h2 + h2))`

This is usually the dominant runtime in the RUL pipeline.

### 4) FOPID Simulation + PSO
- One `simulate_fopid`: `O(T)` (constant work per step).
- PSO-FOPID optimization:
  `O(P_f * I_f * T)`  
  (`code.py` has one extra initial simulation for global-best seeding, so `+ O(T)`).

### Practical Runtime Comparison from Current Settings
- MLP solver iterations:
  - `code.py`: `max_iter=200` during PSO, `300` final train.
  - notebook: `max_iter=1000` for both.
  - Effect: notebook MLP training is roughly ~5x heavier per PSO candidate than `code.py` (same data/architecture).
- FOPID PSO calls with defaults currently used:
  - `code.py main`: `n_particles=10`, `n_iter=10` => `100` objective evaluations (+1 seed eval).
  - notebook run: `n_particles=15`, `n_iter=15` => `225` objective evaluations.
  - Effect: notebook FOPID search is about `2.25x` more simulation-heavy than `code.py` main defaults.

### Parameter-Count Impact (why bounds matter)
Total parameters for 2 hidden layers is approximately:
- `params ~ d*h1 + h1*h2 + h2` (bias terms omitted for simplicity)

Because `code.py` allows up to 200 neurons per hidden layer and notebook up to 100:
- linear term (`d*h1`) can be ~2x larger in `code.py`.
- quadratic hidden interaction (`h1*h2`) can be ~4x larger in worst case.
So `code.py` explores potentially heavier models, but with lower `max_iter`.

## Step-by-Step Commenting Added
- `code.py`: Added step-oriented comments across data loading, RUL creation, sequence building, PSO loops, FOPID simulation, and `main()` flow.
- `measure_control_colab.ipynb`: Added step-oriented comments in each executable code cell (setup, helpers, PSO-MLP, dataset loop, FOPID simulation/optimization, and plotting).

## Recommendation
- Use `code.py` when you want a more robust FOPID implementation and lower per-candidate MLP training cost.
- Use `measure_control_colab.ipynb` for interactive, full RUL experimentation across all FD datasets, accepting higher runtime.
