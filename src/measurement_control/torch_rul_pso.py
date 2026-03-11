"""
torch_rul_pso.py
================

Purpose:
- Provide a PyTorch-based replacement for the scikit-learn MLP RUL pipeline.
- Keep the same ANN structure idea (2 hidden layers) and PSO metaheuristic style
  used in code.py.
- Apply notebook-oriented optimization defaults (for example: hidden bounds
  (10, 100), PSO particles/iterations = 5/5, and high training iterations).

Design goals:
- Keep the implementation explicit and easy to audit.
- Add detailed comments to explain each processing and optimization step.
- Preserve compatibility with CMAPSS dataset conventions used in this project.
"""

from __future__ import annotations

import argparse
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# Step 1: Configure visual and warning behavior once for the entire module.
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 12, "figure.dpi": 300})


@dataclass
class TrainingConfig:
    """
    Centralized configuration for data, model training, and PSO.

    Notebook-aligned optimization defaults:
    - pso_bounds=(10, 100)
    - n_particles=5
    - n_iter=5
    - epochs=100
    """

    # Step 2: Data/sequence configuration.
    seq_len: int = 30
    test_size: float = 0.2
    random_state: int = 42

    # Step 3: PyTorch training configuration.
    batch_size: int = 128
    learning_rate: float = 1e-3
    epochs: int = 100

    # Step 4: PSO configuration (kept close to notebook defaults).
    n_particles: int = 5
    n_iter: int = 5
    pso_bounds: Tuple[int, int] = (10, 100)
    w: float = 0.5
    c1: float = 1.5
    c2: float = 1.5

    # Step 5: Runtime and output behavior.
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "outputs/torch_pytorch"
    verbose: bool = True


def set_global_seed(seed: int) -> None:
    """
    Set all major RNG seeds to make runs more reproducible.

    This does not guarantee perfectly deterministic GPU behavior in every
    low-level kernel, but it significantly reduces run-to-run randomness.
    """

    # Step 6: Seed Python's random module.
    random.seed(seed)
    # Step 7: Seed NumPy random generator.
    np.random.seed(seed)
    # Step 8: Seed PyTorch CPU and CUDA generators.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cmapss(fd_path: Path) -> pd.DataFrame:
    """
    Load one CMAPSS split file as a pandas DataFrame.

    Expected format:
    - whitespace-separated values
    - no header in raw file
    - 26 columns total
    """

    # Step 9: Build standardized column names col_1..col_26.
    col_names = [f"col_{i}" for i in range(1, 27)]
    # Step 10: Parse whitespace-delimited content into a DataFrame.
    df = pd.read_csv(fd_path, sep=r"\s+", header=None, names=col_names)
    return df


def compute_rul(train_df: pd.DataFrame, clip_max: int = 125) -> pd.Series:
    """
    Compute Remaining Useful Life (RUL) target for each row.

    RUL definition:
    RUL(row) = max_cycle_for_engine - current_cycle
    """

    # Step 11: Find the last observed cycle for each engine id.
    max_cycle_by_unit = train_df.groupby("col_1")["col_2"].max()
    # Step 12: Map each row to that engine-specific maximum cycle.
    max_cycle_per_row = train_df["col_1"].map(max_cycle_by_unit)
    # Step 13: Subtract current cycle and clip large values for stability.
    rul = (max_cycle_per_row - train_df["col_2"]).clip(upper=clip_max)
    return rul


def select_features_by_dataset(train_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Apply notebook-style feature filtering policy by dataset.

    Notebook behavior:
    - FD002 and FD004 keep operational settings columns (col_3,col_4,col_5).
    - FD001 and FD003 drop those settings columns.
    """

    # Step 14: Encode dataset-specific rule for keeping operational settings.
    metadata = {
        "FD001": {"keep_settings": False},
        "FD002": {"keep_settings": True},
        "FD003": {"keep_settings": False},
        "FD004": {"keep_settings": True},
    }
    keep_settings = metadata.get(dataset_name, {}).get("keep_settings", False)

    # Step 15: Start from the common low-information columns to drop.
    cols_to_drop = [
        "col_6",
        "col_8",
        "col_9",
        "col_10",
        "col_14",
        "col_15",
        "col_17",
        "col_20",
        "col_21",
        "col_22",
        "col_23",
    ]

    # Step 16: Drop operational settings only for single-regime datasets.
    if not keep_settings:
        cols_to_drop.extend(["col_3", "col_4", "col_5"])

    # Step 17: Return reduced feature table.
    return train_df.drop(columns=cols_to_drop, errors="ignore")


def build_sequences(
    df: pd.DataFrame,
    rul: pd.Series,
    seq_len: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build flattened rolling sequences and aligned RUL targets.

    Result:
    - X shape: (num_windows, seq_len * num_features)
    - y shape: (num_windows,)
    """

    # Step 18: Use all columns except engine id and cycle index as features.
    feature_cols = df.columns[2:]
    units = df["col_1"].unique()
    sequences: List[np.ndarray] = []
    targets: List[float] = []

    # Step 19: Process each engine trajectory separately.
    for unit in units:
        unit_df = df[df["col_1"] == unit]
        unit_rul = rul[unit_df.index]

        # Step 20: Normalize this trajectory independently.
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(unit_df[feature_cols])

        # Step 21: Build windows of length seq_len.
        for i in range(len(unit_df) - seq_len + 1):
            # Step 22: Flatten each window so MLP can consume vector input.
            seq_x = scaled[i : i + seq_len].reshape(-1)
            # Step 23: Use the RUL at the last element in the window as label.
            seq_y = unit_rul.iloc[i + seq_len - 1]
            sequences.append(seq_x)
            targets.append(seq_y)

    # Step 24: Convert lists to contiguous NumPy arrays.
    X = np.array(sequences, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    return X, y


class TorchMLP(nn.Module):
    """
    Two-hidden-layer MLP matching the architecture concept of code.py.
    """

    def __init__(self, input_dim: int, hidden_sizes: Tuple[int, int]):
        super().__init__()
        h1, h2 = hidden_sizes

        # Step 25: Define fully-connected feed-forward network.
        self.network = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 26: Produce scalar regression output per sample.
        return self.network(x).squeeze(-1)


def train_torch_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hidden_sizes: Tuple[int, int],
    config: TrainingConfig,
) -> TorchMLP:
    """
    Train one PyTorch MLP model for the provided hidden layer sizes.
    """

    # Step 27: Build model and move it to the configured device.
    model = TorchMLP(X_train.shape[1], hidden_sizes).to(config.device)
    # Step 28: Use Adam to mirror the solver family from sklearn MLP.
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    # Step 29: Convert NumPy arrays into torch tensors.
    x_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    # Step 30: Wrap tensors in a dataset/loader for mini-batch training.
    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Step 31: Iterate training epochs and update network weights.
    model.train()
    for _ in range(config.epochs):
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(config.device)
            y_batch = y_batch.to(config.device)

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

    return model


def predict_torch_model(model: TorchMLP, X: np.ndarray, device: str) -> np.ndarray:
    """
    Generate predictions for a NumPy design matrix.
    """

    # Step 32: Switch to eval mode and disable gradient tracking.
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        preds = model(x_tensor).detach().cpu().numpy()
    return preds


class Particle:
    """
    PSO particle state for 2D architecture search (hidden1, hidden2).
    """

    def __init__(self, dim: int, bounds: Tuple[int, int]):
        # Step 33: Randomly initialize position in search range.
        self.position = np.random.uniform(bounds[0], bounds[1], size=dim)
        # Step 34: Start with zero velocity.
        self.velocity = np.zeros(dim, dtype=np.float64)
        # Step 35: Track personal best architecture and objective score.
        self.best_position = self.position.copy()
        self.best_score = np.inf


def pso_optimize_torch(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    config: TrainingConfig,
) -> Tuple[Tuple[int, int], float, List[float]]:
    """
    PSO search for the best two-layer ANN hidden sizes.

    PSO dynamics here intentionally follow the same style as code.py:
    - evaluate each particle
    - update personal/global best
    - update velocity/position every loop
    """

    # Step 36: Initialize swarm for 2D hidden-size search.
    dim = 2
    particles = [Particle(dim, config.pso_bounds) for _ in range(config.n_particles)]
    global_best_position = None
    global_best_score = np.inf
    score_history: List[float] = []

    # Step 37: Iterate optimization steps.
    for iteration in range(config.n_iter):
        for particle in particles:
            # Step 38: Convert continuous position to valid integer neurons.
            hidden_sizes = tuple(
                int(max(config.pso_bounds[0], min(config.pso_bounds[1], round(val))))
                for val in particle.position
            )

            # Step 39: Train candidate architecture and compute validation MSE.
            model = train_torch_model(X_train, y_train, hidden_sizes, config)
            preds = predict_torch_model(model, X_val, config.device)
            mse = mean_squared_error(y_val, preds)

            # Step 40: Update personal best if current candidate is better.
            if mse < particle.best_score:
                particle.best_score = mse
                particle.best_position = particle.position.copy()

            # Step 41: Update global best if candidate beats swarm record.
            if mse < global_best_score:
                global_best_score = mse
                global_best_position = particle.position.copy()

            # Step 42: Apply PSO velocity and position update equations.
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            cognitive = config.c1 * r1 * (particle.best_position - particle.position)
            social = config.c2 * r2 * (global_best_position - particle.position)
            particle.velocity = config.w * particle.velocity + cognitive + social
            particle.position += particle.velocity

        # Step 43: Store global best for convergence diagnostics.
        score_history.append(global_best_score)
        if config.verbose:
            print(
                f"PSO Torch MLP Iter {iteration + 1}/{config.n_iter}: "
                f"Best MSE = {global_best_score:.4f}"
            )

    # Step 44: Convert best continuous position into final integer architecture.
    best_hidden = tuple(
        int(round(max(config.pso_bounds[0], min(config.pso_bounds[1], val))))
        for val in global_best_position
    )
    return best_hidden, global_best_score, score_history


def save_mlp_convergence_plot(
    history: List[float],
    dataset_name: str,
    output_dir: Path,
) -> None:
    """
    Save PSO convergence curve for a dataset.
    """

    # Step 45: Create output folder before writing plot files.
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(history) + 1), history, "b-o", linewidth=2)
    plt.title(f"PSO Convergence (PyTorch): {dataset_name}")
    plt.xlabel("Iteration")
    plt.ylabel("Best Validation MSE")
    plt.grid(True)
    plt.savefig(output_dir / f"fig_torch_mlp_convergence_{dataset_name}.png", bbox_inches="tight")
    plt.close()


def save_prediction_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_name: str,
    output_dir: Path,
) -> None:
    """
    Save sorted true-vs-predicted RUL plot.
    """

    # Step 46: Sort indices to make prediction trend easier to interpret.
    sorted_idx = np.argsort(y_true)
    plt.figure(figsize=(10, 5))
    plt.plot(y_true[sorted_idx], "k-", label="True")
    plt.plot(y_pred[sorted_idx], "r--", label="Pred")
    plt.title(f"RUL Prediction (PyTorch): {dataset_name}")
    plt.xlabel("Sample (sorted by true RUL)")
    plt.ylabel("RUL")
    plt.legend()
    plt.grid(True)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"fig_torch_rul_prediction_{dataset_name}.png", bbox_inches="tight")
    plt.close()


def process_dataset(
    dataset_name: str,
    data_root: Path,
    config: TrainingConfig,
) -> Dict[str, object]:
    """
    Run complete PyTorch RUL pipeline for one FD dataset.
    """

    # Step 47: Emit clear processing header for logs.
    if config.verbose:
        print(f"\n{'=' * 40}\nProcessing {dataset_name} (PyTorch)\n{'=' * 40}")

    # Step 48: Resolve CMAPSS file paths for this dataset.
    train_path = data_root / f"train_{dataset_name}.txt"
    test_path = data_root / f"test_{dataset_name}.txt"

    # Step 49: Validate required files before continuing.
    if not train_path.exists() or not test_path.exists():
        if config.verbose:
            print(f"Missing files for {dataset_name}: {train_path} or {test_path}")
        return {}

    # Step 50: Load raw training/testing files.
    train_df = load_cmapss(train_path)
    _ = load_cmapss(test_path)  # Loaded to keep parity with notebook workflow.

    # Step 51: Build RUL labels and select feature subset.
    rul = compute_rul(train_df)
    train_reduced = select_features_by_dataset(train_df, dataset_name)

    # Step 52: Convert time-series table into supervised samples.
    X, y = build_sequences(train_reduced, rul, seq_len=config.seq_len)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    # Step 53: Optimize ANN architecture with PSO.
    if config.verbose:
        print("Optimizing PyTorch MLP architecture with PSO...")
    best_hidden, best_score, history = pso_optimize_torch(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        config=config,
    )
    if config.verbose:
        print(f"Best hidden layers: {best_hidden} | Best validation MSE: {best_score:.4f}")

    # Step 54: Train final model using best architecture and evaluate metrics.
    final_model = train_torch_model(X_train, y_train, best_hidden, config)
    val_preds = predict_torch_model(final_model, X_val, config.device)
    mse = mean_squared_error(y_val, val_preds)
    mae = mean_absolute_error(y_val, val_preds)

    # Step 55: Save dataset-level diagnostic plots.
    output_dir = Path(config.output_dir)
    save_mlp_convergence_plot(history, dataset_name, output_dir)
    save_prediction_plot(y_val, val_preds, dataset_name, output_dir)

    # Step 56: Return compact summary record for this dataset.
    return {
        "Dataset": dataset_name,
        "Best Hidden Layers": str(best_hidden),
        "Validation MSE": float(mse),
        "Validation MAE": float(mae),
    }


def run_all_datasets(
    data_root: Path,
    config: TrainingConfig,
    datasets: List[str] | None = None,
) -> pd.DataFrame:
    """
    Execute the full pipeline for all requested FD datasets.
    """

    # Step 57: Default to all CMAPSS subsets when not explicitly provided.
    if datasets is None:
        datasets = ["FD001", "FD002", "FD003", "FD004"]

    # Step 58: Process each dataset and collect non-empty results.
    results: List[Dict[str, object]] = []
    for ds in datasets:
        record = process_dataset(ds, data_root, config)
        if record:
            results.append(record)

    # Step 59: Convert collected records to DataFrame and persist summary CSV.
    results_df = pd.DataFrame(results)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "torch_rul_summary.csv", index=False)

    # Step 60: Print summary table for quick review.
    if config.verbose:
        print("\nSummary of PyTorch RUL results:")
        print(results_df)

    return results_df


def parse_args() -> argparse.Namespace:
    """
    Build CLI parser so the pipeline can be executed from scripts or terminal.
    """

    parser = argparse.ArgumentParser(
        description=(
            "PyTorch CMAPSS RUL pipeline using PSO for 2-layer MLP architecture search "
            "with notebook-aligned optimization defaults."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/CMAPSSData"),
        help="Path to folder containing train_FD00x.txt and test_FD00x.txt files.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["FD001", "FD002", "FD003", "FD004"],
        help="Datasets to run, e.g. FD001 FD002.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs per model fit (notebook-style default: 1000).",
    )
    parser.add_argument(
        "--n-particles",
        type=int,
        default=5,
        help="PSO swarm size (notebook-style default: 5).",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=5,
        help="Number of PSO iterations (notebook-style default: 5).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/torch_pytorch",
        help="Directory where plots and summary CSV will be saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose logging.",
    )
    return parser.parse_args()


def main() -> None:
    """
    CLI entrypoint.
    """

    # Step 61: Parse runtime options from command line.
    args = parse_args()

    # Step 62: Build configuration object from CLI settings.
    config = TrainingConfig(
        epochs=args.epochs,
        n_particles=args.n_particles,
        n_iter=args.n_iter,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )

    # Step 63: Apply seed before any stochastic process starts.
    set_global_seed(args.seed)

    # Step 64: Run selected datasets and save summary outputs.
    run_all_datasets(
        data_root=args.data_root,
        config=config,
        datasets=args.datasets,
    )


if __name__ == "__main__":
    main()
