"""
torch_rul_pso.py
================

Two-stage CMAPSS RUL pipeline:
- Stage 1 uses PSO as a low-cost ANN structure screening method.
- Stage 2 fully retunes only the top-k Stage 1 structures.
- The official test split is evaluated only once for the final selected model.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import random
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
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


warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 12, "figure.dpi": 300})


@dataclass
class SearchSpaceConfig:
    """Discrete ANN structure search space for Stage 1 PSO."""

    min_hidden_layers: int = 1
    max_hidden_layers: int = 2
    min_neurons: int = 10
    max_neurons: int = 100
    activation_choices: Tuple[str, ...] = ("relu",)
    search_activation: bool = False

    def particle_dimension(self) -> int:
        extra = 1 if self.search_activation and len(self.activation_choices) > 1 else 0
        return 1 + self.max_hidden_layers + extra


@dataclass
class PSOConfig:
    """Inertia-based PSO settings for the low-fidelity screening stage."""

    n_particles: int = 5
    n_iter: int = 5
    inertia: float = 0.5
    c1: float = 1.5
    c2: float = 1.5


@dataclass
class ModelTrainConfig:
    """Training hyperparameters for one model fit."""

    batch_size: int
    learning_rate: float
    weight_decay: float
    epochs: int
    patience: Optional[int]
    device: str
    seed: int


@dataclass
class TrainingConfig:
    """
    Central experiment parameter block.

    The workflow is rerunnable from this single configuration object while still
    allowing Stage 1 and Stage 2 to use different budgets.
    """

    seq_len: int = 30
    validation_size: float = 0.2
    clip_max: int = 125
    random_seed: int = 42

    min_hidden_layers: int = 1
    max_hidden_layers: int = 2
    min_neurons: int = 10
    max_neurons: int = 100
    activation_choices: Tuple[str, ...] = ("relu",)
    search_activation: bool = False

    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    low_fidelity_epochs: int = 15
    low_fidelity_patience: int = 5

    top_k: int = 3
    full_tuning_epochs: int = 100
    full_tuning_patience: int = 15
    retune_repeats: int = 2
    tuning_learning_rates: Tuple[float, ...] = (1e-3, 5e-4)
    tuning_batch_sizes: Tuple[int, ...] = (128,)
    tuning_weight_decays: Tuple[float, ...] = (0.0,)
    tuning_activation_choices: Tuple[str, ...] = ("relu",)

    final_train_epochs: int = 100

    n_particles: int = 5
    n_iter: int = 5
    inertia: float = 0.5
    c1: float = 1.5
    c2: float = 1.5

    complexity_penalty_weight: float = 0.1

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "outputs/torch_pytorch"
    verbose: bool = True

    def __post_init__(self) -> None:
        self.activation_choices = tuple(self.activation_choices)
        self.tuning_learning_rates = tuple(self.tuning_learning_rates)
        self.tuning_batch_sizes = tuple(self.tuning_batch_sizes)
        self.tuning_weight_decays = tuple(self.tuning_weight_decays)
        self.tuning_activation_choices = tuple(self.tuning_activation_choices)
        if not self.tuning_activation_choices:
            self.tuning_activation_choices = self.activation_choices
        if self.min_hidden_layers > self.max_hidden_layers:
            raise ValueError("min_hidden_layers must be <= max_hidden_layers.")
        if self.min_neurons > self.max_neurons:
            raise ValueError("min_neurons must be <= max_neurons.")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1.")
        if self.retune_repeats < 1:
            raise ValueError("retune_repeats must be >= 1.")
        if self.low_fidelity_epochs < 1 or self.full_tuning_epochs < 1 or self.final_train_epochs < 1:
            raise ValueError("All epoch budgets must be >= 1.")

    def search_space(self) -> SearchSpaceConfig:
        return SearchSpaceConfig(
            min_hidden_layers=self.min_hidden_layers,
            max_hidden_layers=self.max_hidden_layers,
            min_neurons=self.min_neurons,
            max_neurons=self.max_neurons,
            activation_choices=self.activation_choices,
            search_activation=self.search_activation,
        )

    def pso_config(self) -> PSOConfig:
        return PSOConfig(
            n_particles=self.n_particles,
            n_iter=self.n_iter,
            inertia=self.inertia,
            c1=self.c1,
            c2=self.c2,
        )

    def low_fidelity_train_config(self, seed: int) -> ModelTrainConfig:
        return ModelTrainConfig(
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            epochs=self.low_fidelity_epochs,
            patience=self.low_fidelity_patience,
            device=self.device,
            seed=seed,
        )

    def final_train_config(
        self,
        learning_rate: float,
        batch_size: int,
        weight_decay: float,
        seed: int,
    ) -> ModelTrainConfig:
        return ModelTrainConfig(
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=self.final_train_epochs,
            patience=None,
            device=self.device,
            seed=seed,
        )


@dataclass(frozen=True)
class ArchitectureConfig:
    """Discrete ANN architecture used by both workflow stages."""

    hidden_layers: Tuple[int, ...]
    activation: str = "relu"

    def signature(self) -> str:
        return f"{self.activation}|{'-'.join(str(v) for v in self.hidden_layers)}"


@dataclass
class ModelConfig:
    """Model specification passed into build_model()."""

    input_dim: int
    architecture: ArchitectureConfig


@dataclass
class CandidateEvaluation:
    """Result of one low-fidelity candidate evaluation."""

    architecture: ArchitectureConfig
    validation_mse: float
    validation_mae: float
    complexity_penalty: float
    objective_score: float
    num_parameters: int
    valid: bool = True
    note: str = ""
    epochs_ran: int = 0


@dataclass
class PSOSearchResult:
    """Collected Stage 1 search output."""

    best_candidate: CandidateEvaluation
    top_candidates: List[CandidateEvaluation]
    evaluated_candidates: List[CandidateEvaluation]
    history: List[float]
    invalid_candidates: List[CandidateEvaluation] = field(default_factory=list)


@dataclass
class RetunedCandidateResult:
    """Aggregated Stage 2 tuning result for one candidate structure."""

    architecture: ArchitectureConfig
    mean_validation_mse: float
    std_validation_mse: float
    mean_validation_mae: float
    selection_score: float
    complexity_penalty: float
    num_parameters: int
    best_tuning: Dict[str, object]
    tuning_runs: List[Dict[str, object]] = field(default_factory=list)


@dataclass
class TrainSplitBundle:
    """Training-only data used for search and model selection."""

    dataset_name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_full_train: np.ndarray
    y_full_train: np.ndarray
    input_dim: int


def set_global_seed(seed: int) -> None:
    """Set all major RNGs used by the workflow."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cmapss(fd_path: Path) -> pd.DataFrame:
    """Load a CMAPSS split file with standardized column names."""

    col_names = [f"col_{i}" for i in range(1, 27)]
    return pd.read_csv(fd_path, sep=r"\s+", header=None, names=col_names)


def load_rul_targets(rul_path: Path) -> np.ndarray:
    """Load the official test-set RUL labels."""

    rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None)
    return rul_df.iloc[:, 0].to_numpy(dtype=np.float32)


def compute_rul(train_df: pd.DataFrame, clip_max: int = 125) -> pd.Series:
    """Compute row-level training RUL labels from the run-to-failure split."""

    max_cycle_by_unit = train_df.groupby("col_1")["col_2"].max()
    max_cycle_per_row = train_df["col_1"].map(max_cycle_by_unit)
    return (max_cycle_per_row - train_df["col_2"]).clip(upper=clip_max)


def select_features_by_dataset(train_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Keep the project preprocessing policy while remaining dataset-aware.

    FD002 and FD004 retain operating settings because they contain multiple
    operating regimes; FD001 and FD003 drop them.
    """

    metadata = {
        "FD001": {"keep_settings": False},
        "FD002": {"keep_settings": True},
        "FD003": {"keep_settings": False},
        "FD004": {"keep_settings": True},
    }
    keep_settings = metadata.get(dataset_name, {}).get("keep_settings", False)
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
    if not keep_settings:
        cols_to_drop.extend(["col_3", "col_4", "col_5"])
    return train_df.drop(columns=cols_to_drop, errors="ignore")


def build_sequences(
    df: pd.DataFrame,
    rul: pd.Series,
    seq_len: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build flattened rolling windows for training from the run-to-failure split."""

    feature_cols = df.columns[2:]
    sequences: List[np.ndarray] = []
    targets: List[float] = []

    for unit in df["col_1"].unique():
        unit_df = df[df["col_1"] == unit]
        unit_rul = rul[unit_df.index]
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(unit_df[feature_cols])

        for i in range(len(unit_df) - seq_len + 1):
            sequences.append(scaled[i : i + seq_len].reshape(-1))
            targets.append(unit_rul.iloc[i + seq_len - 1])

    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)


def build_official_test_samples(
    df: pd.DataFrame,
    rul_targets: np.ndarray,
    seq_len: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build one prediction sample per official test engine.

    If a test trajectory is shorter than seq_len, the first normalized row is
    repeated on the left so that every engine still yields one fixed-size sample.
    """

    feature_cols = df.columns[2:]
    units = sorted(df["col_1"].unique())
    if len(units) != len(rul_targets):
        raise ValueError(
            "Mismatch between official test units and RUL labels: "
            f"{len(units)} units vs {len(rul_targets)} labels."
        )

    samples: List[np.ndarray] = []
    targets: List[float] = []
    for idx, unit in enumerate(units):
        unit_df = df[df["col_1"] == unit]
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(unit_df[feature_cols]).astype(np.float32)
        if len(unit_df) >= seq_len:
            window = scaled[-seq_len:]
        else:
            pad_count = seq_len - len(unit_df)
            pad_block = np.repeat(scaled[:1], pad_count, axis=0)
            window = np.vstack([pad_block, scaled])
        samples.append(window.reshape(-1))
        targets.append(rul_targets[idx])

    return np.array(samples, dtype=np.float32), np.array(targets, dtype=np.float32)


def prepare_training_split(
    dataset_name: str,
    data_root: Path,
    config: TrainingConfig,
) -> TrainSplitBundle:
    """Load and preprocess the official training split and build train/val data."""

    train_path = data_root / f"train_{dataset_name}.txt"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing training file: {train_path}")

    train_df = load_cmapss(train_path)
    rul = compute_rul(train_df, clip_max=config.clip_max)
    reduced = select_features_by_dataset(train_df, dataset_name)
    X_full, y_full = build_sequences(reduced, rul, seq_len=config.seq_len)
    X_train, X_val, y_train, y_val = train_test_split(
        X_full,
        y_full,
        test_size=config.validation_size,
        random_state=config.random_seed,
        shuffle=True,
    )
    return TrainSplitBundle(
        dataset_name=dataset_name,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_full_train=X_full,
        y_full_train=y_full,
        input_dim=X_full.shape[1],
    )


def prepare_official_test_split(
    dataset_name: str,
    data_root: Path,
    config: TrainingConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess the official test split."""

    test_path = data_root / f"test_{dataset_name}.txt"
    rul_path = data_root / f"RUL_{dataset_name}.txt"
    if not test_path.exists() or not rul_path.exists():
        raise FileNotFoundError(
            f"Missing official test files for {dataset_name}: {test_path} or {rul_path}"
        )

    test_df = load_cmapss(test_path)
    rul_targets = load_rul_targets(rul_path)
    reduced = select_features_by_dataset(test_df, dataset_name)
    return build_official_test_samples(reduced, rul_targets, seq_len=config.seq_len)


def activation_from_name(name: str) -> nn.Module:
    """Map a string activation name to the corresponding PyTorch layer."""

    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class TorchMLP(nn.Module):
    """Variable-depth feed-forward regressor used for RUL prediction."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        layers: List[nn.Module] = []
        previous_dim = config.input_dim
        for hidden_units in config.architecture.hidden_layers:
            layers.append(nn.Linear(previous_dim, hidden_units))
            layers.append(activation_from_name(config.architecture.activation))
            previous_dim = hidden_units
        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def build_model(config: ModelConfig) -> TorchMLP:
    """Instantiate one ANN for the requested architecture."""

    return TorchMLP(config)


def predict_torch_model(model: nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    """Generate predictions for a NumPy design matrix."""

    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        preds = model(x_tensor).detach().cpu().numpy()
    return preds


def evaluate_model(
    model: nn.Module,
    data: Tuple[np.ndarray, np.ndarray],
    device: str,
) -> Dict[str, object]:
    """Evaluate an ANN on one dataset split."""

    X, y = data
    preds = predict_torch_model(model, X, device)
    mse = float(mean_squared_error(y, preds))
    mae = float(mean_absolute_error(y, preds))
    return {"mse": mse, "mae": mae, "predictions": preds}


def train_model(
    model: nn.Module,
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Optional[Tuple[np.ndarray, np.ndarray]],
    train_config: ModelTrainConfig,
) -> Dict[str, object]:
    """Train an ANN with optional early stopping on validation MSE."""

    set_global_seed(train_config.seed)
    model = model.to(train_config.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    criterion = nn.MSELoss()

    X_train, y_train = train_data
    x_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    dataset = TensorDataset(x_tensor, y_tensor)
    generator = torch.Generator()
    generator.manual_seed(train_config.seed)
    loader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        generator=generator,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_mse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []

    for epoch in range(train_config.epochs):
        model.train()
        running_loss = 0.0
        sample_count = 0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(train_config.device)
            y_batch = y_batch.to(train_config.device)

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * len(x_batch)
            sample_count += len(x_batch)

        train_loss = running_loss / max(sample_count, 1)
        record = {"epoch": float(epoch + 1), "train_loss": train_loss}

        if val_data is not None:
            metrics = evaluate_model(model, val_data, train_config.device)
            val_mse = float(metrics["mse"])
            record["val_mse"] = val_mse
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if (
                    train_config.patience is not None
                    and epochs_without_improvement >= train_config.patience
                ):
                    history.append(record)
                    break
        else:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

        history.append(record)

    model.load_state_dict(best_state)
    return {
        "model": model,
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "best_val_mse": None if val_data is None else best_val_mse,
        "history": history,
    }


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable ANN parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def validate_candidate(
    architecture: ArchitectureConfig,
    search_space: SearchSpaceConfig,
) -> Tuple[bool, str]:
    """Validate that a decoded candidate lies inside the configured search space."""

    depth = len(architecture.hidden_layers)
    if depth < search_space.min_hidden_layers or depth > search_space.max_hidden_layers:
        return False, f"depth {depth} outside configured bounds"
    if architecture.activation not in search_space.activation_choices:
        return False, f"unsupported activation {architecture.activation}"
    for units in architecture.hidden_layers:
        if units < search_space.min_neurons or units > search_space.max_neurons:
            return False, f"hidden size {units} outside configured bounds"
    return True, ""


def decode_particle(
    position: np.ndarray,
    search_space: SearchSpaceConfig,
) -> ArchitectureConfig:
    """Round and clip a continuous particle position into a valid architecture."""

    n_hidden_layers = int(
        np.clip(round(position[0]), search_space.min_hidden_layers, search_space.max_hidden_layers)
    )
    neuron_slots: List[int] = []
    for raw_value in position[1 : 1 + search_space.max_hidden_layers]:
        neurons = int(np.clip(round(raw_value), search_space.min_neurons, search_space.max_neurons))
        neuron_slots.append(neurons)
    hidden_layers = tuple(neuron_slots[:n_hidden_layers])

    activation = search_space.activation_choices[0]
    if search_space.search_activation and len(search_space.activation_choices) > 1:
        raw_index = position[-1]
        activation_index = int(
            np.clip(round(raw_index), 0, len(search_space.activation_choices) - 1)
        )
        activation = search_space.activation_choices[activation_index]

    return ArchitectureConfig(hidden_layers=hidden_layers, activation=activation)


def particle_bounds(search_space: SearchSpaceConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Create lower and upper vector bounds for PSO."""

    lower = [search_space.min_hidden_layers]
    upper = [search_space.max_hidden_layers]
    for _ in range(search_space.max_hidden_layers):
        lower.append(search_space.min_neurons)
        upper.append(search_space.max_neurons)
    if search_space.search_activation and len(search_space.activation_choices) > 1:
        lower.append(0)
        upper.append(len(search_space.activation_choices) - 1)
    return np.array(lower, dtype=np.float64), np.array(upper, dtype=np.float64)


def objective_low_fidelity(
    candidate_config: ArchitectureConfig,
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    input_dim: int,
    search_space: SearchSpaceConfig,
    config: TrainingConfig,
) -> CandidateEvaluation:
    """
    Evaluate one candidate under a common low-cost training budget.

    The returned objective is for structural screening only. Final model choice
    is deferred to Stage 2 retuning.
    """

    is_valid, note = validate_candidate(candidate_config, search_space)
    if not is_valid:
        return CandidateEvaluation(
            architecture=candidate_config,
            validation_mse=float("inf"),
            validation_mae=float("inf"),
            complexity_penalty=float("inf"),
            objective_score=float("inf"),
            num_parameters=0,
            valid=False,
            note=note,
        )

    model_config = ModelConfig(input_dim=input_dim, architecture=candidate_config)
    model = build_model(model_config)
    num_parameters = count_trainable_parameters(model)
    complexity_penalty = len(candidate_config.hidden_layers) + (num_parameters / 1000.0)
    low_fidelity_train_config = config.low_fidelity_train_config(seed=config.random_seed)

    try:
        outcome = train_model(model, train_data, val_data, low_fidelity_train_config)
        metrics = evaluate_model(outcome["model"], val_data, config.device)
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        return CandidateEvaluation(
            architecture=candidate_config,
            validation_mse=float("inf"),
            validation_mae=float("inf"),
            complexity_penalty=complexity_penalty,
            objective_score=float("inf"),
            num_parameters=num_parameters,
            valid=False,
            note=str(exc),
        )

    objective_score = metrics["mse"] + config.complexity_penalty_weight * complexity_penalty
    return CandidateEvaluation(
        architecture=candidate_config,
        validation_mse=float(metrics["mse"]),
        validation_mae=float(metrics["mae"]),
        complexity_penalty=float(complexity_penalty),
        objective_score=float(objective_score),
        num_parameters=num_parameters,
        valid=True,
        note="",
        epochs_ran=int(outcome["epochs_ran"]),
    )


@dataclass
class Particle:
    """PSO particle for mixed discrete ANN architecture search."""

    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray
    best_score: float


def run_pso_search(
    search_space: SearchSpaceConfig,
    pso_config: PSOConfig,
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    input_dim: int,
    config: TrainingConfig,
) -> PSOSearchResult:
    """
    Stage 1 PSO search.

    The swarm is used only to shrink the structural search space under a cheap,
    common training budget.
    """

    lower_bounds, upper_bounds = particle_bounds(search_space)
    dimension = search_space.particle_dimension()
    particles: List[Particle] = []
    for _ in range(pso_config.n_particles):
        position = np.random.uniform(lower_bounds, upper_bounds)
        particles.append(
            Particle(
                position=position,
                velocity=np.zeros(dimension, dtype=np.float64),
                best_position=position.copy(),
                best_score=float("inf"),
            )
        )

    cache: Dict[str, CandidateEvaluation] = {}
    invalid_candidates: Dict[str, CandidateEvaluation] = {}
    global_best_position: Optional[np.ndarray] = None
    global_best_score = float("inf")
    history: List[float] = []

    for iteration in range(pso_config.n_iter):
        for particle in particles:
            architecture = decode_particle(particle.position, search_space)
            signature = architecture.signature()

            if signature in cache:
                evaluation = cache[signature]
            elif signature in invalid_candidates:
                evaluation = invalid_candidates[signature]
            else:
                evaluation = objective_low_fidelity(
                    candidate_config=architecture,
                    train_data=train_data,
                    val_data=val_data,
                    input_dim=input_dim,
                    search_space=search_space,
                    config=config,
                )
                if evaluation.valid:
                    cache[signature] = evaluation
                else:
                    invalid_candidates[signature] = evaluation
                    if config.verbose:
                        print(f"[Stage 1] Invalid candidate {signature}: {evaluation.note}")

            if evaluation.objective_score < particle.best_score:
                particle.best_score = evaluation.objective_score
                particle.best_position = particle.position.copy()

            if evaluation.objective_score < global_best_score:
                global_best_score = evaluation.objective_score
                global_best_position = particle.position.copy()

            if global_best_position is None:
                global_best_position = particle.position.copy()

            r1 = np.random.rand(dimension)
            r2 = np.random.rand(dimension)
            cognitive = pso_config.c1 * r1 * (particle.best_position - particle.position)
            social = pso_config.c2 * r2 * (global_best_position - particle.position)
            particle.velocity = pso_config.inertia * particle.velocity + cognitive + social
            particle.position = np.clip(particle.position + particle.velocity, lower_bounds, upper_bounds)

        history.append(global_best_score)
        if config.verbose:
            print(
                f"[Stage 1] PSO iter {iteration + 1}/{pso_config.n_iter} | "
                f"best low-fidelity score = {global_best_score:.4f}"
            )

    ranked_candidates = sorted(
        cache.values(),
        key=lambda record: (record.objective_score, record.validation_mse, record.num_parameters),
    )
    if not ranked_candidates:
        raise RuntimeError("PSO search did not produce any valid architecture candidates.")

    return PSOSearchResult(
        best_candidate=ranked_candidates[0],
        top_candidates=ranked_candidates[: max(1, config.top_k)],
        evaluated_candidates=ranked_candidates,
        history=history,
        invalid_candidates=list(invalid_candidates.values()),
    )


def tuning_grid(config: TrainingConfig) -> Iterable[Dict[str, object]]:
    """Yield the Stage 2 hyperparameter combinations."""

    for activation, learning_rate, batch_size, weight_decay in itertools.product(
        config.tuning_activation_choices,
        config.tuning_learning_rates,
        config.tuning_batch_sizes,
        config.tuning_weight_decays,
    ):
        yield {
            "activation": activation,
            "learning_rate": float(learning_rate),
            "batch_size": int(batch_size),
            "weight_decay": float(weight_decay),
        }


def retune_top_candidates(
    candidates: Sequence[CandidateEvaluation],
    tuning_config: TrainingConfig,
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    input_dim: int,
) -> List[RetunedCandidateResult]:
    """
    Stage 2 retuning.

    Only the top-k Stage 1 structures are retrained under a larger budget and,
    optionally, repeated across multiple seeds.
    """

    results: List[RetunedCandidateResult] = []

    for candidate in candidates[: max(1, tuning_config.top_k)]:
        architecture_runs: List[Dict[str, object]] = []
        best_summary: Optional[Dict[str, object]] = None

        for option_index, option in enumerate(tuning_grid(tuning_config)):
            tuned_architecture = ArchitectureConfig(
                hidden_layers=candidate.architecture.hidden_layers,
                activation=str(option["activation"]),
            )
            model_config = ModelConfig(input_dim=input_dim, architecture=tuned_architecture)
            penalty_model = build_model(model_config)
            num_parameters = count_trainable_parameters(penalty_model)
            complexity_penalty = len(tuned_architecture.hidden_layers) + (num_parameters / 1000.0)

            run_metrics: List[Dict[str, float]] = []
            for repeat_index in range(tuning_config.retune_repeats):
                run_seed = tuning_config.random_seed + 1000 + option_index * 100 + repeat_index
                stage2_train_config = ModelTrainConfig(
                    batch_size=int(option["batch_size"]),
                    learning_rate=float(option["learning_rate"]),
                    weight_decay=float(option["weight_decay"]),
                    epochs=tuning_config.full_tuning_epochs,
                    patience=tuning_config.full_tuning_patience,
                    device=tuning_config.device,
                    seed=run_seed,
                )

                model = build_model(model_config)
                outcome = train_model(model, train_data, val_data, stage2_train_config)
                metrics = evaluate_model(outcome["model"], val_data, tuning_config.device)
                selection_score = metrics["mse"] + (
                    tuning_config.complexity_penalty_weight * complexity_penalty
                )
                run_metrics.append(
                    {
                        "seed": float(run_seed),
                        "validation_mse": float(metrics["mse"]),
                        "validation_mae": float(metrics["mae"]),
                        "selection_score": float(selection_score),
                        "epochs_ran": float(outcome["epochs_ran"]),
                    }
                )

            mean_validation_mse = float(np.mean([item["validation_mse"] for item in run_metrics]))
            mean_validation_mae = float(np.mean([item["validation_mae"] for item in run_metrics]))
            std_validation_mse = float(np.std([item["validation_mse"] for item in run_metrics]))
            mean_selection_score = float(np.mean([item["selection_score"] for item in run_metrics]))
            summary = {
                "architecture_signature": tuned_architecture.signature(),
                "activation": tuned_architecture.activation,
                "learning_rate": float(option["learning_rate"]),
                "batch_size": int(option["batch_size"]),
                "weight_decay": float(option["weight_decay"]),
                "mean_validation_mse": mean_validation_mse,
                "std_validation_mse": std_validation_mse,
                "mean_validation_mae": mean_validation_mae,
                "selection_score": mean_selection_score,
                "complexity_penalty": float(complexity_penalty),
                "num_parameters": int(num_parameters),
                "repeat_metrics": run_metrics,
            }
            architecture_runs.append(summary)

            if best_summary is None or (
                summary["selection_score"],
                summary["mean_validation_mse"],
                summary["num_parameters"],
            ) < (
                best_summary["selection_score"],
                best_summary["mean_validation_mse"],
                best_summary["num_parameters"],
            ):
                best_summary = summary

        assert best_summary is not None
        best_architecture = ArchitectureConfig(
            hidden_layers=candidate.architecture.hidden_layers,
            activation=str(best_summary["activation"]),
        )
        results.append(
            RetunedCandidateResult(
                architecture=best_architecture,
                mean_validation_mse=float(best_summary["mean_validation_mse"]),
                std_validation_mse=float(best_summary["std_validation_mse"]),
                mean_validation_mae=float(best_summary["mean_validation_mae"]),
                selection_score=float(best_summary["selection_score"]),
                complexity_penalty=float(best_summary["complexity_penalty"]),
                num_parameters=int(best_summary["num_parameters"]),
                best_tuning={
                    "learning_rate": float(best_summary["learning_rate"]),
                    "batch_size": int(best_summary["batch_size"]),
                    "weight_decay": float(best_summary["weight_decay"]),
                    "activation": str(best_summary["activation"]),
                    "repeats": int(tuning_config.retune_repeats),
                },
                tuning_runs=architecture_runs,
            )
        )

        if tuning_config.verbose:
            print(
                f"[Stage 2] {candidate.architecture.signature()} -> "
                f"{best_architecture.signature()} | "
                f"mean val MSE = {best_summary['mean_validation_mse']:.4f} | "
                f"selection score = {best_summary['selection_score']:.4f}"
            )

    return sorted(
        results,
        key=lambda record: (record.selection_score, record.mean_validation_mse, record.num_parameters),
    )


def save_mlp_convergence_plot(
    history: List[float],
    dataset_name: str,
    output_dir: Path,
) -> None:
    """Save Stage 1 best-objective convergence history."""

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(history) + 1), history, "b-o", linewidth=2)
    plt.title(f"Stage 1 PSO Screening: {dataset_name}")
    plt.xlabel("Iteration")
    plt.ylabel("Best low-fidelity score")
    plt.grid(True)
    plt.savefig(output_dir / f"fig_torch_mlp_convergence_{dataset_name}.png", bbox_inches="tight")
    plt.close()


def save_prediction_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_name: str,
    output_dir: Path,
    split_label: str = "test",
) -> None:
    """Save sorted true-vs-predicted plot for the final evaluated split."""

    sorted_idx = np.argsort(y_true)
    plt.figure(figsize=(10, 5))
    plt.plot(y_true[sorted_idx], "k-", label="True")
    plt.plot(y_pred[sorted_idx], "r--", label="Pred")
    plt.title(f"RUL Prediction ({split_label.title()}): {dataset_name}")
    plt.xlabel("Sample (sorted by true RUL)")
    plt.ylabel("RUL")
    plt.legend()
    plt.grid(True)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"fig_torch_rul_prediction_{dataset_name}.png", bbox_inches="tight")
    plt.close()


def save_experiment_report(
    dataset_name: str,
    output_dir: Path,
    report: Dict[str, object],
) -> Path:
    """Persist a small machine-readable report for one dataset run."""

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"torch_two_stage_report_{dataset_name}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def architecture_to_string(architecture: ArchitectureConfig) -> str:
    """Compact human-readable architecture string."""

    return f"{architecture.activation}:{list(architecture.hidden_layers)}"


def run_full_pipeline(
    dataset_name: str,
    data_root: Path,
    config: TrainingConfig,
) -> Dict[str, object]:
    """
    Run the full two-stage workflow for one dataset.

    The official test split is not evaluated until the final candidate has been
    selected from Stage 2 retuning.
    """

    if config.verbose:
        print(f"\n{'=' * 48}\nDataset {dataset_name}: two-stage ANN + PSO\n{'=' * 48}")

    stage0_start = time.perf_counter()
    split_bundle = prepare_training_split(dataset_name, data_root, config)
    stage0_time = time.perf_counter() - stage0_start

    stage1_start = time.perf_counter()
    search_result = run_pso_search(
        search_space=config.search_space(),
        pso_config=config.pso_config(),
        train_data=(split_bundle.X_train, split_bundle.y_train),
        val_data=(split_bundle.X_val, split_bundle.y_val),
        input_dim=split_bundle.input_dim,
        config=config,
    )
    stage1_time = time.perf_counter() - stage1_start

    stage2_start = time.perf_counter()
    retuned_candidates = retune_top_candidates(
        candidates=search_result.top_candidates,
        tuning_config=config,
        train_data=(split_bundle.X_train, split_bundle.y_train),
        val_data=(split_bundle.X_val, split_bundle.y_val),
        input_dim=split_bundle.input_dim,
    )
    stage2_time = time.perf_counter() - stage2_start
    if not retuned_candidates:
        raise RuntimeError("Stage 2 retuning produced no valid results.")

    selected_candidate = retuned_candidates[0]
    final_model_config = ModelConfig(
        input_dim=split_bundle.input_dim,
        architecture=selected_candidate.architecture,
    )
    final_train_cfg = config.final_train_config(
        learning_rate=float(selected_candidate.best_tuning["learning_rate"]),
        batch_size=int(selected_candidate.best_tuning["batch_size"]),
        weight_decay=float(selected_candidate.best_tuning["weight_decay"]),
        seed=config.random_seed + 5000,
    )

    final_train_start = time.perf_counter()
    final_model = build_model(final_model_config)
    final_outcome = train_model(
        final_model,
        train_data=(split_bundle.X_full_train, split_bundle.y_full_train),
        val_data=None,
        train_config=final_train_cfg,
    )
    final_train_time = time.perf_counter() - final_train_start

    # Touch the official test split only after model selection is complete.
    X_test, y_test = prepare_official_test_split(dataset_name, data_root, config)
    final_test_metrics = evaluate_model(final_outcome["model"], (X_test, y_test), config.device)

    output_dir = Path(config.output_dir)
    save_mlp_convergence_plot(search_result.history, dataset_name, output_dir)
    save_prediction_plot(
        y_true=y_test,
        y_pred=final_test_metrics["predictions"],
        dataset_name=dataset_name,
        output_dir=output_dir,
        split_label="official test",
    )

    report = {
        "dataset_name": dataset_name,
        "rationale": (
            "PSO is used as a computationally efficient first-stage search to "
            "identify promising ANN structures under a common low-cost training "
            "budget; final model selection is based on a second-stage retraining "
            "and tuning of only the best candidates."
        ),
        "config": asdict(config),
        "stage1": {
            "best_candidate": {
                "architecture": architecture_to_string(search_result.best_candidate.architecture),
                "low_fidelity_score": float(search_result.best_candidate.objective_score),
                "validation_mse": float(search_result.best_candidate.validation_mse),
                "validation_mae": float(search_result.best_candidate.validation_mae),
                "complexity_penalty": float(search_result.best_candidate.complexity_penalty),
                "num_parameters": int(search_result.best_candidate.num_parameters),
            },
            "top_candidates": [
                {
                    "architecture": architecture_to_string(candidate.architecture),
                    "low_fidelity_score": float(candidate.objective_score),
                    "validation_mse": float(candidate.validation_mse),
                    "validation_mae": float(candidate.validation_mae),
                    "complexity_penalty": float(candidate.complexity_penalty),
                    "num_parameters": int(candidate.num_parameters),
                }
                for candidate in search_result.top_candidates
            ],
            "evaluated_candidates": len(search_result.evaluated_candidates),
            "invalid_candidates": [
                {
                    "architecture": architecture_to_string(candidate.architecture),
                    "note": candidate.note,
                }
                for candidate in search_result.invalid_candidates
            ],
            "history": [float(value) for value in search_result.history],
            "time_seconds": float(stage1_time),
        },
        "stage2": {
            "retuned_candidates": [
                {
                    "architecture": architecture_to_string(candidate.architecture),
                    "mean_validation_mse": float(candidate.mean_validation_mse),
                    "mean_validation_mae": float(candidate.mean_validation_mae),
                    "selection_score": float(candidate.selection_score),
                    "complexity_penalty": float(candidate.complexity_penalty),
                    "num_parameters": int(candidate.num_parameters),
                    "best_tuning": candidate.best_tuning,
                }
                for candidate in retuned_candidates
            ],
            "selected_candidate": {
                "architecture": architecture_to_string(selected_candidate.architecture),
                "mean_validation_mse": float(selected_candidate.mean_validation_mse),
                "mean_validation_mae": float(selected_candidate.mean_validation_mae),
                "selection_score": float(selected_candidate.selection_score),
                "best_tuning": selected_candidate.best_tuning,
            },
            "time_seconds": float(stage2_time),
        },
        "final_model": {
            "architecture": architecture_to_string(selected_candidate.architecture),
            "best_tuning": selected_candidate.best_tuning,
            "final_train_epochs": int(final_train_cfg.epochs),
            "final_train_time_seconds": float(final_train_time),
            "official_test_mse": float(final_test_metrics["mse"]),
            "official_test_mae": float(final_test_metrics["mae"]),
        },
        "data_prep_time_seconds": float(stage0_time),
    }
    report_path = save_experiment_report(dataset_name, output_dir, report)

    if config.verbose:
        print(
            f"[Summary] Stage 1 best = {architecture_to_string(search_result.best_candidate.architecture)} "
            f"| low-fidelity score = {search_result.best_candidate.objective_score:.4f}"
        )
        print(
            f"[Summary] Stage 2 selected = {architecture_to_string(selected_candidate.architecture)} "
            f"| retuned selection score = {selected_candidate.selection_score:.4f}"
        )
        print(
            f"[Summary] Official test = MSE {final_test_metrics['mse']:.4f} | "
            f"MAE {final_test_metrics['mae']:.4f}"
        )
        print(f"[Summary] Report saved to {report_path}")

    return {
        "Dataset": dataset_name,
        "Stage1 Best Candidate": architecture_to_string(search_result.best_candidate.architecture),
        "Stage1 Low-Fidelity Score": float(search_result.best_candidate.objective_score),
        "Stage1 Validation MSE": float(search_result.best_candidate.validation_mse),
        "Stage2 Selected Architecture": architecture_to_string(selected_candidate.architecture),
        "Stage2 Retuned Score": float(selected_candidate.selection_score),
        "Stage2 Validation MSE": float(selected_candidate.mean_validation_mse),
        "Final Test MSE": float(final_test_metrics["mse"]),
        "Final Test MAE": float(final_test_metrics["mae"]),
        "Stage1 Evaluations": int(len(search_result.evaluated_candidates)),
        "Stage1 Time Sec": float(stage1_time),
        "Stage2 Time Sec": float(stage2_time),
        "Final Train Time Sec": float(final_train_time),
        "Report Path": str(report_path),
    }


def process_dataset(
    dataset_name: str,
    data_root: Path,
    config: TrainingConfig,
) -> Dict[str, object]:
    """Backward-compatible wrapper around the new full pipeline."""

    return run_full_pipeline(dataset_name, data_root, config)


def run_all_datasets(
    data_root: Path,
    config: TrainingConfig,
    datasets: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Execute the two-stage workflow for all requested datasets."""

    datasets = datasets or ["FD001", "FD002", "FD003", "FD004"]
    results: List[Dict[str, object]] = []
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in datasets:
        record = process_dataset(dataset_name, data_root, config)
        if record:
            results.append(record)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "torch_rul_summary.csv", index=False)

    if config.verbose:
        print("\nTwo-stage experiment summary:")
        print(results_df)

    return results_df


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the two-stage workflow."""

    parser = argparse.ArgumentParser(
        description=(
            "Two-stage PyTorch CMAPSS pipeline: low-fidelity PSO screening "
            "followed by selective retuning of top architectures."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/CMAPSSData"),
        help="Path to folder containing train_FD00x.txt, test_FD00x.txt, and RUL_FD00x.txt files.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["FD001", "FD002", "FD003", "FD004"],
        help="Datasets to run, e.g. FD001 FD002.",
    )
    parser.add_argument("--seq-len", type=int, default=30, help="Sequence length.")
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="Validation fraction taken from the official training split.",
    )
    parser.add_argument("--clip-max", type=int, default=125, help="Training RUL clipping threshold.")
    parser.add_argument("--seed", type=int, default=1952, help="Global random seed.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/torch_pytorch",
        help="Directory where plots, CSV summaries, and reports will be saved.",
    )
    parser.add_argument("--min-hidden-layers", type=int, default=1, help="Minimum hidden-layer count.")
    parser.add_argument("--max-hidden-layers", type=int, default=2, help="Maximum hidden-layer count.")
    parser.add_argument("--min-neurons", type=int, default=10, help="Minimum neurons per hidden layer.")
    parser.add_argument("--max-neurons", type=int, default=100, help="Maximum neurons per hidden layer.")
    parser.add_argument(
        "--activation-choices",
        nargs="+",
        default=["relu"],
        help="Activation choices available to the workflow, e.g. relu tanh.",
    )
    parser.add_argument(
        "--tuning-activation-choices",
        nargs="+",
        default=None,
        help="Stage 2 activation choices. Defaults to --activation-choices.",
    )
    parser.add_argument(
        "--search-activation",
        action="store_true",
        help="Allow Stage 1 PSO to search over activation choices when multiple are provided.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Low-fidelity batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Low-fidelity learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Low-fidelity weight decay.")
    parser.add_argument(
        "--low-fidelity-epochs",
        type=int,
        default=15,
        help="Stage 1 epochs per candidate under the screening budget.",
    )
    parser.add_argument(
        "--low-fidelity-patience",
        type=int,
        default=5,
        help="Early-stopping patience for Stage 1 screening.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of Stage 1 structures to retune.")
    parser.add_argument(
        "--full-tuning-epochs",
        type=int,
        default=100,
        help="Stage 2 epochs for each retuning run.",
    )
    parser.add_argument(
        "--full-tuning-patience",
        type=int,
        default=15,
        help="Early-stopping patience for Stage 2 retuning runs.",
    )
    parser.add_argument(
        "--final-train-epochs",
        type=int,
        default=100,
        help="Epochs used to retrain the single final selected model on all training windows.",
    )
    parser.add_argument(
        "--retune-repeats",
        type=int,
        default=2,
        help="Number of repeated Stage 2 fits per tuning setting.",
    )
    parser.add_argument(
        "--tuning-learning-rates",
        nargs="+",
        type=float,
        default=[1e-3, 5e-4],
        help="Learning rates explored during Stage 2.",
    )
    parser.add_argument(
        "--tuning-batch-sizes",
        nargs="+",
        type=int,
        default=[128],
        help="Batch sizes explored during Stage 2.",
    )
    parser.add_argument(
        "--tuning-weight-decays",
        nargs="+",
        type=float,
        default=[0.0],
        help="Weight decays explored during Stage 2.",
    )
    parser.add_argument("--n-particles", type=int, default=5, help="PSO swarm size.")
    parser.add_argument("--n-iter", type=int, default=5, help="PSO iteration count.")
    parser.add_argument("--pso-inertia", type=float, default=0.5, help="PSO inertia coefficient.")
    parser.add_argument("--pso-c1", type=float, default=1.5, help="PSO cognitive coefficient.")
    parser.add_argument("--pso-c2", type=float, default=1.5, help="PSO social coefficient.")
    parser.add_argument(
        "--complexity-penalty-weight",
        type=float,
        default=0.1,
        help="Weight applied to the architecture complexity penalty.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Deprecated alias for --full-tuning-epochs.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose logging.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    full_tuning_epochs = args.full_tuning_epochs if args.epochs is None else args.epochs
    tuning_activation_choices = (
        args.activation_choices
        if args.tuning_activation_choices is None
        else args.tuning_activation_choices
    )
    config = TrainingConfig(
        seq_len=args.seq_len,
        validation_size=args.validation_size,
        clip_max=args.clip_max,
        random_seed=args.seed,
        min_hidden_layers=args.min_hidden_layers,
        max_hidden_layers=args.max_hidden_layers,
        min_neurons=args.min_neurons,
        max_neurons=args.max_neurons,
        activation_choices=tuple(args.activation_choices),
        search_activation=args.search_activation,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        low_fidelity_epochs=args.low_fidelity_epochs,
        low_fidelity_patience=args.low_fidelity_patience,
        top_k=args.top_k,
        full_tuning_epochs=full_tuning_epochs,
        full_tuning_patience=args.full_tuning_patience,
        final_train_epochs=args.final_train_epochs,
        retune_repeats=args.retune_repeats,
        tuning_learning_rates=tuple(args.tuning_learning_rates),
        tuning_batch_sizes=tuple(args.tuning_batch_sizes),
        tuning_weight_decays=tuple(args.tuning_weight_decays),
        tuning_activation_choices=tuple(tuning_activation_choices),
        n_particles=args.n_particles,
        n_iter=args.n_iter,
        inertia=args.pso_inertia,
        c1=args.pso_c1,
        c2=args.pso_c2,
        complexity_penalty_weight=args.complexity_penalty_weight,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )

    set_global_seed(args.seed)
    run_all_datasets(
        data_root=args.data_root,
        config=config,
        datasets=args.datasets,
    )


if __name__ == "__main__":
    main()
