"""Predicción de RUL en C-MAPSS mediante MLP y PSO multifidelidad.

Características principales
---------------------------
1. PSO con tamaño de enjambre decreciente.
2. Presupuesto acumulado de épocas idéntico para todas las arquitecturas de
   una misma etapa.
3. Caché persistente por arquitectura y por número exacto de épocas.
4. Warm start de pesos y del estado de AdamW sin utilizar estados entrenados
   por encima del presupuesto de la etapa actual.
5. Reutilización de arquitecturas repetidas sin volver a entrenarlas dentro
   de la misma etapa.
6. Penalización cuadrática para arquitecturas que exceden un presupuesto de
   parámetros configurable.
7. Entrenamiento con salida lineal y recorte de RUL solamente al predecir.
8. Objetivo y predicciones limitados coherentemente a RUL_MAX en validación
   y test.
9. Gráficas de diagnóstico apropiadas para regresión.

La caché de esta versión es incompatible deliberadamente con versiones
anteriores para evitar reutilizar evaluaciones realizadas con presupuestos
no comparables.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim


# =============================================================================
# 1. Configuración general
# =============================================================================

DATA_DIR = Path("data/CMAPSSData")
DATASETS = ("FD001",)  # Cambiar por ("FD001", "FD002", "FD003", "FD004")

SEQ_LEN = 30
RUL_MAX = 125
VALIDATION_SIZE = 0.20
RANDOM_STATE = 42

# El número indica cuántas partículas quedan en cada etapa.
PARTICLE_SCHEDULE = (100, 50, 25, 12, 6, 4)

# Etapa 1: 100 épocas totales; etapa 2: 200; ...; etapa 6: 600.
SEARCH_EPOCHS_PER_STAGE = 100

# Todas las finalistas terminan con exactamente este total acumulado.
FINAL_TOTAL_EPOCHS = 2000

HIDDEN_BOUNDS = (10, 100)
SEARCH_LEARNING_RATE = 1e-3

# Penalización estructural:
# fitness = MSE + lambda * max(0, (params - objetivo) / objetivo) ** potencia
TARGET_PARAMETERS = 20_000
COMPLEXITY_LAMBDA = 100.0
COMPLEXITY_POWER = 2.0

# Puede fijarse, por ejemplo, en 30_000 para rechazar modelos mayores.
# None deja activa solamente la penalización suave.
MAX_ALLOWED_PARAMETERS: Optional[int] = None

CACHE_VERSION = 2
CACHE_DIR = Path("pso_cache")

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 12, "figure.dpi": 180})


# =============================================================================
# 2. Reproducibilidad
# =============================================================================


def set_global_seed(seed: int) -> None:
    """Configura semillas reproducibles para NumPy y PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # Compatibilidad con versiones antiguas de PyTorch.
        torch.use_deterministic_algorithms(True)


def architecture_seed(base_seed: int, hidden_sizes: Tuple[int, ...]) -> int:
    """Genera una semilla estable a partir de una arquitectura."""
    value = int(base_seed)
    for index, width in enumerate(hidden_sizes, start=1):
        value = (value * 1_000_003 + index * 10_007 + int(width)) % 2_147_483_647
    return value


# =============================================================================
# 3. Modelo MLP
# =============================================================================


class TorchMLPRegressor(nn.Module):
    """MLP de regresión con activaciones ELU y salida lineal."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int = 1,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        in_features = int(input_size)

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, int(hidden_size)))
            layers.append(nn.ELU(alpha=1.0))
            in_features = int(hidden_size)

        layers.append(nn.Linear(in_features, int(output_size)))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No se usa clamp durante el entrenamiento. Un clamp aquí puede dejar
        # gradiente nulo cuando la salida inicial es negativa.
        return self.network(x)


class MLPRegressor:
    """Envoltorio simple para entrenamiento, predicción y warm start."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        learning_rate: float = SEARCH_LEARNING_RATE,
    ) -> None:
        self.input_size = int(input_size)
        self.hidden_sizes = tuple(int(value) for value in hidden_sizes)
        self.learning_rate = float(learning_rate)

        self.model = TorchMLPRegressor(
            input_size=self.input_size,
            hidden_sizes=self.hidden_sizes,
        )
        self.criterion = nn.SmoothL1Loss(beta=1.0)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
        )

    def get_weights(self) -> Dict[str, torch.Tensor]:
        return copy.deepcopy(self.model.state_dict())

    def load_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state_dict)

    def get_optimizer_state(self) -> Dict[str, Any]:
        return copy.deepcopy(self.optimizer.state_dict())

    def load_optimizer_state(self, state_dict: Dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        verbose: bool = False,
        print_every: int = 200,
    ) -> Dict[str, float]:
        """Entrena durante un número adicional de épocas."""
        epochs = int(epochs)
        if epochs < 0:
            raise ValueError("epochs no puede ser negativo.")
        if epochs == 0:
            return {"elapsed": 0.0, "final_loss": float("nan")}

        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        y_tensor = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)

        self.model.train()
        start_time = time.time()
        final_loss = float("nan")

        for epoch in range(1, epochs + 1):
            self.optimizer.zero_grad(set_to_none=True)
            predictions = self.model(X_tensor)
            loss = self.criterion(predictions, y_tensor)
            loss.backward()
            self.optimizer.step()
            final_loss = float(loss.item())

            if verbose and (epoch % print_every == 0 or epoch == epochs):
                print(
                    f"      [Epoch adicional {epoch}/{epochs}] "
                    f"Loss: {final_loss:.4f}"
                )

        elapsed = time.time() - start_time
        if verbose:
            print(
                "    --> Entrenamiento MLP finalizado en "
                f"{elapsed:.2f}s (Loss final: {final_loss:.4f})"
            )

        return {"elapsed": elapsed, "final_loss": final_loss}

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        self.model.eval()

        with torch.no_grad():
            predictions = self.model(X_tensor)

        return predictions.cpu().numpy().reshape(-1)

    def predict(
        self,
        X: np.ndarray,
        clip_min: Optional[float] = 0.0,
        clip_max: Optional[float] = RUL_MAX,
    ) -> np.ndarray:
        predictions = self.predict_raw(X)

        if clip_min is None and clip_max is None:
            return predictions

        lower = -np.inf if clip_min is None else float(clip_min)
        upper = np.inf if clip_max is None else float(clip_max)
        return np.clip(predictions, lower, upper)


# =============================================================================
# 4. Lectura y preparación de C-MAPSS
# =============================================================================


def load_cmapss(path: Path | str) -> pd.DataFrame:
    column_names = [f"col_{index}" for index in range(1, 27)]
    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=column_names,
    )


def load_cmapss_rul(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["rul"],
    )


def compute_rul(
    dataframe: pd.DataFrame,
    base_rul: Optional[pd.Series] = None,
    clip_max: Optional[int] = RUL_MAX,
) -> pd.Series:
    """Calcula RUL por fila, opcionalmente con RUL residual de test."""
    maximum_cycle = dataframe.groupby("col_1")["col_2"].max()

    if base_rul is not None:
        base_rul = base_rul.copy()
        base_rul.index = base_rul.index.astype(maximum_cycle.index.dtype)
        maximum_cycle = maximum_cycle.add(base_rul, fill_value=np.nan)

    mapped_maximum = dataframe["col_1"].map(maximum_cycle)
    rul = mapped_maximum - dataframe["col_2"]

    if clip_max is not None:
        rul = rul.clip(lower=0, upper=int(clip_max))
    else:
        rul = rul.clip(lower=0)

    return rul.astype(float)


def select_non_flat_features(
    dataframe: pd.DataFrame,
    threshold: float = 1e-5,
) -> List[str]:
    feature_columns = list(dataframe.columns[2:])
    standard_deviations = dataframe[feature_columns].std()
    selected = standard_deviations[
        standard_deviations > float(threshold)
    ].index.tolist()

    if not selected:
        raise ValueError("No quedaron variables C-MAPSS no constantes.")

    return selected


def build_sequences(
    dataframe: pd.DataFrame,
    rul: pd.Series,
    seq_len: int = SEQ_LEN,
    scaler: Optional[MinMaxScaler] = None,
    only_last: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construye ventanas aplanadas y conserva el ID de cada unidad."""
    feature_columns = list(dataframe.columns[2:])
    units = dataframe["col_1"].drop_duplicates().to_numpy()

    sequences: List[np.ndarray] = []
    targets: List[float] = []
    sequence_units: List[int] = []

    if scaler is None:
        scaler = MinMaxScaler().fit(dataframe[feature_columns])

    for unit in units:
        unit_dataframe = dataframe[dataframe["col_1"] == unit]
        unit_rul = rul.loc[unit_dataframe.index]
        scaled = scaler.transform(unit_dataframe[feature_columns])

        if only_last:
            if len(unit_dataframe) < seq_len:
                continue

            start = len(unit_dataframe) - seq_len
            sequence = scaled[start : start + seq_len].reshape(-1)
            target = float(unit_rul.iloc[start + seq_len - 1])

            sequences.append(sequence)
            targets.append(target)
            sequence_units.append(int(unit))
            continue

        for start in range(len(unit_dataframe) - seq_len + 1):
            sequence = scaled[start : start + seq_len].reshape(-1)
            target = float(unit_rul.iloc[start + seq_len - 1])

            sequences.append(sequence)
            targets.append(target)
            sequence_units.append(int(unit))

    if not sequences:
        raise ValueError(
            "No se pudieron construir secuencias. Revise SEQ_LEN y los datos."
        )

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(sequence_units, dtype=np.int64),
    )


# =============================================================================
# 5. Métricas y complejidad
# =============================================================================


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def count_mlp_parameters(
    input_size: int,
    hidden_sizes: Sequence[int],
    output_size: int = 1,
) -> int:
    layer_sizes = (
        int(input_size),
        *(int(value) for value in hidden_sizes),
        int(output_size),
    )
    return int(
        sum(
            (in_features + 1) * out_features
            for in_features, out_features in zip(
                layer_sizes[:-1],
                layer_sizes[1:],
            )
        )
    )


def calculate_regularized_score(
    mse: float,
    n_parameters: int,
    target_parameters: int = TARGET_PARAMETERS,
    complexity_lambda: float = COMPLEXITY_LAMBDA,
    complexity_power: float = COMPLEXITY_POWER,
    max_allowed_parameters: Optional[int] = MAX_ALLOWED_PARAMETERS,
) -> Tuple[float, float, float]:
    """Calcula fitness y penaliza solamente el exceso de complejidad."""
    if target_parameters <= 0:
        raise ValueError("target_parameters debe ser positivo.")
    if complexity_lambda < 0:
        raise ValueError("complexity_lambda no puede ser negativo.")
    if complexity_power <= 0:
        raise ValueError("complexity_power debe ser positivo.")

    excess_ratio = max(
        0.0,
        (int(n_parameters) - int(target_parameters))
        / float(target_parameters),
    )

    if (
        max_allowed_parameters is not None
        and n_parameters > max_allowed_parameters
    ):
        return float("inf"), float("inf"), float(excess_ratio)

    penalty = float(
        complexity_lambda * (excess_ratio ** complexity_power)
    )
    fitness = float(mse + penalty)
    return fitness, penalty, float(excess_ratio)


# =============================================================================
# 6. Caché justa por arquitectura y presupuesto exacto
# =============================================================================


ArchitectureCache = Dict[Tuple[int, ...], Dict[str, Any]]


def load_architecture_cache(
    cache_path: Optional[Path | str],
    input_size: int,
    cache_id: str,
) -> ArchitectureCache:
    """Carga solamente una caché compatible con esta versión y estos datos."""
    if cache_path is None:
        return {}

    path = Path(cache_path)
    if not path.exists():
        return {}

    try:
        try:
            package = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            package = torch.load(path, map_location="cpu")
    except Exception as exc:
        print(f"Aviso: no se pudo cargar la caché {path}: {exc}")
        return {}

    if package.get("version") != CACHE_VERSION:
        print(
            "Aviso: se ignoró una caché de una versión anterior para evitar "
            "comparaciones con presupuestos desiguales."
        )
        return {}

    if int(package.get("input_size", -1)) != int(input_size):
        print("Aviso: la caché no coincide con el tamaño de entrada.")
        return {}

    if package.get("cache_id") != cache_id:
        print("Aviso: la caché pertenece a otra preparación de datos.")
        return {}

    raw_entries = package.get("entries", {})
    cache: ArchitectureCache = {}

    for raw_architecture, entry in raw_entries.items():
        if not isinstance(entry, dict):
            continue

        architecture = tuple(int(value) for value in raw_architecture)
        snapshots = entry.get("snapshots", {})
        normalized_snapshots: Dict[int, Dict[str, Any]] = {}

        for raw_epoch, snapshot in snapshots.items():
            if isinstance(snapshot, dict):
                normalized_snapshots[int(raw_epoch)] = snapshot

        cache[architecture] = {
            "snapshots": normalized_snapshots,
            "evaluations": int(entry.get("evaluations", 0)),
        }

    print(
        f"Caché compatible cargada: {len(cache)} arquitecturas desde {path}."
    )
    return cache


def save_architecture_cache(
    cache_path: Optional[Path | str],
    cache: ArchitectureCache,
    input_size: int,
    cache_id: str,
) -> None:
    """Guarda la caché mediante reemplazo atómico."""
    if cache_path is None:
        return

    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    package = {
        "version": CACHE_VERSION,
        "input_size": int(input_size),
        "cache_id": cache_id,
        "entries": cache,
    }

    torch.save(package, temporary_path)
    temporary_path.replace(path)


def _snapshot_at_or_below(
    entry: Dict[str, Any],
    target_total_epochs: int,
) -> Tuple[int, Optional[Dict[str, Any]]]:
    snapshots = entry.get("snapshots", {})
    eligible_epochs = [
        int(epoch)
        for epoch in snapshots
        if int(epoch) <= int(target_total_epochs)
    ]

    if not eligible_epochs:
        return 0, None

    selected_epoch = max(eligible_epochs)
    return selected_epoch, snapshots[selected_epoch]


def evaluate_architecture(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    hidden_sizes: Tuple[int, ...],
    target_total_epochs: int,
    architecture_cache: ArchitectureCache,
    learning_rate: float,
    random_seed: int,
    target_parameters: int,
    complexity_lambda: float,
    complexity_power: float,
    max_allowed_parameters: Optional[int],
    prediction_clip_min: Optional[float] = 0.0,
    prediction_clip_max: Optional[float] = RUL_MAX,
    verbose: bool = False,
    print_every: int = 500,
) -> Tuple[MLPRegressor, Dict[str, Any]]:
    """Evalúa una arquitectura con presupuesto total exacto.

    Nunca se carga un estado entrenado durante más épocas que el presupuesto
    solicitado. Si existe un estado inferior, se continúa desde ese punto.
    Si existe exactamente el presupuesto solicitado, se reutiliza sin volver
    a entrenar.
    """
    target_total_epochs = int(target_total_epochs)
    if target_total_epochs < 1:
        raise ValueError("target_total_epochs debe ser positivo.")

    hidden_sizes = tuple(int(value) for value in hidden_sizes)
    seed = architecture_seed(random_seed, hidden_sizes)
    set_global_seed(seed)

    entry = architecture_cache.setdefault(
        hidden_sizes,
        {"snapshots": {}, "evaluations": 0},
    )
    snapshots: Dict[int, Dict[str, Any]] = entry.setdefault("snapshots", {})

    exact_cache_hit = target_total_epochs in snapshots
    start_epoch, start_snapshot = _snapshot_at_or_below(
        entry,
        target_total_epochs,
    )

    model = MLPRegressor(
        input_size=X_train.shape[1],
        hidden_sizes=hidden_sizes,
        learning_rate=learning_rate,
    )

    optimizer_restored = False
    if start_snapshot is not None:
        model.load_weights(start_snapshot["weights"])
        optimizer_state = start_snapshot.get("optimizer_state")
        if optimizer_state is not None:
            try:
                model.load_optimizer_state(optimizer_state)
                optimizer_restored = True
            except (ValueError, RuntimeError, KeyError) as exc:
                print(
                    f"Aviso: no se restauró AdamW para {hidden_sizes}: {exc}. "
                    "Se conservaron los pesos y se reinició el optimizador."
                )

    epochs_to_train = target_total_epochs - start_epoch
    training_information = model.fit(
        X_train,
        y_train,
        epochs=epochs_to_train,
        verbose=verbose,
        print_every=print_every,
    )

    if exact_cache_hit and epochs_to_train == 0:
        exact_snapshot = snapshots[target_total_epochs]
        validation_mse = float(exact_snapshot["validation_mse"])
        validation_mae = float(exact_snapshot["validation_mae"])
    else:
        validation_predictions = model.predict(
            X_val,
            clip_min=prediction_clip_min,
            clip_max=prediction_clip_max,
        )
        metrics = regression_metrics(y_val, validation_predictions)
        validation_mse = metrics["mse"]
        validation_mae = metrics["mae"]

        snapshots[target_total_epochs] = {
            "weights": model.get_weights(),
            "optimizer_state": model.get_optimizer_state(),
            "validation_mse": validation_mse,
            "validation_mae": validation_mae,
            "training_loss": training_information["final_loss"],
            "seed": seed,
        }

    n_parameters = count_mlp_parameters(
        input_size=X_train.shape[1],
        hidden_sizes=hidden_sizes,
    )
    fitness, penalty, excess_ratio = calculate_regularized_score(
        mse=validation_mse,
        n_parameters=n_parameters,
        target_parameters=target_parameters,
        complexity_lambda=complexity_lambda,
        complexity_power=complexity_power,
        max_allowed_parameters=max_allowed_parameters,
    )

    entry["evaluations"] = int(entry.get("evaluations", 0)) + 1

    status = "cold"
    if exact_cache_hit and epochs_to_train == 0:
        status = "cache-exacta"
    elif start_epoch > 0:
        status = "warm"

    result = {
        "hidden_sizes": hidden_sizes,
        "mse": validation_mse,
        "mae": validation_mae,
        "fitness": fitness,
        "penalty": penalty,
        "excess_ratio": excess_ratio,
        "n_parameters": n_parameters,
        "status": status,
        "start_epoch": start_epoch,
        "target_epoch": target_total_epochs,
        "trained_now": epochs_to_train,
        "optimizer_restored": optimizer_restored,
        "exact_cache_hit": exact_cache_hit,
        "elapsed_training": training_information["elapsed"],
    }
    return model, result


# =============================================================================
# 7. PSO multifidelidad con enjambre decreciente
# =============================================================================


class Particle:
    def __init__(
        self,
        dimension: int,
        bounds: Tuple[int, int],
        rng: np.random.Generator,
    ) -> None:
        self.position = rng.uniform(bounds[0], bounds[1], size=dimension)
        self.velocity = np.zeros(dimension, dtype=float)

        self.best_position = self.position.copy()
        self.best_fitness = float("inf")

        self.last_hidden_sizes: Optional[Tuple[int, ...]] = None
        self.last_mse = float("inf")
        self.last_fitness = float("inf")


def decode_hidden_sizes(
    position: np.ndarray,
    bounds: Tuple[int, int],
) -> Tuple[int, ...]:
    clipped = np.clip(position, bounds[0], bounds[1])
    return tuple(int(round(value)) for value in clipped)


def select_unique_elites(
    particles: Sequence[Particle],
    target_count: int,
    dimension: int,
    bounds: Tuple[int, int],
    rng: np.random.Generator,
) -> List[Particle]:
    """Selecciona élites sin repetir la arquitectura discretizada."""
    selected: List[Particle] = []
    seen: set[Tuple[int, ...]] = set()

    for particle in sorted(particles, key=lambda item: item.last_fitness):
        architecture = particle.last_hidden_sizes
        if architecture is None or architecture in seen:
            continue

        selected.append(particle)
        seen.add(architecture)
        if len(selected) == target_count:
            return selected

    # Normalmente no se ejecuta. Mantiene el tamaño solicitado si varias
    # partículas convergen exactamente a la misma arquitectura.
    while len(selected) < target_count:
        selected.append(Particle(dimension, bounds, rng))

    return selected


def make_decoded_architecture_unique(
    particle: Particle,
    seen_architectures: set[Tuple[int, ...]],
    bounds: Tuple[int, int],
    rng: np.random.Generator,
    max_attempts: int = 100,
) -> Tuple[Tuple[int, ...], bool]:
    """Evita gastar partículas distintas en el mismo punto discretizado."""
    architecture = decode_hidden_sizes(particle.position, bounds)
    if architecture not in seen_architectures:
        return architecture, False

    original_position = particle.position.copy()

    for attempt in range(1, max_attempts + 1):
        scale = 0.5 + 0.10 * attempt
        candidate = original_position + rng.normal(
            loc=0.0,
            scale=scale,
            size=original_position.shape,
        )
        candidate = np.clip(candidate, bounds[0], bounds[1])
        architecture = decode_hidden_sizes(candidate, bounds)

        if architecture not in seen_architectures:
            particle.position = candidate
            return architecture, True

    # Respaldo aleatorio si toda la zona inmediata ya está ocupada.
    for _ in range(max_attempts):
        candidate = rng.uniform(
            bounds[0],
            bounds[1],
            size=original_position.shape,
        )
        architecture = decode_hidden_sizes(candidate, bounds)
        if architecture not in seen_architectures:
            particle.position = candidate
            return architecture, True

    # El espacio discreto es amplio; esta rama solamente protege casos límite.
    particle.position = original_position
    return decode_hidden_sizes(original_position, bounds), False


@dataclass
class OptimizationResult:
    best_hidden_sizes: Tuple[int, ...]
    best_mse: float
    best_mae: float
    best_fitness: float
    best_penalty: float
    best_parameter_count: int
    best_weights: Dict[str, torch.Tensor]
    fitness_history: List[float]
    mse_history: List[float]


def pso_optimize(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    particle_schedule: Sequence[int] = PARTICLE_SCHEDULE,
    search_epochs_per_stage: int = SEARCH_EPOCHS_PER_STAGE,
    final_total_epochs: int = FINAL_TOTAL_EPOCHS,
    bounds: Tuple[int, int] = HIDDEN_BOUNDS,
    learning_rate: float = SEARCH_LEARNING_RATE,
    target_parameters: int = TARGET_PARAMETERS,
    complexity_lambda: float = COMPLEXITY_LAMBDA,
    complexity_power: float = COMPLEXITY_POWER,
    max_allowed_parameters: Optional[int] = MAX_ALLOWED_PARAMETERS,
    cache_path: Optional[Path | str] = None,
    cache_id: str = "default",
    random_state: int = RANDOM_STATE,
) -> OptimizationResult:
    """Ejecuta PSO decreciente con comparación justa por etapa."""
    schedule = tuple(int(value) for value in particle_schedule)

    if not schedule:
        raise ValueError("particle_schedule no puede estar vacío.")
    if any(value < 1 for value in schedule):
        raise ValueError("Todos los tamaños del enjambre deben ser positivos.")
    if any(
        next_value >= current_value
        for current_value, next_value in zip(schedule, schedule[1:])
    ):
        raise ValueError("particle_schedule debe ser estrictamente decreciente.")
    if search_epochs_per_stage < 1:
        raise ValueError("search_epochs_per_stage debe ser positivo.")

    last_search_budget = len(schedule) * int(search_epochs_per_stage)
    if final_total_epochs < last_search_budget:
        raise ValueError(
            "final_total_epochs debe ser al menos igual al presupuesto de "
            "la última etapa de búsqueda."
        )

    set_global_seed(random_state)
    rng = np.random.default_rng(random_state)

    dimension = 2
    particles = [
        Particle(dimension, bounds, rng)
        for _ in range(schedule[0])
    ]

    cache = load_architecture_cache(
        cache_path=cache_path,
        input_size=X_train.shape[1],
        cache_id=cache_id,
    )

    inertia = 0.5
    cognitive = 1.5
    social = 1.5

    global_best_position: Optional[np.ndarray] = None
    fitness_history: List[float] = []
    mse_history: List[float] = []

    print("Función objetivo regularizada:")
    print(
        "  fitness = MSE + λ × "
        "max(0, (parámetros - objetivo) / objetivo)^potencia"
    )
    print(
        f"  objetivo={target_parameters:,}, λ={complexity_lambda:g}, "
        f"potencia={complexity_power:g}, límite duro="
        f"{max_allowed_parameters if max_allowed_parameters is not None else 'ninguno'}"
    )

    optimization_start = time.time()

    for stage_index, target_particle_count in enumerate(schedule):
        stage_start = time.time()
        stage_target_epochs = (
            stage_index + 1
        ) * int(search_epochs_per_stage)

        if stage_index > 0:
            particles = select_unique_elites(
                particles=particles,
                target_count=target_particle_count,
                dimension=dimension,
                bounds=bounds,
                rng=rng,
            )

        current_particle_count = len(particles)
        print(
            f"\n--- PSO Etapa {stage_index + 1}/{len(schedule)} | "
            f"Partículas: {current_particle_count} | "
            f"Presupuesto total justo: {stage_target_epochs} épocas ---"
        )

        # Reutiliza resultados solamente si una arquitectura se repite dentro
        # de esta misma etapa. Normalmente make_decoded_architecture_unique
        # evita la repetición, pero se conserva esta protección.
        stage_results: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        stage_models: Dict[Tuple[int, ...], MLPRegressor] = {}
        seen_architectures: set[Tuple[int, ...]] = set()

        previous_global_best = (
            None
            if global_best_position is None
            else global_best_position.copy()
        )

        for particle_index, particle in enumerate(particles):
            # Se mantiene inmóvil la primera élite de cada etapa para garantizar
            # que la mejor arquitectura previa reciba warm start al presupuesto
            # superior. Las demás partículas actualizan su posición por PSO.
            if (
                stage_index > 0
                and particle_index > 0
                and previous_global_best is not None
            ):
                r1 = rng.random(dimension)
                r2 = rng.random(dimension)
                particle.velocity = (
                    inertia * particle.velocity
                    + cognitive
                    * r1
                    * (particle.best_position - particle.position)
                    + social
                    * r2
                    * (previous_global_best - particle.position)
                )
                particle.position = np.clip(
                    particle.position + particle.velocity,
                    bounds[0],
                    bounds[1],
                )

            hidden_sizes, relocated = make_decoded_architecture_unique(
                particle=particle,
                seen_architectures=seen_architectures,
                bounds=bounds,
                rng=rng,
            )
            seen_architectures.add(hidden_sizes)

            particle_start = time.time()
            duplicate_in_stage = hidden_sizes in stage_results

            if duplicate_in_stage:
                result = copy.deepcopy(stage_results[hidden_sizes])
                model = stage_models[hidden_sizes]
                result["status"] = "duplicada-reutilizada"
                result["trained_now"] = 0
            else:
                model, result = evaluate_architecture(
                    X_train=X_train,
                    X_val=X_val,
                    y_train=y_train,
                    y_val=y_val,
                    hidden_sizes=hidden_sizes,
                    target_total_epochs=stage_target_epochs,
                    architecture_cache=cache,
                    learning_rate=learning_rate,
                    random_seed=random_state,
                    target_parameters=target_parameters,
                    complexity_lambda=complexity_lambda,
                    complexity_power=complexity_power,
                    max_allowed_parameters=max_allowed_parameters,
                    prediction_clip_min=0.0,
                    prediction_clip_max=RUL_MAX,
                )
                stage_results[hidden_sizes] = copy.deepcopy(result)
                stage_models[hidden_sizes] = model

            elapsed = time.time() - particle_start

            particle.last_hidden_sizes = hidden_sizes
            particle.last_mse = float(result["mse"])
            particle.last_fitness = float(result["fitness"])

            # El mejor personal conserva información direccional. Debido a que
            # el presupuesto crece por etapa, el mejor global utilizado para el
            # siguiente movimiento se redefine exclusivamente con la etapa
            # actual, evitando comparar directamente etapas distintas.
            if particle.last_fitness < particle.best_fitness:
                particle.best_fitness = particle.last_fitness
                particle.best_position = particle.position.copy()

            relocation_text = ", reubicada" if relocated else ""
            optimizer_text = (
                ", AdamW restaurado"
                if result["optimizer_restored"]
                else ""
            )
            print(
                f"  [Partícula {particle_index + 1}/{current_particle_count}] "
                f"Hidden: {hidden_sizes} | "
                f"MSE: {result['mse']:.2f} | "
                f"Penalización: {result['penalty']:.2f} | "
                f"Fitness: {result['fitness']:.2f} | "
                f"Params: {result['n_parameters']:,} | "
                f"{result['status']}{optimizer_text}{relocation_text} | "
                f"inicio: {result['start_epoch']} | "
                f"entrenadas ahora: {result['trained_now']} | "
                f"total: {result['target_epoch']} | "
                f"Tiempo: {elapsed:.2f}s"
            )

        stage_best_particle = min(
            particles,
            key=lambda item: item.last_fitness,
        )
        global_best_position = stage_best_particle.position.copy()

        fitness_history.append(stage_best_particle.last_fitness)
        mse_history.append(stage_best_particle.last_mse)

        save_architecture_cache(
            cache_path=cache_path,
            cache=cache,
            input_size=X_train.shape[1],
            cache_id=cache_id,
        )

        stage_elapsed = time.time() - stage_start
        print(
            f"  >> Fin Etapa {stage_index + 1} | "
            f"Mejor arquitectura: {stage_best_particle.last_hidden_sizes} | "
            f"MSE: {stage_best_particle.last_mse:.2f} | "
            f"Fitness: {stage_best_particle.last_fitness:.2f} | "
            f"Arquitecturas en caché: {len(cache)} | "
            f"Tiempo: {stage_elapsed:.2f}s"
        )

        if stage_index + 1 < len(schedule):
            print(
                f"  >> Selección elitista única: sobreviven "
                f"{schedule[stage_index + 1]} partículas."
            )

    # Las partículas de la última etapa ya son únicas. Se conserva una
    # protección adicional por si el espacio se agotara durante la reubicación.
    finalists: List[Tuple[int, ...]] = []
    for particle in sorted(particles, key=lambda item: item.last_fitness):
        architecture = particle.last_hidden_sizes
        if architecture is not None and architecture not in finalists:
            finalists.append(architecture)

    finalist_count = schedule[-1]
    if len(finalists) < finalist_count:
        last_budget = len(schedule) * int(search_epochs_per_stage)
        candidates_from_cache: List[Tuple[float, Tuple[int, ...]]] = []

        for architecture, entry in cache.items():
            snapshot = entry.get("snapshots", {}).get(last_budget)
            if snapshot is None or architecture in finalists:
                continue

            n_parameters = count_mlp_parameters(
                X_train.shape[1],
                architecture,
            )
            fitness, _, _ = calculate_regularized_score(
                mse=float(snapshot["validation_mse"]),
                n_parameters=n_parameters,
                target_parameters=target_parameters,
                complexity_lambda=complexity_lambda,
                complexity_power=complexity_power,
                max_allowed_parameters=max_allowed_parameters,
            )
            candidates_from_cache.append((fitness, architecture))

        for _, architecture in sorted(candidates_from_cache):
            finalists.append(architecture)
            if len(finalists) == finalist_count:
                break

    finalists = finalists[:finalist_count]
    if len(finalists) != finalist_count:
        raise RuntimeError(
            "No se obtuvieron suficientes arquitecturas finalistas únicas."
        )

    print(
        f"\n--- Evaluación final de alta fidelidad | "
        f"{len(finalists)} arquitecturas únicas | "
        f"presupuesto total idéntico: {final_total_epochs} épocas ---"
    )

    final_results: List[Tuple[Tuple[int, ...], Dict[str, Any], MLPRegressor]] = []

    for finalist_index, hidden_sizes in enumerate(finalists, start=1):
        finalist_start = time.time()
        model, result = evaluate_architecture(
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            hidden_sizes=hidden_sizes,
            target_total_epochs=final_total_epochs,
            architecture_cache=cache,
            learning_rate=learning_rate,
            random_seed=random_state,
            target_parameters=target_parameters,
            complexity_lambda=complexity_lambda,
            complexity_power=complexity_power,
            max_allowed_parameters=max_allowed_parameters,
            prediction_clip_min=0.0,
            prediction_clip_max=RUL_MAX,
            verbose=True,
            print_every=500,
        )
        finalist_elapsed = time.time() - finalist_start
        final_results.append((hidden_sizes, result, model))

        print(
            f"  [Finalista {finalist_index}/{len(finalists)}] "
            f"Hidden: {hidden_sizes} | "
            f"MSE: {result['mse']:.2f} | "
            f"MAE: {result['mae']:.2f} | "
            f"Penalización: {result['penalty']:.2f} | "
            f"Fitness: {result['fitness']:.2f} | "
            f"Params: {result['n_parameters']:,} | "
            f"{result['status']} | "
            f"inicio: {result['start_epoch']} | "
            f"entrenadas ahora: {result['trained_now']} | "
            f"total: {result['target_epoch']} | "
            f"Tiempo: {finalist_elapsed:.2f}s"
        )

        save_architecture_cache(
            cache_path=cache_path,
            cache=cache,
            input_size=X_train.shape[1],
            cache_id=cache_id,
        )

    best_hidden_sizes, best_result, best_model = min(
        final_results,
        key=lambda item: item[1]["fitness"],
    )

    # El modelo devuelto corresponde exactamente al snapshot de 2000 épocas.
    best_weights = best_model.get_weights()

    fitness_history.append(float(best_result["fitness"]))
    mse_history.append(float(best_result["mse"]))

    total_elapsed = time.time() - optimization_start
    print(f"\nOptimización finalizada en {total_elapsed:.2f}s")
    print(
        f"Mejor solución regularizada: {best_hidden_sizes} | "
        f"MSE={best_result['mse']:.2f} | "
        f"MAE={best_result['mae']:.2f} | "
        f"Fitness={best_result['fitness']:.2f} | "
        f"Params={best_result['n_parameters']:,}"
    )

    return OptimizationResult(
        best_hidden_sizes=best_hidden_sizes,
        best_mse=float(best_result["mse"]),
        best_mae=float(best_result["mae"]),
        best_fitness=float(best_result["fitness"]),
        best_penalty=float(best_result["penalty"]),
        best_parameter_count=int(best_result["n_parameters"]),
        best_weights=best_weights,
        fitness_history=fitness_history,
        mse_history=mse_history,
    )


# =============================================================================
# 8. Gráficas y evaluación
# =============================================================================


def plot_pso_history(
    fitness_history: Sequence[float],
    mse_history: Sequence[float],
    dataset_name: str,
) -> None:
    stages = np.arange(1, len(fitness_history) + 1)

    plt.figure(figsize=(8, 4.5))
    plt.plot(stages, fitness_history, marker="o", label="Fitness penalizado")
    plt.plot(stages, mse_history, marker="s", label="MSE")
    plt.xlabel("Etapa; el último punto corresponde a alta fidelidad")
    plt.ylabel("Valor")
    plt.title(f"Convergencia PSO multifidelidad: {dataset_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
) -> None:
    lower = float(min(np.min(y_true), np.min(y_pred), 0.0))
    upper = float(max(np.max(y_true), np.max(y_pred), RUL_MAX))

    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, s=12, alpha=0.25)
    plt.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        linewidth=2,
        label="Predicción perfecta",
    )
    plt.xlabel("RUL real")
    plt.ylabel("RUL predicho")
    plt.title(title)
    plt.xlim(lower, upper)
    plt.ylim(lower, upper)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_test_by_engine(
    engine_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_name: str,
) -> None:
    order = np.argsort(engine_ids)

    plt.figure(figsize=(11, 5))
    plt.plot(
        engine_ids[order],
        y_true[order],
        linewidth=1.8,
        label="RUL real",
    )
    plt.plot(
        engine_ids[order],
        y_pred[order],
        linewidth=1.4,
        linestyle="--",
        label="RUL predicho",
    )
    plt.xlabel("ID del motor")
    plt.ylabel("RUL")
    plt.title(f"Predicción de RUL en test: {dataset_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def evaluate_validation_model(
    model: MLPRegressor,
    X_val: np.ndarray,
    y_val: np.ndarray,
    dataset_name: str,
) -> Dict[str, float]:
    raw_predictions = model.predict_raw(X_val)
    clipped_predictions = np.clip(raw_predictions, 0.0, float(RUL_MAX))

    raw_metrics = regression_metrics(y_val, raw_predictions)
    clipped_metrics = regression_metrics(y_val, clipped_predictions)

    print("\nValidación final:")
    print(
        f"  Sin recorte  -> MSE: {raw_metrics['mse']:.2f}, "
        f"RMSE: {raw_metrics['rmse']:.2f}, "
        f"MAE: {raw_metrics['mae']:.2f}"
    )
    print(
        f"  Recorte [0, {RUL_MAX}] -> MSE: {clipped_metrics['mse']:.2f}, "
        f"RMSE: {clipped_metrics['rmse']:.2f}, "
        f"MAE: {clipped_metrics['mae']:.2f}"
    )

    plot_actual_vs_predicted(
        y_true=y_val,
        y_pred=clipped_predictions,
        title=f"RUL real frente a predicho: {dataset_name} (validación)",
    )

    return {
        "validation_mse": clipped_metrics["mse"],
        "validation_rmse": clipped_metrics["rmse"],
        "validation_mae": clipped_metrics["mae"],
        "validation_raw_mse": raw_metrics["mse"],
    }


def test_final_model(
    model: MLPRegressor,
    dataset_name: str,
    columns_to_drop: Sequence[str],
    selected_features: Sequence[str],
    scaler: MinMaxScaler,
    seq_len: int = SEQ_LEN,
) -> Dict[str, float]:
    print(f"\nProcesando conjunto de test para {dataset_name}...")
    start_time = time.time()

    test_path = DATA_DIR / f"test_{dataset_name}.txt"
    rul_path = DATA_DIR / f"RUL_{dataset_name}.txt"

    try:
        test_dataframe = load_cmapss(test_path)
        test_rul_dataframe = load_cmapss_rul(rul_path)
    except FileNotFoundError:
        print(f"No se encontraron archivos de test para {dataset_name}.")
        return {}

    unit_order = test_dataframe["col_1"].drop_duplicates().to_numpy()
    if len(unit_order) != len(test_rul_dataframe):
        raise ValueError(
            "El número de motores en test no coincide con el archivo RUL."
        )

    raw_rul_lookup = pd.Series(
        test_rul_dataframe["rul"].to_numpy(dtype=float),
        index=unit_order,
    )

    # Se calcula el RUL por fila con la misma función objetivo limitada que se
    # utilizó durante entrenamiento y validación.
    test_rul_by_row = compute_rul(
        test_dataframe,
        base_rul=raw_rul_lookup,
        clip_max=RUL_MAX,
    )

    reduced_test = test_dataframe.drop(
        columns=list(columns_to_drop),
        errors="ignore",
    )
    reduced_test = reduced_test[
        ["col_1", "col_2", *selected_features]
    ]

    X_test, _, test_units = build_sequences(
        dataframe=reduced_test,
        rul=test_rul_by_row,
        seq_len=seq_len,
        scaler=scaler,
        only_last=True,
    )

    y_test_raw = raw_rul_lookup.loc[test_units].to_numpy(dtype=float)
    y_test = np.clip(y_test_raw, 0.0, float(RUL_MAX))

    raw_predictions = model.predict_raw(X_test)
    clipped_predictions = np.clip(raw_predictions, 0.0, float(RUL_MAX))

    raw_target_metrics = regression_metrics(y_test_raw, raw_predictions)
    consistent_metrics = regression_metrics(y_test, clipped_predictions)

    elapsed = time.time() - start_time
    print(
        f"  Métrica coherente con RUL limitado a {RUL_MAX}: "
        f"MSE={consistent_metrics['mse']:.2f}, "
        f"RMSE={consistent_metrics['rmse']:.2f}, "
        f"MAE={consistent_metrics['mae']:.2f}"
    )
    print(
        "  Referencia sobre RUL original sin limitar: "
        f"MSE={raw_target_metrics['mse']:.2f}, "
        f"RMSE={raw_target_metrics['rmse']:.2f}, "
        f"MAE={raw_target_metrics['mae']:.2f}"
    )
    print(f"  Tiempo de test: {elapsed:.2f}s")

    plot_test_by_engine(
        engine_ids=test_units,
        y_true=y_test,
        y_pred=clipped_predictions,
        dataset_name=dataset_name,
    )
    plot_actual_vs_predicted(
        y_true=y_test,
        y_pred=clipped_predictions,
        title=f"RUL real frente a predicho: {dataset_name} (test)",
    )

    return {
        "test_mse": consistent_metrics["mse"],
        "test_rmse": consistent_metrics["rmse"],
        "test_mae": consistent_metrics["mae"],
        "test_raw_target_mse": raw_target_metrics["mse"],
    }


# =============================================================================
# 9. Procesamiento completo de cada subconjunto
# =============================================================================


def process_dataset(dataset_name: str) -> Dict[str, Any]:
    dataset_start = time.time()
    print(f"\n{'=' * 72}\nProcesando {dataset_name}\n{'=' * 72}")

    metadata = {
        "FD001": {"keep_settings": False},
        "FD002": {"keep_settings": True},
        "FD003": {"keep_settings": False},
        "FD004": {"keep_settings": True},
    }
    keep_settings = bool(
        metadata.get(dataset_name, {}).get("keep_settings", False)
    )

    train_path = DATA_DIR / f"train_{dataset_name}.txt"
    try:
        train_dataframe = load_cmapss(train_path)
    except FileNotFoundError:
        print(f"No se encontró el archivo: {train_path}")
        return {}

    train_rul = compute_rul(
        train_dataframe,
        clip_max=RUL_MAX,
    )

    columns_to_drop = [
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
        columns_to_drop.extend(["col_3", "col_4", "col_5"])

    reduced = train_dataframe.drop(
        columns=columns_to_drop,
        errors="ignore",
    )

    units = np.sort(reduced["col_1"].unique())
    train_units, validation_units = train_test_split(
        units,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )

    train_reduced = reduced[
        reduced["col_1"].isin(train_units)
    ].copy()
    validation_reduced = reduced[
        reduced["col_1"].isin(validation_units)
    ].copy()

    selected_features = select_non_flat_features(train_reduced)
    train_reduced = train_reduced[
        ["col_1", "col_2", *selected_features]
    ]
    validation_reduced = validation_reduced[
        ["col_1", "col_2", *selected_features]
    ]

    scaler = MinMaxScaler().fit(train_reduced[selected_features])

    print("Construyendo secuencias de entrenamiento y validación...")
    sequence_start = time.time()

    X_train, y_train, _ = build_sequences(
        dataframe=train_reduced,
        rul=train_rul,
        seq_len=SEQ_LEN,
        scaler=scaler,
        only_last=False,
    )
    X_val, y_val, _ = build_sequences(
        dataframe=validation_reduced,
        rul=train_rul,
        seq_len=SEQ_LEN,
        scaler=scaler,
        only_last=False,
    )

    print(f"Secuencias creadas en {time.time() - sequence_start:.2f}s")
    print(
        f"Train windows: {X_train.shape}; "
        f"validation windows: {X_val.shape}"
    )

    cache_id = (
        f"cache-v{CACHE_VERSION}|dataset={dataset_name}|seq={SEQ_LEN}|"
        f"rul_max={RUL_MAX}|split={RANDOM_STATE}|"
        f"val={VALIDATION_SIZE}|lr={SEARCH_LEARNING_RATE}|"
        f"loss=SmoothL1-beta1|features={','.join(selected_features)}"
    )
    cache_path = CACHE_DIR / f"{dataset_name}_fair_architecture_cache_v2.pt"

    print("\nOptimizando arquitectura MLP con PSO multifidelidad...")
    optimization = pso_optimize(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        particle_schedule=PARTICLE_SCHEDULE,
        search_epochs_per_stage=SEARCH_EPOCHS_PER_STAGE,
        final_total_epochs=FINAL_TOTAL_EPOCHS,
        bounds=HIDDEN_BOUNDS,
        learning_rate=SEARCH_LEARNING_RATE,
        target_parameters=TARGET_PARAMETERS,
        complexity_lambda=COMPLEXITY_LAMBDA,
        complexity_power=COMPLEXITY_POWER,
        max_allowed_parameters=MAX_ALLOWED_PARAMETERS,
        cache_path=cache_path,
        cache_id=cache_id,
        random_state=RANDOM_STATE,
    )

    print(
        f"\nBest Config: {optimization.best_hidden_sizes} | "
        f"MSE: {optimization.best_mse:.2f} | "
        f"RMSE: {np.sqrt(optimization.best_mse):.2f} | "
        f"MAE: {optimization.best_mae:.2f} | "
        f"Params: {optimization.best_parameter_count:,} | "
        f"Penalización: {optimization.best_penalty:.2f} | "
        f"Fitness: {optimization.best_fitness:.2f}"
    )

    plot_pso_history(
        fitness_history=optimization.fitness_history,
        mse_history=optimization.mse_history,
        dataset_name=dataset_name,
    )

    # No se realiza un fine-tuning adicional: el ganador ya corresponde al
    # presupuesto final exacto de FINAL_TOTAL_EPOCHS.
    final_model = MLPRegressor(
        input_size=X_train.shape[1],
        hidden_sizes=optimization.best_hidden_sizes,
        learning_rate=SEARCH_LEARNING_RATE,
    )
    final_model.load_weights(optimization.best_weights)

    validation_results = evaluate_validation_model(
        model=final_model,
        X_val=X_val,
        y_val=y_val,
        dataset_name=dataset_name,
    )

    test_results = test_final_model(
        model=final_model,
        dataset_name=dataset_name,
        columns_to_drop=columns_to_drop,
        selected_features=selected_features,
        scaler=scaler,
        seq_len=SEQ_LEN,
    )

    elapsed = time.time() - dataset_start
    print(
        f"\n>> Dataset {dataset_name} completado en "
        f"{elapsed:.2f}s ({elapsed / 60:.2f} min) <<"
    )

    return {
        "Dataset": dataset_name,
        "Best Hidden Layers": str(optimization.best_hidden_sizes),
        "Parameters": optimization.best_parameter_count,
        "Penalty": optimization.best_penalty,
        "Fitness": optimization.best_fitness,
        "Validation MSE": validation_results["validation_mse"],
        "Validation RMSE": validation_results["validation_rmse"],
        "Validation MAE": validation_results["validation_mae"],
        "Test MSE": test_results.get("test_mse", np.nan),
        "Test RMSE": test_results.get("test_rmse", np.nan),
        "Test MAE": test_results.get("test_mae", np.nan),
        "Time_sec": round(elapsed, 2),
    }


# =============================================================================
# 10. Ejecución
# =============================================================================


def main(datasets: Iterable[str] = DATASETS) -> pd.DataFrame:
    script_start = time.time()
    results: List[Dict[str, Any]] = []

    for dataset_name in datasets:
        result = process_dataset(dataset_name)
        if result:
            results.append(result)

    results_dataframe = pd.DataFrame(results)
    total_elapsed = time.time() - script_start

    print(
        f"\n{'=' * 72}\nEjecución total finalizada en "
        f"{total_elapsed:.2f}s ({total_elapsed / 60:.2f} min)"
    )
    print("Resumen de resultados de predicción RUL:")
    print(results_dataframe)

    return results_dataframe


if __name__ == "__main__":
    main()