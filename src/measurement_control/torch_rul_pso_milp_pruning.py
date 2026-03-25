"""
torch_rul_pso_milp_pruning.py
=============================

Experimental CMAPSS pipeline with an additional pruning stage:
- Stage 1 uses PSO for low-cost ANN screening.
- Stage 2 trains cheap dense references and prunes the top-k PSO candidates.
- Stage 3 tunes the pruned candidates and selects the best one.
- Stage 4 compares the final dense and pruned models on the official test split.

Important scope note:
- This module is intentionally separate from torch_rul_pso.py.
- A linear-error MILP is used for tractability with scipy.optimize.milp for
  one- and two-hidden-layer ReLU ANNs.
- If the exact MILP hits the solver time limit or does not return a solution,
  the pipeline falls back to a transparent global magnitude-pruning mask.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from . import torch_rul_pso as base
except ImportError:
    module_path = Path(__file__).resolve().with_name("torch_rul_pso.py")
    spec = importlib.util.spec_from_file_location("torch_rul_pso_base", module_path)
    if spec is None or spec.loader is None:
        raise
    base = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("torch_rul_pso_base", base)
    spec.loader.exec_module(base)


@dataclass
class MILPPruningConfig(base.TrainingConfig):
    """Extension of the baseline config for the pruning experiment."""

    # Use a larger default subset than the earlier 10% fast mode so validation
    # behavior on FD002/FD003 is less brittle while keeping this branch lighter
    # than a full-data final run.
    training_fraction: float = 0.3
    full_tuning_epochs: int = 120
    full_tuning_patience: int = 12
    tuning_learning_rates: Tuple[float, ...] = (1e-3, 5e-4, 3e-4)
    tuning_weight_decays: Tuple[float, ...] = (0.0, 1e-5, 1e-4)
    pruning_keep_fraction: float = 0.5
    pruning_exact_budget: bool = True
    pruning_calibration_size: int = 16
    pruning_time_limit_sec: float = 30.0
    pruning_mip_rel_gap: float = 0.0
    pruning_finetune_epochs: int = 80
    pruning_finetune_patience: int = 10
    pruning_max_stage1_candidates: int = 3

    def __post_init__(self) -> None:
        super().__post_init__()
        if any(str(act).lower() != "relu" for act in self.activation_choices):
            raise ValueError("MILP pruning pipeline currently supports ReLU activations only.")
        if any(str(act).lower() != "relu" for act in self.tuning_activation_choices):
            raise ValueError("Stage 2 tuning must keep ReLU when MILP pruning is enabled.")
        if self.min_hidden_layers < 1 or self.max_hidden_layers > 2:
            raise ValueError("MILP pruning pipeline currently supports between 1 and 2 hidden layers.")
        if not 0 < self.training_fraction <= 1:
            raise ValueError("training_fraction must lie in (0, 1].")
        if not 0 < self.pruning_keep_fraction <= 1:
            raise ValueError("pruning_keep_fraction must lie in (0, 1].")
        if self.pruning_calibration_size < 4:
            raise ValueError("pruning_calibration_size must be at least 4.")
        if self.pruning_time_limit_sec <= 0:
            raise ValueError("pruning_time_limit_sec must be positive.")
        if self.pruning_max_stage1_candidates < 1:
            raise ValueError("pruning_max_stage1_candidates must be at least 1.")


@dataclass
class PrePruningReferenceResult:
    """Cheap dense reference used to derive a pruning mask for one candidate."""

    architecture: base.ArchitectureConfig
    validation_mse: float
    validation_mae: float
    num_parameters: int
    trained_model: nn.Module
    training_time_sec: float
    epochs_ran: int


@dataclass
class MILPPruningResult:
    """Arc-mask solution produced by the pruning stage."""

    layer_masks: List[np.ndarray]
    active_input_arcs: int
    active_output_arcs: int
    active_total_arcs: int
    total_arcs: int
    per_layer_active_arcs: List[int]
    keep_ratio: float
    solve_time_sec: float
    solver_status: str
    success: bool
    objective_value: Optional[float]
    calibration_mae: float
    fallback_used: bool = False
    pruning_method: str = "milp"


@dataclass
class PrunedCandidateResult:
    """Result of prune-then-tune evaluation for one top-k PSO candidate."""

    architecture: base.ArchitectureConfig
    best_tuning: Dict[str, object]
    prepruning_validation_mse: float
    prepruning_validation_mae: float
    validation_mse: float
    validation_mae: float
    selection_score: float
    num_parameters: int
    reference_train_time_sec: float
    pruning_solve_time_sec: float
    tuning_time_sec: float
    total_candidate_time_sec: float
    nonzero_parameters: int
    density: float
    pruning_result: MILPPruningResult
    tuning_runs: List[Dict[str, float]]
    best_history: List[Dict[str, float]]
    model: nn.Module


def select_stage1_candidates(
    search_result: base.PSOSearchResult,
    config: MILPPruningConfig,
) -> List[base.CandidateEvaluation]:
    """Pick the top-k Stage 1 candidates to prune and tune."""

    candidate_limit = max(1, min(config.top_k, config.pruning_max_stage1_candidates))
    candidates = search_result.top_candidates[:candidate_limit]
    if not candidates:
        raise RuntimeError("No valid Stage 1 candidates are available for pruning.")
    return candidates


def get_linear_layers(model: nn.Module) -> List[nn.Linear]:
    """Extract all Linear layers from the baseline TorchMLP."""

    if not hasattr(model, "network"):
        raise TypeError("Expected the TorchMLP model used by the baseline pipeline.")
    linear_layers = [layer for layer in model.network if isinstance(layer, nn.Linear)]
    if len(linear_layers) < 2:
        raise ValueError("Expected at least one hidden layer plus one output layer.")
    return linear_layers


def get_one_hidden_linear_layers(model: nn.Module) -> Tuple[nn.Linear, nn.Linear]:
    """Extract the two Linear layers of a one-hidden-layer TorchMLP."""

    linear_layers = get_linear_layers(model)
    if len(linear_layers) != 2:
        raise ValueError("MILP pruning currently supports exactly one hidden layer.")
    return linear_layers[0], linear_layers[1]


def sample_calibration_subset(
    X: np.ndarray,
    y: np.ndarray,
    size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pick a deterministic target-stratified calibration subset."""

    if size >= len(X):
        return X, y
    sorted_idx = np.argsort(y)
    selected = np.linspace(0, len(sorted_idx) - 1, num=size, dtype=int)
    chosen_idx = np.unique(sorted_idx[selected])
    return X[chosen_idx], y[chosen_idx]


def subsample_supervised_rows(
    X: np.ndarray,
    y: np.ndarray,
    fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Take a reproducible subset of rows for fast experimental runs."""

    original_size = len(X)
    if fraction >= 1.0 or original_size <= 1:
        return X, y, original_size

    sample_size = max(1, int(round(original_size * fraction)))
    sample_size = min(sample_size, original_size)
    rng = np.random.default_rng(seed)
    chosen_idx = np.sort(rng.choice(original_size, size=sample_size, replace=False))
    return X[chosen_idx], y[chosen_idx], original_size


def apply_training_fraction(
    split_bundle: base.TrainSplitBundle,
    config: MILPPruningConfig,
) -> Tuple[base.TrainSplitBundle, Dict[str, int]]:
    """
    Downsample the pruning branch after preprocessing.

    This keeps the official NASA train/test separation intact and only reduces
    the number of windowed samples used by the experimental pruning workflow.
    """

    X_train, y_train, original_train = subsample_supervised_rows(
        split_bundle.X_train,
        split_bundle.y_train,
        config.training_fraction,
        config.random_seed,
    )
    X_val, y_val, original_val = subsample_supervised_rows(
        split_bundle.X_val,
        split_bundle.y_val,
        config.training_fraction,
        config.random_seed + 1,
    )
    X_full, y_full, original_full = subsample_supervised_rows(
        split_bundle.X_full_train,
        split_bundle.y_full_train,
        config.training_fraction,
        config.random_seed + 2,
    )

    sampled_bundle = base.TrainSplitBundle(
        dataset_name=split_bundle.dataset_name,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_full_train=X_full,
        y_full_train=y_full,
        input_dim=split_bundle.input_dim,
        train_unit_count=split_bundle.train_unit_count,
        val_unit_count=split_bundle.val_unit_count,
        full_train_feature_scaler=split_bundle.full_train_feature_scaler,
        split_method=split_bundle.split_method,
    )
    subset_summary = {
        "original_search_train_windows": int(original_train),
        "sampled_search_train_windows": int(len(X_train)),
        "original_validation_windows": int(original_val),
        "sampled_validation_windows": int(len(X_val)),
        "original_full_training_windows": int(original_full),
        "sampled_full_training_windows": int(len(X_full)),
    }
    return sampled_bundle, subset_summary


def summarize_layer_masks(layer_masks: Sequence[np.ndarray]) -> Tuple[List[int], int, int, int]:
    """Summarize mask activity across all linear layers."""

    per_layer_active = [int(np.count_nonzero(mask)) for mask in layer_masks]
    active_total = int(sum(per_layer_active))
    total_arcs = int(sum(mask.size for mask in layer_masks))
    active_input = int(per_layer_active[0]) if per_layer_active else 0
    active_output = int(per_layer_active[-1]) if per_layer_active else 0
    return per_layer_active, active_total, total_arcs, active_input, active_output


def count_pruned_nonzero_parameters(model: nn.Module) -> int:
    """Count nonzero trainable parameters after pruning and fine-tuning."""

    total = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            total += int(torch.count_nonzero(parameter.detach()).item())
    return total


def magnitude_pruning_masks(
    model: nn.Module,
    keep_fraction: float,
) -> List[np.ndarray]:
    """Fallback mask using global weight magnitudes across all dense layers."""

    linear_layers = get_linear_layers(model)
    weight_magnitudes = [np.abs(layer.weight.detach().cpu().numpy()) for layer in linear_layers]
    flat = np.concatenate([weights.reshape(-1) for weights in weight_magnitudes])
    keep_total = max(len(weight_magnitudes), int(round(keep_fraction * flat.size)))
    keep_total = min(keep_total, flat.size)
    threshold = np.partition(flat, -keep_total)[-keep_total]

    layer_masks: List[np.ndarray] = []
    for weights in weight_magnitudes:
        mask = (weights >= threshold).astype(np.float64)
        if np.count_nonzero(mask) == 0:
            best_idx = np.unravel_index(np.argmax(weights), weights.shape)
            mask[best_idx] = 1.0
        layer_masks.append(mask)
    return layer_masks


def ensure_nonempty_layer_masks(
    model: nn.Module,
    layer_masks: Sequence[np.ndarray],
) -> List[np.ndarray]:
    """Ensure each pruned Linear layer keeps at least one active arc."""

    linear_layers = get_linear_layers(model)
    adjusted_masks: List[np.ndarray] = []
    for layer, raw_mask in zip(linear_layers, layer_masks):
        mask = np.array(raw_mask, dtype=np.float64, copy=True)
        if np.count_nonzero(mask) == 0:
            weights = np.abs(layer.weight.detach().cpu().numpy())
            best_idx = np.unravel_index(np.argmax(weights), weights.shape)
            mask[best_idx] = 1.0
        adjusted_masks.append(mask)
    return adjusted_masks


def build_milp_pruning_problem_one_hidden_layer(
    X_calib: np.ndarray,
    y_hat: np.ndarray,
    model: nn.Module,
    keep_fraction: float,
    exact_budget: bool,
) -> Tuple[np.ndarray, np.ndarray, Bounds, List[LinearConstraint], Dict[str, object]]:
    """
    Build an exact MILP for one-hidden-layer ReLU arc pruning.

    The objective minimizes the L1 deviation to the dense reference model on a
    small calibration subset. This is a linear surrogate for a quadratic
    teacher-matching objective because scipy.optimize.milp supports MILP but not
    a native MIQP objective.
    """

    fc1, fc2 = get_one_hidden_linear_layers(model)
    w1 = fc1.weight.detach().cpu().numpy().astype(np.float64)
    b1 = fc1.bias.detach().cpu().numpy().astype(np.float64)
    w2 = fc2.weight.detach().cpu().numpy().reshape(-1).astype(np.float64)
    b2 = float(fc2.bias.detach().cpu().numpy().reshape(-1)[0])

    n_samples, input_dim = X_calib.shape
    hidden_dim = w1.shape[0]
    coeff = X_calib[:, None, :] * w1[None, :, :]
    lower_a = b1[None, :] + np.minimum(coeff, 0.0).sum(axis=2)
    upper_a = b1[None, :] + np.maximum(coeff, 0.0).sum(axis=2)
    upper_h = np.maximum(upper_a, 0.0)
    upper_h = np.maximum(upper_h, 1e-6)

    counts = {
        "z1": hidden_dim * input_dim,
        "z2": hidden_dim,
        "delta": n_samples * hidden_dim,
        "a": n_samples * hidden_dim,
        "h": n_samples * hidden_dim,
        "q": n_samples * hidden_dim,
        "y": n_samples,
        "e": n_samples,
    }
    offsets: Dict[str, int] = {}
    current = 0
    for key in ("z1", "z2", "delta", "a", "h", "q", "y", "e"):
        offsets[key] = current
        current += counts[key]
    n_vars = current

    def idx_z1(j: int, i: int) -> int:
        return offsets["z1"] + j * input_dim + i

    def idx_z2(j: int) -> int:
        return offsets["z2"] + j

    def idx_delta(n: int, j: int) -> int:
        return offsets["delta"] + n * hidden_dim + j

    def idx_a(n: int, j: int) -> int:
        return offsets["a"] + n * hidden_dim + j

    def idx_h(n: int, j: int) -> int:
        return offsets["h"] + n * hidden_dim + j

    def idx_q(n: int, j: int) -> int:
        return offsets["q"] + n * hidden_dim + j

    def idx_y(n: int) -> int:
        return offsets["y"] + n

    def idx_e(n: int) -> int:
        return offsets["e"] + n

    lb = np.full(n_vars, -np.inf, dtype=np.float64)
    ub = np.full(n_vars, np.inf, dtype=np.float64)
    integrality = np.zeros(n_vars, dtype=np.int8)

    lb[offsets["z1"] : offsets["z1"] + counts["z1"]] = 0.0
    ub[offsets["z1"] : offsets["z1"] + counts["z1"]] = 1.0
    integrality[offsets["z1"] : offsets["z1"] + counts["z1"]] = 1

    lb[offsets["z2"] : offsets["z2"] + counts["z2"]] = 0.0
    ub[offsets["z2"] : offsets["z2"] + counts["z2"]] = 1.0
    integrality[offsets["z2"] : offsets["z2"] + counts["z2"]] = 1

    lb[offsets["delta"] : offsets["delta"] + counts["delta"]] = 0.0
    ub[offsets["delta"] : offsets["delta"] + counts["delta"]] = 1.0
    integrality[offsets["delta"] : offsets["delta"] + counts["delta"]] = 1

    lb[offsets["a"] : offsets["a"] + counts["a"]] = lower_a.reshape(-1)
    ub[offsets["a"] : offsets["a"] + counts["a"]] = upper_a.reshape(-1)
    lb[offsets["h"] : offsets["h"] + counts["h"]] = 0.0
    ub[offsets["h"] : offsets["h"] + counts["h"]] = upper_h.reshape(-1)
    lb[offsets["q"] : offsets["q"] + counts["q"]] = 0.0
    ub[offsets["q"] : offsets["q"] + counts["q"]] = upper_h.reshape(-1)
    lb[offsets["e"] : offsets["e"] + counts["e"]] = 0.0

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    lhs: List[float] = []
    rhs: List[float] = []
    row_index = 0

    def add_constraint(entries: Sequence[Tuple[int, float]], lower: float, upper: float) -> None:
        nonlocal row_index
        for col, value in entries:
            rows.append(row_index)
            cols.append(col)
            data.append(float(value))
        lhs.append(float(lower))
        rhs.append(float(upper))
        row_index += 1

    for n in range(n_samples):
        for j in range(hidden_dim):
            a_entries = [(idx_a(n, j), 1.0)]
            for i in range(input_dim):
                coefficient = -float(w1[j, i] * X_calib[n, i])
                if coefficient != 0.0:
                    a_entries.append((idx_z1(j, i), coefficient))
            add_constraint(a_entries, b1[j], b1[j])

            add_constraint([(idx_h(n, j), 1.0), (idx_a(n, j), -1.0)], 0.0, np.inf)
            add_constraint(
                [
                    (idx_h(n, j), 1.0),
                    (idx_a(n, j), -1.0),
                    (idx_delta(n, j), -lower_a[n, j]),
                ],
                -np.inf,
                -lower_a[n, j],
            )
            add_constraint(
                [(idx_h(n, j), 1.0), (idx_delta(n, j), -upper_a[n, j])],
                -np.inf,
                0.0,
            )

            add_constraint([(idx_q(n, j), 1.0), (idx_h(n, j), -1.0)], -np.inf, 0.0)
            add_constraint(
                [(idx_q(n, j), 1.0), (idx_z2(j), -upper_h[n, j])],
                -np.inf,
                0.0,
            )
            add_constraint(
                [
                    (idx_q(n, j), 1.0),
                    (idx_h(n, j), -1.0),
                    (idx_z2(j), -upper_h[n, j]),
                ],
                -upper_h[n, j],
                np.inf,
            )

        y_entries = [(idx_y(n), 1.0)]
        for j in range(hidden_dim):
            coefficient = -float(w2[j])
            if coefficient != 0.0:
                y_entries.append((idx_q(n, j), coefficient))
        add_constraint(y_entries, b2, b2)
        add_constraint([(idx_e(n), 1.0), (idx_y(n), -1.0)], -float(y_hat[n]), np.inf)
        add_constraint([(idx_e(n), 1.0), (idx_y(n), 1.0)], float(y_hat[n]), np.inf)

    total_arcs = hidden_dim * input_dim + hidden_dim
    keep_total = max(1, int(round(keep_fraction * total_arcs)))
    budget_entries = [(offsets["z1"] + i, 1.0) for i in range(counts["z1"])]
    budget_entries.extend((offsets["z2"] + j, 1.0) for j in range(counts["z2"]))
    if exact_budget:
        add_constraint(budget_entries, float(keep_total), float(keep_total))
    else:
        add_constraint(budget_entries, -np.inf, float(keep_total))

    c = np.zeros(n_vars, dtype=np.float64)
    c[offsets["e"] : offsets["e"] + counts["e"]] = 1.0

    A = sparse.coo_matrix((data, (rows, cols)), shape=(row_index, n_vars)).tocsc()
    constraints = [LinearConstraint(A, np.asarray(lhs), np.asarray(rhs))]
    bounds = Bounds(lb, ub)
    meta: Dict[str, object] = {
        "depth": 1,
        "offsets": offsets,
        "mask_shapes": [w1.shape, fc2.weight.detach().cpu().numpy().shape],
        "n_samples": n_samples,
    }
    return c, integrality, bounds, constraints, meta


def build_milp_pruning_problem_two_hidden_layers(
    X_calib: np.ndarray,
    y_hat: np.ndarray,
    model: nn.Module,
    keep_fraction: float,
    exact_budget: bool,
) -> Tuple[np.ndarray, np.ndarray, Bounds, List[LinearConstraint], Dict[str, object]]:
    """
    Build an exact MILP for two-hidden-layer ReLU arc pruning.

    Binary variables select which arcs remain active across all three dense
    layers. Auxiliary q variables linearize the products between bounded ReLU
    activations and binary arc-retention decisions.
    """

    fc1, fc2, fc3 = get_linear_layers(model)
    if fc3.weight.shape[0] != 1:
        raise ValueError("Two-hidden-layer exact MILP currently assumes a scalar output.")

    w1 = fc1.weight.detach().cpu().numpy().astype(np.float64)
    b1 = fc1.bias.detach().cpu().numpy().astype(np.float64)
    w2 = fc2.weight.detach().cpu().numpy().astype(np.float64)
    b2 = fc2.bias.detach().cpu().numpy().astype(np.float64)
    w3 = fc3.weight.detach().cpu().numpy().reshape(-1).astype(np.float64)
    b3 = float(fc3.bias.detach().cpu().numpy().reshape(-1)[0])

    n_samples, input_dim = X_calib.shape
    hidden1_dim = w1.shape[0]
    hidden2_dim = w2.shape[0]

    coeff1 = X_calib[:, None, :] * w1[None, :, :]
    lower_a1 = b1[None, :] + np.minimum(coeff1, 0.0).sum(axis=2)
    upper_a1 = b1[None, :] + np.maximum(coeff1, 0.0).sum(axis=2)
    upper_h1 = np.maximum(upper_a1, 0.0)
    upper_h1 = np.maximum(upper_h1, 1e-6)

    contrib2 = upper_h1[:, None, :] * w2[None, :, :]
    lower_a2 = b2[None, :] + np.minimum(contrib2, 0.0).sum(axis=2)
    upper_a2 = b2[None, :] + np.maximum(contrib2, 0.0).sum(axis=2)
    upper_h2 = np.maximum(upper_a2, 0.0)
    upper_h2 = np.maximum(upper_h2, 1e-6)

    counts = {
        "z1": hidden1_dim * input_dim,
        "z2": hidden2_dim * hidden1_dim,
        "z3": hidden2_dim,
        "delta1": n_samples * hidden1_dim,
        "delta2": n_samples * hidden2_dim,
        "a1": n_samples * hidden1_dim,
        "a2": n_samples * hidden2_dim,
        "h1": n_samples * hidden1_dim,
        "h2": n_samples * hidden2_dim,
        "q12": n_samples * hidden2_dim * hidden1_dim,
        "q23": n_samples * hidden2_dim,
        "y": n_samples,
        "e": n_samples,
    }
    offsets: Dict[str, int] = {}
    current = 0
    for key in ("z1", "z2", "z3", "delta1", "delta2", "a1", "a2", "h1", "h2", "q12", "q23", "y", "e"):
        offsets[key] = current
        current += counts[key]
    n_vars = current

    def idx_z1(j: int, i: int) -> int:
        return offsets["z1"] + j * input_dim + i

    def idx_z2(k: int, j: int) -> int:
        return offsets["z2"] + k * hidden1_dim + j

    def idx_z3(k: int) -> int:
        return offsets["z3"] + k

    def idx_delta1(n: int, j: int) -> int:
        return offsets["delta1"] + n * hidden1_dim + j

    def idx_delta2(n: int, k: int) -> int:
        return offsets["delta2"] + n * hidden2_dim + k

    def idx_a1(n: int, j: int) -> int:
        return offsets["a1"] + n * hidden1_dim + j

    def idx_a2(n: int, k: int) -> int:
        return offsets["a2"] + n * hidden2_dim + k

    def idx_h1(n: int, j: int) -> int:
        return offsets["h1"] + n * hidden1_dim + j

    def idx_h2(n: int, k: int) -> int:
        return offsets["h2"] + n * hidden2_dim + k

    def idx_q12(n: int, k: int, j: int) -> int:
        return offsets["q12"] + n * hidden2_dim * hidden1_dim + k * hidden1_dim + j

    def idx_q23(n: int, k: int) -> int:
        return offsets["q23"] + n * hidden2_dim + k

    def idx_y(n: int) -> int:
        return offsets["y"] + n

    def idx_e(n: int) -> int:
        return offsets["e"] + n

    lb = np.full(n_vars, -np.inf, dtype=np.float64)
    ub = np.full(n_vars, np.inf, dtype=np.float64)
    integrality = np.zeros(n_vars, dtype=np.int8)

    for key in ("z1", "z2", "z3", "delta1", "delta2"):
        lb[offsets[key] : offsets[key] + counts[key]] = 0.0
        ub[offsets[key] : offsets[key] + counts[key]] = 1.0
        integrality[offsets[key] : offsets[key] + counts[key]] = 1

    lb[offsets["a1"] : offsets["a1"] + counts["a1"]] = lower_a1.reshape(-1)
    ub[offsets["a1"] : offsets["a1"] + counts["a1"]] = upper_a1.reshape(-1)
    lb[offsets["a2"] : offsets["a2"] + counts["a2"]] = lower_a2.reshape(-1)
    ub[offsets["a2"] : offsets["a2"] + counts["a2"]] = upper_a2.reshape(-1)

    lb[offsets["h1"] : offsets["h1"] + counts["h1"]] = 0.0
    ub[offsets["h1"] : offsets["h1"] + counts["h1"]] = upper_h1.reshape(-1)
    lb[offsets["h2"] : offsets["h2"] + counts["h2"]] = 0.0
    ub[offsets["h2"] : offsets["h2"] + counts["h2"]] = upper_h2.reshape(-1)

    q12_ub = np.broadcast_to(upper_h1[:, None, :], (n_samples, hidden2_dim, hidden1_dim)).reshape(-1)
    lb[offsets["q12"] : offsets["q12"] + counts["q12"]] = 0.0
    ub[offsets["q12"] : offsets["q12"] + counts["q12"]] = q12_ub

    lb[offsets["q23"] : offsets["q23"] + counts["q23"]] = 0.0
    ub[offsets["q23"] : offsets["q23"] + counts["q23"]] = upper_h2.reshape(-1)
    lb[offsets["e"] : offsets["e"] + counts["e"]] = 0.0

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    lhs: List[float] = []
    rhs: List[float] = []
    row_index = 0

    def add_constraint(entries: Sequence[Tuple[int, float]], lower: float, upper: float) -> None:
        nonlocal row_index
        for col, value in entries:
            rows.append(row_index)
            cols.append(col)
            data.append(float(value))
        lhs.append(float(lower))
        rhs.append(float(upper))
        row_index += 1

    for n in range(n_samples):
        for j in range(hidden1_dim):
            a1_entries = [(idx_a1(n, j), 1.0)]
            for i in range(input_dim):
                coefficient = -float(w1[j, i] * X_calib[n, i])
                if coefficient != 0.0:
                    a1_entries.append((idx_z1(j, i), coefficient))
            add_constraint(a1_entries, b1[j], b1[j])

            add_constraint([(idx_h1(n, j), 1.0), (idx_a1(n, j), -1.0)], 0.0, np.inf)
            add_constraint(
                [
                    (idx_h1(n, j), 1.0),
                    (idx_a1(n, j), -1.0),
                    (idx_delta1(n, j), -lower_a1[n, j]),
                ],
                -np.inf,
                -lower_a1[n, j],
            )
            add_constraint(
                [(idx_h1(n, j), 1.0), (idx_delta1(n, j), -upper_a1[n, j])],
                -np.inf,
                0.0,
            )

        for k in range(hidden2_dim):
            for j in range(hidden1_dim):
                upper_q12 = upper_h1[n, j]
                add_constraint([(idx_q12(n, k, j), 1.0), (idx_h1(n, j), -1.0)], -np.inf, 0.0)
                add_constraint(
                    [(idx_q12(n, k, j), 1.0), (idx_z2(k, j), -upper_q12)],
                    -np.inf,
                    0.0,
                )
                add_constraint(
                    [
                        (idx_q12(n, k, j), 1.0),
                        (idx_h1(n, j), -1.0),
                        (idx_z2(k, j), -upper_q12),
                    ],
                    -upper_q12,
                    np.inf,
                )

            a2_entries = [(idx_a2(n, k), 1.0)]
            for j in range(hidden1_dim):
                coefficient = -float(w2[k, j])
                if coefficient != 0.0:
                    a2_entries.append((idx_q12(n, k, j), coefficient))
            add_constraint(a2_entries, b2[k], b2[k])

            add_constraint([(idx_h2(n, k), 1.0), (idx_a2(n, k), -1.0)], 0.0, np.inf)
            add_constraint(
                [
                    (idx_h2(n, k), 1.0),
                    (idx_a2(n, k), -1.0),
                    (idx_delta2(n, k), -lower_a2[n, k]),
                ],
                -np.inf,
                -lower_a2[n, k],
            )
            add_constraint(
                [(idx_h2(n, k), 1.0), (idx_delta2(n, k), -upper_a2[n, k])],
                -np.inf,
                0.0,
            )

            upper_q23 = upper_h2[n, k]
            add_constraint([(idx_q23(n, k), 1.0), (idx_h2(n, k), -1.0)], -np.inf, 0.0)
            add_constraint(
                [(idx_q23(n, k), 1.0), (idx_z3(k), -upper_q23)],
                -np.inf,
                0.0,
            )
            add_constraint(
                [
                    (idx_q23(n, k), 1.0),
                    (idx_h2(n, k), -1.0),
                    (idx_z3(k), -upper_q23),
                ],
                -upper_q23,
                np.inf,
            )

        y_entries = [(idx_y(n), 1.0)]
        for k in range(hidden2_dim):
            coefficient = -float(w3[k])
            if coefficient != 0.0:
                y_entries.append((idx_q23(n, k), coefficient))
        add_constraint(y_entries, b3, b3)
        add_constraint([(idx_e(n), 1.0), (idx_y(n), -1.0)], -float(y_hat[n]), np.inf)
        add_constraint([(idx_e(n), 1.0), (idx_y(n), 1.0)], float(y_hat[n]), np.inf)

    total_arcs = hidden1_dim * input_dim + hidden2_dim * hidden1_dim + hidden2_dim
    keep_total = max(1, int(round(keep_fraction * total_arcs)))
    budget_entries = [(offsets["z1"] + i, 1.0) for i in range(counts["z1"])]
    budget_entries.extend((offsets["z2"] + i, 1.0) for i in range(counts["z2"]))
    budget_entries.extend((offsets["z3"] + i, 1.0) for i in range(counts["z3"]))
    if exact_budget:
        add_constraint(budget_entries, float(keep_total), float(keep_total))
    else:
        add_constraint(budget_entries, -np.inf, float(keep_total))

    c = np.zeros(n_vars, dtype=np.float64)
    c[offsets["e"] : offsets["e"] + counts["e"]] = 1.0

    A = sparse.coo_matrix((data, (rows, cols)), shape=(row_index, n_vars)).tocsc()
    constraints = [LinearConstraint(A, np.asarray(lhs), np.asarray(rhs))]
    bounds = Bounds(lb, ub)
    meta = {
        "depth": 2,
        "offsets": offsets,
        "mask_shapes": [w1.shape, w2.shape, fc3.weight.detach().cpu().numpy().shape],
        "n_samples": n_samples,
    }
    return c, integrality, bounds, constraints, meta


def build_milp_pruning_problem(
    X_calib: np.ndarray,
    y_hat: np.ndarray,
    model: nn.Module,
    keep_fraction: float,
    exact_budget: bool,
) -> Tuple[np.ndarray, np.ndarray, Bounds, List[LinearConstraint], Dict[str, object]]:
    """Dispatch to the exact MILP builder matching the candidate depth."""

    linear_layers = get_linear_layers(model)
    if len(linear_layers) == 2:
        return build_milp_pruning_problem_one_hidden_layer(
            X_calib=X_calib,
            y_hat=y_hat,
            model=model,
            keep_fraction=keep_fraction,
            exact_budget=exact_budget,
        )
    if len(linear_layers) == 3:
        return build_milp_pruning_problem_two_hidden_layers(
            X_calib=X_calib,
            y_hat=y_hat,
            model=model,
            keep_fraction=keep_fraction,
            exact_budget=exact_budget,
        )
    raise ValueError("Exact MILP pruning currently supports one or two hidden layers only.")


def extract_milp_solution_masks(
    model: nn.Module,
    x_sol: np.ndarray,
    meta: Dict[str, object],
) -> List[np.ndarray]:
    """Extract per-layer pruning masks from an exact MILP solution vector."""

    linear_layers = get_linear_layers(model)
    offsets = meta["offsets"]
    layer_masks: List[np.ndarray] = []
    for layer_index, layer in enumerate(linear_layers, start=1):
        key = f"z{layer_index}"
        shape = layer.weight.detach().cpu().numpy().shape
        start = int(offsets[key])
        stop = start + int(np.prod(shape))
        mask = (x_sol[start:stop].reshape(shape) > 0.5).astype(np.float64)
        layer_masks.append(mask)
    return ensure_nonempty_layer_masks(model, layer_masks)


def solve_milp_pruning(
    model: nn.Module,
    calibration_data: Tuple[np.ndarray, np.ndarray],
    config: MILPPruningConfig,
) -> MILPPruningResult:
    """Solve the exact pruning MILP or fall back to magnitude pruning if needed."""

    linear_layers = get_linear_layers(model)
    if len(linear_layers) not in {2, 3}:
        layer_masks = magnitude_pruning_masks(model, config.pruning_keep_fraction)
        layer_masks = ensure_nonempty_layer_masks(model, layer_masks)
        per_layer_active, active_total, total_arcs, active_input, active_output = summarize_layer_masks(
            layer_masks
        )
        return MILPPruningResult(
            layer_masks=layer_masks,
            active_input_arcs=active_input,
            active_output_arcs=active_output,
            active_total_arcs=active_total,
            total_arcs=total_arcs,
            per_layer_active_arcs=per_layer_active,
            keep_ratio=float(active_total / max(total_arcs, 1)),
            solve_time_sec=0.0,
            solver_status="multilayer_magnitude_fallback",
            success=False,
            objective_value=None,
            calibration_mae=float("nan"),
            fallback_used=True,
            pruning_method="magnitude_unsupported_depth",
        )

    X_calib, _ = calibration_data
    y_hat = base.predict_torch_model(model, X_calib, config.device).astype(np.float64)
    c, integrality, bounds, constraints, meta = build_milp_pruning_problem(
        X_calib=X_calib.astype(np.float64),
        y_hat=y_hat,
        model=model,
        keep_fraction=config.pruning_keep_fraction,
        exact_budget=config.pruning_exact_budget,
    )

    solve_start = time.perf_counter()
    result = milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={
            "time_limit": float(config.pruning_time_limit_sec),
            "mip_rel_gap": float(config.pruning_mip_rel_gap),
            "disp": bool(config.verbose),
        },
    )
    solve_time = time.perf_counter() - solve_start

    if result.x is None:
        layer_masks = magnitude_pruning_masks(model, config.pruning_keep_fraction)
        layer_masks = ensure_nonempty_layer_masks(model, layer_masks)
        per_layer_active, active_total, total_arcs, active_input, active_output = summarize_layer_masks(
            layer_masks
        )
        return MILPPruningResult(
            layer_masks=layer_masks,
            active_input_arcs=active_input,
            active_output_arcs=active_output,
            active_total_arcs=active_total,
            total_arcs=total_arcs,
            per_layer_active_arcs=per_layer_active,
            keep_ratio=float(active_total / max(total_arcs, 1)),
            solve_time_sec=float(solve_time),
            solver_status=str(result.status),
            success=False,
            objective_value=None if result.fun is None else float(result.fun),
            calibration_mae=float("nan"),
            fallback_used=True,
            pruning_method="magnitude_solver_fallback",
        )

    x_sol = np.asarray(result.x, dtype=np.float64)
    offsets = meta["offsets"]
    layer_masks = extract_milp_solution_masks(model, x_sol, meta)
    y_sol = x_sol[offsets["y"] : offsets["y"] + len(X_calib)]
    per_layer_active, active_total, total_arcs, active_input, active_output = summarize_layer_masks(
        layer_masks
    )
    return MILPPruningResult(
        layer_masks=layer_masks,
        active_input_arcs=active_input,
        active_output_arcs=active_output,
        active_total_arcs=active_total,
        total_arcs=total_arcs,
        per_layer_active_arcs=per_layer_active,
        keep_ratio=float(active_total / max(total_arcs, 1)),
        solve_time_sec=float(solve_time),
        solver_status=str(result.status),
        success=True,
        objective_value=None if result.fun is None else float(result.fun),
        calibration_mae=float(np.mean(np.abs(y_sol - y_hat))),
        fallback_used=False,
        pruning_method="milp_exact",
    )


def apply_masks_to_model(
    model: nn.Module,
    layer_masks: Sequence[np.ndarray],
) -> List[torch.Tensor]:
    """Apply pruning masks in-place and return them as tensors."""

    linear_layers = get_linear_layers(model)
    if len(linear_layers) != len(layer_masks):
        raise ValueError("Number of layer masks does not match the number of Linear layers.")

    mask_tensors: List[torch.Tensor] = []
    for layer, mask in zip(linear_layers, layer_masks):
        mask_tensor = torch.tensor(mask, dtype=layer.weight.dtype, device=layer.weight.device)
        layer.weight.data.mul_(mask_tensor)
        mask_tensors.append(mask_tensor)
    return mask_tensors


def masked_train_model(
    model: nn.Module,
    layer_masks: Sequence[np.ndarray],
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Optional[Tuple[np.ndarray, np.ndarray]],
    train_config: base.ModelTrainConfig,
) -> Dict[str, object]:
    """Fine-tune a pruned model while keeping masked arcs fixed at zero."""

    base.set_global_seed(train_config.seed)
    model = model.to(train_config.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    criterion = nn.MSELoss()
    X_train, y_train = train_data
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    generator = torch.Generator()
    generator.manual_seed(train_config.seed)
    loader = DataLoader(dataset, batch_size=train_config.batch_size, shuffle=True, generator=generator)

    linear_layers = get_linear_layers(model)
    mask_tensors = apply_masks_to_model(model, layer_masks)
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
            for layer, mask_tensor in zip(linear_layers, mask_tensors):
                if layer.weight.grad is not None:
                    layer.weight.grad.mul_(mask_tensor)
            optimizer.step()
            for layer, mask_tensor in zip(linear_layers, mask_tensors):
                layer.weight.data.mul_(mask_tensor)
            running_loss += float(loss.item()) * len(x_batch)
            sample_count += len(x_batch)

        record = {"epoch": float(epoch + 1), "train_loss": running_loss / max(sample_count, 1)}
        if val_data is not None:
            metrics = base.evaluate_model(model, val_data, train_config.device)
            val_mse = float(metrics["mse"])
            record["val_mse"] = val_mse
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if train_config.patience is not None and epochs_without_improvement >= train_config.patience:
                    history.append(record)
                    break
        else:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
        history.append(record)

    model.load_state_dict(best_state)
    apply_masks_to_model(model, layer_masks)
    return {
        "model": model,
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "best_val_mse": None if val_data is None else best_val_mse,
        "history": history,
    }


def train_reference_for_pruning(
    candidate: base.CandidateEvaluation,
    config: MILPPruningConfig,
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    input_dim: int,
    seed_offset: int,
) -> PrePruningReferenceResult:
    """Train a cheap dense reference model whose weights will be pruned."""

    model = base.build_model(
        base.ModelConfig(input_dim=input_dim, architecture=candidate.architecture)
    )
    train_cfg = config.low_fidelity_train_config(seed=config.random_seed + seed_offset)
    start = time.perf_counter()
    outcome = base.train_model(model, train_data, val_data, train_cfg)
    training_time = time.perf_counter() - start
    metrics = base.evaluate_model(outcome["model"], val_data, config.device)
    return PrePruningReferenceResult(
        architecture=candidate.architecture,
        validation_mse=float(metrics["mse"]),
        validation_mae=float(metrics["mae"]),
        num_parameters=base.count_trainable_parameters(outcome["model"]),
        trained_model=copy.deepcopy(outcome["model"]).cpu(),
        training_time_sec=float(training_time),
        epochs_ran=int(outcome["epochs_ran"]),
    )


def tune_pruned_candidate(
    candidate: base.CandidateEvaluation,
    reference_result: PrePruningReferenceResult,
    pruning_result: MILPPruningResult,
    config: MILPPruningConfig,
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    candidate_index: int,
) -> PrunedCandidateResult:
    """Tune one pruned top-k candidate and return its best validation summary."""

    best_summary: Optional[PrunedCandidateResult] = None
    tuning_start = time.perf_counter()

    for option_index, option in enumerate(base.tuning_grid(config)):
        repeat_metrics: List[Dict[str, float]] = []
        best_model_for_option: Optional[nn.Module] = None
        best_history_for_option: List[Dict[str, float]] = []
        best_option_score = float("inf")
        best_nonzero = reference_result.num_parameters
        best_density = 1.0

        for repeat_index in range(config.retune_repeats):
            run_seed = config.random_seed + 4000 + candidate_index * 1000 + option_index * 100 + repeat_index
            stage3_cfg = base.ModelTrainConfig(
                batch_size=int(option["batch_size"]),
                learning_rate=float(option["learning_rate"]),
                weight_decay=float(option["weight_decay"]),
                epochs=config.full_tuning_epochs,
                patience=config.full_tuning_patience,
                device=config.device,
                seed=run_seed,
            )
            pruned_model = copy.deepcopy(reference_result.trained_model).cpu()
            outcome = masked_train_model(
                model=pruned_model,
                layer_masks=pruning_result.layer_masks,
                train_data=train_data,
                val_data=val_data,
                train_config=stage3_cfg,
            )
            metrics = base.evaluate_model(outcome["model"], val_data, config.device)
            nonzero_parameters = count_pruned_nonzero_parameters(outcome["model"])
            density = float(nonzero_parameters / max(reference_result.num_parameters, 1))
            selection_score = float(metrics["mse"]) + config.complexity_penalty_weight * (
                1.0 + nonzero_parameters / 1000.0
            )
            repeat_metrics.append(
                {
                    "seed": float(run_seed),
                    "validation_mse": float(metrics["mse"]),
                    "validation_mae": float(metrics["mae"]),
                    "selection_score": float(selection_score),
                    "epochs_ran": float(outcome["epochs_ran"]),
                    "nonzero_parameters": float(nonzero_parameters),
                    "density": float(density),
                }
            )
            if selection_score < best_option_score:
                best_option_score = selection_score
                best_model_for_option = copy.deepcopy(outcome["model"]).cpu()
                best_history_for_option = copy.deepcopy(outcome["history"])
                best_nonzero = nonzero_parameters
                best_density = density

        mean_mse = float(np.mean([item["validation_mse"] for item in repeat_metrics]))
        mean_mae = float(np.mean([item["validation_mae"] for item in repeat_metrics]))
        mean_score = float(np.mean([item["selection_score"] for item in repeat_metrics]))
        assert best_model_for_option is not None
        result = PrunedCandidateResult(
            architecture=candidate.architecture,
            best_tuning={
                "learning_rate": float(option["learning_rate"]),
                "batch_size": int(option["batch_size"]),
                "weight_decay": float(option["weight_decay"]),
                "activation": str(option["activation"]),
                "repeats": int(config.retune_repeats),
            },
            prepruning_validation_mse=float(reference_result.validation_mse),
            prepruning_validation_mae=float(reference_result.validation_mae),
            validation_mse=mean_mse,
            validation_mae=mean_mae,
            selection_score=mean_score,
            num_parameters=int(reference_result.num_parameters),
            reference_train_time_sec=float(reference_result.training_time_sec),
            pruning_solve_time_sec=float(pruning_result.solve_time_sec),
            tuning_time_sec=0.0,
            total_candidate_time_sec=0.0,
            nonzero_parameters=int(best_nonzero),
            density=float(best_density),
            pruning_result=pruning_result,
            tuning_runs=repeat_metrics,
            best_history=best_history_for_option,
            model=best_model_for_option,
        )
        if best_summary is None or (
            result.selection_score,
            result.validation_mse,
            result.nonzero_parameters,
        ) < (
            best_summary.selection_score,
            best_summary.validation_mse,
            best_summary.nonzero_parameters,
        ):
            best_summary = result

    assert best_summary is not None
    best_summary.tuning_time_sec = float(time.perf_counter() - tuning_start)
    best_summary.total_candidate_time_sec = float(
        best_summary.reference_train_time_sec
        + best_summary.pruning_solve_time_sec
        + best_summary.tuning_time_sec
    )
    return best_summary


def prune_and_tune_top_candidates(
    candidates: Sequence[base.CandidateEvaluation],
    config: MILPPruningConfig,
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    input_dim: int,
) -> List[PrunedCandidateResult]:
    """Prune each top-k PSO candidate first, then tune the pruned candidates."""

    candidate_results: List[PrunedCandidateResult] = []
    for candidate_index, candidate in enumerate(candidates):
        reference_result = train_reference_for_pruning(
            candidate=candidate,
            config=config,
            train_data=train_data,
            val_data=val_data,
            input_dim=input_dim,
            seed_offset=2000 + candidate_index,
        )
        calibration_data = sample_calibration_subset(
            train_data[0],
            train_data[1],
            config.pruning_calibration_size,
        )
        pruning_result = solve_milp_pruning(reference_result.trained_model, calibration_data, config)
        candidate_results.append(
            tune_pruned_candidate(
                candidate=candidate,
                reference_result=reference_result,
                pruning_result=pruning_result,
                config=config,
                train_data=train_data,
                val_data=val_data,
                candidate_index=candidate_index,
            )
        )

    candidate_results.sort(
        key=lambda item: (
            item.selection_score,
            item.validation_mse,
            item.nonzero_parameters,
        )
    )
    return candidate_results


def save_candidate_comparison_artifacts(
    dataset_name: str,
    candidate_results: Sequence[PrunedCandidateResult],
    output_dir: Path,
) -> Tuple[Path, Optional[Path]]:
    """Save candidate-level performance/time artifacts for the prune-then-tune stage."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for rank, candidate in enumerate(candidate_results, start=1):
        rows.append(
            {
                "rank": int(rank),
                "architecture": base.architecture_to_string(candidate.architecture),
                "prepruning_validation_mse": float(candidate.prepruning_validation_mse),
                "prepruning_validation_mae": float(candidate.prepruning_validation_mae),
                "validation_mse": float(candidate.validation_mse),
                "validation_mae": float(candidate.validation_mae),
                "selection_score": float(candidate.selection_score),
                "reference_train_time_sec": float(candidate.reference_train_time_sec),
                "pruning_solve_time_sec": float(candidate.pruning_solve_time_sec),
                "tuning_time_sec": float(candidate.tuning_time_sec),
                "total_candidate_time_sec": float(candidate.total_candidate_time_sec),
                "num_parameters": int(candidate.num_parameters),
                "nonzero_parameters": int(candidate.nonzero_parameters),
                "density": float(candidate.density),
                "pruning_method": str(candidate.pruning_result.pruning_method),
                "active_total_arcs": int(candidate.pruning_result.active_total_arcs),
                "keep_ratio": float(candidate.pruning_result.keep_ratio),
                "best_learning_rate": float(candidate.best_tuning["learning_rate"]),
                "best_batch_size": int(candidate.best_tuning["batch_size"]),
                "best_weight_decay": float(candidate.best_tuning["weight_decay"]),
            }
        )

    df = pd.DataFrame(rows)
    csv_path = output_dir / f"torch_milp_pruning_candidates_{dataset_name}.csv"
    df.to_csv(csv_path, index=False)

    plot_path: Optional[Path] = None
    if not df.empty:
        labels = [f"C{idx}" for idx in range(1, len(df) + 1)]
        fig, axes = plt.subplots(1, 2, figsize=(max(8, 2.8 * len(df)), 4.5))
        axes[0].bar(labels, df["validation_mse"], color="#c44e52")
        axes[0].set_title("Pruned Candidate Validation MSE")
        axes[0].set_xlabel("Top-k candidate")
        axes[0].set_ylabel("MSE")
        axes[1].bar(labels, df["total_candidate_time_sec"], color="#4c72b0")
        axes[1].set_title("Prune-Then-Tune Candidate Time")
        axes[1].set_xlabel("Top-k candidate")
        axes[1].set_ylabel("Seconds")
        fig.suptitle(f"Top-k pruning comparison: {dataset_name}")
        fig.tight_layout()
        plot_path = output_dir / f"fig_torch_milp_pruning_candidates_{dataset_name}.png"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return csv_path, plot_path


def save_tuning_history_artifacts(
    dataset_name: str,
    candidate_results: Sequence[PrunedCandidateResult],
    output_dir: Path,
) -> Tuple[Path, Optional[Path]]:
    """Save Stage 3 train/validation histories for the top-k pruned candidates."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for rank, candidate in enumerate(candidate_results, start=1):
        architecture_label = base.architecture_to_string(candidate.architecture)
        for record in candidate.best_history:
            rows.append(
                {
                    "rank": int(rank),
                    "architecture": architecture_label,
                    "epoch": int(record["epoch"]),
                    "train_loss": float(record["train_loss"]),
                    "val_mse": None if "val_mse" not in record else float(record["val_mse"]),
                }
            )

    history_df = pd.DataFrame(rows)
    csv_path = output_dir / f"torch_milp_pruning_histories_{dataset_name}.csv"
    history_df.to_csv(csv_path, index=False)

    plot_path: Optional[Path] = None
    if not history_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for rank, candidate in enumerate(candidate_results, start=1):
            label = f"C{rank}: {base.architecture_to_string(candidate.architecture)}"
            candidate_df = history_df[history_df["rank"] == rank]
            axes[0].plot(candidate_df["epoch"], candidate_df["train_loss"], marker="o", label=label)
            if candidate_df["val_mse"].notna().any():
                axes[1].plot(candidate_df["epoch"], candidate_df["val_mse"], marker="o", label=label)
        axes[0].set_title("Stage 3 Train Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[1].set_title("Stage 3 Validation MSE")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MSE")
        for axis in axes:
            axis.legend(fontsize=8)
        fig.suptitle(f"Top-k pruned candidate histories: {dataset_name}")
        fig.tight_layout()
        plot_path = output_dir / f"fig_torch_milp_pruning_histories_{dataset_name}.png"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return csv_path, plot_path


def finalize_selected_candidate(
    selected_candidate: PrunedCandidateResult,
    split_bundle: base.TrainSplitBundle,
    dataset_name: str,
    data_root: Path,
    config: MILPPruningConfig,
) -> Dict[str, object]:
    """Finalize dense and pruned versions of the selected architecture."""

    final_cfg = config.final_train_config(
        learning_rate=float(selected_candidate.best_tuning["learning_rate"]),
        batch_size=int(selected_candidate.best_tuning["batch_size"]),
        weight_decay=float(selected_candidate.best_tuning["weight_decay"]),
        seed=config.random_seed + 12000,
    )

    architecture = selected_candidate.architecture
    dense_final = base.build_model(base.ModelConfig(input_dim=split_bundle.input_dim, architecture=architecture))
    dense_train_start = time.perf_counter()
    dense_outcome = base.train_model(
        dense_final,
        train_data=(split_bundle.X_full_train, split_bundle.y_full_train),
        val_data=None,
        train_config=final_cfg,
    )
    dense_train_time = time.perf_counter() - dense_train_start

    reference_model = base.build_model(base.ModelConfig(input_dim=split_bundle.input_dim, architecture=architecture))
    reference_cfg = config.low_fidelity_train_config(seed=config.random_seed + 13000)
    reference_train_start = time.perf_counter()
    reference_outcome = base.train_model(
        reference_model,
        train_data=(split_bundle.X_full_train, split_bundle.y_full_train),
        val_data=None,
        train_config=reference_cfg,
    )
    reference_train_time = time.perf_counter() - reference_train_start

    calibration_full = sample_calibration_subset(
        split_bundle.X_full_train,
        split_bundle.y_full_train,
        config.pruning_calibration_size,
    )
    pruning_result = solve_milp_pruning(reference_outcome["model"], calibration_full, config)

    pruned_final = copy.deepcopy(reference_outcome["model"]).cpu()
    pruned_train_cfg = base.ModelTrainConfig(
        batch_size=final_cfg.batch_size,
        learning_rate=final_cfg.learning_rate,
        weight_decay=final_cfg.weight_decay,
        epochs=config.final_train_epochs,
        patience=None,
        device=config.device,
        seed=config.random_seed + 14000,
    )
    pruned_train_start = time.perf_counter()
    pruned_outcome = masked_train_model(
        pruned_final,
        layer_masks=pruning_result.layer_masks,
        train_data=(split_bundle.X_full_train, split_bundle.y_full_train),
        val_data=None,
        train_config=pruned_train_cfg,
    )
    pruned_train_time = time.perf_counter() - pruned_train_start

    X_test, y_test = base.prepare_official_test_split(
        dataset_name,
        data_root,
        config,
        fitted_scaler=split_bundle.full_train_feature_scaler,
    )
    dense_metrics = base.evaluate_model(dense_outcome["model"], (X_test, y_test), config.device)
    pruned_metrics = base.evaluate_model(pruned_outcome["model"], (X_test, y_test), config.device)

    output_dir = Path(config.output_dir)
    base.save_prediction_plot(
        y_true=y_test,
        y_pred=dense_metrics["predictions"],
        dataset_name=f"{dataset_name}_dense",
        output_dir=output_dir,
        split_label="official test",
    )
    base.save_prediction_plot(
        y_true=y_test,
        y_pred=pruned_metrics["predictions"],
        dataset_name=f"{dataset_name}_pruned",
        output_dir=output_dir,
        split_label="official test",
    )

    dense_num_parameters = base.count_trainable_parameters(dense_outcome["model"])
    pruned_nonzero = count_pruned_nonzero_parameters(pruned_outcome["model"])

    return {
        "dense_final": {
            "architecture": base.architecture_to_string(architecture),
            "train_time_sec": float(dense_train_time),
            "official_test_mse": float(dense_metrics["mse"]),
            "official_test_mae": float(dense_metrics["mae"]),
            "num_parameters": int(dense_num_parameters),
        },
        "selected_pruning_pipeline": {
            "reference_train_time_sec": float(reference_train_time),
            "pruning_method": str(pruning_result.pruning_method),
            "solver_status": pruning_result.solver_status,
            "success": bool(pruning_result.success),
            "fallback_used": bool(pruning_result.fallback_used),
            "active_input_arcs": int(pruning_result.active_input_arcs),
            "active_output_arcs": int(pruning_result.active_output_arcs),
            "active_total_arcs": int(pruning_result.active_total_arcs),
            "total_arcs": int(pruning_result.total_arcs),
            "per_layer_active_arcs": pruning_result.per_layer_active_arcs,
            "keep_ratio": float(pruning_result.keep_ratio),
            "solve_time_sec": float(pruning_result.solve_time_sec),
            "calibration_mae": float(pruning_result.calibration_mae),
            "objective_value": pruning_result.objective_value,
        },
        "pruned_final": {
            "train_time_sec": float(pruned_train_time),
            "total_pipeline_time_sec": float(reference_train_time + pruning_result.solve_time_sec + pruned_train_time),
            "official_test_mse": float(pruned_metrics["mse"]),
            "official_test_mae": float(pruned_metrics["mae"]),
            "nonzero_parameters": int(pruned_nonzero),
            "density": float(pruned_nonzero / max(dense_num_parameters, 1)),
        },
    }


def save_pruning_report(
    dataset_name: str,
    output_dir: Path,
    report: Dict[str, object],
) -> Path:
    """Save the pruning experiment report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"torch_milp_pruning_report_{dataset_name}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def hidden_layers_label(architecture: base.ArchitectureConfig) -> str:
    """Render hidden-layer widths as a compact label for tables."""

    return "-".join(str(width) for width in architecture.hidden_layers)


def save_before_after_summary_tables(
    results: Sequence[Dict[str, object]],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Save a dense-vs-pruned summary table across all evaluated datasets."""

    rows: List[Dict[str, object]] = []
    for result in results:
        dataset = str(result["Dataset"])
        architecture = str(result["Selected Candidate"])
        hidden_layers = str(result["Selected Hidden Layers"])
        hidden_layer_count = int(result["Selected Hidden Layer Count"])

        rows.append(
            {
                "dataset": dataset,
                "variant": "before_pruning_dense_ann",
                "architecture": architecture,
                "hidden_layer_count": hidden_layer_count,
                "hidden_layers": hidden_layers,
                "pruning_method": "none",
                "parameters_or_nonzero": int(result["Dense Parameters"]),
                "active_arcs": int(result["Dense Total Arcs"]),
                "total_arcs": int(result["Dense Total Arcs"]),
                "keep_ratio": 1.0,
                "validation_mse": float(result["Selected Dense Validation MSE"]),
                "validation_mae": float(result["Selected Dense Validation MAE"]),
                "official_test_mse": float(result["Dense Final Test MSE"]),
                "official_test_mae": float(result["Dense Final Test MAE"]),
                "fit_time_sec": float(result["Dense Final Train Sec"]),
            }
        )
        rows.append(
            {
                "dataset": dataset,
                "variant": "after_pruning_ann",
                "architecture": architecture,
                "hidden_layer_count": hidden_layer_count,
                "hidden_layers": hidden_layers,
                "pruning_method": str(result["Selected Pruning Method"]),
                "parameters_or_nonzero": int(result["Pruned Nonzero Parameters"]),
                "active_arcs": int(result["Pruned Active Arcs"]),
                "total_arcs": int(result["Pruned Total Arcs"]),
                "keep_ratio": float(result["Selected Keep Ratio"]),
                "validation_mse": float(result["Selected Validation MSE"]),
                "validation_mae": float(result["Selected Validation MAE"]),
                "official_test_mse": float(result["Pruned Final Test MSE"]),
                "official_test_mae": float(result["Pruned Final Test MAE"]),
                "fit_time_sec": float(result["Pruned Final Total Pipeline Sec"]),
            }
        )

    summary_df = pd.DataFrame(rows)
    csv_path = output_dir / "torch_milp_pruning_before_after_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    md_lines = [
        "| dataset | variant | architecture | hidden_layer_count | hidden_layers | pruning_method | parameters_or_nonzero | active_arcs | total_arcs | keep_ratio | validation_mse | validation_mae | official_test_mse | official_test_mae | fit_time_sec |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        md_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["variant"]),
                    str(row["architecture"]),
                    str(row["hidden_layer_count"]),
                    str(row["hidden_layers"]),
                    str(row["pruning_method"]),
                    str(row["parameters_or_nonzero"]),
                    str(row["active_arcs"]),
                    str(row["total_arcs"]),
                    f"{float(row['keep_ratio']):.6f}",
                    f"{float(row['validation_mse']):.6f}",
                    f"{float(row['validation_mae']):.6f}",
                    f"{float(row['official_test_mse']):.6f}",
                    f"{float(row['official_test_mae']):.6f}",
                    f"{float(row['fit_time_sec']):.6f}",
                ]
            )
            + " |"
        )

    md_path = output_dir / "torch_milp_pruning_before_after_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def run_full_pipeline(
    dataset_name: str,
    data_root: Path,
    config: MILPPruningConfig,
) -> Dict[str, object]:
    """Run the experimental top-k prune-then-tune pipeline for one dataset."""

    if config.verbose:
        print(f"\n{'=' * 48}\nDataset {dataset_name}: ANN + PSO + MILP pruning\n{'=' * 48}")

    prep_start = time.perf_counter()
    split_bundle = base.prepare_training_split(dataset_name, data_root, config)
    split_bundle, subset_summary = apply_training_fraction(split_bundle, config)
    prep_time = time.perf_counter() - prep_start

    stage1_start = time.perf_counter()
    search_result = base.run_pso_search(
        search_space=config.search_space(),
        pso_config=config.pso_config(),
        train_data=(split_bundle.X_train, split_bundle.y_train),
        val_data=(split_bundle.X_val, split_bundle.y_val),
        input_dim=split_bundle.input_dim,
        config=config,
    )
    stage1_time = time.perf_counter() - stage1_start

    stage1_candidates = select_stage1_candidates(search_result, config)
    candidate_results = prune_and_tune_top_candidates(
        candidates=stage1_candidates,
        config=config,
        train_data=(split_bundle.X_train, split_bundle.y_train),
        val_data=(split_bundle.X_val, split_bundle.y_val),
        input_dim=split_bundle.input_dim,
    )
    selected_candidate = candidate_results[0]

    final_results = finalize_selected_candidate(
        selected_candidate=selected_candidate,
        split_bundle=split_bundle,
        dataset_name=dataset_name,
        data_root=data_root,
        config=config,
    )

    output_dir = Path(config.output_dir)
    base.save_mlp_convergence_plot(search_result.history, dataset_name, output_dir)
    candidate_csv_path, candidate_plot_path = save_candidate_comparison_artifacts(
        dataset_name,
        candidate_results,
        output_dir,
    )
    history_csv_path, history_plot_path = save_tuning_history_artifacts(
        dataset_name,
        candidate_results,
        output_dir,
    )

    report = {
        "dataset_name": dataset_name,
        "rationale": (
            "PSO performs low-cost structure screening, the top-k architectures are "
            "pruned with cheap dense references on a small calibration subset, "
            "using an exact MILP for one- and two-hidden-layer candidates when "
            "the solver returns a solution, and only the pruned candidates are "
            "tuned before final selection."
        ),
        "config": asdict(config),
        "split_summary": {
            "split_method": split_bundle.split_method,
            "normalization_mode": config.normalization_mode,
            "training_fraction": float(config.training_fraction),
            "official_train_units": int(split_bundle.train_unit_count + split_bundle.val_unit_count),
            "search_train_units": int(split_bundle.train_unit_count),
            "validation_units": int(split_bundle.val_unit_count),
            "search_train_windows": int(len(split_bundle.X_train)),
            "validation_windows": int(len(split_bundle.X_val)),
            "full_training_windows": int(len(split_bundle.X_full_train)),
            **subset_summary,
        },
        "stage1": {
            "best_architecture": base.architecture_to_string(search_result.best_candidate.architecture),
            "best_low_fidelity_score": float(search_result.best_candidate.objective_score),
            "best_validation_mse": float(search_result.best_candidate.validation_mse),
            "best_validation_mae": float(search_result.best_candidate.validation_mae),
            "top_k_candidates": [
                {
                    "architecture": base.architecture_to_string(candidate.architecture),
                    "low_fidelity_score": float(candidate.objective_score),
                    "validation_mse": float(candidate.validation_mse),
                    "validation_mae": float(candidate.validation_mae),
                    "num_parameters": int(candidate.num_parameters),
                }
                for candidate in stage1_candidates
            ],
            "time_seconds": float(stage1_time),
        },
        "stage2_prune_then_tune": {
            "candidate_count": int(len(candidate_results)),
            "candidate_artifacts": {
                "csv_path": str(candidate_csv_path),
                "plot_path": None if candidate_plot_path is None else str(candidate_plot_path),
                "history_csv_path": str(history_csv_path),
                "history_plot_path": None if history_plot_path is None else str(history_plot_path),
            },
            "candidate_results": [
                {
                    "architecture": base.architecture_to_string(candidate.architecture),
                    "best_tuning": candidate.best_tuning,
                    "prepruning_validation_mse": float(candidate.prepruning_validation_mse),
                    "prepruning_validation_mae": float(candidate.prepruning_validation_mae),
                    "validation_mse": float(candidate.validation_mse),
                    "validation_mae": float(candidate.validation_mae),
                    "selection_score": float(candidate.selection_score),
                    "num_parameters": int(candidate.num_parameters),
                    "nonzero_parameters": int(candidate.nonzero_parameters),
                    "density": float(candidate.density),
                    "reference_train_time_sec": float(candidate.reference_train_time_sec),
                    "pruning_solve_time_sec": float(candidate.pruning_solve_time_sec),
                    "tuning_time_sec": float(candidate.tuning_time_sec),
                    "total_candidate_time_sec": float(candidate.total_candidate_time_sec),
                    "pruning": {
                        "pruning_method": str(candidate.pruning_result.pruning_method),
                        "solver_status": candidate.pruning_result.solver_status,
                        "success": bool(candidate.pruning_result.success),
                        "fallback_used": bool(candidate.pruning_result.fallback_used),
                        "active_input_arcs": int(candidate.pruning_result.active_input_arcs),
                        "active_output_arcs": int(candidate.pruning_result.active_output_arcs),
                        "active_total_arcs": int(candidate.pruning_result.active_total_arcs),
                        "total_arcs": int(candidate.pruning_result.total_arcs),
                        "per_layer_active_arcs": candidate.pruning_result.per_layer_active_arcs,
                        "keep_ratio": float(candidate.pruning_result.keep_ratio),
                        "calibration_mae": float(candidate.pruning_result.calibration_mae),
                    },
                    "tuning_runs": candidate.tuning_runs,
                    "best_history": candidate.best_history,
                }
                for candidate in candidate_results
            ],
        },
        "selected_candidate": {
            "architecture": base.architecture_to_string(selected_candidate.architecture),
            "best_tuning": selected_candidate.best_tuning,
            "validation_mse": float(selected_candidate.validation_mse),
            "validation_mae": float(selected_candidate.validation_mae),
            "selection_score": float(selected_candidate.selection_score),
            "nonzero_parameters": int(selected_candidate.nonzero_parameters),
            "density": float(selected_candidate.density),
            "total_candidate_time_sec": float(selected_candidate.total_candidate_time_sec),
            "pruning_method": str(selected_candidate.pruning_result.pruning_method),
            "keep_ratio": float(selected_candidate.pruning_result.keep_ratio),
        },
        "final_comparison": final_results,
        "data_prep_time_seconds": float(prep_time),
    }
    report_path = save_pruning_report(dataset_name, output_dir, report)

    if config.verbose:
        print(
            f"[Dense Final] Test MSE {final_results['dense_final']['official_test_mse']:.4f} | "
            f"MAE {final_results['dense_final']['official_test_mae']:.4f}"
        )
        print(
            f"[Pruned Final] Test MSE {final_results['pruned_final']['official_test_mse']:.4f} | "
            f"MAE {final_results['pruned_final']['official_test_mae']:.4f}"
        )
        print(
            f"[Selected] {base.architecture_to_string(selected_candidate.architecture)} | "
            f"Val MSE {selected_candidate.validation_mse:.4f} | "
            f"Candidate time {selected_candidate.total_candidate_time_sec:.2f}s"
        )
        print(f"[Summary] Report saved to {report_path}")

    return {
        "Dataset": dataset_name,
        "Normalization Mode": config.normalization_mode,
        "Split Method": split_bundle.split_method,
        "Training Fraction": float(config.training_fraction),
        "Search Train Windows": int(len(split_bundle.X_train)),
        "Validation Windows": int(len(split_bundle.X_val)),
        "Full Training Windows": int(len(split_bundle.X_full_train)),
        "Stage1 Best Candidate": base.architecture_to_string(search_result.best_candidate.architecture),
        "Stage1 Low-Fidelity Score": float(search_result.best_candidate.objective_score),
        "Top-K Candidate Count": int(len(candidate_results)),
        "Selected Candidate": base.architecture_to_string(selected_candidate.architecture),
        "Selected Hidden Layer Count": int(len(selected_candidate.architecture.hidden_layers)),
        "Selected Hidden Layers": hidden_layers_label(selected_candidate.architecture),
        "Selected Dense Validation MSE": float(selected_candidate.prepruning_validation_mse),
        "Selected Dense Validation MAE": float(selected_candidate.prepruning_validation_mae),
        "Selected Validation MSE": float(selected_candidate.validation_mse),
        "Selected Validation MAE": float(selected_candidate.validation_mae),
        "Selected Pruning Method": str(selected_candidate.pruning_result.pruning_method),
        "Dense Parameters": int(final_results["dense_final"]["num_parameters"]),
        "Dense Total Arcs": int(final_results["selected_pruning_pipeline"]["total_arcs"]),
        "Dense Final Test MSE": float(final_results["dense_final"]["official_test_mse"]),
        "Dense Final Test MAE": float(final_results["dense_final"]["official_test_mae"]),
        "Pruned Nonzero Parameters": int(final_results["pruned_final"]["nonzero_parameters"]),
        "Pruned Density": float(final_results["pruned_final"]["density"]),
        "Pruned Active Arcs": int(final_results["selected_pruning_pipeline"]["active_total_arcs"]),
        "Pruned Total Arcs": int(final_results["selected_pruning_pipeline"]["total_arcs"]),
        "Pruned Final Test MSE": float(final_results["pruned_final"]["official_test_mse"]),
        "Pruned Final Test MAE": float(final_results["pruned_final"]["official_test_mae"]),
        "Selected Keep Ratio": float(selected_candidate.pruning_result.keep_ratio),
        "Selected Candidate Time Sec": float(selected_candidate.total_candidate_time_sec),
        "Final Pruning Solve Time Sec": float(final_results["selected_pruning_pipeline"]["solve_time_sec"]),
        "Dense Final Train Sec": float(final_results["dense_final"]["train_time_sec"]),
        "Pruned Final Train Sec": float(final_results["pruned_final"]["train_time_sec"]),
        "Pruned Final Total Pipeline Sec": float(final_results["pruned_final"]["total_pipeline_time_sec"]),
        "Report Path": str(report_path),
    }


def run_all_datasets(
    data_root: Path,
    config: MILPPruningConfig,
    datasets: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Execute the pruning pipeline on all requested datasets."""

    datasets = datasets or ["FD001", "FD002", "FD003", "FD004"]
    results: List[Dict[str, object]] = []
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset_name in datasets:
        results.append(run_full_pipeline(dataset_name, data_root, config))
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "torch_milp_pruning_summary.csv", index=False)
    comparison_csv_path, comparison_md_path = save_before_after_summary_tables(results, output_dir)
    if config.verbose:
        print("\nMILP pruning experiment summary:")
        print(df)
        print(f"[Summary Table] CSV saved to {comparison_csv_path}")
        print(f"[Summary Table] Markdown saved to {comparison_md_path}")
    return df


def parse_args() -> argparse.Namespace:
    """CLI for the experimental pruning pipeline."""

    parser = argparse.ArgumentParser(
        description=(
            "Experimental CMAPSS pipeline with PSO screening and a SciPy-MILP "
            "arc-pruning stage for 1-to-2-hidden-layer ReLU ANNs "
            "(exact MILP when the solver succeeds, magnitude fallback otherwise)."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/CMAPSSData"))
    parser.add_argument("--datasets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument(
        "--normalization-mode",
        type=str,
        default="global_standard",
        choices=["global_standard", "global_minmax", "per_unit_minmax"],
    )
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--clip-max", type=int, default=125)
    parser.add_argument("--seed", type=int, default=1952)
    parser.add_argument("--output-dir", type=str, default="outputs/torch_pytorch_milp_pruning")
    parser.add_argument("--training-fraction", type=float, default=0.3)
    parser.add_argument("--min-hidden-layers", type=int, default=1)
    parser.add_argument("--max-hidden-layers", type=int, default=2)
    parser.add_argument("--min-neurons", type=int, default=10)
    parser.add_argument("--max-neurons", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--low-fidelity-epochs", type=int, default=10)
    parser.add_argument("--low-fidelity-patience", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--full-tuning-epochs", type=int, default=120)
    parser.add_argument("--full-tuning-patience", type=int, default=12)
    parser.add_argument("--final-train-epochs", type=int, default=100)
    parser.add_argument("--retune-repeats", type=int, default=1)
    parser.add_argument("--tuning-learning-rates", nargs="+", type=float, default=[1e-3, 5e-4, 3e-4])
    parser.add_argument("--tuning-batch-sizes", nargs="+", type=int, default=[128])
    parser.add_argument("--tuning-weight-decays", nargs="+", type=float, default=[0.0, 1e-5, 1e-4])
    parser.add_argument("--n-particles", type=int, default=3)
    parser.add_argument("--n-iter", type=int, default=5)
    parser.add_argument("--pso-inertia", type=float, default=0.5)
    parser.add_argument("--pso-c1", type=float, default=1.5)
    parser.add_argument("--pso-c2", type=float, default=1.5)
    parser.add_argument("--complexity-penalty-weight", type=float, default=0.1)
    parser.add_argument("--pruning-keep-fraction", type=float, default=0.5)
    parser.add_argument("--pruning-calibration-size", type=int, default=16)
    parser.add_argument("--pruning-time-limit-sec", type=float, default=30.0)
    parser.add_argument("--pruning-finetune-epochs", type=int, default=80)
    parser.add_argument("--pruning-finetune-patience", type=int, default=10)
    parser.add_argument("--pruning-exact-budget", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for the pruning experiment."""

    args = parse_args()
    config = MILPPruningConfig(
        seq_len=args.seq_len,
        validation_size=args.validation_size,
        clip_max=args.clip_max,
        random_seed=args.seed,
        normalization_mode=args.normalization_mode,
        training_fraction=args.training_fraction,
        min_hidden_layers=args.min_hidden_layers,
        max_hidden_layers=args.max_hidden_layers,
        min_neurons=args.min_neurons,
        max_neurons=args.max_neurons,
        activation_choices=("relu",),
        search_activation=False,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        low_fidelity_epochs=args.low_fidelity_epochs,
        low_fidelity_patience=args.low_fidelity_patience,
        top_k=args.top_k,
        full_tuning_epochs=args.full_tuning_epochs,
        full_tuning_patience=args.full_tuning_patience,
        final_train_epochs=args.final_train_epochs,
        retune_repeats=args.retune_repeats,
        tuning_learning_rates=tuple(args.tuning_learning_rates),
        tuning_batch_sizes=tuple(args.tuning_batch_sizes),
        tuning_weight_decays=tuple(args.tuning_weight_decays),
        tuning_activation_choices=("relu",),
        n_particles=args.n_particles,
        n_iter=args.n_iter,
        inertia=args.pso_inertia,
        c1=args.pso_c1,
        c2=args.pso_c2,
        complexity_penalty_weight=args.complexity_penalty_weight,
        output_dir=args.output_dir,
        verbose=not args.quiet,
        pruning_keep_fraction=args.pruning_keep_fraction,
        pruning_exact_budget=args.pruning_exact_budget,
        pruning_calibration_size=args.pruning_calibration_size,
        pruning_time_limit_sec=args.pruning_time_limit_sec,
        pruning_finetune_epochs=args.pruning_finetune_epochs,
        pruning_finetune_patience=args.pruning_finetune_patience,
        pruning_max_stage1_candidates=args.top_k,
    )
    base.set_global_seed(args.seed)
    run_all_datasets(args.data_root, config, args.datasets)


if __name__ == "__main__":
    main()
