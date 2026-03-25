"""
Generate reproducible figures and LaTeX table snippets for docs/article.tex.

The script consumes the pruning workflow artifacts under
``outputs/torch_pytorch_milp_pruning`` and writes:

- combined paper figures under ``docs/generated``
- LaTeX table snippets under ``docs/generated``

When raw CSV artifacts are available, the script regenerates standardized
figures directly from the data. If some older runs only contain per-dataset
PNG plots, the script falls back to those images to preserve reproducibility.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import ConnectionPatch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DATASETS = ["FD001", "FD002", "FD003", "FD004"]
PALETTE = {
    "dense": "#2F6690",
    "pruned": "#F28E2B",
    "truth": "#2A2A2A",
    "accent": "#59A14F",
    "muted": "#B07AA1",
    "grid": "#D9D9D9",
}
VARIANT_LABELS = {
    "before_pruning_dense_ann": "Dense",
    "after_pruning_ann": "Pruned",
}
VARIANT_COLORS = {
    "Dense": PALETTE["dense"],
    "Pruned": PALETTE["pruned"],
}
DATASET_COLORS = {
    "FD001": "#4C78A8",
    "FD002": "#F58518",
    "FD003": "#54A24B",
    "FD004": "#E45756",
}


def latex_escape(value: object) -> str:
    """Escape a small subset of LaTeX-sensitive characters for table cells."""

    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def compact_pruning_method(raw_value: str) -> str:
    """Shorten long pruning-method labels for paper tables."""

    mapping = {
        "none": "Dense baseline",
        "magnitude_solver_fallback_after_reduced_exact": "Fallback magnitude",
        "magnitude_multilayer": "Magnitude heuristic",
        "milp_exact": "Exact MILP",
        "milp_exact_reduced_neighborhood": "Reduced exact MILP",
    }
    return mapping.get(raw_value, raw_value.replace("_", " "))


def compact_strategy(raw_value: str) -> str:
    """Shorten solver-strategy labels for paper tables."""

    mapping = {
        "reduced_neighborhood_exact": "Reduced exact",
        "full_exact": "Full exact",
    }
    return mapping.get(raw_value, raw_value.replace("_", " "))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_report(results_dir: Path, dataset: str) -> Dict[str, object]:
    report_path = results_dir / f"torch_milp_pruning_report_{dataset}.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing report file: {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def panel_tag(ax: plt.Axes, text: str) -> None:
    """Add a small in-panel label without using subplot titles."""

    ax.text(
        0.02,
        0.96,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )


def parse_architecture_string(raw_value: str) -> Tuple[str, List[int]]:
    """Parse strings like ``relu:[98, 87]`` into activation and widths."""

    activation, layers_text = raw_value.split(":", maxsplit=1)
    stripped = layers_text.strip().strip("[]")
    if not stripped:
        return activation.strip(), []
    hidden_layers = [int(part.strip()) for part in stripped.split(",") if part.strip()]
    return activation.strip(), hidden_layers


def build_config_from_report(report: Dict[str, object]):
    """Rehydrate the pruning config from a saved report."""

    from src.measurement_control import torch_rul_pso_milp_pruning as pruning

    config_dict = copy.deepcopy(report["config"])
    tuple_keys = [
        "activation_choices",
        "tuning_learning_rates",
        "tuning_batch_sizes",
        "tuning_weight_decays",
        "tuning_activation_choices",
    ]
    for key in tuple_keys:
        if key in config_dict:
            config_dict[key] = tuple(config_dict[key])
    return pruning.MILPPruningConfig(**config_dict)


def grouped_bar(ax: plt.Axes, datasets: Sequence[str], dense_values: Sequence[float], pruned_values: Sequence[float], ylabel: str, panel_text: str) -> None:
    """Draw a consistent dense-vs-pruned grouped bar chart."""

    x = np.arange(len(datasets), dtype=float)
    width = 0.36
    ax.bar(x - width / 2, dense_values, width=width, label="Dense", color=VARIANT_COLORS["Dense"])
    ax.bar(x + width / 2, pruned_values, width=width, label="Pruned", color=VARIANT_COLORS["Pruned"])
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(ylabel)
    panel_tag(ax, panel_text)


def generate_metric_overview(summary_df: pd.DataFrame, before_after_df: pd.DataFrame, out_path: Path) -> None:
    """Create a 2x2 overview of dense-vs-pruned performance across datasets."""

    before_after = before_after_df.copy()
    before_after["variant_label"] = before_after["variant"].map(VARIANT_LABELS)
    before_after = before_after[before_after["dataset"].isin(DATASETS)]

    def pivot_metric(metric: str) -> pd.DataFrame:
        table = before_after.pivot(index="dataset", columns="variant_label", values=metric)
        return table.reindex(DATASETS)

    val_mse = pivot_metric("validation_mse")
    test_mse = pivot_metric("official_test_mse")
    test_mae = pivot_metric("official_test_mae")
    fit_time = pivot_metric("fit_time_sec")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    grouped_bar(
        axes[0, 0],
        DATASETS,
        test_mse["Dense"].tolist(),
        test_mse["Pruned"].tolist(),
        ylabel="MSE",
        panel_text="(a) Official-test MSE",
    )
    grouped_bar(
        axes[0, 1],
        DATASETS,
        test_mae["Dense"].tolist(),
        test_mae["Pruned"].tolist(),
        ylabel="MAE",
        panel_text="(b) Official-test MAE",
    )
    grouped_bar(
        axes[1, 0],
        DATASETS,
        val_mse["Dense"].tolist(),
        val_mse["Pruned"].tolist(),
        ylabel="MSE",
        panel_text="(c) Validation MSE",
    )
    grouped_bar(
        axes[1, 1],
        DATASETS,
        fit_time["Dense"].tolist(),
        fit_time["Pruned"].tolist(),
        ylabel="Seconds",
        panel_text="(d) Fit time",
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_rul_cap_example(out_path: Path) -> None:
    """Create a simple piecewise-linear RUL target illustration."""

    sns.set_theme(style="whitegrid")
    max_cycle = 220
    clip_max = 125
    cycles = np.arange(1, max_cycle + 1)
    uncapped = max_cycle - cycles
    capped = np.minimum(uncapped, clip_max)
    transition_cycle = max_cycle - clip_max

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(cycles, uncapped, color=PALETTE["muted"], linewidth=2.0, linestyle="--", label="Uncapped RUL")
    ax.plot(cycles, capped, color=PALETTE["dense"], linewidth=2.4, label="Capped target")
    ax.axvspan(cycles.min(), transition_cycle, color=PALETTE["accent"], alpha=0.10)
    ax.axvline(transition_cycle, color=PALETTE["accent"], linewidth=1.2, linestyle=":")
    ax.annotate(
        "Healthy-phase cap",
        xy=(transition_cycle * 0.55, clip_max),
        xytext=(12, 12),
        textcoords="offset points",
        fontsize=9,
        color=PALETTE["accent"],
    )
    ax.annotate(
        "Linear decay near failure",
        xy=(transition_cycle + 55, 55),
        xytext=(12, -18),
        textcoords="offset points",
        fontsize=9,
        color=PALETTE["dense"],
    )
    ax.set_xlabel("Cycle")
    ax.set_ylabel("RUL target")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_window_embedding_example(out_path: Path) -> None:
    """Create a visual explanation of the sliding-window flattening step."""

    sns.set_theme(style="white")
    window = np.array(
        [
            [0.18, 0.44, 0.73, 0.66],
            [0.22, 0.47, 0.69, 0.61],
            [0.29, 0.53, 0.63, 0.55],
            [0.34, 0.58, 0.56, 0.48],
            [0.40, 0.64, 0.49, 0.41],
        ],
        dtype=float,
    )
    flattened = window.reshape(1, -1)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1.2, 1.8]})
    im0 = axes[0].imshow(window, cmap="Blues", aspect="auto", vmin=0.0, vmax=1.0)
    axes[0].set_xlabel("Retained feature")
    axes[0].set_ylabel("Windowed cycle")
    axes[0].set_xticks(range(window.shape[1]))
    axes[0].set_xticklabels([f"$f_{j+1}$" for j in range(window.shape[1])])
    axes[0].set_yticks(range(window.shape[0]))
    axes[0].set_yticklabels([f"$t-{window.shape[0]-1-j}$" for j in range(window.shape[0])])
    panel_tag(axes[0], "(a) Window matrix")

    im1 = axes[1].imshow(flattened, cmap="Blues", aspect="auto", vmin=0.0, vmax=1.0)
    axes[1].set_xlabel("Flattened coordinate")
    axes[1].set_yticks([])
    tick_positions = [0, 3, 4, 7, 8, 11, 12, 15, 16, 19]
    tick_labels = ["$x_{t-4}$", "", "$x_{t-3}$", "", "$x_{t-2}$", "", "$x_{t-1}$", "", "$x_t$", ""]
    axes[1].set_xticks(tick_positions)
    axes[1].set_xticklabels(tick_labels, rotation=0)
    panel_tag(axes[1], "(b) Flattened embedding")

    connector = ConnectionPatch(
        xyA=(1.02, 0.50),
        coordsA=axes[0].transAxes,
        xyB=(-0.02, 0.50),
        coordsB=axes[1].transAxes,
        arrowstyle="->",
        linewidth=1.5,
        color=PALETTE["dense"],
    )
    fig.add_artist(connector)
    cbar = fig.colorbar(im1, ax=axes, fraction=0.03, pad=0.03)
    cbar.set_label("Normalized value")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_stage3_histories(results_dir: Path, out_path: Path) -> None:
    """Create a 4x2 grid of Stage 3 train-loss and validation-MSE histories."""

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(12, 13), sharex=False)

    for row, dataset in enumerate(DATASETS):
        history_path = results_dir / f"torch_milp_pruning_histories_{dataset}.csv"
        history_df = pd.read_csv(history_path)
        selected_df = history_df[history_df["rank"] == 1].copy()
        color = DATASET_COLORS[dataset]

        if selected_df.empty:
            continue

        best_epoch = None
        best_val = None
        if selected_df["val_mse"].notna().any():
            best_idx = selected_df["val_mse"].astype(float).idxmin()
            best_epoch = float(selected_df.loc[best_idx, "epoch"])
            best_val = float(selected_df.loc[best_idx, "val_mse"])

        axes[row, 0].plot(selected_df["epoch"], selected_df["train_loss"], color=color, linewidth=2.0)
        axes[row, 0].scatter(selected_df["epoch"].iloc[-1], selected_df["train_loss"].iloc[-1], color=color, s=22)
        axes[row, 0].set_ylabel(f"{dataset}\nLoss")
        axes[row, 0].set_xlabel("Epoch")
        panel_tag(axes[row, 0], f"{dataset} train")

        axes[row, 1].plot(selected_df["epoch"], selected_df["val_mse"], color=color, linewidth=2.0)
        if best_epoch is not None and best_val is not None:
            axes[row, 1].scatter(best_epoch, best_val, color=color, s=30, zorder=3)
            axes[row, 1].annotate(
                f"best={best_epoch:.0f}",
                xy=(best_epoch, best_val),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        axes[row, 1].set_ylabel("MSE")
        axes[row, 1].set_xlabel("Epoch")
        panel_tag(axes[row, 1], f"{dataset} validation")

        for axis in axes[row]:
            axis.grid(alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_image_grid(image_paths: Sequence[Path], titles: Sequence[str], nrows: int, ncols: int, figsize: Sequence[float], suptitle: str, out_path: Path) -> None:
    """Compose a grid of existing PNG artifacts into one paper figure."""

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_array = np.asarray(axes).reshape(-1)
    for ax, image_path, title in zip(axes_array, image_paths, titles):
        image = mpimg.imread(image_path)
        crop_rows = int(image.shape[0] * 0.08)
        cropped = image[crop_rows:, ...] if crop_rows > 0 else image
        ax.imshow(cropped)
        panel_tag(ax, title)
        ax.axis("off")
    for ax in axes_array[len(image_paths) :]:
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_stage1_convergence(results_dir: Path, out_path: Path) -> None:
    """Create a 2x2 Stage 1 convergence figure from CSVs or existing PNGs."""

    csv_paths = [results_dir / f"torch_milp_pruning_stage1_history_{dataset}.csv" for dataset in DATASETS]
    if all(path.exists() for path in csv_paths):
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for ax, dataset, csv_path in zip(axes.flat, DATASETS, csv_paths):
            df = pd.read_csv(csv_path)
            color = DATASET_COLORS[dataset]
            ax.plot(df["iteration"], df["best_low_fidelity_score"], color=color, linewidth=2.0, marker="o")
            ax.set_xlabel("PSO iteration")
            ax.set_ylabel("Best low-fidelity score")
            panel_tag(ax, dataset)
            ax.grid(alpha=0.25)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    png_paths = [results_dir / f"fig_torch_mlp_convergence_{dataset}.png" for dataset in DATASETS]
    build_image_grid(
        image_paths=png_paths,
        titles=DATASETS,
        nrows=2,
        ncols=2,
        figsize=(11, 8),
        suptitle="Stage 1 PSO convergence across the four subsets",
        out_path=out_path,
    )


def generate_prediction_grid(results_dir: Path, out_path: Path) -> None:
    """Create a dataset-wide official-test prediction figure for all subsets."""

    dense_csv_paths = [results_dir / f"torch_milp_pruning_predictions_{dataset}_dense.csv" for dataset in DATASETS]
    pruned_csv_paths = [results_dir / f"torch_milp_pruning_predictions_{dataset}_pruned.csv" for dataset in DATASETS]

    if all(path.exists() for path in dense_csv_paths + pruned_csv_paths):
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
        for ax, dataset, dense_path, pruned_path in zip(axes.flat, DATASETS, dense_csv_paths, pruned_csv_paths):
            dense_df = pd.read_csv(dense_path)
            pruned_df = pd.read_csv(pruned_path)
            merged = dense_df.merge(
                pruned_df[["sample_index", "y_pred"]].rename(columns={"y_pred": "y_pred_pruned"}),
                on="sample_index",
                how="inner",
            )
            y_true = merged["y_true"].to_numpy(dtype=float)
            dense_pred = merged["y_pred"].to_numpy(dtype=float)
            pruned_pred = merged["y_pred_pruned"].to_numpy(dtype=float)
            diagonal_max = max(float(np.max(y_true)), float(np.max(dense_pred)), float(np.max(pruned_pred)))

            ax.scatter(y_true, dense_pred, color=VARIANT_COLORS["Dense"], alpha=0.42, s=28, edgecolors="none", label="Dense")
            ax.scatter(y_true, pruned_pred, color=VARIANT_COLORS["Pruned"], alpha=0.42, s=28, marker="^", edgecolors="none", label="Pruned")
            ax.plot([0, diagonal_max], [0, diagonal_max], color=PALETTE["truth"], linewidth=1.2, linestyle="--", label="Ideal")
            ax.set_xlabel("True RUL")
            ax.set_ylabel("Predicted RUL")
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.25)
            panel_tag(ax, dataset)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles[:3], labels[:3], loc="upper center", ncol=3, frameon=False)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    image_paths: List[Path] = []
    titles: List[str] = []
    for dataset in DATASETS:
        image_paths.append(results_dir / f"fig_torch_rul_prediction_{dataset}_dense.png")
        titles.append(f"{dataset} dense")
        image_paths.append(results_dir / f"fig_torch_rul_prediction_{dataset}_pruned.png")
        titles.append(f"{dataset} pruned")
    build_image_grid(
        image_paths=image_paths,
        titles=titles,
        nrows=4,
        ncols=2,
        figsize=(12, 16),
        suptitle="Official-test prediction panels for dense and pruned models",
        out_path=out_path,
    )


def write_experiment_config_table(reports: Sequence[Dict[str, object]], out_path: Path) -> None:
    """Generate a LaTeX table with the shared pruning configuration."""

    if not reports:
        raise ValueError("At least one report is required to build the config table.")
    base_report = reports[0]
    config = base_report["config"]
    rows = [
        ("Normalization", config["normalization_mode"]),
        ("Sequence length", config["seq_len"]),
        ("RUL clip", config["clip_max"]),
        ("Validation split", f"{float(config['validation_size']):.2f}"),
        ("Training fraction", f"{float(config['training_fraction']):.2f}"),
        ("PSO particles", int(config["n_particles"])),
        ("PSO iterations", int(config["n_iter"])),
        ("Stage 1 epochs", int(config["low_fidelity_epochs"])),
        ("Stage 1 patience", int(config["low_fidelity_patience"])),
        ("Top-k candidates", int(config["top_k"])),
        ("Stage 3 tuning epochs", int(config["full_tuning_epochs"])),
        ("Stage 3 tuning patience", int(config["full_tuning_patience"])),
        ("Final retraining epochs", int(config["final_train_epochs"])),
        ("Pruning keep fraction", f"{float(config['pruning_keep_fraction']):.2f}"),
        ("Calibration subset size", int(config["pruning_calibration_size"])),
        ("MILP time limit (s)", f"{float(config['pruning_time_limit_sec']):.1f}"),
        ("Two-layer pruning strategy", compact_strategy(str(config["two_hidden_milp_strategy"]))),
    ]
    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \caption{Common experimental configuration used by the measurement-only ANN + PSO + pruning study.}",
        r"    \label{tab:experiment_config}",
        r"    \begin{tabular}{ll}",
        r"        \toprule",
        r"        \textbf{Setting} & \textbf{Value} \\",
        r"        \midrule",
    ]
    for setting, value in rows:
        lines.append(f"        {latex_escape(setting)} & {latex_escape(value)} \\\\")
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"\end{table}",
        ]
    )
    write_text(out_path, "\n".join(lines) + "\n")


def write_pruning_results_table(before_after_df: pd.DataFrame, out_path: Path) -> None:
    """Generate a LaTeX results table with dense-vs-pruned validation/test metrics."""

    rows = []
    for dataset in DATASETS:
        dense_row = before_after_df[
            (before_after_df["dataset"] == dataset) & (before_after_df["variant"] == "before_pruning_dense_ann")
        ].iloc[0]
        pruned_row = before_after_df[
            (before_after_df["dataset"] == dataset) & (before_after_df["variant"] == "after_pruning_ann")
        ].iloc[0]
        rows.append(
            (
                dataset,
                dense_row["validation_mse"],
                pruned_row["validation_mse"],
                dense_row["official_test_mse"],
                pruned_row["official_test_mse"],
                dense_row["official_test_mae"],
                pruned_row["official_test_mae"],
            )
        )

    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \caption{Dense-versus-pruned performance for the selected architecture in each C-MAPSS subset. Validation metrics come from the internal held-out engine split; official-test metrics use the benchmark test partition once at the end.}",
        r"    \label{tab:pruning_results}",
        r"    \resizebox{\textwidth}{!}{%",
        r"    \begin{tabular}{lrrrrrr}",
        r"        \toprule",
        r"        \textbf{Dataset} & \textbf{Dense Val. MSE} & \textbf{Pruned Val. MSE} & \textbf{Dense Test MSE} & \textbf{Pruned Test MSE} & \textbf{Dense Test MAE} & \textbf{Pruned Test MAE} \\",
        r"        \midrule",
    ]
    for row in rows:
        dataset, dense_val, pruned_val, dense_test_mse, pruned_test_mse, dense_test_mae, pruned_test_mae = row
        lines.append(
            "        "
            + " & ".join(
                [
                    latex_escape(dataset),
                    f"{dense_val:.2f}",
                    f"{pruned_val:.2f}",
                    f"{dense_test_mse:.2f}",
                    f"{pruned_test_mse:.2f}",
                    f"{dense_test_mae:.2f}",
                    f"{pruned_test_mae:.2f}",
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}}",
            r"\end{table}",
        ]
    )
    write_text(out_path, "\n".join(lines) + "\n")


def write_topology_table(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Generate a LaTeX table summarizing topology and compression ratios."""

    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \caption{Selected dense topologies and resulting pruning compression for each subset. The reported pruning method corresponds to the final structure actually used in the selected candidate.}",
        r"    \label{tab:topology_summary}",
        r"    \resizebox{\textwidth}{!}{%",
        r"    \begin{tabular}{lrrrrrrl}",
        r"        \toprule",
        r"        \textbf{Dataset} & \textbf{Layers} & \textbf{Widths} & \textbf{Dense Params.} & \textbf{Nonzero Params.} & \textbf{Active Arcs} & \textbf{Keep Ratio} & \textbf{Method} \\",
        r"        \midrule",
    ]
    for dataset in DATASETS:
        row = summary_df[summary_df["Dataset"] == dataset].iloc[0]
        method_label = compact_pruning_method(str(row["Selected Pruning Method"]))
        lines.append(
            "        "
            + " & ".join(
                [
                    latex_escape(dataset),
                    f"{int(row['Selected Hidden Layer Count'])}",
                    latex_escape(row["Selected Hidden Layers"]),
                    f"{int(row['Dense Parameters'])}",
                    f"{int(row['Pruned Nonzero Parameters'])}",
                    f"{int(row['Pruned Active Arcs'])}",
                    f"{float(row['Selected Keep Ratio']):.3f}",
                    latex_escape(method_label),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}}",
            r"\end{table}",
        ]
    )
    write_text(out_path, "\n".join(lines) + "\n")


def write_reduced_neighborhood_table(reports: Sequence[Dict[str, object]], out_path: Path) -> None:
    """Generate a LaTeX table summarizing the heuristic neighborhood parameters."""

    if not reports:
        raise ValueError("At least one report is required to build the reduced-neighborhood table.")
    config = reports[0]["config"]
    rows = [
        (r"$\alpha$", "Global pruning ratio", f"{1.0 - float(config['pruning_keep_fraction']):.2f}"),
        (r"$\rho_{\mathrm{free}}$", "Free-arc fraction before clipping", f"{float(config['reduced_milp_free_arc_fraction']):.2f}"),
        (r"$F_{\min}$", "Minimum free binary arcs", int(config["reduced_milp_min_free_arcs"])),
        (r"$F_{\max}$", "Maximum free binary arcs", int(config["reduced_milp_max_free_arcs"])),
        (r"$R_{\mathrm{LS}}$", "Local-search rounds", int(config["activation_local_search_rounds"])),
        (r"$P_{\mathrm{LS}}$", "Swap pool size per side", int(config["activation_local_search_pool_size"])),
        (r"$E_{\max}$", "Maximum swap evaluations per round", int(config["activation_local_search_max_evals"])),
        ("Time limit", "MILP time limit per solve (s)", f"{float(config['pruning_time_limit_sec']):.1f}"),
        ("Calibration", "Teacher-matching subset size", int(config["pruning_calibration_size"])),
    ]
    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \caption{Parameters of the activation-aware reduced-neighborhood heuristic used before the exact two-hidden-layer MILP.}",
        r"    \label{tab:reduced_neighborhood_params}",
        r"    \begin{tabular}{lll}",
        r"        \toprule",
        r"        \textbf{Symbol} & \textbf{Meaning} & \textbf{Value} \\",
        r"        \midrule",
    ]
    for symbol, meaning, value in rows:
        lines.append(f"        {symbol} & {latex_escape(meaning)} & {latex_escape(value)} \\\\")
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"\end{table}",
        ]
    )
    write_text(out_path, "\n".join(lines) + "\n")


def ensure_prediction_csvs(results_dir: Path, data_root: Path) -> None:
    """Rebuild missing dense/pruned prediction CSVs from saved reports."""

    missing_pairs = []
    for dataset in DATASETS:
        dense_path = results_dir / f"torch_milp_pruning_predictions_{dataset}_dense.csv"
        pruned_path = results_dir / f"torch_milp_pruning_predictions_{dataset}_pruned.csv"
        if not dense_path.exists() or not pruned_path.exists():
            missing_pairs.append((dataset, dense_path, pruned_path))
    if not missing_pairs:
        return

    from src.measurement_control import torch_rul_pso as base
    from src.measurement_control import torch_rul_pso_milp_pruning as pruning

    for dataset, dense_path, pruned_path in missing_pairs:
        report = load_report(results_dir, dataset)
        config = build_config_from_report(report)
        activation, hidden_layers = parse_architecture_string(str(report["selected_candidate"]["architecture"]))
        best_tuning = report["selected_candidate"]["best_tuning"]

        split_bundle = base.prepare_training_split(dataset, data_root, config)
        split_bundle, _ = pruning.apply_training_fraction(split_bundle, config)
        architecture = base.ArchitectureConfig(hidden_layers=hidden_layers, activation=activation)
        model_config = base.ModelConfig(input_dim=split_bundle.input_dim, architecture=architecture)
        final_cfg = config.final_train_config(
            learning_rate=float(best_tuning["learning_rate"]),
            batch_size=int(best_tuning["batch_size"]),
            weight_decay=float(best_tuning["weight_decay"]),
            seed=config.random_seed + 12000,
        )

        dense_model = base.build_model(model_config)
        dense_outcome = base.train_model(
            dense_model,
            train_data=(split_bundle.X_full_train, split_bundle.y_full_train),
            val_data=None,
            train_config=final_cfg,
        )

        calibration_full = pruning.sample_calibration_subset(
            split_bundle.X_full_train,
            split_bundle.y_full_train,
            config.pruning_calibration_size,
        )
        pruning_result = pruning.solve_milp_pruning(dense_outcome["model"], calibration_full, config)
        pruned_model = copy.deepcopy(dense_outcome["model"])
        pruned_cfg = base.ModelTrainConfig(
            batch_size=final_cfg.batch_size,
            learning_rate=final_cfg.learning_rate,
            weight_decay=final_cfg.weight_decay,
            epochs=config.pruning_finetune_epochs,
            patience=None,
            device=config.device,
            seed=config.random_seed + 13000,
        )
        pruned_outcome = pruning.masked_train_model(
            pruned_model,
            layer_masks=pruning_result.layer_masks,
            train_data=(split_bundle.X_full_train, split_bundle.y_full_train),
            val_data=None,
            train_config=pruned_cfg,
        )

        X_test, y_test = base.prepare_official_test_split(
            dataset,
            data_root,
            config,
            fitted_scaler=split_bundle.full_train_feature_scaler,
        )
        dense_metrics = base.evaluate_model(dense_outcome["model"], (X_test, y_test), config.device)
        pruned_metrics = base.evaluate_model(pruned_outcome["model"], (X_test, y_test), config.device)

        dense_df = pd.DataFrame({"sample_index": np.arange(len(y_test)), "y_true": y_test, "y_pred": dense_metrics["predictions"]})
        pruned_df = pd.DataFrame({"sample_index": np.arange(len(y_test)), "y_true": y_test, "y_pred": pruned_metrics["predictions"]})
        dense_df.to_csv(dense_path, index=False)
        pruned_df.to_csv(pruned_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reproducible paper figures and tables from pruning outputs.")
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/torch_pytorch_milp_pruning"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--data-root", type=Path, default=Path("data/CMAPSSData"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    generated_dir = args.docs_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(results_dir / "torch_milp_pruning_summary.csv")
    before_after_df = pd.read_csv(results_dir / "torch_milp_pruning_before_after_summary.csv")
    reports = [load_report(results_dir, dataset) for dataset in DATASETS]

    write_experiment_config_table(reports, generated_dir / "table_experiment_config.tex")
    write_pruning_results_table(before_after_df, generated_dir / "table_pruning_results.tex")
    write_topology_table(summary_df, generated_dir / "table_topology_summary.tex")
    write_reduced_neighborhood_table(reports, generated_dir / "table_reduced_neighborhood_params.tex")

    generate_rul_cap_example(generated_dir / "fig_rul_target_example.png")
    generate_window_embedding_example(generated_dir / "fig_window_embedding_example.png")
    generate_stage1_convergence(results_dir, generated_dir / "fig_stage1_convergence_grid.png")
    generate_metric_overview(summary_df, before_after_df, generated_dir / "fig_metric_overview.png")
    generate_stage3_histories(results_dir, generated_dir / "fig_stage3_histories_grid.png")
    ensure_prediction_csvs(results_dir, args.data_root)
    generate_prediction_grid(results_dir, generated_dir / "fig_prediction_grid.png")

    manifest = {
        "results_dir": str(results_dir),
        "generated_dir": str(generated_dir),
        "tables": [
            "table_experiment_config.tex",
            "table_pruning_results.tex",
            "table_topology_summary.tex",
            "table_reduced_neighborhood_params.tex",
        ],
        "figures": [
            "fig_rul_target_example.png",
            "fig_window_embedding_example.png",
            "fig_stage1_convergence_grid.png",
            "fig_metric_overview.png",
            "fig_stage3_histories_grid.png",
            "fig_prediction_grid.png",
        ],
    }
    write_text(generated_dir / "article_assets_manifest.json", json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
