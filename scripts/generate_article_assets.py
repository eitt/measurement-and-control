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
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DATASETS = ["FD001", "FD002", "FD003", "FD004"]
VARIANT_LABELS = {
    "before_pruning_dense_ann": "Dense",
    "after_pruning_ann": "Pruned",
}
VARIANT_COLORS = {
    "Dense": "#4C78A8",
    "Pruned": "#F58518",
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


def grouped_bar(ax: plt.Axes, datasets: Sequence[str], dense_values: Sequence[float], pruned_values: Sequence[float], ylabel: str, title: str) -> None:
    """Draw a consistent dense-vs-pruned grouped bar chart."""

    x = np.arange(len(datasets), dtype=float)
    width = 0.36
    ax.bar(x - width / 2, dense_values, width=width, label="Dense", color=VARIANT_COLORS["Dense"])
    ax.bar(x + width / 2, pruned_values, width=width, label="Pruned", color=VARIANT_COLORS["Pruned"])
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


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
        title="Official Test MSE",
    )
    grouped_bar(
        axes[0, 1],
        DATASETS,
        test_mae["Dense"].tolist(),
        test_mae["Pruned"].tolist(),
        ylabel="MAE",
        title="Official Test MAE",
    )
    grouped_bar(
        axes[1, 0],
        DATASETS,
        val_mse["Dense"].tolist(),
        val_mse["Pruned"].tolist(),
        ylabel="MSE",
        title="Validation MSE",
    )
    grouped_bar(
        axes[1, 1],
        DATASETS,
        fit_time["Dense"].tolist(),
        fit_time["Pruned"].tolist(),
        ylabel="Seconds",
        title="Fit Time",
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Dense vs pruned performance across the four C-MAPSS subsets", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
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
        if row == 0:
            axes[row, 0].set_title("Stage 3 Train Loss")
        axes[row, 0].set_xlabel("Epoch")

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
        if row == 0:
            axes[row, 1].set_title("Stage 3 Validation MSE")
        axes[row, 1].set_xlabel("Epoch")

        for axis in axes[row]:
            axis.grid(alpha=0.25)

    fig.suptitle("Selected pruned-candidate training histories across FD001-FD004", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_image_grid(image_paths: Sequence[Path], titles: Sequence[str], nrows: int, ncols: int, figsize: Sequence[float], suptitle: str, out_path: Path) -> None:
    """Compose a grid of existing PNG artifacts into one paper figure."""

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_array = np.asarray(axes).reshape(-1)
    for ax, image_path, title in zip(axes_array, image_paths, titles):
        image = mpimg.imread(image_path)
        ax.imshow(image)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for ax in axes_array[len(image_paths) :]:
        ax.axis("off")
    fig.suptitle(suptitle, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
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
            ax.set_title(dataset)
            ax.set_xlabel("PSO iteration")
            ax.set_ylabel("Best low-fidelity score")
            ax.grid(alpha=0.25)
        fig.suptitle("Stage 1 PSO convergence across the four subsets", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
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
    """Create a combined prediction figure for all datasets."""

    dense_csv_paths = [results_dir / f"torch_milp_pruning_predictions_{dataset}_dense.csv" for dataset in DATASETS]
    pruned_csv_paths = [results_dir / f"torch_milp_pruning_predictions_{dataset}_pruned.csv" for dataset in DATASETS]

    if all(path.exists() for path in dense_csv_paths + pruned_csv_paths):
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, dataset, dense_path, pruned_path in zip(axes.flat, DATASETS, dense_csv_paths, pruned_csv_paths):
            dense_df = pd.read_csv(dense_path).sort_values("y_true").reset_index(drop=True)
            pruned_df = pd.read_csv(pruned_path).sort_values("y_true").reset_index(drop=True)
            ax.plot(dense_df.index, dense_df["y_true"], color="#222222", linewidth=2.0, label="True")
            ax.plot(dense_df.index, dense_df["y_pred"], color=VARIANT_COLORS["Dense"], linestyle="--", linewidth=1.8, label="Dense")
            ax.plot(pruned_df.index, pruned_df["y_pred"], color=VARIANT_COLORS["Pruned"], linestyle="-.", linewidth=1.8, label="Pruned")
            ax.set_title(dataset)
            ax.set_xlabel("Sample (sorted by true RUL)")
            ax.set_ylabel("RUL")
            ax.grid(alpha=0.25)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
        fig.suptitle("Official-test predictions for dense and pruned models", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reproducible paper figures and tables from pruning outputs.")
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/torch_pytorch_milp_pruning"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
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

    generate_stage1_convergence(results_dir, generated_dir / "fig_stage1_convergence_grid.png")
    generate_metric_overview(summary_df, before_after_df, generated_dir / "fig_metric_overview.png")
    generate_stage3_histories(results_dir, generated_dir / "fig_stage3_histories_grid.png")
    generate_prediction_grid(results_dir, generated_dir / "fig_prediction_grid.png")

    manifest = {
        "results_dir": str(results_dir),
        "generated_dir": str(generated_dir),
        "tables": [
            "table_experiment_config.tex",
            "table_pruning_results.tex",
            "table_topology_summary.tex",
        ],
        "figures": [
            "fig_stage1_convergence_grid.png",
            "fig_metric_overview.png",
            "fig_stage3_histories_grid.png",
            "fig_prediction_grid.png",
        ],
    }
    write_text(generated_dir / "article_assets_manifest.json", json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
