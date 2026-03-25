"""
Top-level package for measurement-focused CMAPSS utilities.

The package path is preserved for backward compatibility, but the maintained
workflows in this repository are now RUL measurement pipelines.
"""

from .torch_rul_pso import (
    TrainingConfig,
    process_dataset,
    run_full_pipeline,
    run_all_datasets,
)
from .torch_rul_pso_milp_pruning import (
    MILPPruningConfig,
    run_full_pipeline as run_full_pipeline_milp_pruning,
    run_all_datasets as run_all_datasets_milp_pruning,
)

__all__ = [
    "TrainingConfig",
    "MILPPruningConfig",
    "process_dataset",
    "run_full_pipeline",
    "run_all_datasets",
    "run_full_pipeline_milp_pruning",
    "run_all_datasets_milp_pruning",
]
