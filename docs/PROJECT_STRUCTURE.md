# Project Organization Guide

This repository now uses a clearer structure to improve maintainability and onboarding.

## Suggested Structure

```text
measurement-and-control/
|-- data/                         # Raw CMAPSS and related datasets
|-- docs/
|   |-- PROJECT_STRUCTURE.md      # This organization guide
|-- outputs/
|   |-- torch_pytorch/            # Generated plots + CSV from PyTorch pipeline
|-- scripts/
|   |-- run_torch_pipeline.py     # Entry script for PyTorch PSO-ANN pipeline
|-- src/
|   |-- measurement_control/
|       |-- __init__.py
|       |-- torch_rul_pso.py      # Fully commented PyTorch implementation
|-- code.py                       # Existing reference/fixed implementation
|-- main_universal.py             # Existing universal workflow
|-- measure_control_colab.ipynb   # Existing notebook workflow
|-- requirements.txt              # Python dependencies
```

## Management Principles

- Keep reusable logic in `src/measurement_control/`.
- Keep runnable entrypoints in `scripts/`.
- Keep analysis explanations and decisions in `docs/`.
- Keep generated artifacts in `outputs/` (instead of project root whenever possible).
- Keep original baseline scripts (`code.py`, notebook) for reproducibility and comparison.

## New PyTorch Pipeline

- Main implementation: `src/measurement_control/torch_rul_pso.py`
- Entrypoint script: `scripts/run_torch_pipeline.py`
- Uses:
  - two-hidden-layer ANN in PyTorch
  - PSO architecture search following `code.py` style
  - notebook-aligned optimization defaults (`bounds=(10,100)`, `n_particles=5`, `n_iter=5`, `epochs=1000`)
