import torch
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

@dataclass
class TrainingState:
    """Estado de continuación y mejor checkpoint observado."""

    total_epochs: int
    resume_weights: Dict[str, torch.Tensor]
    optimizer_state: Dict[str, Any]
    best_weights: Dict[str, torch.Tensor]
    best_epoch: int
    best_metrics: Dict[str, float]
    final_training_loss: float
