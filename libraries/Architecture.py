from dataclasses import dataclass, field
from typing import Any, Tuple

@dataclass(frozen=True)
class Architecture:
    """Descripción discreta de una MLP candidata."""

    hidden_sizes: Tuple[int, ...]
    activation: str
    dropout: float
    loss: str

    def key(self) -> Tuple[Any, ...]:
        """Clave estable y serializable para caché y eliminación de duplicados."""
        return (
            tuple(int(value) for value in self.hidden_sizes),
            str(self.activation),
            round(float(self.dropout), 4),
            str(self.loss),
        )

    def label(self) -> str:
        return (
            f"hidden={self.hidden_sizes}, activation={self.activation}, "
            f"dropout={self.dropout:.2f}, loss={self.loss}"
        )
