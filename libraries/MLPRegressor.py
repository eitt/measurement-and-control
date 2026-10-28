class MLPRegressor:
    """Envoltorio para entrenamiento, predicción y restauración de estados."""

    def __init__(
        self,
        input_size: int,
        architecture: Architecture,
        learning_rate: float,
        weight_decay: float,
        device: torch.device,
    ) -> None:
        self.input_size = int(input_size)
        self.architecture = architecture
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.device = device

        self.model = TorchMLPRegressor(
            input_size=self.input_size,
            architecture=self.architecture,
        ).to(self.device)

        # El criterio de pérdida estará atado al objetivo seleccionado. Si se quiere usar SmoothL1Loss para entrenamiento ordinario, se puede pasar None.
        self.criterion = make_criterion(architecture.loss)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def get_weights(self) -> Dict[str, torch.Tensor]:
        return move_nested_tensors_to_cpu(self.model.state_dict())

    def load_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    def get_optimizer_state(self) -> Dict[str, Any]:
        return move_nested_tensors_to_cpu(self.optimizer.state_dict())

    def load_optimizer_state(self, state_dict: Dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict)
        move_optimizer_state_to_device(self.optimizer, self.device)

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        X_tensor = torch.as_tensor(
            X,
            dtype=torch.float32,
            device=self.device,
        )
        self.model.eval()

        with torch.no_grad():
            predictions = self.model(X_tensor)

        return predictions.detach().cpu().numpy().reshape(-1)

    def predict_clipped(self, X: np.ndarray) -> np.ndarray:
        return np.clip(self.predict_raw(X), 0.0, float(RUL_MAX))