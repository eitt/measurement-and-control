# Step 1: Import required libraries.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Configure plotting style for consistent visuals.

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})

class TorchMLPRegressor(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=1):
        super().__init__()

        layers = []
        in_features = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size

        layers.append(nn.Linear(in_features, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class MLPRegressor:
    def __init__(self, input_size, hidden_sizes, learning_rate=0.001, epochs=1000):
        self.model = TorchMLPRegressor(input_size, hidden_sizes)
        self.epochs = epochs
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).view(-1,1)

        self.model.train()

        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            pred = self.model(X)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        self.model.eval()

        with torch.no_grad():
            pred = self.model(X)

        return pred.numpy().flatten()

# Step 3: Define data-loading and RUL-target helper functions.

def load_cmapss(fd_path: str) -> pd.DataFrame:
    # Step 3.1: Define the CMAPSS column schema.
    col_names = [f"col_{i}" for i in range(1, 27)]
    df = pd.read_csv(fd_path, sep=r"\s+", header=None, names=col_names)
    return df

def load_cmapss_rul(fd_path: str) -> pd.DataFrame:
    col_names = ['rul']
    df = pd.read_csv(fd_path, sep=r"\s+", header=None, names=col_names)
    return df

def compute_rul(train_df: pd.DataFrame, base_rul: pd.Series = None, clip_max: int = 125) -> pd.Series:
    # Step 3.2: Compute max cycle by engine id.
    grouped = train_df.groupby('col_1')['col_2'].max()
    if base_rul is not None:
        grouped = grouped + base_rul
    max_cycle = train_df['col_1'].map(grouped)
    rul = (max_cycle - train_df['col_2']).clip(upper=clip_max)
    return rul

def select_non_flat_features(df: pd.DataFrame, threshold: float = 1e-5) -> List[str]:
    # Step 3.3: Select features using standard deviations from training data only.
    feature_cols = list(df.columns[2:])
    stds = df[feature_cols].std()
    selected = stds[stds > threshold].index.tolist()
    if not selected:
        raise ValueError("No non-constant CMAPSS features remain.")
    return selected

# Step 4: Build normalized rolling sequences for MLP training.

def build_sequences(df: pd.DataFrame, rul: pd.Series, seq_len: int = 30, scaler: MinMaxScaler = None, only_last: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    # Build flattened windows using a scaler fitted on training data.
    feature_cols = df.columns[2:]
    units = df['col_1'].unique()
    sequences: List[np.ndarray] = []
    targets: List[float] = []
    if scaler is None:
        scaler = MinMaxScaler().fit(df[feature_cols])
    # Step 4.1: Process one engine trajectory at a time.
    for unit in units:
        unit_df = df[df['col_1'] == unit]
        unit_rul = rul[unit_df.index]
        scaled = scaler.transform(unit_df[feature_cols])
        # Step 4.2: Slide sequence window and collect labels.
        if only_last:
            if len(unit_df) < seq_len:
                continue
            start = len(unit_df) - seq_len
            seq_x = scaled[start:start + seq_len].reshape(-1)
            seq_y = unit_rul.iloc[start + seq_len - 1]
            sequences.append(seq_x)
            targets.append(seq_y)
        else:
            for i in range(len(unit_df) - seq_len + 1):
                seq_x = scaled[i:i + seq_len].reshape(-1)
                seq_y = unit_rul.iloc[i + seq_len - 1]
                sequences.append(seq_x)
                targets.append(seq_y)
    X = np.array(sequences)
    y = np.array(targets)
    return X, y

# Step 5: Define PSO particle structure and optimization routine.

class Particle:
    def __init__(self, dim: int, bounds: Tuple[int, int]):
        self.position = np.random.uniform(bounds[0], bounds[1], size=dim)
        self.velocity = np.zeros(dim)
        self.best_position = self.position.copy()
        self.best_score = np.inf

def pso_optimize(X_train, X_val, y_train, y_val, n_particles=5, n_iter=5, bounds=(10, 100)):
    dim = 2
    # Step 5.1: Initialize swarm in architecture space.
    particles = [Particle(dim, bounds) for _ in range(n_particles)]
    global_best_position = None
    global_best_score = np.inf
    score_history = []
    w, c1, c2 = 0.5, 1.5, 1.5

    # Step 5.2: Iterate PSO updates and MLP evaluations.
    for iter in range(n_iter):
        for p in particles:
            if iter > 0 and global_best_position is not None:
                r1, r2 = np.random.rand(dim), np.random.rand(dim)
                p.velocity = w * p.velocity + c1 * r1 * (p.best_position - p.position) + c2 * r2 * (global_best_position - p.position)
                p.position += p.velocity

            # If no velocity variation, skip iteration for the particle
            if iter > 0 and np.array_equal(p.velocity, np.zeros(dim)):
                continue

            hidden_sizes = tuple(int(max(bounds[0], min(bounds[1], round(val)))) for val in p.position)
            model = MLPRegressor(
                input_size=X_train.shape[1],
                hidden_sizes=hidden_sizes,
                epochs=1000
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            mse = mean_squared_error(y_val, preds)
            # Step 5.3: Update personal/global best scores.
            if mse < p.best_score:
                p.best_score = mse
                p.best_position = p.position.copy()
            if mse < global_best_score:
                global_best_score = mse
                global_best_position = p.position.copy()

        score_history.append(global_best_score)
    best_hidden = tuple(int(round(max(bounds[0], min(bounds[1], val)))) for val in global_best_position)
    return best_hidden, global_best_score, score_history

# Step 6: Train final MLP using best architecture from PSO.

def train_final_model(X_train, y_train, hidden_sizes):
    model = MLPRegressor(
        input_size=X_train.shape[1],
        hidden_sizes=hidden_sizes,
        epochs=1000
    )
    model.fit(X_train, y_train)
    return model

# Step 7: Run complete RUL pipeline per CMAPSS subset.
# Step 8: Compare model quality and aggregate metrics.

def test_final_model(model, dataset_name, cols_to_drop, selected_features, scaler):
    # Run original test set over the final model to compare with literature results.
    test_path = f'data/CMAPSSData/test_{dataset_name}.txt'
    rul_path = f'data/CMAPSSData/RUL_{dataset_name}.txt'
    try:
        test_df = load_cmapss(test_path)
        test_rul_df = load_cmapss_rul(rul_path)
        test_rul_series = test_rul_df['rul']
        test_rul_series.index = test_df['col_1'].unique()  # Align RUL with engine IDs
        test_rul = compute_rul(test_df, base_rul=test_rul_series)

        df_test_red = test_df.drop(columns=cols_to_drop, errors='ignore')

        # Remove flat sensors and fit normalization on training engines only.
        df_test_red = df_test_red[['col_1', 'col_2', *selected_features]]

        # compute_rul applies the piecewise-linear target clipping at 125.
        X_test, _ = build_sequences(df_test_red, test_rul, seq_len=30, scaler=scaler, only_last=True)
        y_test = test_rul_df['rul'].to_numpy()

        test_preds = model.predict(X_test)

        test_mae = mean_absolute_error(y_test, test_preds)
        test_mse = mean_squared_error(y_test, test_preds)
        test_rmse = test_mse**0.5
        print(f"Final Test MSE: {test_mse:.2f}, RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}")

        # Plot Predictions
        plt.figure(figsize=(10, 5))
        plt.plot(y_test, '-', label='True', color='red')
        plt.plot(test_preds, '-', label='Pred', color='blue')
        plt.title(f'RUL Prediction: {dataset_name}')
        plt.xlabel('Engine ID')
        plt.ylabel('RUL')
        plt.legend(fontsize='xx-small', loc='upper right')
        plt.show()
    except FileNotFoundError:
        print(f"Files for {dataset_name} not found.")

def process_dataset(dataset_name: str) -> Dict:
    print(f"\n{'='*40}\nProcessing {dataset_name}\n{'='*40}")
    
    # Metadata & Settings check
    # Step 7.1: Load dataset-specific feature policy.
    metadata = {
        'FD001': {'keep_settings': False},
        'FD002': {'keep_settings': True},
        'FD003': {'keep_settings': False},
        'FD004': {'keep_settings': True}
    }
    keep_settings = metadata.get(dataset_name, {}).get('keep_settings', False)
    
    train_path = f'data/CMAPSSData/train_{dataset_name}.txt'
    try:
        train_df = load_cmapss(train_path)
    except FileNotFoundError:
        print(f"Files for {dataset_name} not found.")
        return {}
    
    # Step 7.2: Compute Remaining Useful Life targets.
    train_rul = compute_rul(train_df)
    # Feature Selection
    cols_to_drop = ['col_6', 'col_8', 'col_9', 'col_10', 'col_14', 'col_15', 'col_17', 'col_20', 'col_21', 'col_22', 'col_23']
    if not keep_settings:
        cols_to_drop.extend(['col_3', 'col_4', 'col_5'])
    
    reduced = train_df.drop(columns=cols_to_drop, errors='ignore')
    
    # Corrección del objetivo RUL
    
    # During the initial stages of operation, the machine/motor is completely healthy. However, a strictly linear RUL assigns
    # a decreasing value from the first cycle (e.g., RUL = 250). Since the sensors show no signs of degradation at the beginning,
    # it is impossible for the model to learn the relationship between healthy sensors and an RUL of 250,
    # generating a huge mean squared error (MSE) during the early stages.

    # Split complete engine trajectories before creating sliding windows.
    units = np.sort(reduced['col_1'].unique())
    train_units, val_units = train_test_split(
        units, test_size=0.2, random_state=42
    )
    df_train_red = reduced[reduced['col_1'].isin(train_units)].copy()
    df_val_red = reduced[reduced['col_1'].isin(val_units)].copy()

    # Remove flat sensors and fit normalization on training engines only.
    selected_features = select_non_flat_features(df_train_red)
    df_train_red = df_train_red[['col_1', 'col_2', *selected_features]]
    df_val_red = df_val_red[['col_1', 'col_2', *selected_features]]
    scaler = MinMaxScaler().fit(df_train_red[selected_features])

    # compute_rul applies the piecewise-linear target clipping at 125.
    X_train, y_train = build_sequences(df_train_red, train_rul, seq_len=30, scaler=scaler)
    X_val, y_val = build_sequences(df_val_red, train_rul, seq_len=30, scaler=scaler)

    print(f"Train windows: {X_train.shape}; validation windows: {X_val.shape}")

    # Optimization & Training
    print("Optimizing MLP Architecture...")
    # Step 5.3: Optimize MLP hidden layers with PSO.
    best_hidden, best_score, mlp_history = pso_optimize(X_train, X_val, y_train, y_val, n_particles=5, n_iter=5)
    print(f"Best Config: {best_hidden}, MSE: {best_score:.2f}, RMSE: {(best_score**0.5):.2f}")
    
    # Plot Convergence
    plt.figure(figsize=(8, 4))
    plt.plot(mlp_history, 'b-o')
    plt.title(f'PSO Convergence: {dataset_name}')
    plt.ylabel('MSE')
    plt.show()
    
    # Final Model
    # Step 5.4: Train final model and evaluate predictions.
    model = train_final_model(X_train, y_train, best_hidden)
    preds = model.predict(X_val)
    
    mae = mean_absolute_error(y_val, preds)
    mse = mean_squared_error(y_val, preds)
    rmse = mse**0.5
    print(f"Final Validation MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    
    # Plot Predictions
    plt.figure(figsize=(10, 5))
    indices = np.argsort(y_val)
    plt.plot(preds[indices], 'r--', label='Pred')
    plt.plot(y_val[indices], 'k-', label='True')
    plt.title(f'RUL Prediction: {dataset_name}')
    plt.legend()
    plt.show()

    test_final_model(model, dataset_name, cols_to_drop, selected_features, scaler)

    return {'Dataset': dataset_name, 'Best Hidden Layers': str(best_hidden), 'MSE': mse, 'MAE': mae, 'RMSE': rmse}

# Executing for all datasets
results = []
for ds in ['FD001']: #, 'FD002', 'FD003', 'FD004']:
    res = process_dataset(ds)
    if res:
        results.append(res)

results_df = pd.DataFrame(results)
print("\nSummary of RUL Prediction Results:")
