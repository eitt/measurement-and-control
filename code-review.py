# Step 1: Import required libraries.
# Step 2: Configure plotting style for consistent visuals.

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

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})

# Step 1: Define data-loading and RUL-target helper functions.
# Step 2: Build normalized rolling sequences for MLP training.

def load_cmapss(fd_path: str) -> pd.DataFrame:
    # Step 1.1: Define the CMAPSS column schema.
    col_names = [f"col_{i}" for i in range(1, 27)]
    df = pd.read_csv(fd_path, sep=r"\s+", header=None, names=col_names)
    return df

def compute_rul(train_df: pd.DataFrame, clip_max: int = 125) -> pd.Series:
    # Step 1.2: Compute max cycle by engine id.
    grouped = train_df.groupby('col_1')['col_2'].max()
    max_cycle = train_df['col_1'].map(grouped)
    rul = (max_cycle - train_df['col_2']).clip(upper=clip_max)
    return rul

def build_sequences(df: pd.DataFrame, rul: pd.Series, seq_len: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    feature_cols = df.columns[2:]
    units = df['col_1'].unique()
    sequences: List[np.ndarray] = []
    targets: List[float] = []
    # Step 2.1: Process one engine trajectory at a time.
    for unit in units:
        unit_df = df[df['col_1'] == unit]
        unit_rul = rul[unit_df.index]
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(unit_df[feature_cols])
        # Step 2.2: Slide sequence window and collect labels.
        for i in range(len(unit_df) - seq_len + 1):
            seq_x = scaled[i:i + seq_len].reshape(-1)
            seq_y = unit_rul.iloc[i + seq_len - 1]
            sequences.append(seq_x)
            targets.append(seq_y)
    X = np.array(sequences)
    y = np.array(targets)
    return X, y

# Step 3: Define PSO particle structure and optimization routine.
# Step 4: Train final MLP using best architecture from PSO.

class Particle:
    def __init__(self, dim: int, bounds: Tuple[int, int]):
        self.position = np.random.uniform(bounds[0], bounds[1], size=dim)
        self.velocity = np.zeros(dim)
        self.best_position = self.position.copy()
        self.best_score = np.inf

def pso_optimize(X_train, X_val, y_train, y_val, n_particles=5, n_iter=5, bounds=(10, 100)):
    dim = 2
    # Step 3.1: Initialize swarm in architecture space.
    particles = [Particle(dim, bounds) for _ in range(n_particles)]
    global_best_position = None
    global_best_score = np.inf
    score_history = []
    w, c1, c2 = 0.5, 1.5, 1.5

    # Step 3.2: Iterate PSO updates and MLP evaluations.
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
            # Step 3.3: Update personal/global best scores.
            if mse < p.best_score:
                p.best_score = mse
                p.best_position = p.position.copy()
            if mse < global_best_score:
                global_best_score = mse
                global_best_position = p.position.copy()

        score_history.append(global_best_score)
    best_hidden = tuple(int(round(max(bounds[0], min(bounds[1], val)))) for val in global_best_position)
    return best_hidden, global_best_score, score_history

def train_final_model(X_train, y_train, hidden_sizes):
    model = MLPRegressor(
        input_size=X_train.shape[1],
        hidden_sizes=hidden_sizes,
        epochs=1000
    )
    model.fit(X_train, y_train)
    return model

# Step 5: Run complete RUL pipeline per CMAPSS subset.
# Step 6: Compare model quality and aggregate metrics.

def process_dataset(dataset_name: str) -> Dict:
    print(f"\n{'='*40}\nProcessing {dataset_name}\n{'='*40}")
    
    # Metadata & Settings check
    # Step 5.1: Load dataset-specific feature policy.
    metadata = {
        'FD001': {'keep_settings': False},
        'FD002': {'keep_settings': True},
        'FD003': {'keep_settings': False},
        'FD004': {'keep_settings': True}
    }
    keep_settings = metadata.get(dataset_name, {}).get('keep_settings', False)
    
    train_path = f'data/CMAPSSData/train_{dataset_name}.txt'
    test_path = f'data/CMAPSSData/test_{dataset_name}.txt'
    try:
        train_df = load_cmapss(train_path)
        test_df = load_cmapss(test_path)
    except FileNotFoundError:
        print(f"Files for {dataset_name} not found.")
        return {}
    
    # Step 5.2: Compute Remaining Useful Life targets.
    rul = compute_rul(train_df)
    
    # Feature Selection
    cols_to_drop = ['col_6', 'col_8', 'col_9', 'col_10', 'col_14', 'col_15', 'col_17', 'col_20', 'col_21', 'col_22', 'col_23']
    if not keep_settings:
        cols_to_drop.extend(['col_3', 'col_4', 'col_5'])
    
    df_red = train_df.drop(columns=cols_to_drop, errors='ignore')
    
    X, y = build_sequences(df_red, rul, seq_len=30)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Optimization & Training
    print("Optimizing MLP Architecture...")
    # Step 5.3: Optimize MLP hidden layers with PSO.
    best_hidden, best_score, mlp_history = pso_optimize(X_train, X_val, y_train, y_val, n_particles=5, n_iter=5)
    print(f"Best Config: {best_hidden}, MSE: {best_score:.2f}")
    
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
    print(f"Final Validation MSE: {mse:.2f}")
    
    # Plot Predictions
    plt.figure(figsize=(10, 5))
    indices = np.argsort(y_val)
    plt.plot(preds[indices], 'r--', label='Pred')
    plt.plot(y_val[indices], 'k-', label='True')
    plt.title(f'RUL Prediction: {dataset_name}')
    plt.legend()
    plt.show()
    
    return {'Dataset': dataset_name, 'Best Hidden Layers': str(best_hidden), 'MSE': mse, 'MAE': mae}

# Executing for all datasets
results = []
for ds in ['FD001']: #, 'FD002', 'FD003', 'FD004']:
    res = process_dataset(ds)
    if res:
        results.append(res)

results_df = pd.DataFrame(results)
print("\nSummary of RUL Prediction Results:")
display(results_df)