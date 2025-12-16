"""
code.py
========
FIXED VERSION
- Solved 'complex power' error in FOPID simulation using abs() and sign().
- Solved 'NoneType' error in PSO initialization.
- Skipped Neural Network retraining (logic commented out in main).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set style for english publication quality figures
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})


def load_cmapss(fd_path: str) -> pd.DataFrame:
    col_names = [f"col_{i}" for i in range(1, 27)]
    df = pd.read_csv(fd_path, sep=r"\s+", header=None, names=col_names)
    return df


def compute_rul(train_df: pd.DataFrame, clip_max: int = 125) -> pd.Series:
    grouped = train_df.groupby('col_1')['col_2'].max()
    max_cycle = train_df['col_1'].map(grouped)
    rul = (max_cycle - train_df['col_2']).clip(upper=clip_max)
    return rul


def drop_uninformative(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = ['col_3', 'col_4', 'col_5', 'col_6', 'col_8', 'col_9',
                    'col_10', 'col_14', 'col_15', 'col_17', 'col_20',
                    'col_21', 'col_22', 'col_23']
    df_reduced = df.drop(columns=cols_to_drop)
    return df_reduced


def build_sequences(df: pd.DataFrame, rul: pd.Series, seq_len: int = 30
                    ) -> Tuple[np.ndarray, np.ndarray]:
    feature_cols = df.columns[2:] 
    units = df['col_1'].unique()
    sequences: List[np.ndarray] = []
    targets: List[float] = []
    for unit in units:
        unit_df = df[df['col_1'] == unit]
        unit_rul = rul[unit_df.index]
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(unit_df[feature_cols])
        for i in range(len(unit_df) - seq_len + 1):
            seq_x = scaled[i:i + seq_len].reshape(-1)
            seq_y = unit_rul.iloc[i + seq_len - 1]
            sequences.append(seq_x)
            targets.append(seq_y)
    X = np.array(sequences)
    y = np.array(targets)
    return X, y


class Particle:
    def __init__(self, dim: int, bounds: Tuple[int, int]):
        self.position = np.random.uniform(bounds[0], bounds[1], size=dim)
        self.velocity = np.zeros(dim)
        self.best_position = self.position.copy()
        self.best_score = np.inf


def pso_optimize(X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray,
                 y_val: np.ndarray, n_particles: int = 5, n_iter: int = 5,
                 bounds: Tuple[int, int] = (10, 200)) -> Tuple[Tuple[int, int], float]:
    dim = 2 
    particles = [Particle(dim, bounds) for _ in range(n_particles)]
    global_best_position = None
    global_best_score = np.inf
    score_history = []
    w, c1, c2 = 0.5, 1.5, 1.5
    
    for _ in range(n_iter):
        for p in particles:
            hidden_sizes = tuple(int(max(bounds[0], min(bounds[1], round(val))))
                                 for val in p.position)
            model = MLPRegressor(hidden_layer_sizes=hidden_sizes,
                                 activation='relu', solver='adam', max_iter=200,
                                 random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            mse = mean_squared_error(y_val, preds)
            
            if mse < p.best_score:
                p.best_score = mse
                p.best_position = p.position.copy()
            
            if mse < global_best_score:
                global_best_score = mse
                global_best_position = p.position.copy()
            
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            cognitive = c1 * r1 * (p.best_position - p.position)
            social = c2 * r2 * (global_best_position - p.position)
            p.velocity = w * p.velocity + cognitive + social
            p.position += p.velocity
        score_history.append(global_best_score)
        print(f"PSO MLP Iter {_+1}/{n_iter}: Best MSE = {global_best_score:.4f}")

    best_hidden = tuple(int(round(max(bounds[0], min(bounds[1], val))))
                        for val in global_best_position)
    return best_hidden, global_best_score, score_history


def train_final_model(X_train: np.ndarray, y_train: np.ndarray,
                      hidden_sizes: Tuple[int, int]) -> MLPRegressor:
    model = MLPRegressor(hidden_layer_sizes=hidden_sizes,
                         activation='relu', solver='adam', max_iter=300,
                         random_state=42)
    model.fit(X_train, y_train)
    return model


def simulate_fopid(ps_params: Tuple[float, float, float, float, float],
                   duration: float = 10.0, dt: float = 0.01) -> float:
    """
    FIXED: Uses np.abs() and np.sign() to prevent complex numbers when
    taking fractional powers of negative error derivatives.
    """
    Kp, Ki, Kd, lam, mu = ps_params
    x1, x2 = 0.0, 0.0
    integ = 0.0
    deriv_prev = 0.0
    ise = 0.0
    n_steps = int(duration / dt)
    
    for _ in range(n_steps):
        error = 1.0 - x1  
        
        # --- FIX START ---
        # Fractional Integral Approximation
        # integ = integ + (error * dt) ** lam  <-- OLD ERROR SOURCE
        term_i = error * dt
        integ = integ + np.sign(term_i) * (np.abs(term_i) ** lam)
        
        # Fractional Derivative Approximation
        # deriv = ((error - deriv_prev) / dt) ** mu <-- OLD ERROR SOURCE
        term_d = (error - deriv_prev) / dt
        deriv = np.sign(term_d) * (np.abs(term_d) ** mu)
        # --- FIX END ---
        
        deriv_prev = error
        u = Kp * error + Ki * integ + Kd * deriv
        
        # Stability check
        if np.isnan(u) or np.isinf(u) or abs(u) > 1e6:
            return 1e9

        # System: x'' + x' + x = u
        a = u - x1 - x2
        x2 = x2 + a * dt
        x1 = x1 + x2 * dt
        ise += error ** 2 * dt
        
    return ise if not np.isnan(ise) else 1e9


def simulate_fopid_response(ps_params: Tuple[float, float, float, float, float],
                            duration: float = 10.0, dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    Kp, Ki, Kd, lam, mu = ps_params
    x1, x2 = 0.0, 0.0
    integ = 0.0
    deriv_prev = 0.0
    n_steps = int(duration / dt)
    t_values = np.linspace(0, duration, n_steps)
    y_values = []
    
    for _ in range(n_steps):
        error = 1.0 - x1
        
        term_i = error * dt
        integ = integ + np.sign(term_i) * (np.abs(term_i) ** lam)
        
        term_d = (error - deriv_prev) / dt
        deriv = np.sign(term_d) * (np.abs(term_d) ** mu)
        
        deriv_prev = error
        u = Kp * error + Ki * integ + Kd * deriv
        
        a = u - x1 - x2
        x2 = x2 + a * dt
        x1 = x1 + x2 * dt
        y_values.append(x1)
        
    return t_values, np.array(y_values)


def pso_fopid(n_particles: int = 5, n_iter: int = 5) -> Tuple[Tuple[float, float, float, float, float], float]:
    """
    FIXED: Initializes global best properly to avoid NoneType error.
    """
    dim = 5
    bounds = [(0.0, 5.0), (0.0, 2.0), (0.0, 2.0), (0.1, 1.9), (0.1, 1.9)]
    
    particles = []
    for _ in range(n_particles):
        pos = np.array([np.random.uniform(low, high) for low, high in bounds])
        vel = np.zeros(dim)
        p = {
            'position': pos,
            'velocity': vel,
            'best_position': pos.copy(),
            'best_cost': np.inf
        }
        particles.append(p)
    
    # --- FIX START ---
    # Initialize global best with the first particle to avoid NoneType later
    g_best_pos = particles[0]['position'].copy()
    g_best_cost = simulate_fopid(tuple(g_best_pos))
    # --- FIX END ---
    
    cost_history = []
    w, c1, c2 = 0.5, 1.5, 1.5
    
    for i in range(n_iter):
        for p in particles:
            cost = simulate_fopid(tuple(p['position']))
            
            if cost < p['best_cost']:
                p['best_cost'] = cost
                p['best_position'] = p['position'].copy()
            
            if cost < g_best_cost:
                g_best_cost = cost
                g_best_pos = p['position'].copy()
                
        for p in particles:
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            cognitive = c1 * r1 * (p['best_position'] - p['position'])
            social = c2 * r2 * (g_best_pos - p['position'])
            p['velocity'] = w * p['velocity'] + cognitive + social
            p['position'] += p['velocity']
            
            # Boundary check
            for j, (low, high) in enumerate(bounds):
                p['position'][j] = max(low, min(high, p['position'][j]))
                
        cost_history.append(g_best_cost)
        print(f"PSO FOPID Iter {i+1}/{n_iter}: Best ISE = {g_best_cost:.4f}")
        
    return tuple(g_best_pos), g_best_cost, cost_history


def process_dataset(dataset_name: str):
    # (Kept function definition for compatibility, but logic moved out)
    pass 

def main():
    # --- SKIPPING NEURAL NETWORK TRAINING AS REQUESTED ---
    # We assume the user has already run this or wants to use existing logs.
    print("Skipping Neural Network Training (using provided log info)...")
    
    # --- FOPID OPTIMIZATION (Running this because it crashed last time) ---
    print("\n" + "="*40)
    print("Iniciando optimización PSO para FOPID (Sistema de Control)...")
    print("="*40)
    
    try:
        best_params, best_cost, fopid_history = pso_fopid(n_particles=10, n_iter=10)
        print(f'Mejor FOPID encontrado: {best_params}, ISE={best_cost:.4f}')

        # Plot Convergence
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(fopid_history) + 1), fopid_history, 'g-o', linewidth=2)
        plt.title('FOPID PSO Optimization Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Best ISE Cost')
        plt.grid(True)
        plt.savefig('fig_fopid_convergence.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Plot Step Response
        t_opt, y_opt = simulate_fopid_response(best_params)
        mean_params = (1.0, 0.5, 0.5, 1.0, 0.5) # Baseline
        t_base, y_base = simulate_fopid_response(mean_params)
        
        plt.figure(figsize=(10, 6))
        plt.plot(t_opt, np.ones_like(t_opt), 'k--', label='Setpoint')
        plt.plot(t_base, y_base, 'b:', label='Baseline', linewidth=1.5)
        plt.plot(t_opt, y_opt, 'r-', label='Optimized FOPID', linewidth=2)
        plt.title('Step Response Comparison')
        plt.xlabel('Time (s)')
        plt.ylabel('System Output')
        plt.legend()
        plt.grid(True)
        plt.savefig('fig_fopid_response.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("FOPID plots generated successfully.")
        
    except Exception as e:
        print(f"Critical Error in FOPID optimization: {e}")

if __name__ == '__main__':
    main()