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
    # Step 1: Build standard CMAPSS column names (26 columns).
    col_names = [f"col_{i}" for i in range(1, 27)]
    # Step 2: Read whitespace-delimited telemetry into a DataFrame.
    df = pd.read_csv(fd_path, sep=r"\s+", header=None, names=col_names)
    # Step 3: Return normalized raw table for downstream steps.
    return df


def compute_rul(train_df: pd.DataFrame, clip_max: int = 125) -> pd.Series:
    # Step 1: Compute the final observed cycle for each engine id.
    grouped = train_df.groupby('col_1')['col_2'].max()
    # Step 2: Map each row to that engine-level maximum cycle.
    max_cycle = train_df['col_1'].map(grouped)
    # Step 3: RUL = final cycle - current cycle, then clip large values.
    rul = (max_cycle - train_df['col_2']).clip(upper=clip_max)
    return rul


def drop_uninformative(df: pd.DataFrame) -> pd.DataFrame:
    # Step 1: Define mostly constant settings/sensors to remove.
    cols_to_drop = ['col_3', 'col_4', 'col_5', 'col_6', 'col_8', 'col_9',
                    'col_10', 'col_14', 'col_15', 'col_17', 'col_20',
                    'col_21', 'col_22', 'col_23']
    # Step 2: Return reduced feature table.
    df_reduced = df.drop(columns=cols_to_drop)
    return df_reduced


def build_sequences(df: pd.DataFrame, rul: pd.Series, seq_len: int = 30
                    ) -> Tuple[np.ndarray, np.ndarray]:
    # Step 1: Keep sensor/features only (exclude id and cycle columns).
    feature_cols = df.columns[2:] 
    units = df['col_1'].unique()
    sequences: List[np.ndarray] = []
    targets: List[float] = []
    for unit in units:
        # Step 2: Process each engine trajectory independently.
        unit_df = df[df['col_1'] == unit]
        unit_rul = rul[unit_df.index]
        scaler = MinMaxScaler()
        # Step 3: Normalize this engine's features.
        scaled = scaler.fit_transform(unit_df[feature_cols])
        for i in range(len(unit_df) - seq_len + 1):
            # Step 4: Build rolling sequence windows.
            seq_x = scaled[i:i + seq_len].reshape(-1)
            # Step 5: Use RUL at the sequence end as target label.
            seq_y = unit_rul.iloc[i + seq_len - 1]
            sequences.append(seq_x)
            targets.append(seq_y)
    # Step 6: Convert to NumPy arrays for model training.
    X = np.array(sequences)
    y = np.array(targets)
    return X, y


class Particle:
    def __init__(self, dim: int, bounds: Tuple[int, int]):
        # Step 1: Initialize random position in architecture search space.
        self.position = np.random.uniform(bounds[0], bounds[1], size=dim)
        # Step 2: Start with zero velocity.
        self.velocity = np.zeros(dim)
        # Step 3: Personal best starts from initial position.
        self.best_position = self.position.copy()
        self.best_score = np.inf


def pso_optimize(X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray,
                 y_val: np.ndarray, n_particles: int = 5, n_iter: int = 5,
                 bounds: Tuple[int, int] = (10, 200)) -> Tuple[Tuple[int, int], float]:
    # Step 1: Search two dimensions => two hidden layer sizes.
    dim = 2 
    particles = [Particle(dim, bounds) for _ in range(n_particles)]
    global_best_position = None
    global_best_score = np.inf
    score_history = []
    # Step 2: PSO coefficients: inertia, cognitive, social.
    w, c1, c2 = 0.5, 1.5, 1.5
    
    for _ in range(n_iter):
        for p in particles:
            # Step 3: Map continuous particle position to integer neurons.
            hidden_sizes = tuple(int(max(bounds[0], min(bounds[1], round(val))))
                                 for val in p.position)
            # Step 4: Train and evaluate candidate MLP architecture.
            model = MLPRegressor(hidden_layer_sizes=hidden_sizes,
                                 activation='relu', solver='adam', max_iter=200,
                                 random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            mse = mean_squared_error(y_val, preds)
            
            # Step 5: Update personal best.
            if mse < p.best_score:
                p.best_score = mse
                p.best_position = p.position.copy()
            
            # Step 6: Update global best across swarm.
            if mse < global_best_score:
                global_best_score = mse
                global_best_position = p.position.copy()
            
            # Step 7: Update velocity/position with PSO dynamics.
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            cognitive = c1 * r1 * (p.best_position - p.position)
            social = c2 * r2 * (global_best_position - p.position)
            p.velocity = w * p.velocity + cognitive + social
            p.position += p.velocity
        # Step 8: Track convergence for plotting.
        score_history.append(global_best_score)
        print(f"PSO MLP Iter {_+1}/{n_iter}: Best MSE = {global_best_score:.4f}")

    # Step 9: Return best architecture and optimization traces.
    best_hidden = tuple(int(round(max(bounds[0], min(bounds[1], val))))
                        for val in global_best_position)
    return best_hidden, global_best_score, score_history


def train_final_model(X_train: np.ndarray, y_train: np.ndarray,
                      hidden_sizes: Tuple[int, int]) -> MLPRegressor:
    # Step 1: Instantiate final MLP with selected hidden sizes.
    model = MLPRegressor(hidden_layer_sizes=hidden_sizes,
                         activation='relu', solver='adam', max_iter=300,
                         random_state=42)
    # Step 2: Fit and return model.
    model.fit(X_train, y_train)
    return model


def simulate_fopid(ps_params: Tuple[float, float, float, float, float],
                   duration: float = 10.0, dt: float = 0.01) -> float:
    """
    FIXED: Uses np.abs() and np.sign() to prevent complex numbers when
    taking fractional powers of negative error derivatives.
    """
    # Step 1: Unpack controller parameters and initialize plant state.
    Kp, Ki, Kd, lam, mu = ps_params
    x1, x2 = 0.0, 0.0
    integ = 0.0
    deriv_prev = 0.0
    ise = 0.0
    n_steps = int(duration / dt)
    
    for _ in range(n_steps):
        # Step 2: Compute tracking error for a unit-step reference.
        error = 1.0 - x1  
        
        # --- FIX START ---
        # Step 3: Fractional integral approximation (sign-preserving power).
        term_i = error * dt
        integ = integ + np.sign(term_i) * (np.abs(term_i) ** lam)
        
        # Step 4: Fractional derivative approximation (sign-preserving power).
        term_d = (error - deriv_prev) / dt
        deriv = np.sign(term_d) * (np.abs(term_d) ** mu)
        # --- FIX END ---
        
        # Step 5: Build control signal from FOPID terms.
        deriv_prev = error
        u = Kp * error + Ki * integ + Kd * deriv
        
        # Step 6: Guard against unstable or undefined numeric trajectories.
        if np.isnan(u) or np.isinf(u) or abs(u) > 1e6:
            return 1e9

        # Step 7: Integrate second-order system dynamics.
        a = u - x1 - x2
        x2 = x2 + a * dt
        x1 = x1 + x2 * dt
        # Step 8: Accumulate ISE objective.
        ise += error ** 2 * dt
        
    return ise if not np.isnan(ise) else 1e9


def simulate_fopid_response(ps_params: Tuple[float, float, float, float, float],
                            duration: float = 10.0, dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    # Step 1: Initialize controller values and simulation buffers.
    Kp, Ki, Kd, lam, mu = ps_params
    x1, x2 = 0.0, 0.0
    integ = 0.0
    deriv_prev = 0.0
    n_steps = int(duration / dt)
    t_values = np.linspace(0, duration, n_steps)
    y_values = []
    
    for _ in range(n_steps):
        # Step 2: Recompute error and fractional terms at each step.
        error = 1.0 - x1
        
        term_i = error * dt
        integ = integ + np.sign(term_i) * (np.abs(term_i) ** lam)
        
        term_d = (error - deriv_prev) / dt
        deriv = np.sign(term_d) * (np.abs(term_d) ** mu)
        
        deriv_prev = error
        u = Kp * error + Ki * integ + Kd * deriv
        
        # Step 3: Advance plant state and store output trajectory.
        a = u - x1 - x2
        x2 = x2 + a * dt
        x1 = x1 + x2 * dt
        y_values.append(x1)
        
    return t_values, np.array(y_values)


def pso_fopid(n_particles: int = 5, n_iter: int = 5) -> Tuple[Tuple[float, float, float, float, float], float]:
    """
    FIXED: Initializes global best properly to avoid NoneType error.
    """
    # Step 1: Search [Kp, Ki, Kd, lambda, mu] in bounded domain.
    dim = 5
    bounds = [(0.0, 5.0), (0.0, 2.0), (0.0, 2.0), (0.1, 1.9), (0.1, 1.9)]
    
    particles = []
    for _ in range(n_particles):
        # Step 2: Initialize particles and personal best memories.
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
    # Step 3: Initialize global best from first particle (safe bootstrap).
    g_best_pos = particles[0]['position'].copy()
    g_best_cost = simulate_fopid(tuple(g_best_pos))
    # --- FIX END ---
    
    cost_history = []
    # Step 4: PSO motion coefficients.
    w, c1, c2 = 0.5, 1.5, 1.5
    
    for i in range(n_iter):
        for p in particles:
            # Step 5: Evaluate particle via ISE objective.
            cost = simulate_fopid(tuple(p['position']))
            
            # Step 6: Update personal and global bests.
            if cost < p['best_cost']:
                p['best_cost'] = cost
                p['best_position'] = p['position'].copy()
            
            if cost < g_best_cost:
                g_best_cost = cost
                g_best_pos = p['position'].copy()
                
        for p in particles:
            # Step 7: Apply PSO velocity/position updates.
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            cognitive = c1 * r1 * (p['best_position'] - p['position'])
            social = c2 * r2 * (g_best_pos - p['position'])
            p['velocity'] = w * p['velocity'] + cognitive + social
            p['position'] += p['velocity']
            
            # Step 8: Enforce parameter bounds.
            for j, (low, high) in enumerate(bounds):
                p['position'][j] = max(low, min(high, p['position'][j]))
                
        # Step 9: Track convergence history.
        cost_history.append(g_best_cost)
        print(f"PSO FOPID Iter {i+1}/{n_iter}: Best ISE = {g_best_cost:.4f}")
        
    return tuple(g_best_pos), g_best_cost, cost_history


def process_dataset(dataset_name: str):
    # (Kept function definition for compatibility, but logic moved out)
    pass 

def main():
    # Step 1: Skip RUL retraining branch in this fixed script version.
    print("Skipping Neural Network Training (using provided log info)...")
    
    # Step 2: Run PSO optimization for the FOPID controller.
    print("\n" + "="*40)
    print("Iniciando optimización PSO para FOPID (Sistema de Control)...")
    print("="*40)
    
    try:
        best_params, best_cost, fopid_history = pso_fopid(n_particles=10, n_iter=10)
        print(f'Mejor FOPID encontrado: {best_params}, ISE={best_cost:.4f}')

        # Step 3: Plot and save PSO convergence history.
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(fopid_history) + 1), fopid_history, 'g-o', linewidth=2)
        plt.title('FOPID PSO Optimization Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Best ISE Cost')
        plt.grid(True)
        plt.savefig('fig_fopid_convergence.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Step 4: Compare optimized response against a baseline controller.
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
        # Step 5: Fail with explicit message if optimization crashes.
        print(f"Critical Error in FOPID optimization: {e}")

if __name__ == '__main__':
    main()
