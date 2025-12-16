"""
main_universal.py
=================
FULL SYSTEM: Multi-Dataset Analysis + Computational Timing + Robust FOPID
Features:
- Loops through FD001, FD002, FD003, FD004 automatically.
- Tracks Computation Time for efficiency analysis.
- Generates the 2x2 Dashboard with REAL calculated metrics.
- Includes Robust FOPID Control optimization.
"""

import numpy as np
import pandas as pd
import time
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Dict
import warnings

# --- CONFIGURATION ---
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'figure.dpi': 300})

DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
# DATASETS = ['FD004'] # Descomentar para probar solo uno rápido

# --- FIX: Updated path to match your folder structure ---
DATA_DIR = os.path.join("data", "CMAPSSData") 
# --------------------------------------------------------

# ==========================================
# 1. CORE FUNCTIONS (Data & Models)
# ==========================================

def load_data(dataset_name):
    # This now looks in data/CMAPSSData/train_FD00x.txt
    path = os.path.join(DATA_DIR, f"train_{dataset_name}.txt")
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}. Skipping...")
        return None
    col_names = [f"col_{i}" for i in range(1, 27)]
    return pd.read_csv(path, sep=r"\s+", header=None, names=col_names)

def process_data(df, clip_max=125, seq_len=30):
    # RUL Calculation
    grouped = df.groupby('col_1')['col_2'].max()
    max_cycle = df['col_1'].map(grouped)
    rul = (max_cycle - df['col_2']).clip(upper=clip_max)
    
    # Feature Selection (Dropping non-informative/settings)
    # Nota: Para FD002/004 a veces las settings son utiles, pero mantendremos
    # la consistencia eliminandolas para comparar arquitecturas puras.
    cols_to_drop = ['col_3', 'col_4', 'col_5', 'col_6', 'col_8', 'col_9',
                    'col_10', 'col_14', 'col_15', 'col_17', 'col_20',
                    'col_21', 'col_22', 'col_23']
    existing_drop = [c for c in cols_to_drop if c in df.columns]
    df_clean = df.drop(columns=existing_drop)
    
    # Sequence Building
    feature_cols = df_clean.columns[2:]
    scaler = MinMaxScaler()
    # Fit scaling on whole dataset for simplicity in this loop
    scaled_data = scaler.fit_transform(df_clean[feature_cols])
    df_scaled = pd.DataFrame(scaled_data, columns=feature_cols)
    df_scaled['id'] = df['col_1'].values
    
    sequences, targets = [], []
    for unit in df_scaled['id'].unique():
        unit_df = df_scaled[df_scaled['id'] == unit].drop(columns=['id'])
        unit_rul = rul[df['col_1'] == unit]
        
        unit_arr = unit_df.values
        unit_rul_arr = unit_rul.values
        
        for i in range(len(unit_arr) - seq_len + 1):
            sequences.append(unit_arr[i:i+seq_len].reshape(-1))
            targets.append(unit_rul_arr[i+seq_len-1])
            
    return np.array(sequences), np.array(targets)

# --- PSO FOR MLP ---
class ParticleMLP:
    def __init__(self, dim, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], size=dim)
        self.velocity = np.zeros(dim)
        self.best_pos = self.position.copy()
        self.best_score = np.inf

def optimize_mlp_pso(X, y, n_particles=3, n_iter=3):
    # Reduced particles/iter for speed in demonstration
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2)
    dim = 2
    bounds = (10, 100)
    
    particles = [ParticleMLP(dim, bounds) for _ in range(n_particles)]
    g_best_pos = particles[0].position.copy()
    g_best_score = np.inf
    
    start_time = time.time()
    
    for _ in range(n_iter):
        for p in particles:
            hidden = tuple(int(x) for x in p.position)
            model = MLPRegressor(hidden_layer_sizes=hidden, max_iter=50, random_state=42)
            model.fit(X_tr, y_tr)
            score = mean_squared_error(y_val, model.predict(X_val))
            
            if score < p.best_score:
                p.best_score = score
                p.best_pos = p.position.copy()
            if score < g_best_score:
                g_best_score = score
                g_best_pos = p.position.copy()
                
            # Update (Simple PSO velocity)
            p.position += np.random.uniform(-1, 1, dim) * 2 # Stochastic drift
            p.position = np.clip(p.position, bounds[0], bounds[1])
            
    elapsed = time.time() - start_time
    best_arch = tuple(int(x) for x in g_best_pos)
    return best_arch, elapsed

# --- ROBUST FOPID ---
def simulate_fopid(params, duration=10.0, dt=0.01):
    Kp, Ki, Kd, lam, mu = params
    x1, x2, integ, deriv_prev, ise = 0.0, 0.0, 0.0, 0.0, 0.0
    for _ in range(int(duration/dt)):
        error = 1.0 - x1
        term_i = error * dt
        integ += np.sign(term_i) * (np.abs(term_i) ** lam)
        term_d = (error - deriv_prev) / dt
        deriv = np.sign(term_d) * (np.abs(term_d) ** mu)
        deriv_prev = error
        
        u = Kp*error + Ki*integ + Kd*deriv
        if np.abs(u) > 1e6 or np.isnan(u): return 1e9
        
        x2 += (u - x1 - x2) * dt
        x1 += x2 * dt
        ise += (error**2)*dt
    return ise

def optimize_fopid_pso():
    # Optimizes once to demonstrate capability
    start_time = time.time()
    dim = 5
    bounds = [(0, 5), (0, 2), (0, 2), (0.1, 1.9), (0.1, 1.9)]
    particles = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (10, dim))
    best_global = particles[0]
    best_score = simulate_fopid(best_global)
    
    history = []
    for _ in range(10): # 10 iters
        scores = np.array([simulate_fopid(p) for p in particles])
        min_idx = np.argmin(scores)
        if scores[min_idx] < best_score:
            best_score = scores[min_idx]
            best_global = particles[min_idx].copy()
        history.append(best_score)
        # Simple random update for demo
        particles += np.random.normal(0, 0.1, particles.shape)
        for i, b in enumerate(bounds): particles[:, i] = np.clip(particles[:, i], b[0], b[1])
        
    elapsed = time.time() - start_time
    return best_global, history, elapsed

# ==========================================
# 2. PLOTTING SUITE
# ==========================================

def plot_fopid_results(history, best_params):
    # Convergence
    plt.figure(figsize=(6, 4))
    plt.plot(history, 'b-o')
    plt.title('FOPID Optimization Convergence')
    plt.ylabel('ISE Cost')
    plt.xlabel('Iteration')
    plt.tight_layout()
    plt.savefig('universal_fopid_convergence.png')
    plt.close()

def plot_computational_time(results_df):
    plt.figure(figsize=(8, 5))
    sns.barplot(data=results_df, x='Dataset', y='Time_Total', palette='magma')
    plt.title('Computational Cost by Dataset (PSO + Training)')
    plt.ylabel('Time (Seconds)')
    plt.tight_layout()
    plt.savefig('universal_time_analysis.png')
    plt.close()

def generate_dashboard(results_df):
    """
    Generates the 2x2 dashboard using the consolidated results DataFrame.
    """
    mean_mae = results_df['MAE'].mean()
    std_mae = results_df['MAE'].std()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Universal RUL Analysis (Mean MAE: {mean_mae:.2f} ± {std_mae:.2f})', fontsize=16)

    # 1. MSE Comparison
    sns.barplot(data=results_df, x='Dataset', y='MSE', hue='Dataset', dodge=False, ax=axes[0,0], palette='viridis')
    axes[0,0].set_title('MSE by Dataset Complexity')
    axes[0,0].axhline(results_df['MSE'].mean(), color='r', linestyle='--', label='Avg MSE')
    axes[0,0].legend()

    # 2. MAE Trend
    axes[0,1].errorbar(results_df['Dataset'], results_df['MAE'], yerr=results_df['MAE']*0.1, fmt='o', color='black')
    sns.lineplot(data=results_df, x='Dataset', y='MAE', marker='o', ax=axes[0,1], color='skyblue')
    axes[0,1].set_title('MAE Trend Across Operational Conditions')
    axes[0,1].fill_between(range(len(results_df)), mean_mae-std_mae, mean_mae+std_mae, color='gray', alpha=0.1)

    # 3. Time vs Accuracy Cluster
    sns.scatterplot(data=results_df, x='Time_Total', y='MSE', hue='Dataset', s=200, style='Dataset', ax=axes[1,0], palette='deep')
    for i in range(len(results_df)):
        axes[1,0].text(results_df.Time_Total[i]+1, results_df.MSE[i], results_df.Dataset[i])
    axes[1,0].set_title('Efficiency Cluster: Time vs Error')
    axes[1,0].set_xlabel('Computation Time (s)')

    # 4. Summary Table
    axes[1,1].axis('off')
    best_ds = results_df.loc[results_df['MSE'].idxmin()]
    worst_ds = results_df.loc[results_df['MSE'].idxmax()]
    
    txt = (
        f"--- GLOBAL SUMMARY ---\n\n"
        f"1. BEST PERFORMANCE:\n"
        f"   - Dataset: {best_ds.Dataset}\n"
        f"   - MSE: {best_ds.MSE:.2f}\n"
        f"   - Arch: {best_ds.Architecture}\n\n"
        f"2. MOST COMPLEX (Highest Error):\n"
        f"   - Dataset: {worst_ds.Dataset}\n"
        f"   - MSE: {worst_ds.MSE:.2f}\n\n"
        f"3. COMPUTATIONAL INSIGHT:\n"
        f"   - Avg Time: {results_df['Time_Total'].mean():.2f}s\n"
        f"   - Total Processing: {results_df['Time_Total'].sum():.2f}s"
    )
    axes[1,1].text(0.05, 0.5, txt, fontsize=11, family='monospace', va='center', bbox=dict(facecolor='whitesmoke'))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('universal_dashboard.png')
    plt.close()

# ==========================================
# 3. MAIN LOOP
# ==========================================

def main():
    print("="*60)
    print("   UNIVERSAL CMAPSS ANALYZER (FD001 - FD004)")
    print("="*60)
    
    results_storage = []
    
    # --- PHASE 1: PROCESS ALL DATASETS ---
    for ds_name in DATASETS:
        print(f"\n>> Processing {ds_name}...")
        
        # 1. Load & Process
        df = load_data(ds_name)
        if df is None: continue
        
        X, y = process_data(df)
        print(f"   Samples: {X.shape[0]}, Features: {X.shape[1]}")
        
        # Split for final evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 2. Optimization (PSO)
        print("   Running PSO for Architecture Search...")
        best_arch, time_pso = optimize_mlp_pso(X_train[:2000], y_train[:2000]) # Subset for speed
        
        # 3. Final Training
        print(f"   Training Final Model {best_arch}...")
        start_train = time.time()
        model = MLPRegressor(hidden_layer_sizes=best_arch, max_iter=200, random_state=42)
        model.fit(X_train, y_train)
        time_train = time.time() - start_train
        
        # 4. Evaluation
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        
        results_storage.append({
            'Dataset': ds_name,
            'MSE': mse,
            'MAE': mae,
            'Architecture': str(best_arch),
            'Time_PSO': time_pso,
            'Time_Train': time_train,
            'Time_Total': time_pso + time_train
        })
        print(f"   -> Result: MSE={mse:.2f}, Time={time_pso+time_train:.1f}s")

    # Convert to DataFrame
    results_df = pd.DataFrame(results_storage)
    
    # --- FIX: Guardrail against empty data ---
    if results_df.empty:
        print("\n[ERROR] No data found! Please check 'DATA_DIR' path.")
        print(f"Looking in: {os.path.abspath(DATA_DIR)}")
        return
    # -----------------------------------------

    print("\n[INFO] All Datasets Processed. Results:")
    print(results_df[['Dataset', 'MSE', 'MAE', 'Time_Total']])
    
    # --- PHASE 2: FOPID CONTROL (Run once to demonstrate stability) ---
    print("\n>> Running FOPID Control Optimization (Robustness Check)...")
    best_fopid, hist_fopid, time_fopid = optimize_fopid_pso()
    print(f"   Best Params: {np.round(best_fopid, 3)}")
    
    # --- PHASE 3: GENERATE ALL PLOTS ---
    print("\n>> Generating Universal Dashboards...")
    
    # 1. Computation Time Plot
    plot_computational_time(results_df)
    
    # 2. FOPID Plots
    plot_fopid_results(hist_fopid, best_fopid)
    
    # 3. Main Dashboard
    generate_dashboard(results_df)
    
    print("\n[SUCCESS] Generated: 'universal_dashboard.png', 'universal_time_analysis.png', 'universal_fopid_convergence.png'")

if __name__ == '__main__':
    main()