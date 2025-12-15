"""
code.py
========

Este script carga el subconjunto FD001 del conjunto de datos C,MAPSS, prepara
secuencias de entrenamiento, entrena un modelo de perceptrón multicapa (MLP)
para predecir la vida útil remanente (RUL) de motores turbofan y ajusta sus
hiperparámetros mediante Particle Swarm Optimization (PSO).  Adicionalmente
incluye una demostración simplificada del ajuste de un controlador de orden
fraccionario (FOPID) mediante PSO utilizando un sistema de segundo orden.

Las funciones están pensadas para ser reproducibles y permiten variar la
longitud de la secuencia y los parámetros de búsqueda.  Este script sirve
como soporte computacional para el artículo asociado.
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

# Set style for english publication quality figures
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})


def load_cmapss(fd_path: str) -> pd.DataFrame:
    """Carga un archivo FD001 o FD00x del conjunto C,MAPSS.

    Parameters
    ----------
    fd_path : str
        Ruta al archivo de entrenamiento (por ejemplo 'train_FD001.txt').

    Returns
    -------
    df : DataFrame
        Conjunto de datos con columnas indexadas de 1 a 26.
    """
    # Carga del archivo delimitado por espacios múltiples
    col_names = [f"col_{i}" for i in range(1, 27)]
    df = pd.read_csv(fd_path, sep=r"\s+", header=None, names=col_names)
    return df


def compute_rul(train_df: pd.DataFrame, clip_max: int = 125) -> pd.Series:
    """Calcula el Remaining Useful Life (RUL) para cada fila del conjunto de
    entrenamiento y aplica un límite superior.

    Parameters
    ----------
    train_df : DataFrame
        Datos de entrenamiento con identificador de motor en col_1 y ciclo en col_2.
    clip_max : int, optional
        Valor máximo para truncar el RUL, by default 125.

    Returns
    -------
    pd.Series
        Serie con RUL calculado por fila.
    """
    # El RUL es el ciclo máximo por motor menos el ciclo actual
    grouped = train_df.groupby('col_1')['col_2'].max()
    max_cycle = train_df['col_1'].map(grouped)
    rul = (max_cycle - train_df['col_2']).clip(upper=clip_max)
    return rul


def drop_uninformative(df: pd.DataFrame) -> pd.DataFrame:
    """Descarta variables de ajuste operativas y sensores con baja variación.

    Esta selección sigue criterios reportados en el cargador de datos de Nixtla
    (phm2008.py), eliminando col_3, col_4, col_5 y los sensores sin
    información suficiente.

    Parameters
    ----------
    df : DataFrame
        Conjunto de datos original.

    Returns
    -------
    DataFrame
        Conjunto reducido con variables más informativas.
    """
    # Columnas a eliminar: ajustes operativos (col_3, col_4, col_5) y sensores
    # 6, 8, 9, 10, 14, 15, 17, 20, 21, 22, 23 sin variabilidad significativa
    cols_to_drop = ['col_3', 'col_4', 'col_5', 'col_6', 'col_8', 'col_9',
                    'col_10', 'col_14', 'col_15', 'col_17', 'col_20',
                    'col_21', 'col_22', 'col_23']
    df_reduced = df.drop(columns=cols_to_drop)
    return df_reduced


def build_sequences(df: pd.DataFrame, rul: pd.Series, seq_len: int = 30
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Construye secuencias de longitud fija para entrenamiento de redes.

    Parameters
    ----------
    df : DataFrame
        Conjunto de datos con identificador de motor, ciclo y variables de
        sensores.
    rul : Series
        Valores de RUL por fila.
    seq_len : int, optional
        Longitud de la secuencia en ciclos, by default 30.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Matriz X de forma (n_samples, seq_len * n_features) y vector y con
        valores de RUL.
    """
    feature_cols = df.columns[2:]  # Excluir identificador y ciclo
    units = df['col_1'].unique()
    sequences: List[np.ndarray] = []
    targets: List[float] = []
    for unit in units:
        unit_df = df[df['col_1'] == unit]
        unit_rul = rul[unit_df.index]
        # Normalización por unidad
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(unit_df[feature_cols])
        # Construcción de secuencias
        for i in range(len(unit_df) - seq_len + 1):
            seq_x = scaled[i:i + seq_len].reshape(-1)
            seq_y = unit_rul.iloc[i + seq_len - 1]
            sequences.append(seq_x)
            targets.append(seq_y)
    X = np.array(sequences)
    y = np.array(targets)
    return X, y


class Particle:
    """Representa una partícula para el algoritmo PSO."""
    def __init__(self, dim: int, bounds: Tuple[int, int]):
        self.position = np.random.uniform(bounds[0], bounds[1], size=dim)
        self.velocity = np.zeros(dim)
        self.best_position = self.position.copy()
        self.best_score = np.inf


def pso_optimize(X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray,
                 y_val: np.ndarray, n_particles: int = 5, n_iter: int = 5,
                 bounds: Tuple[int, int] = (10, 200)) -> Tuple[Tuple[int, int], float]:
    """Optimiza el tamaño de las capas ocultas de un MLP mediante PSO.

    Parameters
    ----------
    X_train, X_val, y_train, y_val
        Conjuntos de entrenamiento y validación.
    n_particles : int, optional
        Número de partículas, by default 5.
    n_iter : int, optional
        Iteraciones del algoritmo PSO, by default 5.
    bounds : Tuple[int, int], optional
        Límite inferior y superior para el número de neuronas, by default (10, 200).

    Returns
    -------
    Tuple[Tuple[int, int], float, List[float]]
        Mejor configuración de capas ocultas, su error MSE, y la historia de convergencia.
    """
    dim = 2  # Dos capas ocultas
    particles = [Particle(dim, bounds) for _ in range(n_particles)]
    global_best_position = None
    global_best_score = np.inf
    score_history = []
    # Coeficientes PSO
    w = 0.5  # inercia
    c1 = 1.5
    c2 = 1.5
    for _ in range(n_iter):
        for p in particles:
            # Ajuste de hiperparámetros (enteros)
            hidden_sizes = tuple(int(max(bounds[0], min(bounds[1], round(val))))
                                 for val in p.position)
            # Entrenamiento de MLP
            model = MLPRegressor(hidden_layer_sizes=hidden_sizes,
                                 activation='relu', solver='adam', max_iter=200,
                                 random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            mse = mean_squared_error(y_val, preds)
            # Actualizar mejor local
            if mse < p.best_score:
                p.best_score = mse
                p.best_position = p.position.copy()
            # Actualizar mejor global
            if mse < global_best_score:
                global_best_score = mse
                global_best_position = p.position.copy()
            # Actualizar velocidad y posición
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
    """Entrena un MLP con la configuración proporcionada sobre todos los datos.

    Parameters
    ----------
    X_train : ndarray
        Matriz de entrenamiento.
    y_train : ndarray
        Vector de RUL.
    hidden_sizes : Tuple[int, int]
        Número de neuronas en cada capa oculta.

    Returns
    -------
    MLPRegressor
        Modelo entrenado.
    """
    model = MLPRegressor(hidden_layer_sizes=hidden_sizes,
                         activation='relu', solver='adam', max_iter=300,
                         random_state=42)
    model.fit(X_train, y_train)
    return model


def simulate_fopid(ps_params: Tuple[float, float, float, float, float],
                   duration: float = 10.0, dt: float = 0.01) -> float:
    """Evalúa un controlador FOPID simplificado en un sistema de segundo orden.

    Esta función implementa una simulación discretizada de un sistema
    G(s) = 1/(s^2 + s + 1) controlado por un FOPID.  La fracción de orden
    se aproxima empleando exponentes en los términos integral y derivativo.
    Se calcula el error integral cuadrático (ISE) del seguimiento del escalón.

    Parameters
    ----------
    ps_params : Tuple[float, float, float, float, float]
        Parámetros del FOPID (Kp, Ki, Kd, lambda, mu).
    duration : float, optional
        Duración de la simulación, by default 10.0 segundos.
    dt : float, optional
        Paso de integración, by default 0.01 segundos.

    Returns
    -------
    float
        Valor de la integral cuadrática del error (ISE).
    """
    Kp, Ki, Kd, lam, mu = ps_params
    # Inicialización de estados
    x1 = 0.0  # posición del sistema
    x2 = 0.0  # velocidad del sistema
    integ = 0.0  # integral fraccionaria aproximada
    deriv_prev = 0.0
    ise = 0.0
    n_steps = int(duration / dt)
    for _ in range(n_steps):
        error = 1.0 - x1  # escalón unitario
        # integral de orden lam (aproximación: sumatorio con potencia lam)
        integ = integ + (error * dt) ** lam
        # derivada de orden mu (aproximación: diferencia finita con potencia mu)
        deriv = ((error - deriv_prev) / dt) ** mu
        deriv_prev = error
        u = Kp * error + Ki * integ + Kd * deriv
        # Modelo de sistema: x'' + x' + x = u
        # Discretización por método de Euler
        a = u - x1 - x2
        x2 = x2 + a * dt
        x1 = x1 + x2 * dt
        ise += error ** 2 * dt
    return ise


def simulate_fopid_response(ps_params: Tuple[float, float, float, float, float],
                            duration: float = 10.0, dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """Simula la respuesta al escalón para graficar."""
    Kp, Ki, Kd, lam, mu = ps_params
    x1, x2 = 0.0, 0.0
    integ = 0.0
    deriv_prev = 0.0
    n_steps = int(duration / dt)
    t_values = np.linspace(0, duration, n_steps)
    y_values = []
    
    for _ in range(n_steps):
        error = 1.0 - x1
        integ = integ + (error * dt) ** lam
        deriv = ((error - deriv_prev) / dt) ** mu
        deriv_prev = error
        u = Kp * error + Ki * integ + Kd * deriv
        # Euler integration
        a = u - x1 - x2
        x2 = x2 + a * dt
        x1 = x1 + x2 * dt
        y_values.append(x1)
        
    return t_values, np.array(y_values)


def pso_fopid(n_particles: int = 5, n_iter: int = 5) -> Tuple[Tuple[float, float, float, float, float], float]:
    """Optimiza los parámetros del FOPID mediante PSO minimizando el ISE.

    Parameters
    ----------
    n_particles : int, optional
        Número de partículas, by default 5.
    n_iter : int, optional
        Iteraciones del algoritmo PSO, by default 5.

    Returns
    -------
    Tuple[Tuple[float, float, float, float, float], float, List[float]]
        Mejor vector de parámetros, costo alcanzado e historial.
    """
    dim = 5
    bounds = [(0.0, 2.0),  # Kp
              (0.0, 1.0),  # Ki
              (0.0, 1.0),  # Kd
              (0.5, 1.5),  # lambda (orden integral)
              (0.0, 1.0)]  # mu (orden derivativo)
    # Inicialización de partículas
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
    g_best_pos = None
    g_best_cost = np.inf
    cost_history = []
    w, c1, c2 = 0.5, 1.5, 1.5
    for _ in range(n_iter):
        for p in particles:
            # Evaluar costo
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
            for i, (low, high) in enumerate(bounds):
                p['position'][i] = max(low, min(high, p['position'][i]))
        cost_history.append(g_best_cost)
        print(f"PSO FOPID Iter {_+1}/{n_iter}: Best ISE = {g_best_cost:.4f}")
        
    return tuple(g_best_pos), g_best_cost, cost_history


def experiment_name(dataset: str) -> str:
    return f"{dataset}"

def process_dataset(dataset_name: str):
    print(f"\n{'='*40}")
    print(f"Processing Data Set: {dataset_name}")
    print(f"{'='*40}")

    # Dataset metadata
    metadata = {
        'FD001': {'conditions': 'ONE (Sea Level)', 'faults': 'ONE (HPC Degradation)', 'keep_settings': False},
        'FD002': {'conditions': 'SIX', 'faults': 'ONE (HPC Degradation)', 'keep_settings': True},
        'FD003': {'conditions': 'ONE (Sea Level)', 'faults': 'TWO (HPC, Fan Degradation)', 'keep_settings': False},
        'FD004': {'conditions': 'SIX', 'faults': 'TWO (HPC, Fan Degradation)', 'keep_settings': True}
    }
    info = metadata.get(dataset_name, {'conditions': 'Unknown', 'faults': 'Unknown', 'keep_settings': False})
    print(f"Conditions: {info['conditions']}")
    print(f"Fault Modes: {info['faults']}")

    # Paths
    train_path = f'data/CMAPSSData/train_{dataset_name}.txt'
    test_path = f'data/CMAPSSData/test_{dataset_name}.txt'
    rul_path = f'data/CMAPSSData/RUL_{dataset_name}.txt'

    # Check validity
    try:
        train_df = load_cmapss(train_path)
        test_df = load_cmapss(test_path)
    except FileNotFoundError:
        print(f"Error: Files for {dataset_name} not found in data/CMAPSSData/")
        return

    # Basic stats
    print(f"Train trajectories: {train_df['col_1'].nunique()}")
    print(f"Test trajectories: {test_df['col_1'].nunique()}")

    # RUL Calculation
    train_rul = compute_rul(train_df)

    # Feature Selection
    # Drop settings if singular condition, keep if multiple
    cols_to_drop = ['col_6', 'col_8', 'col_9', 'col_10', 'col_14', 'col_15', 'col_17', 'col_20', 'col_21', 'col_22', 'col_23']
    if not info['keep_settings']:
        cols_to_drop.extend(['col_3', 'col_4', 'col_5'])
    
    print(f"Dropping columns: {cols_to_drop}")
    train_df_red = train_df.drop(columns=cols_to_drop, errors='ignore')
    test_df_red = test_df.drop(columns=cols_to_drop, errors='ignore')

    # Build sequences
    X, y = build_sequences(train_df_red, train_rul, seq_len=30)
    
    # Train/Val Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # PSO Optimization for MLP
    print(f"[{dataset_name}] Starting PSO Optimization for MLP...")
    best_hidden, best_score, mlp_history = pso_optimize(X_train, X_val, y_train, y_val,
                                          n_particles=5, n_iter=5,
                                          bounds=(10, 100)) # Reduced bounds/iter for speed in demo
    print(f'[{dataset_name}] Best Hidden Config: {best_hidden}, MSE={best_score:.2f}')

    # Plot Convergence
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(mlp_history) + 1), mlp_history, 'b-o', linewidth=2)
    plt.title(f'MLP PSO Convergence - {dataset_name}', fontsize=14)
    plt.xlabel('Iteration')
    plt.ylabel('Best MSE (Validation)')
    plt.grid(True)
    plt.savefig(f'fig_mlp_convergence_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Final Training
    model = train_final_model(X_train, y_train, best_hidden)
    preds_val = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds_val)
    mse = mean_squared_error(y_val, preds_val)
    print(f'[{dataset_name}] Validation MAE: {mae:.2f}, MSE: {mse:.2f}')

    # Plot RUL Prediction
    plt.figure(figsize=(10, 6))
    indices = np.argsort(y_val)
    plt.plot(y_val[indices], 'k-', label='True RUL', linewidth=2)
    plt.plot(preds_val[indices], 'r--', label='Predicted RUL', alpha=0.7)
    plt.title(f'RUL Prediction - {dataset_name}', fontsize=14)
    plt.xlabel('Sample Index (Sorted by RUL)')
    plt.ylabel('Remaining Useful Life (cycles)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'fig_rul_prediction_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Ejecuta el flujo para todos los datasets."""
    datasets = ['FD001', 'FD002', 'FD003', 'FD004']
    
    for ds in datasets:
        process_dataset(ds)

    # Optimización del FOPID (Independiente del dataset, simulación de control)
    print("\n" + "="*40)
    print("Iniciando optimización PSO para FOPID (Sistema de Control)...")
    print("="*40)
    best_params, best_cost, fopid_history = pso_fopid(n_particles=10, n_iter=10)
    print(f'Mejor FOPID encontrado: {best_params}, ISE={best_cost:.4f}')

    # Gráfica Convergencia FOPID
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(fopid_history) + 1), fopid_history, 'g-o', linewidth=2)
    plt.title('FOPID PSO Optimization Convergence', fontsize=14)
    plt.xlabel('Iteration')
    plt.ylabel('Best ISE Cost')
    plt.grid(True)
    plt.savefig('fig_fopid_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Gráfica Respuesta FOPID
    t_opt, y_opt = simulate_fopid_response(best_params)
    mean_params = (1.0, 0.5, 0.5, 1.0, 0.5)
    t_base, y_base = simulate_fopid_response(mean_params)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_opt, np.ones_like(t_opt), 'k--', label='Setpoint')
    plt.plot(t_base, y_base, 'b:', label='Baseline Controller', linewidth=1.5)
    plt.plot(t_opt, y_opt, 'r-', label='Optimized FOPID', linewidth=2)
    plt.title('Step Response Comparison', fontsize=14)
    plt.xlabel('Time (s)')
    plt.ylabel('System Output')
    plt.legend()
    plt.grid(True)
    plt.savefig('fig_fopid_response.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()