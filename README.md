#Universal CMAPSS RUL Prediction & Robust FOPID ControlThis project implements a comprehensive framework for **Remaining Useful Life (RUL)** prediction across all four NASA CMAPSS jet engine datasets (`FD001` through `FD004`). It integrates a Particle Swarm Optimization (PSO) tuned Neural Network (MLP) for prognosis and a robust **Fractional Order PID (FOPID)** controller for system stability.

##Key Features* **Multi-Dataset Analysis:** Automatically loops through and processes `FD001`, `FD002`, `FD003`, and `FD004` to compare performance across varying operating conditions and fault modes.
* **PSO-Optimized Architectures:** Uses Particle Swarm Optimization to dynamically find the optimal hidden layer structure for the Neural Network based on the complexity of each dataset.
* **Robust FOPID Control:** Implements a fractional calculus controller with stability safeguards (handling complex number errors) to simulate engine control response based on RUL predictions.
* **Computational Benchmarking:** Tracks and visualizes the training vs. optimization time for each dataset.
* **Universal Dashboard:** Automatically generates a 2x2 summary dashboard comparing MSE/MAE, error clustering, and computational efficiency.

##Datasets (NASA CMAPSS)The project models and synthesizes the following datasets:

| Dataset | Train Traj. | Conditions | Fault Modes | Complexity |
| --- | --- | --- | --- | --- |
| **FD001** | 100 | 1 (Sea Level) | 1 (HPC) | Low |
| **FD002** | 260 | 6 (Mixed) | 1 (HPC) | Medium |
| **FD003** | 100 | 1 (Sea Level) | 2 (HPC, Fan) | Medium |
| **FD004** | 248 | 6 (Mixed) | 2 (HPC, Fan) | High |

##Data FormatInput files (`train_FD00x.txt`) are space-separated text files with 26 columns:

1. **Unit Number**
2. **Time (Cycles)**
3. **Op. Setting 1**
4. **Op. Setting 2**
5. **Op. Setting 3**
6. **Sensor Measurement 1**
...
7. **Sensor Measurement 21**

##Usage###1. PrerequisitesEnsure you have the required Python libraries installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib

```

###2. SetupPlace the dataset files (`train_FD001.txt` ... `train_FD004.txt`) in the same directory as the script.

###3. ExecutionRun the universal main script to process all datasets and generate reports:

```bash
python main_universal.py

```

###4. OutputsThe script generates the following analysis files:

* `universal_dashboard.png`: A 2x2 Summary of Global Performance (MSE/MAE trends and Error Clustering).
* `universal_time_analysis.png`: Bar chart comparing computational cost across datasets.
* `universal_fopid_convergence.png`: Optimization curve for the control parameters.
* Console Logs: Real-time training metrics and architectural choices.

##Methodology###1. RUL Prediction (MLP + PSO)* **Preprocessing:** MinMax Scaling, Sequence generation (window size: 30), and feature selection.
* **Architecture Search:** A PSO algorithm explores the hyperparameter space (hidden neurons) to minimize validation MSE.
* **Training:** The best architecture found is retrained on the full training set.

###2. Control System (FOPID)* **Simulation:** Simulates a plant response using a Fractional Order PID controller: C(s) = K_p + K_i s^{-\lambda} + K_d s^{\mu}.
* **Robustness:** Mathematical safeguards (`abs()` and `sign()`) prevent complex number instability during fractional differentiation.

##ReferenceA. Saxena, K. Goebel, D. Simon, and N. Eklund, “Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation”, in the Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.