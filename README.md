# CMAPSS RUL Prediction and FOPID Optimization

This project implements a pipeline for Remaining Useful Life (RUL) prediction using the NASA CMAPSS datasets and optimizes a Fractional Order PID (FOPID) controller.

## Datasets

The project models and synthesizes the following datasets from the CMAPSS Data:

**Data Set: FD001**
- Train trajectories: 100
- Test trajectories: 100
- Conditions: ONE (Sea Level)
- Fault Modes: ONE (HPC Degradation)

**Data Set: FD002**
- Train trajectories: 260
- Test trajectories: 259
- Conditions: SIX
- Fault Modes: ONE (HPC Degradation)

**Data Set: FD003**
- Train trajectories: 100
- Test trajectories: 100
- Conditions: ONE (Sea Level)
- Fault Modes: TWO (HPC Degradation, Fan Degradation)

**Data Set: FD004**
- Train trajectories: 248
- Test trajectories: 249
- Conditions: SIX
- Fault Modes: TWO (HPC Degradation, Fan Degradation)

## Data Format

The data are provided as a zip-compressed text file with 26 columns of numbers, separated by spaces. Each row is a snapshot of data taken during a single operational cycle, each column is a different variable. The columns correspond to:

1.	unit number
2.	time, in cycles
3.	operational setting 1
4.	operational setting 2
5.	operational setting 3
6.	sensor measurement 1
7.	sensor measurement 2
...
26.	sensor measurement 26

## Reference

A. Saxena, K. Goebel, D. Simon, and N. Eklund, “Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation”, in the Proceedings of the Ist International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.

## Usage

Run the main script to process all datasets:

```bash
python code.py
```

Or use the provided Jupyter Notebook `measure_control_colab.ipynb`.
