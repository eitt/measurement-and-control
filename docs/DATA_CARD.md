# Data Card: NASA C-MAPSS Jet Engine Simulator Data

## 1. Dataset Overview

This data card documents the local copy of the NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) benchmark stored under `data/CMAPSSData`. The benchmark contains multivariate run-to-failure trajectories for turbofan engines and is widely used for Remaining Useful Life (RUL) prediction.

- **Source:** [NASA Open Data Portal](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
- **Local files:** `train_FD001..004.txt`, `test_FD001..004.txt`, `RUL_FD001..004.txt`, `readme.txt`
- **Generator:** Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)
- **Domain:** Predictive maintenance, prognostics, aviation

The benchmark is partitioned into four subsets that increase in difficulty by combining operating-regime shifts and fault-mode diversity.

| Dataset | Operating conditions | Fault modes | Practical interpretation |
|---|---:|---:|---|
| FD001 | 1 | 1 | Baseline case: single regime, single fault |
| FD002 | 6 | 1 | Regime shifts must be separated from degradation |
| FD003 | 1 | 2 | Multiple fault signatures appear in one regime |
| FD004 | 6 | 2 | Combined regime-shift and multi-fault difficulty |

## 2. Dataset Structure and Split Logic

- **Training files (`train_FD00*.txt`):** full run-to-failure trajectories. These are used to compute supervised RUL labels directly from the last observed cycle of each engine.
- **Testing files (`test_FD00*.txt`):** truncated trajectories for unseen engines. The true failure cycle is not present in the file itself.
- **Ground-truth labels (`RUL_FD00*.txt`):** one remaining-life value per engine in the corresponding test file.
- **Integrity check on the local copy:** all train and test files load with the expected 26-column whitespace-delimited structure, and no missing values were found in any train or test matrix.

## 3. Data Schema

Each row in the `train` and `test` files represents a single operational cycle and has 26 columns.

| Column | Name | Data Type | Description |
| :--- | :--- | :--- | :--- |
| 1 | `unit_number` | Integer | Engine identifier within the subset. |
| 2 | `time_in_cycles` | Integer | Operational cycle index. |
| 3-5 | `operational_setting_*` | Float | Three operating-condition variables that affect engine behavior. |
| 6-26 | `sensor_measurement_*` | Float | Twenty-one sensor channels capturing temperatures, pressures, flows, speeds, and related measurements. |

In the codebase, these variables are usually renamed to `col_1` through `col_26` for consistent loading across all subsets.

## 4. Empirical Descriptive Statistics (Local Copy)

All statistics below were computed from the files in `data/CMAPSSData` in this workspace. Subset-level counts use train, test, and RUL files. Column-level tables use the training split because those trajectories run to failure and are the basis for most supervised preprocessing pipelines.

### 4.1 Subset-Level Summary

| Dataset | Train rows | Train units | Test rows | Test units | Train mean cycles/unit | Train std | Train min-max | Test mean cycles/unit | Test std | Test min-max | Test RUL mean | Test RUL std | Test RUL min-max |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---|
| FD001 | 20631 | 100 | 13096 | 100 | 206.31 | 46.34 | 128-362 | 130.96 | 53.59 | 31-303 | 75.52 | 41.76 | 7-145 |
| FD002 | 53759 | 260 | 33991 | 259 | 206.77 | 46.78 | 128-378 | 131.24 | 63.09 | 21-367 | 81.19 | 53.88 | 6-194 |
| FD003 | 24720 | 100 | 16596 | 100 | 247.20 | 86.48 | 145-525 | 165.96 | 86.89 | 38-475 | 75.32 | 41.60 | 6-145 |
| FD004 | 61249 | 249 | 41214 | 248 | 245.98 | 73.11 | 128-543 | 166.19 | 91.61 | 19-486 | 86.55 | 54.63 | 6-195 |

- **Important note on FD004 counts:** the bundled `data/CMAPSSData/readme.txt` swaps the FD004 train/test trajectory counts (`248/249`), but the actual files contain `249` training engines and `248` test engines. The tables in this data card follow the file contents.

### 4.2 Column-Level Descriptive Statistics for the Training Split

The row labels below follow the naming convention used in the code: `col_1 = unit_number`, `col_2 = time_in_cycles`, `col_3..col_5 = operational settings`, and `col_6..col_26 = sensor measurements 1..21`.

#### FD001

| Column | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| col_1 | 51.5066 | 29.2276 | 1.0000 | 100.0000 |
| col_2 | 108.8079 | 68.8810 | 1.0000 | 362.0000 |
| col_3 | -0.0000 | 0.0022 | -0.0087 | 0.0087 |
| col_4 | 0.0000 | 0.0003 | -0.0006 | 0.0006 |
| col_5 | 100.0000 | 0.0000 | 100.0000 | 100.0000 |
| col_6 | 518.6700 | 0.0000 | 518.6700 | 518.6700 |
| col_7 | 642.6809 | 0.5001 | 641.2100 | 644.5300 |
| col_8 | 1590.5231 | 6.1311 | 1571.0400 | 1616.9100 |
| col_9 | 1408.9338 | 9.0006 | 1382.2500 | 1441.4900 |
| col_10 | 14.6200 | 0.0000 | 14.6200 | 14.6200 |
| col_11 | 21.6098 | 0.0014 | 21.6000 | 21.6100 |
| col_12 | 553.3677 | 0.8851 | 549.8500 | 556.0600 |
| col_13 | 2388.0967 | 0.0710 | 2387.9000 | 2388.5600 |
| col_14 | 9065.2429 | 22.0829 | 9021.7300 | 9244.5900 |
| col_15 | 1.3000 | 0.0000 | 1.3000 | 1.3000 |
| col_16 | 47.5412 | 0.2671 | 46.8500 | 48.5300 |
| col_17 | 521.4135 | 0.7376 | 518.6900 | 523.3800 |
| col_18 | 2388.0962 | 0.0719 | 2387.8800 | 2388.5600 |
| col_19 | 8143.7527 | 19.0762 | 8099.9400 | 8293.7200 |
| col_20 | 8.4421 | 0.0375 | 8.3249 | 8.5848 |
| col_21 | 0.0300 | 0.0000 | 0.0300 | 0.0300 |
| col_22 | 393.2107 | 1.5488 | 388.0000 | 400.0000 |
| col_23 | 2388.0000 | 0.0000 | 2388.0000 | 2388.0000 |
| col_24 | 100.0000 | 0.0000 | 100.0000 | 100.0000 |
| col_25 | 38.8163 | 0.1807 | 38.1400 | 39.4300 |
| col_26 | 23.2897 | 0.1083 | 22.8942 | 23.6184 |

#### FD002

| Column | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| col_1 | 131.0830 | 74.4639 | 1.0000 | 260.0000 |
| col_2 | 109.1547 | 69.1806 | 1.0000 | 378.0000 |
| col_3 | 23.9984 | 14.7474 | 0.0000 | 42.0080 |
| col_4 | 0.5721 | 0.3100 | 0.0000 | 0.8420 |
| col_5 | 94.0460 | 14.2377 | 60.0000 | 100.0000 |
| col_6 | 472.9102 | 26.3897 | 445.0000 | 518.6700 |
| col_7 | 579.6724 | 37.2894 | 535.5300 | 644.5200 |
| col_8 | 1419.9710 | 105.9463 | 1243.7300 | 1612.8800 |
| col_9 | 1205.4420 | 119.1234 | 1023.7700 | 1439.2300 |
| col_10 | 8.0320 | 3.6138 | 3.9100 | 14.6200 |
| col_11 | 11.6007 | 5.4318 | 5.7100 | 21.6100 |
| col_12 | 282.6068 | 146.0053 | 136.8000 | 555.8200 |
| col_13 | 2228.8792 | 145.2098 | 1914.7700 | 2388.3900 |
| col_14 | 8525.2008 | 335.8120 | 7985.5600 | 9215.6600 |
| col_15 | 1.0950 | 0.1275 | 0.9300 | 1.3000 |
| col_16 | 42.9852 | 3.2324 | 36.2300 | 48.5100 |
| col_17 | 266.0690 | 137.6595 | 129.1200 | 523.3700 |
| col_18 | 2334.5573 | 128.0683 | 2027.6100 | 2390.4800 |
| col_19 | 8066.5977 | 84.8379 | 7848.3600 | 8268.5000 |
| col_20 | 9.3297 | 0.7493 | 8.3357 | 11.0669 |
| col_21 | 0.0233 | 0.0047 | 0.0200 | 0.0300 |
| col_22 | 348.3095 | 27.7545 | 303.0000 | 399.0000 |
| col_23 | 2228.8064 | 145.3280 | 1915.0000 | 2388.0000 |
| col_24 | 97.7568 | 5.3641 | 84.9300 | 100.0000 |
| col_25 | 20.7893 | 9.8693 | 10.1800 | 39.3400 |
| col_26 | 12.4734 | 5.9216 | 6.0105 | 23.5901 |

#### FD003

| Column | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| col_1 | 48.6319 | 29.3490 | 1.0000 | 100.0000 |
| col_2 | 139.0771 | 98.8467 | 1.0000 | 525.0000 |
| col_3 | -0.0000 | 0.0022 | -0.0086 | 0.0086 |
| col_4 | 0.0000 | 0.0003 | -0.0006 | 0.0007 |
| col_5 | 100.0000 | 0.0000 | 100.0000 | 100.0000 |
| col_6 | 518.6700 | 0.0000 | 518.6700 | 518.6700 |
| col_7 | 642.4579 | 0.5230 | 640.8400 | 645.1100 |
| col_8 | 1588.0792 | 6.8104 | 1564.3000 | 1615.3900 |
| col_9 | 1404.4712 | 9.7732 | 1377.0600 | 1441.1600 |
| col_10 | 14.6200 | 0.0000 | 14.6200 | 14.6200 |
| col_11 | 21.5958 | 0.0181 | 21.4500 | 21.6100 |
| col_12 | 555.1438 | 3.4373 | 549.6100 | 570.4900 |
| col_13 | 2388.0716 | 0.1583 | 2386.9000 | 2388.6000 |
| col_14 | 9064.1108 | 19.9803 | 9017.9800 | 9234.3500 |
| col_15 | 1.3012 | 0.0035 | 1.2900 | 1.3200 |
| col_16 | 47.4157 | 0.3001 | 46.6900 | 48.4400 |
| col_17 | 523.0509 | 3.2553 | 517.7700 | 537.4000 |
| col_18 | 2388.0716 | 0.1581 | 2386.9300 | 2388.6100 |
| col_19 | 8144.2029 | 16.5041 | 8099.6800 | 8290.5500 |
| col_20 | 8.3962 | 0.0605 | 8.1563 | 8.5705 |
| col_21 | 0.0300 | 0.0000 | 0.0300 | 0.0300 |
| col_22 | 392.5665 | 1.7615 | 388.0000 | 399.0000 |
| col_23 | 2388.0000 | 0.0000 | 2388.0000 | 2388.0000 |
| col_24 | 100.0000 | 0.0000 | 100.0000 | 100.0000 |
| col_25 | 38.9886 | 0.2489 | 38.1700 | 39.8500 |
| col_26 | 23.3930 | 0.1492 | 22.8726 | 23.9505 |

#### FD004

| Column | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| col_1 | 124.3252 | 71.9953 | 1.0000 | 249.0000 |
| col_2 | 134.3114 | 89.7834 | 1.0000 | 543.0000 |
| col_3 | 23.9998 | 14.7807 | 0.0000 | 42.0080 |
| col_4 | 0.5713 | 0.3107 | 0.0000 | 0.8420 |
| col_5 | 94.0316 | 14.2520 | 60.0000 | 100.0000 |
| col_6 | 472.8824 | 26.4368 | 445.0000 | 518.6700 |
| col_7 | 579.4201 | 37.3426 | 535.4800 | 644.4200 |
| col_8 | 1417.8966 | 106.1676 | 1242.6700 | 1613.0000 |
| col_9 | 1201.9154 | 119.3276 | 1024.4200 | 1440.7700 |
| col_10 | 8.0316 | 3.6229 | 3.9100 | 14.6200 |
| col_11 | 11.5895 | 5.4440 | 5.6700 | 21.6100 |
| col_12 | 283.3286 | 146.8802 | 136.1700 | 570.8100 |
| col_13 | 2228.6860 | 145.3482 | 1914.7200 | 2388.6400 |
| col_14 | 8524.6733 | 336.9275 | 7984.5100 | 9196.8100 |
| col_15 | 1.0964 | 0.1277 | 0.9300 | 1.3200 |
| col_16 | 42.8745 | 3.2435 | 36.0400 | 48.3600 |
| col_17 | 266.7357 | 138.4791 | 128.3100 | 537.4900 |
| col_18 | 2334.4276 | 128.1979 | 2027.5700 | 2390.4900 |
| col_19 | 8067.8118 | 85.6705 | 7845.7800 | 8261.6500 |
| col_20 | 9.2856 | 0.7504 | 8.1757 | 11.0663 |
| col_21 | 0.0233 | 0.0047 | 0.0200 | 0.0300 |
| col_22 | 347.7600 | 27.8083 | 302.0000 | 399.0000 |
| col_23 | 2228.6133 | 145.4725 | 1915.0000 | 2388.0000 |
| col_24 | 97.7514 | 5.3694 | 84.9300 | 100.0000 |
| col_25 | 20.8643 | 9.9364 | 10.1600 | 39.8900 |
| col_26 | 12.5190 | 5.9627 | 6.0843 | 23.8852 |

## 5. Contrast With Cited Literature

- **Canonical subset taxonomy is empirically visible in the raw numbers.** Saxena et al. (2008) describes FD001 and FD003 as single-condition subsets and FD002 and FD004 as six-condition subsets. The local descriptive statistics support that directly: FD001 and FD003 have near-zero spread in `col_3..col_5`, while FD002 and FD004 show wide operational-setting dispersion (`col_3` std about `14.75`, `col_4` about `0.31`, `col_5` about `14.24-14.25`).
- **The trajectory counts align with the literature summary reproduced in the project paper draft.** The local files yield FD001 `100/100`, FD002 `260/259`, FD003 `100/100`, and FD004 `249/248` train/test engines, matching the subset summary in `docs/article.tex`.
- **The two-fault subsets also exhibit longer degradation histories.** Mean train cycles per engine rise from about `206` in FD001 and FD002 to about `246-247` in FD003 and FD004. This is consistent with later literature treating FD003 and FD004 as harder prognostic scenarios because fault interactions unfold over longer and more variable trajectories.
- **The feature-pruning rationale in the literature review is partly confirmed by the raw statistics.** The review section in `docs/article.tex` notes a common preprocessing choice: drop operational settings and a fixed set of low-information sensors to retain a compact 10-sensor subset. In FD001 and FD003, `col_6`, `col_10`, `col_21`, and `col_23` are exactly constant, and `col_15` is constant or nearly constant, which supports that simplification. In FD002 and FD004 those channels vary more, so the same fixed-drop rule should be understood as a universal baseline rather than a universally optimal choice.
- **Most model-comparison papers emphasize predictive performance rather than descriptive statistics.** Works such as Peringal et al. (2024), Chen (2024), Elsherif et al. (2025), and Xue et al. (2025) focus on architecture design and accuracy on C-MAPSS. The statistics above complement those studies by making the underlying distributional differences across subsets explicit.

## 6. Related Articles and Citations

- Saxena, A., Goebel, K., Simon, D., and Eklund, N. (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*. Proceedings of PHM08. Original benchmark description for C-MAPSS and the train/test/RUL split logic.
- Peringal, A., Mohiuddin, M. B., and Hassan, A. (2024). *Remaining Useful Life Prediction for Aircraft Engines using LSTM*. Uses C-MAPSS to compare sequence models against simpler baselines.
- Chen, X. (2024). *A Novel Transformer-Based Deep-Learning Model Enhanced by Position-Sensitive Attention and Gated Hierarchical LSTM for Aero-Engine Remaining Useful Life Prediction*. Example of recent high-capacity architectures evaluated on C-MAPSS.
- Elsherif, S. M., Hafiz, B., Makhlouf, M. A., and Farouk, O. (2025). *A Deep Learning-Based Prognostic Approach for Predicting Turbofan Engine Degradation and Remaining Useful Life*. Uses all four C-MAPSS subsets and helps motivate the complexity ordering adopted in this project.
- Xue, F., Jin, G., Tan, L., Zhang, C., and Yu, Y. (2025). *Predictive Maintenance Programs for Aircraft Engines Based on Remaining Useful Life Prediction*. Illustrates current work on automated architecture design for C-MAPSS-based prognostics.

A broader narrative literature review for this project is maintained in `docs/article.tex` and `docs/references.bib`. The numeric values in this data card are computed from the local dataset files rather than copied from those papers.
