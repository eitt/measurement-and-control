| dataset | variant | architecture | hidden_layer_count | hidden_layers | pruning_method | parameters_or_nonzero | active_arcs | total_arcs | keep_ratio | validation_mse | validation_mae | official_test_mse | official_test_mae | fit_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FD001 | before_pruning_dense_ann | relu:[49] | 1 | 49 | none | 14799 | 14749 | 14749 | 1.000000 | 7163.345215 | 74.253494 | 7434.659180 | 75.705902 | 0.022536 |
| FD001 | after_pruning_ann | relu:[49] | 1 | 49 | magnitude_fallback | 7424 | 7374 | 14749 | 0.499966 | 7134.569336 | 74.155083 | 7334.388184 | 75.051491 | 1.440105 |
| FD002 | before_pruning_dense_ann | relu:[98, 21, 89] | 3 | 98-21-89 | none | 42445 | 42236 | 42236 | 1.000000 | 7969.223633 | 78.044197 | 9430.245117 | 80.861916 | 0.058315 |
| FD002 | after_pruning_ann | relu:[98, 21, 89] | 3 | 98-21-89 | magnitude_multilayer | 21327 | 21118 | 42236 | 0.500000 | 7946.356445 | 77.900467 | 9434.047852 | 80.879417 | 0.120050 |
| FD003 | before_pruning_dense_ann | relu:[98, 21, 89] | 3 | 98-21-89 | none | 33625 | 33416 | 33416 | 1.000000 | 8935.865234 | 84.401108 | 7326.764160 | 74.923103 | 0.034779 |
| FD003 | after_pruning_ann | relu:[98, 21, 89] | 3 | 98-21-89 | magnitude_multilayer | 16917 | 16708 | 33416 | 0.500000 | 8911.624023 | 84.272316 | 7320.314941 | 74.856422 | 0.065693 |
| FD004 | before_pruning_dense_ann | relu:[98, 21, 89] | 3 | 98-21-89 | none | 42445 | 42236 | 42236 | 1.000000 | 9373.557617 | 87.748749 | 10393.030273 | 86.147873 | 0.058138 |
| FD004 | after_pruning_ann | relu:[98, 21, 89] | 3 | 98-21-89 | magnitude_multilayer | 21327 | 21118 | 42236 | 0.500000 | 9338.375977 | 87.546799 | 10394.474609 | 86.147987 | 0.124125 |
