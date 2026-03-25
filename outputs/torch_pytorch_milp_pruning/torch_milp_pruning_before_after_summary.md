| dataset | variant | architecture | hidden_layer_count | hidden_layers | pruning_method | parameters_or_nonzero | active_arcs | total_arcs | keep_ratio | validation_mse | validation_mae | official_test_mse | official_test_mae | fit_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FD001 | before_pruning_dense_ann | relu:[98, 87] | 2 | 98-87 | none | 38199 | 38013 | 38013 | 1.000000 | 464.434357 | 17.149757 | 300.798706 | 13.411980 | 28.124877 |
| FD001 | after_pruning_ann | relu:[98, 87] | 2 | 98-87 | magnitude_solver_fallback_after_reduced_exact | 3987 | 3801 | 38013 | 0.099992 | 256.761749 | 12.142019 | 318.835510 | 13.179683 | 62.849944 |
| FD002 | before_pruning_dense_ann | relu:[100, 23] | 2 | 100-23 | none | 41447 | 41323 | 41323 | 1.000000 | 963.841309 | 26.040960 | 1302.345825 | 26.938828 | 50.811186 |
| FD002 | after_pruning_ann | relu:[100, 23] | 2 | 100-23 | magnitude_solver_fallback_after_reduced_exact | 4256 | 4132 | 41323 | 0.099993 | 787.797058 | 22.885542 | 1398.657104 | 27.888117 | 86.015809 |
| FD003 | before_pruning_dense_ann | relu:[100, 25] | 2 | 100-25 | none | 32651 | 32525 | 32525 | 1.000000 | 287.682800 | 12.790985 | 249.695892 | 12.043966 | 22.876329 |
| FD003 | after_pruning_ann | relu:[100, 25] | 2 | 100-25 | magnitude_solver_fallback_after_reduced_exact | 3378 | 3252 | 32525 | 0.099985 | 211.776840 | 10.964574 | 296.196014 | 13.534018 | 58.038320 |
| FD004 | before_pruning_dense_ann | relu:[100, 17] | 2 | 100-17 | none | 40835 | 40717 | 40717 | 1.000000 | 654.864746 | 20.621372 | 1413.673340 | 30.227074 | 58.237404 |
| FD004 | after_pruning_ann | relu:[100, 17] | 2 | 100-17 | magnitude_solver_fallback_after_reduced_exact | 4190 | 4072 | 40717 | 0.100007 | 642.360840 | 20.687126 | 1470.834717 | 31.450867 | 106.841812 |
