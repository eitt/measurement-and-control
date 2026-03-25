| dataset | variant | architecture | hidden_layer_count | hidden_layers | pruning_method | parameters_or_nonzero | active_arcs | total_arcs | keep_ratio | validation_mse | validation_mae | official_test_mse | official_test_mae | fit_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FD001 | before_pruning_dense_ann | relu:[98, 90] | 2 | 98-90 | none | 38499 | 38310 | 38310 | 1.000000 | 1466.675537 | 30.628197 | 393.556763 | 14.512111 | 7.345848 |
| FD001 | after_pruning_ann | relu:[98, 90] | 2 | 98-90 | magnitude_multilayer | 19344 | 19155 | 38310 | 0.500000 | 270.992493 | 12.197764 | 363.861053 | 13.918473 | 8.288148 |
| FD002 | before_pruning_dense_ann | relu:[99, 100] | 2 | 99-100 | none | 48810 | 48610 | 48610 | 1.000000 | 1519.796021 | 33.505192 | 1658.454346 | 31.919640 | 21.437601 |
| FD002 | after_pruning_ann | relu:[99, 100] | 2 | 99-100 | magnitude_multilayer | 24505 | 24305 | 48610 | 0.500000 | 767.408264 | 22.650566 | 1453.463623 | 28.914360 | 21.184443 |
| FD003 | before_pruning_dense_ann | relu:[94, 99] | 2 | 94-99 | none | 37799 | 37605 | 37605 | 1.000000 | 636.594177 | 20.557026 | 270.296204 | 12.712638 | 9.886196 |
| FD003 | after_pruning_ann | relu:[94, 99] | 2 | 94-99 | magnitude_multilayer | 18996 | 18802 | 37605 | 0.499987 | 249.019623 | 12.177488 | 309.977753 | 13.924731 | 11.334037 |
| FD004 | before_pruning_dense_ann | relu:[95, 100] | 2 | 95-100 | none | 46846 | 46650 | 46650 | 1.000000 | 1147.435791 | 28.519026 | 1656.772705 | 32.305058 | 25.703888 |
| FD004 | after_pruning_ann | relu:[95, 100] | 2 | 95-100 | magnitude_multilayer | 23521 | 23325 | 46650 | 0.500000 | 794.514832 | 23.160389 | 1609.881592 | 32.358215 | 32.352943 |
