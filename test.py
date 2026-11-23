from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from IPython.display import HTML, display
from matplotlib.lines import Line2D

OBJECTIVES = ('rmse', 'nasa', 'both')
OBJECTIVE_COLORS = {
    'rmse': '#2878B5',
    'nasa': '#E07A2D',
    'both': '#3A923A',
}

def _required_number(mapping, key, context):
    if key not in mapping:
        raise KeyError(f'Falta el campo {key!r} en {context}.')
    value = float(mapping[key])
    if not np.isfinite(value):
        raise ValueError(f'El campo {key!r} no es finito en {context}.')
    return value


def compare_candidate_results(
    dataset_name,
    results_dir='mopso_results',
    test_rul_path=None,
    min_parameters=None,
    max_parameters=None,
    min_rmse=None,
    max_rmse=None,
    min_nasa=None,
    max_nasa=None
):
    """Grafica candidatos; filtros NASA usan el promedio por motor."""
    dataset_name = str(dataset_name).strip().upper()
    supported_datasets = {f'FD00{i}' for i in range(1, 5)}
    if dataset_name not in supported_datasets:
        choices = ', '.join(sorted(supported_datasets))
        raise ValueError(
            f'Dataset no admitido: {dataset_name!r}. Use uno de: {choices}.'
        )

    filter_bounds = {
        'parameter_count': (min_parameters, max_parameters),
        'rmse': (min_rmse, max_rmse),
        'nasa_score': (min_nasa, max_nasa),
    }
    normalized_bounds = {}
    for metric, (minimum, maximum) in filter_bounds.items():
        converted = []
        for label, value in (('mínimo', minimum), ('máximo', maximum)):
            if value is None:
                converted.append(None)
                continue
            number = float(value)
            if not np.isfinite(number):
                raise ValueError(f'El {label} de {metric} debe ser finito.')
            converted.append(number)
        if converted[0] is not None and converted[1] is not None and converted[0] > converted[1]:
            raise ValueError(f'El mínimo de {metric} no puede superar el máximo.')
        normalized_bounds[metric] = tuple(converted)

    results_dir = Path(results_dir)
    if test_rul_path is None:
        rul_filename = f'RUL_{dataset_name}.txt'
        rul_candidates = (
            Path('data') / 'CMAPSSData' / rul_filename,
            Path('data') / 'backup_root_files' / rul_filename,
            results_dir.parent / 'data' / 'CMAPSSData' / rul_filename,
            results_dir.parent / 'data' / 'backup_root_files' / rul_filename,
        )
        test_rul_path = next(
            (path for path in rul_candidates if path.is_file()), None
        )
        if test_rul_path is None:
            searched = ', '.join(str(path) for path in rul_candidates)
            raise FileNotFoundError(
                f'No se encontró {rul_filename}; rutas comprobadas: {searched}. '
                'Pase test_rul_path explícitamente.'
            )
    else:
        test_rul_path = Path(test_rul_path)
        if not test_rul_path.is_file():
            raise FileNotFoundError(f'No se encontró {test_rul_path}.')

    with test_rul_path.open('r', encoding='utf-8') as rul_file:
        test_engine_count = sum(1 for line in rul_file if line.strip())
    if test_engine_count <= 0:
        raise ValueError(f'No se encontraron motores en {test_rul_path}.')

    records = []
    for objective in OBJECTIVES:
        json_path = results_dir / (
            f'{dataset_name}_mopso_final_candidates_v6_{objective}.json'
        )
        if not json_path.is_file():
            raise FileNotFoundError(
                f'No se encontró {json_path}. Pase la carpeta de resultados '
                "con results_dir si no es 'mopso_results'."
            )

        with json_path.open('r', encoding='utf-8') as json_file:
            package = json.load(json_file)
        if package.get('dataset') != dataset_name:
            raise ValueError(
                f'El dataset declarado en {json_path} no coincide con '
                f'{dataset_name}.'
            )
        if package.get('objective') != objective:
            raise ValueError(
                f'El objetivo declarado en {json_path} no coincide con '
                f'{objective}.'
            )

        candidates = package.get('candidates')
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f'No hay candidatos en {json_path}.')

        for candidate in candidates:
            candidate_id = candidate.get('candidate_id')
            architecture = candidate.get('architecture', {})
            hidden_sizes = architecture.get('hidden_sizes')
            metrics = candidate.get('test_results', {})
            context = f'{json_path.name}, candidato {candidate_id!r}'
            if not candidate_id:
                raise ValueError(f'Falta candidate_id en {context}.')
            if not isinstance(hidden_sizes, list):
                raise ValueError(f'Falta architecture.hidden_sizes en {context}.')
            if not isinstance(metrics, dict):
                raise ValueError(f'test_results no es válido en {context}.')

            parameter_count = int(
                _required_number(candidate, 'parameter_count', context)
            )
            if parameter_count <= 0:
                raise ValueError(f'parameter_count debe ser positivo en {context}.')

            rmse = _required_number(metrics, 'test_rmse', context)
            nasa = _required_number(metrics, 'test_nasa', context)
            rmse_low = _required_number(metrics, 'test_rmse_ci95_lower', context)
            rmse_high = _required_number(metrics, 'test_rmse_ci95_upper', context)
            nasa_low = _required_number(metrics, 'test_nasa_ci95_lower', context)
            nasa_high = _required_number(metrics, 'test_nasa_ci95_upper', context)
            if rmse_low > rmse_high or nasa_low > nasa_high:
                raise ValueError(f'Los límites de confianza están invertidos en {context}.')
            records.append({
                'objective': objective,
                'candidate_id': str(candidate_id),
                'hidden_sizes': tuple(int(size) for size in hidden_sizes),
                'parameter_count': parameter_count,
                'rmse': rmse,
                'rmse_low': rmse_low,
                'rmse_high': rmse_high,
                'nasa_score': nasa,
                'nasa': nasa / test_engine_count,
                'nasa_low': nasa_low / test_engine_count,
                'nasa_high': nasa_high / test_engine_count,
                'selected_winner': bool(candidate.get('selected_winner', False)),
            })

    records = [
        record for record in records
        if all(
            (minimum is None or record[metric] >= minimum)
            and (maximum is None or record[metric] <= maximum)
            for metric, (minimum, maximum) in normalized_bounds.items()
        )
    ]
    if not records:
        raise ValueError(
            'Ningún candidato cumple los límites solicitados; amplíe o quite filtros.'
        )

    parameter_logs = np.log10([record['parameter_count'] for record in records])
    log_min = float(parameter_logs.min())
    log_range = float(parameter_logs.max() - log_min)
    bubble_areas = [
        70.0 + 230.0 * ((value - log_min) / log_range if log_range else 0.5)
        for value in parameter_logs
    ]

    bubble_sizes = [np.sqrt(area) for area in bubble_areas]
    hidden_layer_counts = sorted({len(record['hidden_sizes']) for record in records})
    fig = go.Figure()

    for hidden_layer_count in hidden_layer_counts:
        layer_legend_label = (
            f'{hidden_layer_count} capa oculta'
            if hidden_layer_count == 1
            else f'{hidden_layer_count} capas ocultas'
        )
        layer_legend_group = f'hidden-layers-{hidden_layer_count}'
        layer_has_legend_entry = False
        for objective in OBJECTIVES:
            objective_data = [
                record for record in records
                if record['objective'] == objective
                and len(record['hidden_sizes']) == hidden_layer_count
            ]
            if not objective_data:
                continue
            objective_sizes = [
                size for record, size in zip(records, bubble_sizes)
                if record['objective'] == objective
                and len(record['hidden_sizes']) == hidden_layer_count
            ]
            customdata = [
                [
                    record['candidate_id'],
                    str(record['hidden_sizes']),
                    record['parameter_count'],
                    record['rmse_low'],
                    record['rmse_high'],
                    record['nasa_low'],
                    record['nasa_high'],
                    record['nasa_score'],
                    objective,
                    record['selected_winner'],
                ]
                for record in objective_data
            ]
            fig.add_trace(go.Scatter(
                x=[record['rmse'] for record in objective_data],
                y=[record['nasa'] for record in objective_data],
                mode='markers+text',
                text=[f"{objective} {record['hidden_sizes']}" for record in objective_data],
                textposition='top right',
                textfont={'size': 9},
                customdata=customdata,
                name=layer_legend_label,
                legendgroup=layer_legend_group,
                showlegend=not layer_has_legend_entry,
                marker={
                    'size': [
                        13 if record['selected_winner'] else size
                        for record, size in zip(objective_data, objective_sizes)
                    ],
                    'symbol': [
                        'star' if record['selected_winner'] else 'circle'
                        for record in objective_data
                    ],
                    'color': OBJECTIVE_COLORS[objective],
                    'opacity': 0.72,
                    'line': {'color': 'white', 'width': 1},
                },
                error_x={
                    'type': 'data',
                    'symmetric': False,
                    'array': [record['rmse_high'] - record['rmse'] for record in objective_data],
                    'arrayminus': [record['rmse'] - record['rmse_low'] for record in objective_data],
                    'color': OBJECTIVE_COLORS[objective],
                    'thickness': 1,
                    'width': 4,
                },
                error_y={
                    'type': 'data',
                    'symmetric': False,
                    'array': [record['nasa_high'] - record['nasa'] for record in objective_data],
                    'arrayminus': [record['nasa'] - record['nasa_low'] for record in objective_data],
                    'color': OBJECTIVE_COLORS[objective],
                    'thickness': 1,
                    'width': 4,
                },
                hovertemplate=(
                    '<b>%{customdata[8]}</b> · candidato %{customdata[0]}'
                    '<br>Arquitectura: %{customdata[1]}'
                    '<br>Parámetros: %{customdata[2]:,.0f}'
                    '<br>RMSE: %{x:.4f} (IC 95%: %{customdata[3]:.4f}–%{customdata[4]:.4f})'
                    '<br>NASA medio por motor: %{y:.4g} '
                    '(IC 95%: %{customdata[5]:.4g}–%{customdata[6]:.4g})'
                    '<br>NASA sumativo: %{customdata[7]:.4g}'
                    '<extra></extra>'
                ),
            ))
            layer_has_legend_entry = True

    fig.update_layout(
        title=(
            f'Comparación interactiva de candidatos — {dataset_name}'
            '<br><sup>Color: rmse (azul), nasa (naranja), both (verde) · ★ = seleccionado</sup>'
        ),
        xaxis_title='RMSE de test',
        yaxis_title='NASA score medio por motor',
        yaxis_type='linear',
        template='plotly_white',
        width=1200,
        height=760,
        margin={'l': 80, 'r': 300, 't': 100, 'b': 70},
        legend={
            'title': 'Capas ocultas (clic para filtrar)',
            'x': 1.01,
            'y': 1.12,
            'xanchor': 'left',
            'groupclick': 'togglegroup',
        },
        hoverlabel={'namelength': -1},
    )
    #display(HTML(fig.to_html(include_plotlyjs='cdn')))
    return fig
