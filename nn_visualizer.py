from collections.abc import Mapping
from math import ceil
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import numpy as np
import networkx as nx


class NeuralNetworkVisualizer:
    def __init__(self, model: Mapping[str, Any]) -> None:
        self.model = model

    def visualize_weights(self) -> None:
        layers = self.model.get("layers")
        if not isinstance(layers, list) or not layers:
            raise ValueError("The model structure must contain a non-empty layers list.")

        model_input = self.model.get("input", {})
        if not isinstance(model_input, Mapping):
            raise ValueError("The model input metadata must be a mapping.")
        input_features = model_input.get("features", [])
        if not isinstance(input_features, (list, tuple)):
            raise ValueError("Input feature names must be provided as a list.")
        matrices = []
        previous_size = model_input.get("size")

        for index, layer in enumerate(layers):
            if not isinstance(layer, Mapping):
                raise ValueError(f"Layer {index} must be a mapping.")

            weights = np.asarray(layer.get("weights"), dtype=np.float64)
            if weights.ndim != 2:
                raise ValueError(f"Layer {index} weights must be a 2D matrix.")
            if not np.isfinite(weights).all():
                raise ValueError(f"Layer {index} weights contain non-finite values.")

            output_size, input_size = weights.shape
            if input_size < 1 or output_size < 1:
                raise ValueError(f"Layer {index} must have non-empty dimensions.")
            if (
                layer.get("output_size") != output_size
                or layer.get("input_size") != input_size
            ):
                raise ValueError(
                    f"Layer {index} declared dimensions do not match its weights."
                )
            if previous_size is not None and input_size != previous_size:
                raise ValueError(
                    f"Layer {index} expects {input_size} inputs, "
                    f"but the previous layer has {previous_size} outputs."
                )

            bias = np.asarray(layer.get("bias"), dtype=np.float64)
            if bias.shape != (output_size,):
                raise ValueError(
                    f"Layer {index} bias must have shape ({output_size},)."
                )
            if not np.isfinite(bias).all():
                raise ValueError(f"Layer {index} bias contains non-finite values.")

            matrices.append(weights)
            previous_size = output_size

        if (
            model_input.get("size") is not None
            and int(model_input["size"]) != matrices[0].shape[1]
        ):
            raise ValueError(
                "The declared model input size does not match the first layer."
            )
        if input_features and len(input_features) != matrices[0].shape[1]:
            raise ValueError(
                "The number of input feature names does not match the input size."
            )

        max_abs_weight = max(float(np.max(np.abs(weights))) for weights in matrices)
        color_limit = max_abs_weight if max_abs_weight > 0.0 else 1.0
        norm = Normalize(vmin=-color_limit, vmax=color_limit)
        cmap = LinearSegmentedColormap.from_list(
            "weight_polarity",
            [
                (0.0, "#8b0000"),
                (0.45, "#fff2a6"),
                (0.5, "#ffffff"),
                (0.55, "#d9f2d9"),
                (1.0, "#001f5b"),
            ],
        )
        cmap.set_bad("#ffffff")

        for index, (layer, weights) in enumerate(zip(layers, matrices)):
            input_size = weights.shape[1]
            output_size = weights.shape[0]
            input_layer_name = (
                "input"
                if index == 0
                else layers[index - 1].get("name", f"layer_{index}")
            )
            output_layer_name = layer.get("name", f"layer_{index + 1}")

            width = max(8.0, min(16.0, 4.0 + 0.18 * input_size))
            height = max(3.0, min(12.0, 2.0 + 0.18 * output_size))
            figure, axis = plt.subplots(figsize=(width, height))
            image = axis.imshow(
                np.ma.masked_equal(weights, 0.0),
                cmap=cmap,
                norm=norm,
                aspect="auto",
                interpolation="nearest",
            )

            x_step = max(1, ceil(input_size / 20))
            y_step = max(1, ceil(output_size / 20))
            x_indices = np.arange(0, input_size, x_step)
            y_indices = np.arange(0, output_size, y_step)

            if index == 0 and input_features:
                x_labels = [str(input_features[i]) for i in x_indices]
            else:
                x_labels = [str(i) for i in x_indices]

            axis.set_xticks(x_indices, labels=x_labels, rotation=90)
            axis.set_yticks(y_indices, labels=[str(i) for i in y_indices])
            axis.set_xlabel(f"Input neurons: {input_layer_name}")
            axis.set_ylabel(f"Output neurons: {output_layer_name}")
            axis.set_title(
                f"Weights: {input_layer_name} to {output_layer_name} "
                f"({input_size} x {output_size})"
            )
            figure.colorbar(image, ax=axis, label="Weight value")
            figure.tight_layout()

        plt.show()

    def visualize_graph(self) -> nx.DiGraph:
        layers = self.model.get("layers")
        if not isinstance(layers, list) or not layers:
            raise ValueError("The model structure must contain a non-empty layers list.")

        model_input = self.model.get("input", {})
        if not isinstance(model_input, Mapping):
            raise ValueError("The model input metadata must be a mapping.")

        input_size = model_input.get("size")
        input_features = model_input.get("features", [])
        if not isinstance(input_features, (list, tuple)):
            raise ValueError("Input feature names must be provided as a list.")
        if input_size is None:
            input_size = layers[0].get("input_size")
        if not isinstance(input_size, int) or input_size < 1:
            raise ValueError("The model must declare a positive input size.")
        if input_features and len(input_features) != input_size:
            raise ValueError(
                "The number of input feature names does not match the input size."
            )

        graph = nx.DiGraph(
            dataset=self.model.get("dataset"),
            candidate_id=self.model.get("candidate_id"),
            representative_seed=self.model.get("seed"),
        )
        layer_nodes = []
        previous_size = input_size

        for neuron_index in range(input_size):
            node = f"L0_N{neuron_index}"
            attributes = {
                "layer_index": 0,
                "layer_name": "input",
                "neuron_index": neuron_index,
                "bias": None,
                "activation": "input",
            }
            if input_features:
                attributes["feature"] = str(input_features[neuron_index])
            graph.add_node(node, **attributes)
        layer_nodes.append([f"L0_N{index}" for index in range(input_size)])

        matrices = []
        for layer_index, layer in enumerate(layers, start=1):
            if not isinstance(layer, Mapping):
                raise ValueError(f"Layer {layer_index - 1} must be a mapping.")
            if layer.get("type", "dense") != "dense":
                raise ValueError(
                    f"Layer {layer_index - 1} has unsupported type "
                    f"{layer.get('type')!r}; only dense layers are supported."
                )

            weights = np.asarray(layer.get("weights"), dtype=np.float64)
            if weights.ndim != 2:
                raise ValueError(
                    f"Layer {layer_index - 1} weights must be a 2D matrix."
                )
            if not np.isfinite(weights).all():
                raise ValueError(
                    f"Layer {layer_index - 1} weights contain non-finite values."
                )

            output_size, layer_input_size = weights.shape
            if layer_input_size != previous_size:
                raise ValueError(
                    f"Layer {layer_index - 1} expects {layer_input_size} inputs, "
                    f"but the preceding layer has {previous_size} outputs."
                )
            if (
                layer.get("input_size") != layer_input_size
                or layer.get("output_size") != output_size
            ):
                raise ValueError(
                    f"Layer {layer_index - 1} declared dimensions do not match "
                    "its weight matrix."
                )
            if layer.get("weights_layout", "[output_neuron][input_neuron]") != (
                "[output_neuron][input_neuron]"
            ):
                raise ValueError(
                    f"Layer {layer_index - 1} uses an unsupported weight layout."
                )

            biases = np.asarray(layer.get("bias"), dtype=np.float64)
            if biases.shape != (output_size,):
                raise ValueError(
                    f"Layer {layer_index - 1} bias must have shape "
                    f"({output_size},)."
                )
            if not np.isfinite(biases).all():
                raise ValueError(
                    f"Layer {layer_index - 1} bias contains non-finite values."
                )

            layer_name = str(layer.get("name", f"layer_{layer_index}"))
            activation = str(layer.get("activation", "linear"))
            current_nodes = []
            for neuron_index, bias in enumerate(biases):
                node = f"L{layer_index}_N{neuron_index}"
                graph.add_node(
                    node,
                    layer_index=layer_index,
                    layer_name=layer_name,
                    neuron_index=neuron_index,
                    bias=float(bias),
                    activation=activation,
                )
                current_nodes.append(node)

            matrices.append(weights)
            layer_nodes.append(current_nodes)
            previous_size = output_size

        for layer_index, weights in enumerate(matrices):
            source_nodes = layer_nodes[layer_index]
            target_nodes = layer_nodes[layer_index + 1]
            for target_index, target_node in enumerate(target_nodes):
                for source_index, source_node in enumerate(source_nodes):
                    graph.add_edge(
                        source_node,
                        target_node,
                        weight=float(weights[target_index, source_index]),
                    )

        positions = {}
        for layer_index, nodes in enumerate(layer_nodes):
            vertical_positions = (
                np.linspace(0.5, -0.5, num=len(nodes))
                if len(nodes) > 1
                else np.array([0.0])
            )
            positions.update(
                {
                    node: (layer_index, float(vertical_position))
                    for node, vertical_position in zip(nodes, vertical_positions)
                }
            )

        edge_weights = np.asarray(
            [attributes["weight"] for _, _, attributes in graph.edges(data=True)],
            dtype=np.float64,
        )
        max_abs_weight = float(np.max(np.abs(edge_weights)))
        color_limit = max_abs_weight if max_abs_weight > 0.0 else 1.0
        norm = Normalize(vmin=-color_limit, vmax=color_limit)
        cmap = LinearSegmentedColormap.from_list(
            "weight_polarity",
            [
                (0.0, "#8b0000"),
                (0.45, "#fff2a6"),
                (0.5, "#ffffff"),
                (0.55, "#d9f2d9"),
                (1.0, "#001f5b"),
            ],
        )

        figure, axis = plt.subplots(
            figsize=(max(10.0, 3.5 * len(layer_nodes)), 8.0)
        )
        node_colors = ["#a6cee3", "#b2df8a", "#cab2d6", "#fdbf6f"]
        for layer_index, nodes in enumerate(layer_nodes):
            nx.draw_networkx_nodes(
                graph,
                positions,
                nodelist=nodes,
                node_size=48,
                node_color=node_colors[layer_index % len(node_colors)],
                edgecolors="#303030",
                linewidths=0.35,
                ax=axis,
            )

        if max_abs_weight > 0.0:
            edge_list = list(graph.edges(data=True))
            edge_colors = [
                cmap(norm(attributes["weight"]))
                for _, _, attributes in edge_list
            ]
            edge_widths = [
                0.2 + 1.0 * abs(attributes["weight"]) / max_abs_weight
                for _, _, attributes in edge_list
            ]
            nx.draw_networkx_edges(
                graph,
                positions,
                edgelist=[(source, target) for source, target, _ in edge_list],
                edge_color=edge_colors,
                width=edge_widths,
                alpha=0.45,
                arrows=False,
                ax=axis,
            )

        for layer_index, nodes in enumerate(layer_nodes):
            if layer_index == 0:
                layer_name = "input"
                detail = f"{len(nodes)} features"
            else:
                layer = layers[layer_index - 1]
                layer_name = str(layer.get("name", f"layer_{layer_index}"))
                detail = (
                    f"{len(nodes)} neurons, "
                    f"{layer.get('activation', 'linear')}"
                )
            axis.text(
                layer_index,
                0.57,
                f"{layer_name}\n{detail}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        axis.set_title(
            "Neural network graph"
            + (
                f" — {self.model['candidate_id']}"
                if self.model.get("candidate_id")
                else ""
            )
        )
        axis.set_axis_off()
        figure.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            ax=axis,
            label="Connection weight",
            fraction=0.025,
            pad=0.02,
        )
        axis.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    color="w",
                    markerfacecolor=node_colors[index % len(node_colors)],
                    markeredgecolor="#303030",
                    label=(
                        "input"
                        if index == 0
                        else str(layers[index - 1].get("name", f"layer_{index}"))
                    ),
                )
                for index in range(len(layer_nodes))
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.04),
            ncol=min(len(layer_nodes), 5),
            frameon=False,
        )
        figure.tight_layout()
        plt.show()
        return graph
