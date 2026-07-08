import onnx
from onnx import numpy_helper

def load_initializers(graph):
    """
    Convert ONNX initializer tensors into numpy arrays
    """
    initializers = {}

    for init in graph.initializer:
        initializers[init.name] = numpy_helper.to_array(init)
    
    return initializers


def inspect_nodes(graph):
    """
    Print ONNX operators in order
    """
    for i, node in enumerate(graph.node):
        print(f"nNode {i}")
        print(" name:   ", node.name)
        print(" op_type:", node.op_type)
        print(" inputs: ", list(node.input))
        print(" outputs:", list(node.output))


def extract_dense_layers(onnx_path):
    """
    Extract a simple network representation
    """
    model = onnx.load(onnx_path)
    graph = model.graph

    initializers = load_initializers(graph)

    layers = []

    for node in graph.node:
        if node.op_type == "Gemm":
            input_name = node.input[0]
            weight_name = node.input[1]
            bias_name = node.input[2] if len(node.input) > 2 else None
            output_name = node.output[0]

            weights = initializers.get(weight_name)
            bias = initializers.get(bias_name) if bias_name else None

            layer = {
                "type": "dense",
                "onnx_node_name": node.name,
                "input_name": input_name,
                "weight_name": weight_name,
                "output_name": output_name,
                "bias_name": bias_name,
                "weights": weights,
                "bias": bias,
                "activation": None,
            }

            layers.append(layer)

        elif node.op_type in {"Relu", "Sigmoid", "Tanh"}:
            if layers:
                layers[-1]["activation"] = node.op_type

    return layers

def subscript(n):
    """
    Convert small integers to Unicode subscripts.
    """
    table = str.maketrans("0123456789-", "₀₁₂₃₄₅₆₇₈₉₋")
    return str(n).translate(table)


def node_id(layer_index, neuron_index):
    return f"L{layer_index}_N{neuron_index}"


def generate_dot(layers, show_weights=True):
    """
    Generate DOT for a simple fully-connected feed-forward network.
    """
    dot = []

    dot.append("digraph NeuralNet {")
    dot.append("    rankdir=LR;")
    dot.append("    splines=line;")
    dot.append("    nodesep=1;")
    dot.append("    ranksep=3;")
    dot.append("")
    dot.append('    node [shape=circle, style=filled, fillcolor="lightgray"];')
    dot.append("")

    # Infer layer sizes.
    #
    # For first layer weights:
    # Depending on exporter, Gemm weight shape may be:
    #   (input_size, output_size)
    # or
    #   (output_size, input_size)
    #
    # This starter assumes weights.shape == (input_size, output_size).

    input_size = layers[0]["weights"].shape[1]

    layer_sizes = [input_size]

    for layer in layers:
        weights = layer["weights"]
        output_size = layer["weights"].shape[0]
        layer_sizes.append(output_size)

    # Create clustered layers.
    for layer_index, size in enumerate(layer_sizes):
        if layer_index == 0:
            label = "Input Layer"
            prefix = "x"
        elif layer_index == len(layer_sizes) - 1:
            label = "Output Layer"
            prefix = "y"
        else:
            activation = layers[layer_index - 1]["activation"]
            label = f"Hidden Layer {layer_index}"
            if activation:
                label += f"\\n{activation}"
            prefix = "h"

        dot.append(f"    subgraph cluster_layer_{layer_index} {{")
        dot.append(f'        label="{label}";')
        dot.append("        style=rounded;")

        for neuron_index in range(size):
            nid = node_id(layer_index, neuron_index)

            if layer_index == 0:
                display = f"x{subscript(neuron_index + 1)}"
            elif layer_index == len(layer_sizes) - 1:
                display = f"y{subscript(neuron_index + 1)}"
            else:
                display = f"h{subscript(layer_index)},{subscript(neuron_index + 1)}"

            # Bias belongs to non-input neurons.
            if layer_index > 0:
                bias = layers[layer_index - 1]["bias"]
                if bias is not None and neuron_index < len(bias):
                    display += f"\\nb={bias[neuron_index]:.2f}"

            dot.append(f'        {nid} [label="{display}"];')

        dot.append("    }")
        dot.append("")

    # Create weighted edges.
    for layer_index, layer in enumerate(layers):
        weights = layer["weights"]

        src_layer = layer_index
        dst_layer = layer_index + 1

        output_count = weights.shape[0]
        input_count = weights.shape[1]

        max_abs_weight = max(abs(weights).flatten())

        for i in range(input_count):
            for j in range(output_count):
                src = node_id(src_layer, i)
                dst = node_id(dst_layer, j)
                weight = weights[j, i]

                raw_strength = abs(weight) / max_abs_weight if max_abs_weight != 0 else 0

                strength = raw_strength ** 3

                alpha = int(5 + 250 * strength)

                if weight >= 0:
                    color = f"#0000FF{alpha:02X}"
                else:
                    color = f"#FFA500{alpha:02X}"

                dot.append(
                    f'     {src} -> {dst} '
                    f'[color="{color}"]'
                )

        dot.append("")

    add_edge_opacity_legend(dot)
    dot.append("}")

    return "\n".join(dot)


def add_edge_opacity_legend(dot):
    dot.append("    subgraph cluster_legend {")
    dot.append('        label="Weight Magnitude";')
    dot.append("        style=rounded;")
    dot.append("")

    # Positive weights (blue)
    dot.append('        pos_label [label="Positive", shape=plaintext];')
    dot.append('        pos_w [label="Weak", shape=plaintext];')
    dot.append('        pos_m [label="Medium", shape=plaintext];')
    dot.append('        pos_s [label="Strong", shape=plaintext];')
    dot.append('        pos_end [label="", shape=plaintext, width=0.01];')

    dot.append('        pos_w -> pos_m [color="#0000FF33"];')
    dot.append('        pos_m -> pos_s [color="#0000FF99"];')
    dot.append('        pos_s -> pos_end [color="#0000FFFF"];')

    dot.append("")

    # Negative weights (orange)
    dot.append('        neg_label [label="Negative", shape=plaintext];')
    dot.append('        neg_w [label="Weak", shape=plaintext];')
    dot.append('        neg_m [label="Medium", shape=plaintext];')
    dot.append('        neg_s [label="Strong", shape=plaintext];')
    dot.append('        neg_end [label="", shape=plaintext, width=0.01];')

    dot.append('        neg_w -> neg_m [color="#FFA50033"];')
    dot.append('        neg_m -> neg_s [color="#FFA50099"];')
    dot.append('        neg_s -> neg_end [color="#FFA500FF"];')

    dot.append("")

    # Force rows
    dot.append("        { rank=same; pos_label; pos_w; pos_m; pos_s; pos_end; }")
    dot.append("        { rank=same; neg_label; neg_w; neg_m; neg_s; neg_end; }")

    dot.append("    }")
    dot.append("")



def main():
    onnx_path = "ACC_controller_5_20.onnx"

    layers = extract_dense_layers(onnx_path)

    print(f"Extracted {len(layers)} dense layers.")

    for i, layer in enumerate(layers):
        print(f"\nLayer {i}")
        print("  weights:", layer["weights"].shape)
        print("  bias:", None if layer["bias"] is None else layer["bias"].shape)
        print("  activation:", layer["activation"])

    dot_text = generate_dot(layers, show_weights=False)

    with open("network.dot", "w", encoding="utf-8") as f:
        f.write(dot_text)

    print("\nWrote network.dot")
    print("Render with:")
    print("dot -Tpng network.dot -o network.png")


if __name__ == "__main__":
    main()
