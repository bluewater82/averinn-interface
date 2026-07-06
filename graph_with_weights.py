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
    Print ONNX operators in order for debugging
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
    # inspect_nodes(graph)
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
    dot.append("    nodesep=1.0;")
    dot.append("    ranksep=10.0;")
    dot.append("")
    dot.append('    node [shape=circle, style=filled, fillcolor="lightgray"];')
    dot.append("")



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

        for i in range(input_count):
            for j in range(output_count):
                src = node_id(src_layer, i)
                dst = node_id(dst_layer, j)
                weight = weights[j, i]

                color = "blue" if weight >= 0 else "orange"
                penwidth = 0.35

                if show_weights:
                    dot.append(
                        f'    {src} -> {dst} '
                        f'[taillabel="{weight:.2f}", '
                        f'fontcolor="{color}", '
                        f'fontsize=8, '
                        f'labeldistance=4.0, '
                        f'labelangle=0, '
                        f'color="{color}", '
                        f'penwidth={penwidth:.2f}];'
                    )
                else:
                    dot.append(
                        f'    {src} -> {dst} '
                        f'[color="{color}", penwidth={penwidth:.2f}];'
                    )

        dot.append("")

    dot.append("}")

    return "\n".join(dot)


def main():
    onnx_path = "ACC_controller_5_20.onnx"

    layers = extract_dense_layers(onnx_path)
    

    print(f"Extracted {len(layers)} dense layers.")

    for i, layer in enumerate(layers):
        print(f"\nLayer {i}")
        print("  weights:", layer["weights"].shape)
        print("  bias:", None if layer["bias"] is None else layer["bias"].shape)
        print("  activation:", layer["activation"])

    dot_text = generate_dot(layers, show_weights=True)

    with open("network.dot", "w", encoding="utf-8") as f:
        f.write(dot_text)

    print("\nWrote network.dot")
    print("Render with:")
    print("dot -Tpng network.dot -o network.png")


if __name__ == "__main__":
    main()