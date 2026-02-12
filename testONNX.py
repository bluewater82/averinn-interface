from typing import Dict

import onnx
import numpy as np
from onnx import numpy_helper
import onnx
import numpy as np
from onnx import numpy_helper, AttributeProto
import numpy.typing as npt
import numpy as np
def get_attr_int(node, name, default=0):
    for a in node.attribute:
        if a.name == name and a.type == AttributeProto.INT:
            return int(a.i)
    return default

def extract_layerwise_Wb(onnx_path: str):
    model = onnx.load(onnx_path)
    g = model.graph
    init = {t.name: numpy_helper.to_array(t) for t in g.initializer}

    layers = []
    for node in g.node:
        if node.op_type == "Gemm":
            if len(node.input) < 2 or node.input[1] not in init:
                continue
            W = init[node.input[1]]
            b = init[node.input[2]] if len(node.input) > 2 and node.input[2] in init else None

            # Convert to (out, in) for verification-friendly form
            transB = get_attr_int(node, "transB", 0)
            if W.ndim == 2 and transB == 0:
                W = W.T

            layers.append((node.name or "Gemm", W, b))

        elif node.op_type == "Conv":
            if len(node.input) < 2 or node.input[1] not in init:
                continue
            W = init[node.input[1]]
            b = init[node.input[2]] if len(node.input) > 2 and node.input[2] in init else None
            layers.append((node.name or "Conv", W, b))

    return layers

layers = extract_layerwise_Wb("./resources/nncs-benchmarks/unicycle/controller.onnx")
f= open("temp.txt", "w")

dictWeight: Dict[int, npt] = dict()
dictBias: Dict[int, npt] = dict()
for i, (lname, W, b) in enumerate(layers):
    print(f"\nLayer {i}: {lname}")
    print("W shape:", W.shape)
    tempW = []
    if i == 0:
        for j in range(len(W)):
            tempW.append(W[j][0][0])
    else:
        for j in range(len(W)):
            temp = []
            for k in range(len(W[j])):
                temp.append(W[j][k][0][0])
            tempW.append(temp)
    dictWeight[i+1] = tempW
    dictBias[i+2] = b

f.write(str(dictBias))
f.close()