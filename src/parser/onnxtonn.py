"""
Author: Ratan Lal
Date : January 27, 2025
Note: some of the functionality are copied from Stanley Bak
"""
from abc import ABC
from typing import Dict
import onnx
from onnx import numpy_helper
from onnxruntime import InferenceSession
import numpy as np
import numpy.typing as npt

from src.gnn.gnn import GNN
from src.parser.parseruts import ParserUTS
from src.types.boundtype import BoundType
from src.types.datatype import DataType
from src.parser.parser import Parser


class ONNX(Parser, ABC):
    """
    Read neural network input expressed in a special form called onnx
    """

    def __init__(self, strFilePath: str):

        W, b, l = self.readOnnx_to_NN(strFilePath)
        self.__dictNeurons__: Dict[int, int] = self.__getDictNeurons__(l)
        self.__dictWeights__: Dict[int, npt.ArrayLike] = self.__dictWeight__(W)
        self.__dictBiases__: Dict[int, npt.ArrayLike] = self.__dictBias__(b)

    def getNetwork(self) -> GNN:
        """
        Get a neural network as an instance of GNN
        :return: (objGNN: GNN)
        An instance of GNN
        """

        # Return  neural network as an instance of GNN
        return ParserUTS.toStandardGNN(self.__dictNeurons__, self.__dictWeights__, self.__dictBiases__)

    def getDictNeurons(self) -> Dict[int, int]:
        """
        Get dictionary between layer number and its number of neurons
        :return: (dictNeurons: Dict[int,int])
        """
        return self.__dictNeurons__

    def getDictWeights(self, boundType: BoundType) -> Dict[int, npt.ArrayLike]:
        """
        Get dictionary between layer number i and weight matrix where each row represents
        weight array from a neuron at i+1 and all neurons at layer i
        :return: (dictWeights: Dict[int,npt.ArrayLike])
        """
        return self.__dictWeights__

    def getDictBiases(self, boundType: BoundType) -> Dict[int, npt.ArrayLike]:
        """
        Get dictionary between layer number i starting from 2 and an array of biases at layer i
        :return: (dictBiases: Dict[int,npt.ArrayLike])
        """
        return self.__dictBiases__

    def __dictBias__(self, b: npt.ArrayLike) -> Dict[int, npt.ArrayLike]:
        """
        Convert an array of Bias arrays into dictionary between layer number
        and its Bias array
        :param b: npt.ArrayLike
        :type b: npt.ArrayLike
        :return: (dictBias -> Dict[int, npt.ArrayLike])
        dictionary between layer number and its Bias array
        """
        dictBias: Dict[int, npt.ArrayLike] = dict()
        for i in range(len(b)):
            dictBias[i + 2] = self.__toDecimalArray__(b[i])

        return dictBias

    def __dictWeight__(self, W: npt.ArrayLike) -> Dict[int, npt.ArrayLike]:
        """
        Convert an array of weight matrix to dictionary between layer number and
        weight matrix from layer+1 to layer
        :param W: an array of weight matrix
        :type W: npt.ArrayLike
        :return: (dictWeight->Dict[int, npt.ArrayLike])
        dictionary between layer number and weight matrix from layer+1 to layer
        """
        dictWeight: Dict[int, npt.ArrayLike] = {}
        for i in range(len(W)):
            dictWeight[i + 1] = self.__toDecimalArray__(W[i])

        return dictWeight

    def __toDecimalArray__(self, W: npt.ArrayLike) -> npt.ArrayLike:
        """
        Convert a numpy array of weight matrix to numpy array of RealType
        :param W: a numpy array
        :type W: npt.ArrayLike
        :return: (WP -> npt.ArrayLike)
        """
        dim: int = len(W.shape)
        if dim == 1:
            row: int = W.shape[0]
            WP: npt.ArrayLike = np.array([DataType.RealType(0.0) for i in range(row)], dtype=object)
            for i in range(row):
                WP[i] = np.round(DataType.RealType(W[i]), 2)
            return WP
        else:
            row: int = W.shape[0]
            col: int = W.shape[1]
            WP: npt.ArrayLike = np.array([[DataType.RealType(0.0) for i in range(col)]
                                          for j in range(row)], dtype=object)
            for i in range(row):
                for j in range(col):
                    WP[i][j] = np.round(DataType.RealType(W[i][j]),2)
            return WP

    def __getDictNeurons__(self, l: npt.ArrayLike) -> Dict[int, int]:
        """
        Convert an array into dictionary between layer numbers and its neurons
        :param l: an array of neurons
        :type l: npt.ArrayLike
        :return: (dictNeurons -> Dict[int, int])
        dictionary between layer numbers and its neurons
        """
        dictNeurons: Dict[int, int] = {}
        for i in range(len(l)):
            dictNeurons[i + 1] = l[i]

        return dictNeurons

    def tll_wbl(self, w, b, l, op_seq):
        """
        Converts tll benchmark neural network model to standard form by multiplying composing
        affine transformations
        """
        new_w = []
        new_b = []
        new_l = [l[0]]

        def compose_affine(temp_wt, temp_bs, ind):

            if ind in op_seq:
                temp_wt = w[ind]
                temp_bs = b[ind]
                ind = ind + 1
            else:
                temp_wt = np.matmul(w[ind], temp_wt)
                temp_bs = np.matmul(w[ind], temp_bs) + b[ind]
                ind = ind + 1

            if ind in op_seq:
                return temp_wt, temp_bs
            else:
                return compose_affine(temp_wt, temp_bs, ind)

        for j in op_seq[0:-1]:
            new_weight, new_bias = compose_affine(None, None, j)
            new_w.append(new_weight)
            new_b.append(new_bias)
            new_l.append(len(new_bias))

        return new_w, new_b, new_l

    def find_node_with_input(self, graph, input_name):
        'find the unique onnx node with the given input, can return None'

        rv = None

        for n in graph.node:
            for i in n.input:
                if i == input_name:
                    assert rv is None, f"multiple onnx nodes accept network input {input_name}"
                    rv = n

        return rv

    def readOnnx_to_NN(self, path):
        """The main onnx parsing function.
        Most of the code edited from nnenum's(By Stanley Bak) onnx parser.

        Can only correctly parse FFNN's.
        Assumes FFNN's are of standard architecture,
        i.e (Weights->Bias->Relu) except for last layer which is just (weights->bias)

        Can also parse tll benchmark which is (Weight->Bias->weight->Bias....->Relu) by connverting to standard form

        Returns:
             Weights: A list of weight matrices(n-d arrays)
             Bias: A list of biases(1-d arrays)
             Layers: A list of number of nodes in each layer.

        Note: Returned Weights have dimensions in the format (#(output layer nodes), #(input layer nodes))
        The returned weights may not match the shape as stored in the onnx model and displayed in netron.
        """
        onnx_model = onnx.load(path)

        weights = []
        bias = []

        layers = []

        graph = onnx_model.graph

        all_input_names = sum([[str(i) for i in n.input] for n in graph.node], [])

        all_initializer_names = [i.name for i in graph.initializer]
        all_output_names = sum([[str(o) for o in n.output] for n in graph.node], [])

        network_input = None

        for i in all_input_names:
            if i not in all_initializer_names and i not in all_output_names:
                network_input = i

        network_output = graph.output[0].name

        input_map = {i.name: i for i in graph.input}
        init_map = {i.name: i for i in graph.initializer}

        i = input_map[network_input]
        i_shape = [c.dim_value for c in i.type.tensor_type.shape.dim]

        num_inp_nodes = max(i_shape)

        # find the node which takes the input (probably node 0)
        cur_node = self.find_node_with_input(graph, network_input)
        cur_input_name = network_input

        layers.append(num_inp_nodes)

        tll_flag = 0  # Special flag for the tll_verify benchmarks
        op_sequence = [0]  # Ignore for other benchmarks
        op_index = 0  # Ignore for other benchmarks
        last_matmul_relu = None  # Ignore for other benchmarks

        while cur_node is not None:
            op = cur_node.op_type
            if op in ['MatMul', 'Relu', 'Gemm']:
                # Special sub-code just for tll benchmark. Ignore for others
                if last_matmul_relu == op:
                    tll_flag = 1

                last_matmul_relu = op

                if op == 'MatMul':
                    op_index += 1
                else:
                    op_sequence.append(op_index)

            if op in ['Add']:
                init = init_map[cur_node.input[1]]

                data = np.frombuffer(init.raw_data, dtype='<f4')  # little endian float32
                shape = tuple(d for d in init.dims)  # note dims reversed, acasxu has 5, 50 but want 5 cols

                bias.append(data)
                layers.append(shape[0])

            elif op == 'Flatten':
                pass

            elif op == 'MatMul':
                init = init_map[cur_node.input[1]]

                data = np.frombuffer(init.raw_data, dtype='<f4')  # little endian float32
                shape = tuple(d for d in init.dims)
                weights.append(np.transpose(data.reshape(shape)))

            elif op == 'Relu':
                pass

            elif op == 'Gemm':
                assert len(cur_node.input) == 3

                weight_init = init_map[cur_node.input[1]]
                bias_init = init_map[cur_node.input[2]]

                # weight
                data1 = np.frombuffer(weight_init.raw_data, dtype='<f4')  # little endian float32
                shape1 = tuple(d for d in weight_init.dims)  # note dims reversed, acasxu has 5, 50 but want 5 cols
                weights.append(data1.reshape(shape1))

                # bias
                data2 = np.frombuffer(bias_init.raw_data, dtype='<f4')  # little endian float32
                shape2 = tuple(
                    d for d in reversed(bias_init.dims))  # note dims reversed, acasxu has 5, 50 but want 5 cols

                bias.append(data2)
                layers.append(shape2[0])

            else:
                pass

            cur_input_name = cur_node.output[0]

            cur_node = self.find_node_with_input(graph, cur_input_name)

        if tll_flag == 1:
            op_sequence.append(op_index)
            weights, bias, layers = self.tll_wbl(weights, bias, layers, op_sequence)

        '''
        print(layers)
        for w in weights:
            print(w.shape)

        for w in weights:
            print(w[0][0:3])

        print()
        for b in bias:
            print(b[0:3])

        '''

        return weights, bias, layers
