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
from onnx import numpy_helper, AttributeProto
from src.gnn.gnn import GNN
from src.parser.parseruts import ParserUTS
from src.types.boundtype import BoundType
from src.types.datatype import DataType
from src.parser.parser import Parser


class NEWONNX(Parser, ABC):
    """
    Read neural network input expressed in a special form called onnx
    """

    def __init__(self, strFilePath: str):
        self.__dictWeights__: Dict[int, npt.ArrayLike] = self.__extractDictWeights__(strFilePath)
        self.__dictBiases__: Dict[int, npt.ArrayLike] = self.__extractDictBiases__(strFilePath)
        self.__dictNeurons__: Dict[int, int] = self.__extractDictNeurons__()

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

    def __get_attr_int__(self, node, name, default=0):
        for a in node.attribute:
            if a.name == name and a.type == AttributeProto.INT:
                return int(a.i)
        return default

    def __extract_layerwise_Wb__(self, onnx_path: str):
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
                transB = self.__get_attr_int__(node, "transB", 0)
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

    def __extractDictWeights__(self, srtPath: str) -> Dict[int, npt.ArrayLike]:
            """
            :param strPath: path of the network file
            :type strPath: str
            """
            dictWeight: Dict[int, npt] = dict()
            layers = self.__extract_layerwise_Wb__(srtPath)
            for i, (lname, W, b) in enumerate(layers):
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
                dictWeight[i + 1] = np.array(tempW, dtype=np.float64)

            return dictWeight

    def __extractDictBiases__(self, srtPath: str) -> Dict[int, npt.ArrayLike]:
        """
        :param strPath: path of the network file
        :type strPath: str
        """
        dictBias: Dict[int, npt] = dict()
        layers = self.__extract_layerwise_Wb__(srtPath)
        for i, (lname, W, b) in enumerate(layers):
            dictBias[i+2] = np.array(b, dtype=np.float64)

        return dictBias

    def __extractDictNeurons__(self) -> Dict[int, int]:
        """
        Return dictionary of neurons at each layers
        """
        dictNeurons: Dict[int, int] = dict()
        for key in self.__dictWeights__.keys():
            row, col = self.__dictWeights__[key].shape
            dictNeurons[key] = col
            dictNeurons[key + 1] = row

        return dictNeurons

'''
objONNX = NEWONNX("./../../resources/nncs-benchmarks/unicycle/controller.onnx")
print(objONNX.getDictNeurons())
'''