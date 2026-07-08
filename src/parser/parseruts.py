"""
Author: Ratan Lal
Date : November 4, 2024
"""
import numpy as np

from src.gnn.number import Number
from src.gnn.ireal import IReal
from src.gnn.gnn import GNN
from typing import Dict, Tuple
import numpy.typing as npt
from src.gnn.layer import Layer
from src.gnn.node import Node
from src.types.activationtype import ActivationType
from src.gnn.connection import Connection
from src.gnn.edge import Edge
from src.gnn.real import Real
from src.types.datatype import DataType
from src.types.networktype import NetworkType


class ParserUTS:
    """
    All common functions related to PARSER modules
    """

    @staticmethod
    def toRobustGNN(objGNN: GNN, robustValue: np.float64) -> GNN:
        """
        Change weights and biases
        :param objGNN: A general neural network
        :type objGNN: GNN
        :param robustValue: Robust value
        :type robustValue: float
        :return: (objGNNRobust-> GNN)
        """
        dictNeurons: Dict[int, int] = objGNN.getDictNumNeurons()
        dictWeightLow: Dict[int, npt.ArrayLike] = {}
        dictWeightHigh: Dict[int, npt.ArrayLike] = {}
        dictBiasLow: Dict[int, npt.ArrayLike] = {}
        dictBiasHigh: Dict[int, npt.ArrayLike] = {}

        numOfLayers: int = objGNN.getNumOfLayers()
        for i in range(1, numOfLayers, 1):
            arrayWeightLow = objGNN.getLowerMatrixByLayer(i)
            arrayWeightHigh = objGNN.getUpperMatrixByLayer(i)
            arrayBiasLow = objGNN.getLowerBiasByLayer(i + 1)
            arrayBiasHigh = objGNN.getUpperBiasByLayer(i + 1)
            row, col = arrayWeightLow.shape
            for j in range(row):
                for k in range(col):
                    arrayWeightLow[j][k] = arrayWeightLow[j][k] - robustValue
                    arrayWeightHigh[j][k] = arrayWeightHigh[j][k] + robustValue
                arrayBiasLow[j] = arrayBiasLow[j] - robustValue
                arrayBiasHigh[j] = arrayBiasHigh[j] + robustValue

            dictWeightLow[i] = arrayWeightLow
            dictWeightHigh[i] = arrayWeightHigh
            dictBiasLow[i + 1] = arrayBiasLow
            dictBiasHigh[i + 1] = arrayBiasHigh

        objRobustGNN: GNN = ParserUTS.toIntervalGNN(dictNeurons, dictWeightLow, dictWeightHigh, dictBiasLow,
                                                    dictBiasHigh)

        return objRobustGNN

    @staticmethod
    def toStandardGNN(dictNeurons: Dict[int, int], dictWeight: Dict[int, npt.ArrayLike],
                      dictBias: Dict[int, npt.ArrayLike]) -> GNN:
        """
        Use weights and biases to create NeuralNetwork instance
        :param dictNeurons: dictionary between layer number and
        number of neurons at the layer number
        :type dictNeurons: Dict[int, int]
        :param dictWeight: dictionary between layer numbers and weight matrices
        :type dictWeight: Dict[int, npt.ArrayLike]
        :param dictBias: dictionary between layer numbers and bias arrays
        :type dictBias: Dict[int, npt.ArrayLike]
        :return: (objGNN -> GNN)
        an instance of GNN class
        """
        # Create dictionary between layer numbers and Layer instances
        dictLayers: Dict[int, Layer] = ParserUTS.__createStandardDictLayers(dictNeurons,
                                                                            dictBias)
        # Create dictionary between layer numbers and Connection instances
        dictConnections: Dict[int, Connection] = \
            ParserUTS.__createStandardDictConnections(dictLayers, dictWeight)
        # Number of layers
        intNumOfLayers: int = len(dictLayers)
        # Create an instance of NeuralNetwork
        objGNN: GNN = GNN(dictLayers, dictConnections, intNumOfLayers, dictNeurons, NetworkType.STANDARD)
        # Return NeuralNetwork instance
        return objGNN

    @staticmethod
    def toIntervalGNN(dictNeurons: Dict[int, int], dictWeightLow: Dict[int, npt.ArrayLike],
                      dictWeightHigh: Dict[int, npt.ArrayLike], dictBiasLow: Dict[int, npt.ArrayLike],
                      dictBiasHigh: Dict[int, npt.ArrayLike]) -> GNN:
        """
        Use weights and biases to create NeuralNetwork instance
        :param dictNeurons: dictionary between layer number and
        number of neurons at the layer number
        :type dictNeurons: Dict[int, int]
        :param dictWeightLow: dictionary between layer numbers and lower weight matrices
        :type dictWeightLow: Dict[int, npt.ArrayLike]
        :param dictWeightHigh: dictionary between layer numbers and lower weight matrices
        :type dictWeightLow: Dict[int, npt.ArrayLike]
        :param dictBiasLow: dictionary between layer numbers and lower bias arrays
        :type dictBiasLow: Dict[int, npt.ArrayLike]
        :param dictBiasHigh: dictionary between layer numbers and upper bias arrays
        :type dictBiasHigh: Dict[int, npt.ArrayLike]
        :return: (objGNN -> GNN)
        an instance of GNN class
        """
        # Create dictionary between layer numbers and Layer instances
        dictLayers: Dict[int, Layer] = ParserUTS.__createIntervalDictLayers(dictNeurons,
                                                                            dictBiasLow, dictBiasHigh)
        # Create dictionary between layer numbers and Connection instances
        dictConnections: Dict[int, Connection] = \
            ParserUTS.__createIntervalDictConnections(dictLayers, dictWeightLow, dictWeightHigh)
        # Number of layers
        intNumOfLayers: int = len(dictLayers)
        # Create an instance of NeuralNetwork
        objGNN: GNN = GNN(dictLayers, dictConnections, intNumOfLayers, dictNeurons, NetworkType.INTERVAL)
        # Return NeuralNetwork instance
        return objGNN

    @staticmethod
    def __createStandardDictLayers(dictNeurons: Dict[int, int], dictBias: Dict[int, npt.ArrayLike]) \
            -> Dict[int, Layer]:
        """
        Create dictionary between layer number and Layer instances
        to capture neurons' information
        :param dictNeurons: mapping between layer number and number of neurons
        :type dictNeurons: Dict[int, int]
        :param dictBias: mapping between layer number and bias arrays
        :type dictBias: Dict[int, npt.ArrayLike]
        :return: (dictLayer -> Dict[int, Layer])
         mapping of layer number and Layer instances
        """
        # Dictionary between layer number and Layer instances
        dictLayers: Dict[int, Layer] = {}
        intNumOfLayers: int = len(dictNeurons)
        for i in range(1, intNumOfLayers + 1, 1):
            intNumOfNeuronsAti: int = dictNeurons[i]
            # Create a dictionary between neurons' ids and Node instances for layer i
            dictNodes: Dict[int, Node] = {}
            for j in range(intNumOfNeuronsAti):
                # Create an instance of Node
                objNode: Node = None
                # Input neurons do not have any activation function
                if i == 1:
                    bias: Number = Real(DataType.RealType(0.0))
                    objNode: Node = Node(ActivationType.UNKNOWN, bias, 1, j + 1)
                else:
                    bias: Number = Real(dictBias[i][j])
                    objNode: Node = Node(ActivationType.RELU, bias, 1, j + 1)
                dictNodes[j + 1] = objNode
            # Create Layer instance for layer i and store them in dictLayers
            dictLayers[i] = Layer(dictNodes)

        # Return dictionary between layer number and Layer instances
        return dictLayers
    @staticmethod
    def __createStandardDictConnections(dictLayers: Dict[int, Layer],
                                        dictWeight: Dict[int, npt.ArrayLike]) -> Dict[int, Connection]:
        """
        Create dictionary between layer numbers and Connection instances
        :param dictLayers: mapping between layer number and Layer instances
        :type dictLayers: Dict[int, Layer]
        :param dictWeight: mapping between layer number and lower weight matrices
        :type dictWeight: Dict[int, npt.ArrayLike]
        :return: (dictConnections -> Dict[int, Connection])
        dictionary between layer number and Connection instances
        """
        # Create dictionary between layer numbers and Connection instances
        dictConnections: Dict[int, Connection] = {}
        for i in range(1, len(dictLayers), 1):
            # Create dictionary between layer numbers and Edge instances
            dictEdges: Dict[Tuple[int, int], Edge] = {}
            for sourceNode in dictLayers[i].dictNodes.values():
                for targetNode in dictLayers[i + 1].dictNodes.values():
                    # sourceINode id and targetINode id
                    sid: int = sourceNode.intId - 1
                    tid: int = targetNode.intId - 1
                    weight: Number = Real(float(dictWeight[i][tid][sid]))
                    objEdge: Edge = Edge(sourceNode, targetNode, weight)
                    dictEdges[(sid + 1, tid + 1)] = objEdge
            # Create Connection instance between layer i and layer i+1 and store them in
            # dictConnections
            dictConnections[i] = Connection(dictEdges)
        # Return dictionary between layer numbers and Connection instances
        return dictConnections

    @staticmethod
    def __createIntervalDictLayers(dictNeurons: Dict[int, int], dictBiasLow: Dict[int, npt.ArrayLike],
                                   dictBiasHigh: Dict[int, npt.ArrayLike]) -> Dict[int, Layer]:
        """
        Create dictionary between layer number and Layer instances
        to capture neurons' information
        :param dictNeurons: mapping between layer number and number of neurons
        :type dictNeurons: Dict[int, int]
        :param dictBiasLow: mapping between layer number and lower bias arrays
        :type dictBiasLow: Dict[int, npt.ArrayLike]
        :param dictBiasHigh: mapping between layer number and upper bias arrays
        :type dictBiasHigh: Dict[int, npt.ArrayLike]
        :return: (dictLayer -> Dict[int, Layer])
         mapping of layer number and Layer instances
        """
        # Dictionary between layer number and Layer instances
        dictLayers: Dict[int, Layer] = {}
        intNumOfLayers: int = len(dictNeurons)
        for i in range(1, intNumOfLayers + 1, 1):
            intNumOfNeuronsAti: int = dictNeurons[i]
            # Create a dictionary between neurons' ids and Node instances for layer i
            dictNodes: Dict[int, Node] = {}
            for j in range(intNumOfNeuronsAti):
                # Create an instance of Node
                objNode: Node = None
                # Input neurons do not have any activation function
                if i == 1:
                    bias: Number = Real(DataType.RealType(0.0))
                    objNode: Node = Node(ActivationType.UNKNOWN, bias, 1, j + 1)
                else:
                    bias: Number = IReal(dictBiasLow[i][j], dictBiasHigh[i][j])
                    objNode: Node = Node(ActivationType.RELU, bias, 1, j + 1)
                dictNodes[j + 1] = objNode
            # Create Layer instance for layer i and store them in dictLayers
            dictLayers[i] = Layer(dictNodes)

        # Return dictionary between layer number and Layer instances
        return dictLayers

    @staticmethod
    def __createIntervalDictConnections(dictLayers: Dict[int, Layer], dictWeightLow: Dict[int, npt.ArrayLike],
                                        dictWeightHigh: Dict[int, npt.ArrayLike]) -> Dict[int, Connection]:
        """
        Create dictionary between layer numbers and Connection instances
        :param dictLayers: mapping between layer number and Layer instances
        :type dictLayers: Dict[int, Layer]
        :param dictWeightLow: mapping between layer number and lower weight matrices
        :type dictWeightLow: Dict[int, npt.ArrayLike]
        :param dictWeightHigh: mapping between layer number and upper weight matrices
        :type dictWeightHigh: Dict[int, npt.ArrayLike]
        :return: (dictConnections -> Dict[int, Connection])
        dictionary between layer number and Connection instances
        """
        # Create dictionary between layer numbers and Connection instances
        dictConnections: Dict[int, Connection] = {}
        for i in range(1, len(dictLayers), 1):
            # Create dictionary between layer numbers and Edge instances
            dictEdges: Dict[Tuple[int, int], Edge] = {}
            for sourceNode in dictLayers[i].dictNodes.values():
                for targetNode in dictLayers[i + 1].dictNodes.values():
                    # sourceINode id and targetINode id
                    sid: int = sourceNode.intId - 1
                    tid: int = targetNode.intId - 1
                    weight: Number = IReal(DataType.RealType(dictWeightLow[i][tid][sid]),
                                           DataType.RealType(dictWeightHigh[i][tid][sid]))
                    objEdge: Edge = Edge(sourceNode, targetNode, weight)
                    dictEdges[(sid + 1, tid + 1)] = objEdge
            # Create Connection instance between layer i and layer i+1 and store them in
            # dictConnections
            dictConnections[i] = Connection(dictEdges)
        # Return dictionary between layer numbers and Connection instances
        return dictConnections
