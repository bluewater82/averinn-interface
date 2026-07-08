"""
Author: Ratan Lal
Date : November 4, 2024
"""
import copy
import numpy as np
from src.activation.relu import Relu
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
from src.types.lastrelu import LastRelu
from src.types.networktype import NetworkType


class GNNUTS:
    """
    All common functions related to PARSER modules
    """

    @staticmethod
    def getM(objGNN: GNN, arrayInput: npt.ArrayLike, lastRelu: LastRelu) -> Dict[int, Dict[int, DataType.RealType]]:
        """
        Compute upper bound for a standard neural network
        :param objGNN: an instance of GNN
        :type objGNN: GNN
        :param arrayInput: a one dimensional array
        :type arrayInput: npt.ArrayLike
        :param lastRelu: an instance of LastRelu
        :type lastRelu: LastRelu
        :return: (dictM: Dict[int, Dict[int, DataType.RealType]])
        dictionary of dictionaries between node index and its upper bound
        """
        dictM: Dict[int, Dict[int, DataType.RealType]] = {}
        intNumLayers: int = objGNN.getNumOfLayers()
        dictSourceM: Dict[int, DataType.RealType] = dict()
        for i in range(len(arrayInput)):
            dictSourceM[i + 1] = arrayInput[i]
        dictM[1] = dictSourceM
        dictLayers: Dict[int, Layer] = objGNN.getDictLayers()
        for intLayerNum in range(1, intNumLayers, 1):
            dictNodes: Dict[int, Node] = dictLayers[intLayerNum].dictNodes
            # Size of Nodes at layer intLayerNum
            arraySize: npt.ArrayLike = np.array([dictNodes[intId].intSize for intId in range(1, len(dictNodes) + 1, 1)])
            W: npt.ArrayLike = objGNN.getUpperMatrixByLayer(intLayerNum)
            rows: int = W.shape[0]
            for i in range(rows):
                W[i] = W[i] * arraySize
            bias: npt.ArrayLike = objGNN.getUpperBiasByLayer(intLayerNum + 1)
            arrayOutput: npt.ArrayLike = np.dot(W, arrayInput)
            arrayOutput = arrayOutput + bias
            if intLayerNum == intNumLayers - 1:
                if lastRelu == LastRelu.YES:
                    dictM[intLayerNum + 1] = {index + 1: Relu.point(value)
                                              for index, value in enumerate(arrayOutput)}
                else:
                    dictM[intLayerNum + 1] = {index + 1: value for index, value in enumerate(arrayOutput)}
            else:
                dictM[intLayerNum + 1] = {index + 1: Relu.point(value)
                                          for index, value in enumerate(arrayOutput)}
            arrayInput = arrayOutput

        # Return dictM
        return dictM

    @staticmethod
    def ToAGNN(objGNN: GNN) -> GNN:
        """
        Create an absolute neural network as an instance of GNN
        from an interval neural network as an instance of GNN
        :param objGNN: an instance of GNN
        :type objGNN: GNN
        :return: (objANN -> GNN)
        """
        # create dictionary of lower and upper Layer instances
        dictLayers: Dict[int, Layer] = objGNN.getDictLayers()
        dictLayersA: Dict[int, Layer] = GNNUTS.__fromDictLayerToDictLayerA(dictLayers)
        # create dictionary of absolute Connection instances
        dictConnections: Dict[int, Connection] = copy.deepcopy(objGNN.getDictConnections())
        dictConnectionsA: Dict[int, Connection] = \
            GNNUTS.__fromDictConnectionToDictConnectionA(dictConnections, dictLayersA)
        # create an instance of GNN
        intNumLayers: int = objGNN.getNumOfLayers()
        dictNumNeurons: Dict[int, int] = objGNN.getDictNumNeurons()
        objAGNN: GNN = \
            GNN(dictLayersA, dictConnectionsA, intNumLayers,
                dictNumNeurons, NetworkType.STANDARD)

        # return GNN instance
        return objAGNN


    @staticmethod
    def __fromDictLayerToDictLayerA(dictLayers: Dict[int, Layer]) \
            -> Dict[int, Layer]:
        """
        Convert dictionary of Layer instances into a pair of dictionary of
        Layer instances for absolute neural network
        :param dictLayers: a mapping from layer number and Layer class instances
        :return: (dictLayersA-> Dict[int, Layer]) a dictionary of Layer instances
        """
        # create dictionary of Layer instances
        dictLayersA: Dict[int, Layer] = {}
        for key in dictLayers.keys():
            objLayer: Layer = dictLayers[key]
            # Absolute layer
            objLayerA: Layer = GNNUTS.__fromLayerToLayerA(objLayer)
            dictLayersA[key] = objLayerA

        # return dictionary of Layer instance
        return dictLayersA

    @staticmethod
    def __fromDictConnectionToDictConnectionA(dictConnections: Dict[int, Connection],
                                              dictLayersA: Dict[int, Layer]) \
            -> Dict[int, Connection]:
        """
        Convert dictionary of Connection instances into a
        dictionary of absolute Connection instances
        :param dictConnections: Dict[int, Layer]
        :type dictConnections: Dict[int, IConnection]
        :param dictLayersA: absolute mapping from layer number and Layer instances
        :type dictLayersA: Dict[int, Layer]
        :return: (dictConnectionsA: Dict[int, Connection]) an instance of
        dictionary of absolute Connection
        """
        dictConnectionsA: Dict[int, Connection] = {}
        for key in dictConnections.keys():
            objConnection: Connection = dictConnections[key]
            objConnectionA: Connection = GNNUTS.__fromConnectionToConnectionA(objConnection, dictLayersA, key)
            dictConnectionsA[key] = objConnectionA

        # return dictionary of GNN Connection instances
        return dictConnectionsA

    @staticmethod
    def __fromLayerToLayerA(objLayer: Layer) -> Layer:
        """
        Convert an instance of Layer class into an instance of absolute Layer class
        :param objLayer: an instance of Layer class
        :type objLayer: Layer
        :return: (objLayerA)-> Layer
        """
        # create dictionary of Node
        dictNodesA: Dict[int, Node] = {}

        for key in objLayer.dictNodes.keys():
            # convert an Node instance to equivalent absolute Node class instances
            objNode: Node = objLayer.dictNodes[key]
            bias: DataType.RealType = max(abs(objNode.bias.getLower()), abs(objNode.bias.getUpper()))
            objNodeA: Node = Node(objNode.enumAction, Real(bias), objNode.intSize, objNode.intId)
            dictNodesA[objNodeA.intId] = objNodeA
        objLayerA: Layer = Layer(dictNodesA)

        # return an instance of an absolute Layer class
        return objLayerA

    @staticmethod
    def __fromConnectionToConnectionA(objConnection: Connection,
                                      dictLayersA: Dict[int, Layer],
                                      intLayerNum: int) -> Connection:
        """
        Convert an instance of Connection into an absolute Connection instances
        :param objConnection: an instance of Connection class
        :type objConnection: IConnection
        :param dictLayersA: absolute mapping between layer number and Layer class instances
        :type dictLayersA: Dict[int, Layer]
        :param intLayerNum: layer number of the source Layer for the objConnection
        :type intLayerNum: int
        :return: ((objConnectionA: Connection) an instance of absolute Connection class
        """
        # create dictionary of absolute Edges
        dictEdgesA: Dict[Tuple[int, int], Edge] = {}
        for key in objConnection.dictEdges.keys():
            # convert an IEdge instance to equivalent lower and upper Edge instance
            objEdge: Edge = objConnection.dictEdges[key]
            weight: DataType.RealType = max(abs(objEdge.weight.getLower()), abs(objEdge.weight.getUpper()))
            objEdgeA: Edge = Edge(dictLayersA[intLayerNum].dictNodes[objEdge.nodeSource.intId],
                                  dictLayersA[intLayerNum + 1].dictNodes[objEdge.nodeTarget.intId],
                                  Real(weight))

            # add Edge instances to appropriate dictionary
            dictEdgesA[(objEdgeA.nodeSource.intId, objEdgeA.nodeTarget.intId)] = objEdgeA

        # create Connection for absolute neural network
        objConnectionA: Connection = Connection(dictEdgesA)

        # return instance of absolute Connection class
        return objConnectionA

