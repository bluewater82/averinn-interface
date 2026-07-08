"""
Author: Ratan Lal
Date : November 7, 2024
"""
from typing import Dict

import numpy as np

from src.gnn.ireal import IReal
from src.gnn.number import Number
from src.gnn.connection import Connection
from src.gnn.edge import Edge
from src.gnn.layer import Layer
from src.gnn.gnn import GNN
from src.gnn.node import Node
from src.types.abstype import AbsType
from src.types.networktype import NetworkType


class Abstraction:
    """
    It captures different abstraction of GNN class
    """

    def __init__(self, objGNN: GNN, dictPartition: Dict[int, Dict[int, set[int]]],
                 absType: AbsType):
        """
        It creates an instance of GNN that will have an abstraction of another
        instance of GNN class
        :param objGNN: An instance of the GNN class
        :type objGNN: GNN
        :param dictPartition: dictionary between abstract neurons' ids and sets of concrete
        neurons' ids
        :type dictPartition: Dict[int, Dict[int, set[int]]]
        :param absType: a specific type of abstraction
        :type absType: AbsType
        """
        self.__objGNN__ = objGNN
        self.__dictPartition__ = dictPartition
        self.__absType__ = absType

    def getAbstraction(self) -> GNN:
        """
        Return the abstraction of the GNN class
        :return: (objGNNAbs -> GNN)
        Abstraction of GNN class
        """
        return self.__abstraction__()

    def __abstraction__(self) -> GNN:
        """
        Find an abstraction of the GNN based on abstraction types
        :return: (objGNNAbs -> GNN)
        An abstraction of GNN instance
        """
        objGNNAbs: GNN = None
        if self.__absType__ == AbsType.INTERVAL:
            objGNNAbs: GNN = self.__intervalAbstraction__()
        # Return an abstraction of GNN
        return objGNNAbs

    def __intervalAbstraction__(self) -> GNN:
        """
        Construct an instance of GNN class
        based on a given partition of NodeIds of another instance of GNN
        :return: (objGNNAbs -> GNN)
        An abstraction of GNN instance
        """
        # Create the following parameters for creating an Interval abstraction
        # Create dictionary between Layer number and its Layer class instance
        # Number of layers including input and output layers
        intNumLayersAbs: int = self.__objGNN__.getNumOfLayers()
        dictLayersAbs: Dict[int, Layer] = self.__intervalDictLayers__()
        # Create dictionary between Layer number and its Connection class instance
        dictConnectionsAbs: Dict[int, Connection] = self.__intervalDictConnections__(dictLayersAbs)
        # Crete dictionary between Layer number and its number of neurons
        dictNumNeuronsAbs: Dict[int, int] = self.__intervalDictNumOfNeurons__(dictLayersAbs)
        # Create NeuralNetwork instance for the abstract system
        objGNNAbs: GNN = GNN(dictLayersAbs, dictConnectionsAbs,
                             intNumLayersAbs, dictNumNeuronsAbs, NetworkType.INTERVAL)
        # Return an instance of GNN for the abstract system
        return objGNNAbs

    def __intervalDictLayers__(self) -> Dict[int, Layer]:
        """
        Construct Layer instances for interval abstraction as an instance of GNN
        dictionary between layer numbers and Layer instances for an abstract GNN
        """
        # Create a dictionary between the layer numbers and Layer instances
        # for an abstract GNN
        dictLayersAbs: Dict[int, Layer] = {}
        # Get  the dictionary between  the layer numbers and Layer instances
        # for the original GNN
        dictLayers: Dict[int, Layer] = self.__objGNN__.getDictLayers()
        # Get the number of layers from original GNN
        intNumLayers: int = self.__objGNN__.getNumOfLayers()
        for i in range(1, self.__objGNN__.getNumOfLayers() + 1, 1):
            # Dictionary of abstract Node instances for the layer i
            dictNodesAbs: Dict[int, Node] = {}
            for intIdAbs in self.__dictPartition__[i].keys():
                # Compute intervalBiasAbs
                low: np.float64 = \
                    min([dictLayers[i].dictNodes[j].bias.getLower()
                         for j in self.__dictPartition__[i][intIdAbs]])
                high: np.float64 = max([dictLayers[i].dictNodes[j].bias.getUpper()
                                   for j in self.__dictPartition__[i][intIdAbs]])
                biasAbs: Number = IReal(low, high)
                # Compute action
                j = list(self.__dictPartition__[i][intIdAbs])[0]
                enumAction = dictLayers[i].dictNodes[j].enumAction

                # Compute size for an abstract INode instance
                intSizeAbs = 0
                for j in self.__dictPartition__[i][intIdAbs]:
                    intSizeAbs += dictLayers[i].dictNodes[j].intSize

                # Create an abstract node
                dictNodesAbs[intIdAbs] = Node(enumAction, biasAbs, intSizeAbs, intIdAbs)

            # Update the dictionary for the layer
            dictLayersAbs[i] = Layer(dictNodesAbs)

        # Return  dictILayersAbs
        return dictLayersAbs

    def __intervalDictConnections__(self, dictLayersAbs: Dict[int, Layer]) \
            -> Dict[int, Connection]:
        """
        Construct Connection instances for interval abstraction as an instance of GNN
        :param dictLayersAbs:  dictionary between layer numbers and Layer instances for
        the abstract NeuralNetwork
        :type dictLayersAbs: Dict[int, Layer]
        :return: (dictConnectionsAbs -> Dict[int, Connection])
        dictionary between layer numbers and Connection instances for an abstract GNN
        """
        # Create a dictionary between layer numbers and Connection instances
        # for an abstract GNN
        dictConnectionsAbs: Dict[int, Connection] = {}
        # Get the dictionary between layer numbers and Connection instances
        # for the original GNN
        dictConnections: Dict[int, Connection] = self.__objGNN__.getDictConnections()

        # Iterate over all the layers
        intNumLayers: int = self.__objGNN__.getNumOfLayers()

        for i in range(1, intNumLayers, 1):
            # Dictionary for the abstract edges between layer i and i+1
            dictEdgesAbs: Dict[(int, int), Edge] = {}
            for intIdSourceAbs in self.__dictPartition__[i].keys():
                for intTargetIdAbs in self.__dictPartition__[i + 1].keys():
                    # Compute interval weight for an abstract edge
                    low: np.float64 = min([dictConnections[i].dictEdges[(j, k)].weight.getLower()
                                      for j in self.__dictPartition__[i][intIdSourceAbs]
                                      for k in self.__dictPartition__[i + 1][intTargetIdAbs]])
                    high: np.float64 = max([dictConnections[i].dictEdges[(j, k)].weight.getUpper()
                                       for j in self.__dictPartition__[i][intIdSourceAbs]
                                       for k in self.__dictPartition__[i + 1][intTargetIdAbs]])

                    # Create an abstract edge
                    weightAbs: Number = IReal(low, high)
                    dictEdgesAbs[(intIdSourceAbs, intTargetIdAbs)] = \
                        Edge(dictLayersAbs[i].dictNodes[intIdSourceAbs],
                             dictLayersAbs[i + 1].dictNodes[intTargetIdAbs], weightAbs)

            # update the dictIConnectionsAbs for the abstract GNN
            dictConnectionsAbs[i] = Connection(dictEdgesAbs)

        # Return dictIConnectionsAbs
        return dictConnectionsAbs

    def __intervalDictNumOfNeurons__(self, dictLayersAbs: Dict[int, Layer]) -> Dict[int, int]:
        """
        Construct dictionary between layer numbers and numbers of neurons
        for the abstract NeuralNetwork
        :param dictLayersAbs: Dictionary between layer numbers and Layer instances
        :type dictLayersAbs: Dict[int, Layer]
        :return: (dictNumOfNeurons -> Dict[int, int])
        dictionary between layer numbers and numbers of neurons for
        the abstract GNN
        """
        # Create dictionary between layer numbers and number of neurons
        dictNumOfNeuronsAbs: Dict[int, int] = {}

        # Update dictNumOfNeurons
        for intLayerNum in dictLayersAbs.keys():
            dictNumOfNeuronsAbs[intLayerNum] = dictLayersAbs[intLayerNum].intNumNodes

        # Return dictNumOfNeuronsAbs
        return dictNumOfNeuronsAbs
