"""
Author: Ratan Lal
Date : November 4, 2024
"""
import random
from typing import Dict, List
from src.gnn.gnn import GNN
from src.gnn.number import Number
from src.types.datatype import DataType
from src.types.partitiontype import PartitionType
import numpy.typing as npt
from sklearn.cluster import KMeans


class Partition:
    """
    Partition neurons of GNN Instance
    """

    def __init__(self, objGNN: GNN, partitionType: PartitionType, numOfAbsNodes: int, listOfClusters: npt.ArrayLike):
        """
        Partition a specific type of neural network based on a given partition type
        :param objGNN: an instance of the GNN class
        :type objGNN: GNN
        :param partitionType: a type of partition
        :type partitionType: PartitionType
        :param numOfAbsNodes: number of abstract nodes in each layer
        :type numOfAbsNodes: int
        :param listOfClusters: list of clusters
        :type listOfClusters: list
        :return: (dictPartition:Dict[int, Dict[int, set[int]]])
        A partition of nodes of GNN
        """
        self.__dictPartition__: Dict[int, Dict[int, set[int]]] \
            = self.__partition__(objGNN, partitionType, numOfAbsNodes, listOfClusters)

    def getPartition(self) -> Dict[int, Dict[int, set[int]]]:
        """
        Return the partition of NodeIds for GNN instance
        :return:(dictPartition:Dict[int, Dict[int, set[int]]])
        A partition of nodes of GNN class
        """
        return self.__dictPartition__

    def __partition__(self, objGNN: GNN, partitionType: PartitionType, numOfAbsNodes: int, listOfClusters: npt.ArrayLike) \
            -> Dict[int, Dict[int, set[int]]]:
        """
        Partition a specific type of neural network based on a given partition type
        :param objGNN: an instance of th GNN class
        :type objGNN: GNN
        :param partitionType: a type of partition
        :type partitionType: PartitionType
        :param numOfAbsNodes: number of abstract nodes in each layer
        :type numOfAbsNodes: int
        :param listOfClusters: list of clusters
        :type listOfClusters: list
        :return: (dictPartition:Dict[int, Dict[int, set[int]]])
        A partition of neurons of a specific type of neural network
        """
        if partitionType == PartitionType.FIXED:
            return self.__fixedPartition__(objGNN, numOfAbsNodes)
        elif partitionType == PartitionType.RANDOM:
            return self.__randomPartition__(objGNN, numOfAbsNodes)
        elif partitionType == PartitionType.PRESUM:
            return self.__preSumPartition__(objGNN, listOfClusters)
        else:
            return None

    def __fixedPartition__(self, objGNN: GNN, numOfAbsNodes: int) \
            -> Dict[int, Dict[int, set[int]]]:
        """
        Create a partition of neurons of a specific type of neural network where
        division of neurons will be fixed each time this function is called
        :param objGNN: an instance the GNN class
        :type objGNN: GNN
        :param numOfAbsNodes: number of abstract nodes in each layer
        :type numOfAbsNodes: int
        :return: (dictPartition:Dict[int, Dict[int, set[int]]])
        A partition of neurons of Neural Network
        """
        # Collect all the INodes' ids of objANN into a dictionary
        dictNodeIds: Dict[int, List[int]] = dict()
        for i in range(1, objGNN.getNumOfLayers() + 1, 1):
            dictNodeIds[i] = [node.intId for node in
                               objGNN.getDictLayers()[i].dictNodes.values()]
        # Create partition of NodeIds
        dictPartition: Dict[int, Dict[int, set[int]]] = dict()
        for i in range(1, objGNN.getNumOfLayers() + 1, 1):
            # Number of NodeIds that needs to be merged
            intNumOfNodes: int = len(dictNodeIds[i])
            intSize: int = intNumOfNodes//numOfAbsNodes
            if i == 1 or i == objGNN.getNumOfLayers():
                dictLayerPartition: Dict[int, set[int]] = dict()
                # Iterate the list
                k: int = 0
                j: int = 1
                listIds = dictNodeIds[i]
                intLength: int = len(listIds)
                size = 1
                while k < intLength:
                    if k + size <= intLength:
                        dictLayerPartition[j] = set(listIds[k:k + size])
                    else:
                        dictLayerPartition[j] = set(listIds[k:intLength])
                    k += size
                    j += 1

                dictPartition[i] = dictLayerPartition
            elif intSize == 0:
                intSize = 1
                dictPartition[i] = self.__layerwisePartition__(dictNodeIds[i], intSize, numOfAbsNodes)
            else:
                dictPartition[i] = self.__layerwisePartition__(dictNodeIds[i], intSize, numOfAbsNodes)
        # Return dictPartition
        return dictPartition

    def __randomPartition__(self, objGNN: GNN, numOfAbsNodes: int) \
            -> Dict[int, Dict[int, set[int]]]:
        """
        Create a partition of neurons of a specific type of neural network where
        division of neurons will be fixed each time this function is called
        :param objGNN: an instance the GNN class
        :type objGNN: GNN
        :param numOfAbsNodes: number of abstract nodes in each layer
        :type numOfAbsNodes: int
        :return: (dictPartition:Dict[int, Dict[int, set[int]]])
        A partition of NodeIds of GNN class
        """
        # Collect all the Nodes' ids of objNN into a dictionary
        dictNodeIds: Dict[int, List[int]] = dict()
        for i in range(1, objGNN.getNumOfLayers() + 1, 1):
            dictNodeIds[i] = [node.intId for node in
                               objGNN.getDictLayers()[i].dictNodes.values()]
        # Create partition of INodeIds
        dictPartition: Dict[int, Dict[int, set[int]]] = dict()
        for i in range(1, objGNN.getNumOfLayers() + 1, 1):
            # Number of INodeIds that needs to be merged

            intNumOfINodes: int = len(dictNodeIds[i])
            intSize: int = intNumOfINodes // numOfAbsNodes
            if i == 1 or i == objGNN.getNumOfLayers() or intSize == 0:
                intSize = 1
            dictPartition[i] = self.__layerwisePartition__(random.sample(dictNodeIds[i],
                                                                       intNumOfINodes), intSize, numOfAbsNodes)
        # Return dictPartition
        return dictPartition

    def __preSumPartition__(self, objGNN: GNN, listOfClusters) -> Dict[int, Dict[int, set[int]]]:
        """
        Create a partition of neurons of a GNN instance based on clustering
        :param objGNN: an instance of GNN class
        :type objGNN: GNN
        :param listOfClusters: list of cluster sizes, each size corresponds to a layer
        :type listOfClusters: npt.ArrayLike
        :return: (dictPartition:Dict[int, Dict[int, set[int]]])
        A partition of NodeIds of GNN class
        """
        # Create dictionaries of dictionary of node id and  array with bias and presum
        dictPreSum = self.__preSum__(objGNN)
        intNumOfLayers: int = objGNN.getNumOfLayers()
        dictPartition: Dict[int, Dict[int, set[int]]] = dict()
        for intLayer in range(1, intNumOfLayers + 1, 1):
            A = []
            dictPartition[intLayer] = dict()
            for id in dictPreSum[intLayer].keys():
                A.append(dictPreSum[intLayer][id])
            kmeans = KMeans(n_clusters= listOfClusters[intLayer-1], random_state=42)
            print(A)
            kmeans.fit(A)
            clusterAssignment = kmeans.labels_
            for i in range(len(clusterAssignment)):
                if clusterAssignment[i]+1 not in dictPartition[intLayer].keys():
                    dictPartition[intLayer][clusterAssignment[i]+1] = set([i+1])
                else:
                    dictPartition[intLayer][clusterAssignment[i]+1].add(i+1)

        return dictPartition

    def __layerwisePartition__(self, lstINodeIds: List[int], intSize: int, numOfAbsNodes: int) -> Dict[int, set[int]]:
        """
        Create a partition of INodeIds of IntervalNeuralNetwork for a specific layer
        :param lstINodeIds: List of INodeIds for a specific layer
        :type lstINodeIds: List[int]
        :param intSize: number of INodeIds that needs to be merged
        :type intSize: int
        :return: (dictLayerPartition:Dict[int, set[int]])
        Dictionary between abstract INodeIds and set of concrete INodeIds of IntervalNeuralNetwork
        """
        # Number of INodeIds in a specific layer
        intNumOfINodeIds: int = len(lstINodeIds)
        # Number of remaining nodes beyond considering for the abstract nodes
        intLeftNodes: int = intNumOfINodeIds - numOfAbsNodes*intSize
        # Partition for a specific layer
        dictLayerPartition: Dict[int, set[int]] = dict()
        j: int = 1
        k: int = 0
        p: int = 1
        q: int = 0
        while k < intNumOfINodeIds:
            if p <= intLeftNodes:
                p += 1
                q = 1
            else:
                q = 0
            if k + intSize + q <= intNumOfINodeIds:
                dictLayerPartition[j] = set(lstINodeIds[k:k + intSize + q])
            else:
                dictLayerPartition[j] = set(lstINodeIds[k:intNumOfINodeIds])
            j += 1
            k += intSize + q
        # Return partition of a specific layer
        return dictLayerPartition

    def __preSum__(self, objGNN: GNN) -> dict[int, dict[int, npt.ArrayLike]]:
        """
        Compute presum for each neuron from layer 2 to the lastlayer-1
        :param objGNN: an instance of an GNN class
        :type objGNN: GNN
        :return: (dictPreSum: Dict[int, Dict[int, DataType.RealType]])
        """
        dictRange: Dict[int, Number] = {}
        dictPreSum: Dict[int, Dict[int, DataType.RealType]] = {}
        intNumLayers: int = objGNN.getNumOfLayers()
        for intLayer in range(1, intNumLayers+1, 1):
            dictPreSum[intLayer] = {}
            if intLayer ==1 or intLayer ==intNumLayers:
                numOfNeurons: int = objGNN.getDictNumNeurons()[intLayer]
                temp = 0.5
                for i in range(numOfNeurons):
                    dictPreSum[intLayer][i+1] = [0.0+i*temp, 0.0+i*temp]
            else:
                arrayBias: npt.ArrayLike = objGNN.getLowerBiasByLayer(intLayer)
                matWeightLow: npt.ArrayLike = objGNN.getLowerMatrixByLayer(intLayer-1)
                row, col = matWeightLow.shape
                for i in range(row):
                    tempList: npt.ArrayLike = [arrayBias[i]]
                    sum: DataType.RealType = 0.0
                    for j in range(col):
                        sum += matWeightLow[i][j]
                    tempList.append(sum)
                    dictPreSum[intLayer][i + 1] = tempList

        return dictPreSum


