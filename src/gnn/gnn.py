"""
Author: Ratan Lal
Date : November 4, 2023
"""
from src.gnn.connection import Connection
from src.gnn.layer import Layer
from typing import Dict, List
import numpy.typing as npt
import numpy as np
from src.types.datatype import DataType
from src.types.networktype import NetworkType
from itertools import zip_longest
from src.utilities.log import Log


class GNN:
    """
    The class GNN captures a neural network
    """

    def __init__(self, dictLayers: Dict[int, Layer],
                 dictConnections: Dict[int, Connection], intNumLayers: int,
                 dictNumNeurons: Dict[int, int], networkType: NetworkType):
        """
        Initialize an instance of  GNN class
        :param dictLayers: dictionary of mapping between layer number and
        respective object of Layer class
        :type dictLayers: Dict[int, Layer]
        :param dictConnections: dictionary of mapping layer number and
        respective object of Connection class between the layer number and
         its next layer number
        :type dictConnections: Dict[int, Connection]
        :param intNumLayers: number of layers
        :type intNumLayers: int
        :param dictNumNeurons: dictionary between layer number and
        number of neurons at the layer number
        :type dictNumNeurons: Dict[int, int]
        :param networkType: type of network
        :type networkType: NetworkType
        """
        # Dictionary between Layer number and its Layer class instance
        self.__dictLayers__: Dict[int, Layer] = dictLayers
        # Dictionary between Layer number and its Connection class instance
        self.__dictConnections__: Dict[int, Connection] = dictConnections
        # Number of layers including input and output layers
        self.__intNumLayers__: int = intNumLayers
        # Dictionary between Layer number and its number of neurons
        self.__dictNumNeurons__: Dict[int, int] = dictNumNeurons
        # Type of neural network
        self.__networkType__: NetworkType = networkType

    def getNumOfLayers(self) -> int:
        """
        Get number of layers
        :return: (intNumOfLayers -> int)
        number of layers
        """
        return self.__intNumLayers__

    def getDictNumNeurons(self) -> Dict[int, int]:
        """
        Get dictionary between layer numbers and number of neurons
        :return: (dictLayers -> Dict[int, int])
        dictionary between layer numbers and number of neurons
        """
        return self.__dictNumNeurons__

    def getDictLayers(self) -> Dict[int, Layer]:
        """
        Get dictionary between layer numbers and Layer instances for GNN
        :return: (dictLayers -> Dict[int, Layer])
        dictionary between layer numbers and Layer instances
        """
        return self.__dictLayers__

    def getDictConnections(self) -> Dict[int, Connection]:
        """
        Get dictionary between layer numbers and Connection instances for GNN
        :return: (dictConnections -> Dict[int, Connection])
        dictionary between layer numbers and Connection instances
        """
        return self.__dictConnections__

    def getNetworkType(self) -> NetworkType:
        """
        Get type of network
        :return: (networkType -> NetworkType)
        """
        return self.__networkType__

    def display(self):
        """
        Display Neural Network
        :return: None
        """
        # Collect all lower weight matrices
        matricesLow = []
        for i in range(1, self.__intNumLayers__, 1):
            matricesLow.append(self.getLowerMatrixByLayer(i).tolist())
        BiasesLow = []
        for i in range(1, self.__intNumLayers__, 1):
            BiasesLow.append(self.getLowerBiasByLayer(i+1).tolist())
        if self.__networkType__ == NetworkType.STANDARD:
            Log.message("       Matrices and Biases\n")
            self.__printWeightsAndBiases__(matricesLow, BiasesLow)
        elif self.__networkType__ == NetworkType.INTERVAL:
            Log.message("       Lower Matrices and Biases\n")
            self.__printWeightsAndBiases__(matricesLow, BiasesLow)
            # Collect all upper weight matrices
            matricesHigh = []
            for i in range(1, self.__intNumLayers__, 1):
                matricesHigh.append(self.getUpperMatrixByLayer(i).tolist())
            BiasesHigh = []
            for i in range(1, self.__intNumLayers__, 1):
                BiasesHigh.append(self.getUpperBiasByLayer(i + 1).tolist())
            Log.message("       Upper Matrices and Biases\n")
            self.__printWeightsAndBiases__(matricesHigh, BiasesHigh)

    def __printWeightsAndBiases__(self, matrices: List[List[List[DataType.RealType]]], biases: List[List[DataType.RealType]]):
        """
        Print matrices in one rows
        :param matrices: matrices of weights
        :type matrices: List[List[List[float]]]
        :return: None
        """
        # Calculate the number of columns for each matrix
        # Assuming each matrix has consistent row lengths
        max_columns = [len(matrix[0]) for matrix in matrices]

        # Find the maximum number of rows across all matrices
        max_rows = max(len(matrix) for matrix in matrices)

        # Add layer information
        layerMsg = "        "
        for i in range(1, self.__intNumLayers__, 1):
            layerMsg += "L"+str(i+1)+" to L"+str(i)+"                              "
        Log.message(layerMsg+"\n")
        # Prepare the matrices by adding rows of placeholder values where necessary
        # Fill missing rows in each matrix with placeholders (empty lists or None)
        padded_matrices = []
        padded_biases = []
        j:int = 0
        for matrix in matrices:
            padded_matrix = matrix + [['-'] * max_columns[matrices.index(matrix)]] * (max_rows - len(matrix))
            padded_matrices.append(padded_matrix)
            padded_bias: List[DataType.RealType] = biases[j] + ['-']*max_columns[matrices.index(matrix)]
            padded_biases.append(padded_bias)
            j += 1
        # Print the n matrices in parallel
        #print(padded_biases)
        for row_index in range(max_rows):
            row_elements = []
            for i, matrix in enumerate(padded_matrices):
                row_elements.append("       "+str(row_index + 1) + ":" + str(matrix[row_index]))
                # For large network, it creates problem
               # row_elements.append("   "+str(padded_biases[i][row_index]))
            Log.message(str('     '.join(row_elements))+"\n")

    def getLowerMatrixByLayer(self, intLayer: int) -> npt.ArrayLike:
        """
        Find lower weight matrix from layer intLayer+1 to layer intLayer
        :param intLayer: layer number
        :type intLayer: int
        :return: (matWeight: npt.ArrayLike)
        Weight Matrix from layer intLayer+1 to layer intLayer
        """
        # Number of neurons at layer intLayer + 1 (source layer)
        numOfSN: int = self.__dictLayers__[intLayer + 1].intNumNodes
        # Number of neurons at layer intLayer (target layer)
        numOfTN: int = self.__dictLayers__[intLayer].intNumNodes
        # Initialization of a matrix to store lower weight of edges
        # from source layer to target layer
        matWeight: npt.ArrayLike = np.array([[DataType.RealType(0.0) for j in range(numOfTN)]
                                             for i in range(numOfSN)], dtype=object)
        # Extract weight from dictIConnections
        for i in range(numOfSN):
            for j in range(numOfTN):
                matWeight[i][j] = self.__dictConnections__[intLayer].dictEdges[
                    (j + 1, i + 1)].weight.getLower()
        # Return lower weight matrix from layer intLayer + 1 to intLayer
        return matWeight

    def getUpperMatrixByLayer(self, intLayer: int) -> npt.ArrayLike:
        """
        Find upper weight matrix from layer intLayer+1 to layer intLayer
        :param intLayer: layer number
        :type intLayer: int
        :return: (matWeight: npt.ArrayLike)
        Weight Matrix from layer intLayer+1 to layer intLayer
        """
        # Number of neurons at layer intLayer + 1 (source layer)
        numOfSN: int = self.__dictLayers__[intLayer + 1].intNumNodes
        # Number of neurons at layer intLayer (target layer)
        numOfTN: int = self.__dictLayers__[intLayer].intNumNodes
        # Initialization of a matrix to store lower weight of edges
        # from source layer to target layer
        matWeight: npt.ArrayLike = np.array([[DataType.RealType(0.0) for j in range(numOfTN)]
                                             for i in range(numOfSN)], dtype=object)
        # Extract weight from dictConnections
        for i in range(numOfSN):
            for j in range(numOfTN):
                matWeight[i][j] = self.__dictConnections__[intLayer].dictEdges[
                    (j + 1, i + 1)].weight.getUpper()

        # Return upper weight matrix from layer intLayer + 1 to intLayer
        return matWeight

    def getLowerMatrixByLayerWithSize(self, intLayer: int) -> npt.ArrayLike:
        """
        Find lower weight matrix from layer intLayer+1 to layer intLayer
        :param intLayer: layer number
        :type intLayer: int
        :return: (matWeight: npt.ArrayLike)
        Weight Matrix from layer intLayer+1 to layer intLayer
        """
        # Number of neurons at layer intLayer + 1 (source layer)
        numOfSN: int = self.__dictLayers__[intLayer + 1].intNumNodes
        # Number of neurons at layer intLayer (target layer)
        numOfTN: int = self.__dictLayers__[intLayer].intNumNodes
        # Initialization of a matrix to store lower weight of edges
        # from source layer to target layer
        matWeight: npt.ArrayLike = np.array([[DataType.RealType(0.0) for j in range(numOfTN)]
                                             for i in range(numOfSN)], dtype=object)
        # Extract weight from dictIConnections
        for i in range(numOfSN):
            for j in range(numOfTN):
                matWeight[i][j] = self.__dictConnections__[intLayer].dictEdges[
                    (j + 1, i + 1)].weight.getLower()*self.getDictLayers()[intLayer].dictNodes[j+1].intSize
        # Return lower weight matrix from layer intLayer + 1 to intLayer
        return matWeight

    def getUpperMatrixByLayerWithSize(self, intLayer: int) -> npt.ArrayLike:
        """
        Find upper weight matrix from layer intLayer+1 to layer intLayer
        :param intLayer: layer number
        :type intLayer: int
        :return: (matWeight: npt.ArrayLike)
        Weight Matrix from layer intLayer+1 to layer intLayer
        """
        # Number of neurons at layer intLayer + 1 (source layer)
        numOfSN: int = self.__dictLayers__[intLayer + 1].intNumNodes
        # Number of neurons at layer intLayer (target layer)
        numOfTN: int = self.__dictLayers__[intLayer].intNumNodes
        # Initialization of a matrix to store lower weight of edges
        # from source layer to target layer
        matWeight: npt.ArrayLike = np.array([[DataType.RealType(0.0) for j in range(numOfTN)]
                                             for i in range(numOfSN)], dtype=object)
        # Extract weight from dictConnections
        for i in range(numOfSN):
            for j in range(numOfTN):
                matWeight[i][j] = self.__dictConnections__[intLayer].dictEdges[
                    (j + 1, i + 1)].weight.getUpper()*self.getDictLayers()[intLayer].dictNodes[j+1].intSize

        # Return upper weight matrix from layer intLayer + 1 to intLayer
        return matWeight

    def getLowerBiasByLayer(self, intLayer: int) -> npt.ArrayLike:
        """
        Find lower Bias array for the layer number intLayer
        :param intLayer: layer number
        :type intLayer: int
        :return: (arrayBiasLow: npt.ArrayLike)
        An array of lower biases for the layer number intLayer
        """
        # Number of neurons at later intLayer
        numOfN: int = self.__dictLayers__[intLayer].intNumNodes
        # Initialize a numpy array with size numOfN
        arrayBiasLow: npt.ArrayLike = np.array([DataType.RealType(0.0) for i in range(numOfN)], dtype=object)
        # Extract lower biases from the layer intLayer
        for i in range(numOfN):
            arrayBiasLow[i] = self.__dictLayers__[intLayer].dictNodes[i + 1].bias.getLower()
        # Return lower Biases array
        return arrayBiasLow

    def getUpperBiasByLayer(self, intLayer: int) -> npt.ArrayLike:
        """
        Find upper Bias array for the layer number intLayer
        :param intLayer: layer number
        :type intLayer: int
        :return: (arrayBiasLow: npt.ArrayLike)
        An array of upper biases for the layer number intLayer
        """
        # Number of neurons at later intLayer
        numOfN: int = self.__dictLayers__[intLayer].intNumNodes
        # Initialize a numpy array with size numOfN
        arrayBiasLow: npt.ArrayLike = np.array([DataType.RealType(0.0) for i in range(numOfN)], dtype=object)
        # Extract lower biases from the layer intLayer
        for i in range(numOfN):
            arrayBiasLow[i] = self.__dictLayers__[intLayer].dictNodes[i + 1].bias.getUpper()
        # Return lower Biases array
        return arrayBiasLow

    def binarization(self):
        """
        Convert weight and biases between -1 and 1
        """
        # Get the number of layer
        intNumLayers: int = self.__intNumLayers__
        # convert weight between -1 and 1
        for intLayer in range(1, intNumLayers, 1):
            for key in self.__dictConnections__[intLayer].dictEdges.keys():
                if self.__dictConnections__[intLayer].dictEdges[key].weight.getLower() < 0.0:
                    self.__dictConnections__[intLayer].dictEdges[key].weight.setLower(0.0)
                else:
                    self.__dictConnections__[intLayer].dictEdges[key].weight.setLower(1.0)

                if self.__dictConnections__[intLayer].dictEdges[key].weight.getUpper() < 0.0:
                    self.__dictConnections__[intLayer].dictEdges[key].weight.setUpper(0.0)
                else:
                    self.__dictConnections__[intLayer].dictEdges[key].weight.setUpper(1.0)

        # Convert biases between -1 and 1
        for intLayer in range(2, intNumLayers + 1, 1):
            for key in self.__dictLayers__[intLayer].dictNodes.keys():
                if self.__dictLayers__[intLayer].dictNodes[key].bias.getLower() < 0.0:
                    self.__dictLayers__[intLayer].dictNodes[key].bias.setLower(0.0)
                else:
                    self.__dictLayers__[intLayer].dictNodes[key].bias.setLower(1.0)
                if self.__dictLayers__[intLayer].dictNodes[key].bias.getUpper() < 0.0:
                    self.__dictLayers__[intLayer].dictNodes[key].bias.setUpper(0.0)
                else:
                    self.__dictLayers__[intLayer].dictNodes[key].bias.setUpper(1.0)
