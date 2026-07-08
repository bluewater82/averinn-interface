from unittest import TestCase
from typing import Dict
import numpy.typing as npt
import numpy as np
from src.gnn.gnn import GNN
from src.gnn.gnnuts import GNNUTS
from src.parser.parseruts import ParserUTS


class TestGNNUTS(TestCase):

    def test_to_standard_gnn(self):
        """
        Test conversion of NeuralNetwork instance from dictionary of neurons, weight and biases
        :return: None
        """
        # Expected Neural Network
        dictNeurons: Dict[int, int] = {1: 1, 2: 4, 3: 4, 4: 1}
        dictWeight: Dict[int, npt.ArrayLike] = {1: np.array([[-1], [1], [-1], [1]], dtype='f'),
                                                2: np.array([[-1, 1, -1, 1], [-1, 1, -1, 1], [-1, 1, -1, 1],
                                                             [-1, 1, -1, 1]], dtype='f'),
                                                3: np.array([[-1, 1, -1, 1]], dtype='f')
                                                }
        dictBias: Dict[int, npt.ArrayLike] = {2: np.array([1, 1, 1, 1], dtype='f'),
                                              3: np.array([1, 1, 1, 1], dtype='f'),
                                              4: np.array([1], dtype='f')}

        objGNN: GNN = ParserUTS.toStandardGNN(dictNeurons, dictWeight, dictBias)
        # Test neural network configuration
        self.assertDictEqual(dictNeurons, objGNN.getDictNumNeurons(),
                             "Network configuration is not the same")
        # Test number of layers in NeuralNetwork instance
        intNumOfLayers: int = objGNN.getNumOfLayers()
        self.assertEqual(4, intNumOfLayers,
                         "Number of layers are not the same")
        # Test weight matrices
        # Test lower Weight arrays
        for i in range(1, intNumOfLayers, 1):
            np.testing.assert_array_equal(dictWeight[i], objGNN.getLowerMatrixByLayer(i),
                                          "Lower Weight matrices are not equal")
        # Test Bias arrays
        for i in range(2, intNumOfLayers + 1, 1):
            np.testing.assert_array_equal(dictBias[i], objGNN.getLowerBiasByLayer(i),
                                          "Bias arrays are not equal")

    def test_to_interval_gnn(self):
        """
              Test conversion of GNN instance from dictionary of neurons, weight and biases
              :return: None
              """
        # Expected Neural Network
        dictNeurons: Dict[int, int] = {1: 1, 2: 4, 3: 4, 4: 1}
        dictWeight: Dict[int, npt.ArrayLike] = {1: np.array([[-1], [1], [-1], [1]], dtype='f'),
                                                2: np.array([[-1, 1, -1, 1], [-1, 1, -1, 1], [-1, 1, -1, 1],
                                                             [-1, 1, -1, 1]], dtype='f'),
                                                3: np.array([[-1, 1, -1, 1]], dtype='f')
                                                }
        dictBias: Dict[int, npt.ArrayLike] = {2: np.array([1, 1, 1, 1], dtype='f'),
                                              3: np.array([1, 1, 1, 1], dtype='f'),
                                              4: np.array([1], dtype='f')}

        # Actual GNN instance creation
        objGNN: GNN = ParserUTS.toIntervalGNN(dictNeurons, dictWeight, dictWeight, dictBias, dictBias)
        # Test interval neural network configuration
        self.assertDictEqual(dictNeurons, objGNN.getDictNumNeurons(),
                             "Network configuration is not the same")
        # Test number of layers in NeuralNetwork instance
        intNumOfLayers: int = objGNN.getNumOfLayers()
        self.assertEqual(4, intNumOfLayers,
                         "Number of layers are not the same")
        # Test weight matrices
        # Test lower Weight arrays
        for i in range(1, intNumOfLayers, 1):
            np.testing.assert_array_equal(dictWeight[i], objGNN.getLowerMatrixByLayer(i),
                                          "Lower Weight matrices are not equal")
        # Test upper weight arrays which is the same as lower weight arrays
        for i in range(1, intNumOfLayers, 1):
            np.testing.assert_array_equal(dictWeight[i], objGNN.getUpperMatrixByLayer(i),
                                          "Lower Weight matrices are not equal")
        # Test lower Bias arrays
        for i in range(2, intNumOfLayers + 1, 1):
            np.testing.assert_array_equal(dictBias[i], objGNN.getLowerBiasByLayer(i),
                                          "Bias arrays are not equal")
        # Test upper Bias arrays which is the same as lower Bias arrays
        for i in range(2, intNumOfLayers + 1, 1):
            np.testing.assert_array_equal(dictBias[i], objGNN.getUpperBiasByLayer(i),
                                          "Bias arrays are not equal")

