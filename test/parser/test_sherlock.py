from unittest import TestCase

from src.types.boundtype import BoundType
from src.parser.parser import Parser
from src.parser.sherlock import Sherlock
from typing import Dict
import numpy.typing as npt
import numpy as np


class TestSherlock(TestCase):
    def test_network(self):
        """
        It tests getNetwork function of the Sherlock class
        :return: None
        """
        # Expected Neural Network
        dictNeurons: Dict[int, int] = {1: 1, 2: 4, 3: 4, 4: 1}
        # Weights from (i+1)th to ith layer
        dictWeights: Dict[int, npt.ArrayLike] = {1: np.array([[-1], [1], [-1], [1]], dtype='f'),
                                                2: np.array([[-1, 1, -1, 1], [-1, 1, -1, 1], [-1, 1, -1, 1],
                                                             [-1, 1, -1, 1]], dtype='f'),
                                                3: np.array([[-1, 1, -1, 1]], dtype='f')
                                                }
        # Bias at ith layer
        dictBiases: Dict[int, npt.ArrayLike] = {2: np.array([1, 1, 1, 1], dtype='f'),
                                              3: np.array([1, 1, 1, 1], dtype='f'),
                                              4: np.array([1], dtype='f')}
        # Path of a neural network
        file = "/Users/ratanlal/Desktop/repositories/github/AVERINN/resources/test/nn.txt"

        # Parser neural netowrk in sherlock format
        objParser: Parser = Sherlock(file)

        # Test dictNeurons
        self.assertDictEqual(dictNeurons, objParser.getDictNeurons(),
                             "Neurons' configuration are not the same")

        # Test dictBiases
        intNumOfLayers: int = len(objParser.getDictNeurons())
        for i in range(2, intNumOfLayers + 1, 1):
            np.testing.assert_array_equal(dictBiases[i], objParser.getDictBiases(BoundType.LOW)[i],
                                          "Bias arrays are not equal")
        # Test dictWeights
        for i in range(1, intNumOfLayers, 1):
            np.testing.assert_array_equal(dictWeights[i], objParser.getDictWeights(BoundType.LOW)[i],
                                          "Weight matrices are not equal")