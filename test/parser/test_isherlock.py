from typing import Dict
from unittest import TestCase
import numpy.typing as npt
import numpy as np

from src.types.boundtype import BoundType
from src.parser.isherlock import ISherlock
from src.parser.parser import Parser


class TestISherlock(TestCase):
    def test_network(self):
        """
         It tests getNetwork function of the ISherlock class
        :return: None
        """
        # Expected Neural Network
        dictNeurons: Dict[int, int] = {1: 2, 2: 4, 3: 4, 4: 2}
        dictWeightsLow: Dict[int, npt.ArrayLike] = {1: np.array([[0, 0], [0, 1], [0, 1], [1, 0]], dtype='f'),
                                                2: np.array([[1, 0, 1, 0], [1, 1, 1, 1], [0, 1, 0, 1],
                                                             [1, 0, 1, 0]], dtype='f'),
                                                3: np.array([[1, 0, 1, 1], [0, 0, 0, 1]], dtype='f')
                                                }
        dictWeightsHigh: Dict[int, npt.ArrayLike] = {1: np.array([[0, 0], [1, 1], [1, 1], [2, 0]], dtype='f'),
                                                   2: np.array([[1, 1, 1, 2], [1, 1, 1, 1], [1, 1, 1, 1],
                                                                [1, 1, 1, 1]], dtype='f'),
                                                   3: np.array([[1, 0, 1, 1], [1, 1, 1, 1]], dtype='f')
                                                   }
        dictBiasesLow: Dict[int, npt.ArrayLike] = {2: np.array([0, 1, 0, 1], dtype='f'),
                                              3: np.array([0, 1, 1, 1], dtype='f'),
                                              4: np.array([1, 1], dtype='f')}
        dictBiasesHigh: Dict[int, npt.ArrayLike] = {2: np.array([0, 1, 1, 1], dtype='f'),
                                                 3: np.array([0, 1, 1, 1], dtype='f'),
                                                 4: np.array([1, 1], dtype='f')}

        # Path of a neural network
        file = "/Users/ratanlal/Desktop/repositories/github/AVERINN/resources/test/inn.txt"

        # Parse interval neural network given in sherlock format
        objParser: Parser = ISherlock(file)

        # Test GNN instance configuration
        self.assertDictEqual(dictNeurons, objParser.getDictNeurons(),
                             "Neurons' configuration are not the same")

        # Test dictBiasesLow
        intNumOfLayers: int = len(objParser.getDictNeurons())
        for i in range(2, intNumOfLayers + 1, 1):
            np.testing.assert_array_equal(dictBiasesLow[i], objParser.getDictBiases(BoundType.LOW)[i],
                                          "Lower Bias arrays are not equal")
        # Test dictBiasesHigh
        for i in range(2, intNumOfLayers + 1, 1):
            np.testing.assert_array_equal(dictBiasesHigh[i], objParser.getDictBiases(BoundType.HIGH)[i],
                                          "Upper Bias arrays are not equal")

        # Test dictWeightsLow
        for i in range(1, intNumOfLayers, 1):
            np.testing.assert_array_equal(dictWeightsLow[i], objParser.getDictWeights(BoundType.LOW)[i],
                                          "Lower Weight matrices are not equal")

        # Test dictWeightsHigh
        for i in range(1, intNumOfLayers, 1):
            np.testing.assert_array_equal(dictWeightsHigh[i], objParser.getDictWeights(BoundType.HIGH)[i],
                                          "Upper Weight matrices are not equal")
