from unittest import TestCase

from src.types.boundtype import BoundType
from src.parser.nnet import Nnet
from src.parser.parser import Parser
from typing import Dict
import numpy.typing as npt
import numpy as np


class TestNnet(TestCase):

    def test_network(self):
        """
        It tests getNetwork function of the Nnet class
        :return: None
        """
        # Expected Neural Network
        dictNeurons: Dict[int, int] = {1: 1, 2: 4, 3: 4, 4: 1}
        dictWeights: Dict[int, npt.ArrayLike] = {1: np.array([[-1], [1], [-1], [1]], dtype='f'),
                                                2: np.array([[-1, 1, -1, 1], [-1, 1, -1, 1], [-1, 1, -1, 1],
                                                             [-1, 1, -1, 1]], dtype='f'),
                                                3: np.array([[-1, 1, -1, 1]], dtype='f')
                                                }
        dictBiases: Dict[int, npt.ArrayLike] = {2: np.array([1, 1, 1, 1], dtype='f'),
                                              3: np.array([1, 1, 1, 1], dtype='f'),
                                              4: np.array([1], dtype='f')}
        # Path of a neural network
        file = "/Users/ratanlal/Desktop/repositories/github/AVERINN/resources/test/nn.nnet"
        # Parse neural network in .nnet format
        objParser: Parser = Nnet(file)

        # Test dictNeurons
        self.assertDictEqual(dictNeurons, objParser.getDictNeurons(),
                             "Neurons' configuration are not the same")

        # Test Bias arrays
        intNumOfLayers: int = len(objParser.getDictNeurons())
        for i in range(2, intNumOfLayers + 1, 1):
            np.testing.assert_array_equal(dictBiases[i], objParser.getDictBiases(BoundType.LOW)[i],
                                          "Bias arrays are not equal")
        # Test Weight arrays
        for i in range(1, intNumOfLayers, 1):
            np.testing.assert_array_equal(dictWeights[i], objParser.getDictWeights(BoundType.LOW)[i],
                                          "Weight matrices are not equal")


