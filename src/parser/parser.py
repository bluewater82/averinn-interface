"""
Author: Ratan Lal
Date : November 4, 2023
"""

from abc import ABCMeta, abstractmethod
from typing import Dict
import numpy.typing as npt
from src.gnn.gnn import GNN
from src.types.boundtype import BoundType


class Parser(metaclass=ABCMeta):
    """
    Abstract class for parsing different input formats of nn
    """

    @abstractmethod
    def getNetwork(self) -> GNN:
        """
        Get a neural network of any type
        :return: (objGNN: GNN)
        An instance of GNN class
        """
        pass

    @abstractmethod
    def getDictNeurons(self) -> Dict[int, int]:
        """
        Get dictionary between layer number and its number of neurons
        :return: (dictNeurons: Dict[int,int])
        """
        pass

    @abstractmethod
    def getDictWeights(self, boundType: BoundType) -> Dict[int, npt.ArrayLike]:
        """
        Get dictionary between layer number i and weight matrix where each row represents
        weight array from a neuron at i+1 and all neurons at layer i
        :return: (dictWeights: Dict[int,npt.ArrayLike])
        """
        pass

    @abstractmethod
    def getDictBiases(self, boundType: BoundType) -> Dict[int, npt.ArrayLike]:
        """
        Get dictionary between layer number i starting from 2 and an array of biases at layer i
        :return: (dictBiases: Dict[int,npt.ArrayLike])
        """
        pass

