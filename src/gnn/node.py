"""
Author: Ratan Lal
Date : November 4, 2023
"""
from src.gnn.number import Number
from src.types.activationtype import ActivationType


class Node:
    """
    The class Node captures a neuron of a neural network model
    """

    def __init__(self, enumAction: ActivationType, objNumber: Number, intSize: int, intId: int):
        """
        Initialize an object of the class Node
        :param enumAction: name of activation function
        :type enumAction: ActivationType
        :param objNumber: bias associated with the neuron
        :type objNumber: ANumber
        :param intSize: a natural number for the annotation of a neuron
        :type intSize: int
        :param intId: a unique number associated with each neuron
        :type intId: int
        """
        # Activation function associated with the current Node instance
        self.enumAction: ActivationType = enumAction
        # Bias associated with the current Node instance
        self.bias: Number = objNumber
        # Number of concrete Node instances merged into the current Node instance
        self.intSize: int = intSize
        # Identifier for the current Node instance
        self.intId: int = intId
