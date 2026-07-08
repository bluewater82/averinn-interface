"""
Author: Ratan Lal
Date : November 4, 2023
"""
from src.gnn.number import Number
from src.gnn.node import Node


class Edge:
    """
    The class Edge captures an edge between two neurons in consecutive layers
    """

    def __init__(self, nodeSource: Node, nodeTarget: Node, objNumber: Number):
        """"
        Initialize an object of the class Edge
        :param nodeSource: an instance of Node class for the source neuron
        :type nodeSource: Node class
        :param nodeTarget: an instance of Node class for the target neuron
        :type nodeTarget: Node class
        :param objNumber: weight between source and target neurons
        :type objNumber: ANUmber
        """
        # An instance of Node class for the source neuron
        self.nodeSource: Node = nodeSource
        # An instance of Node class for the target neuron
        self.nodeTarget: Node = nodeTarget
        # Weight between source and target neurons
        self.weight: Number = objNumber
