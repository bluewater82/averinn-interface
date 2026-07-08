"""
Author: Ratan Lal
Date : November 4, 2023
"""
from typing import Dict
from src.gnn.node import Node


class Layer:
    """
    The class Layer captures all the nodes in a layer
    """

    def __init__(self, dictNodes: Dict[int, Node]):
        """
        Initialize an object of the class Layer
        :param dictNodes: dictionary of mapping between Node instance's ids and
        Node instances at a layer, that is, {id1: objNode1, ..., idn: objNoden}
        :type dictNodes: Dict[int, Node]
        """
        # Dictionary of INode instance ids and INode instances
        self.dictNodes: Dict[int, Node] = dictNodes
        # Number of INode instance in the current layer
        self.intNumNodes: int = len(dictNodes)
