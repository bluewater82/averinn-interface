"""
Author: Ratan Lal
Date : November 4, 2023
"""
from typing import Dict, Tuple
from src.gnn.edge import Edge


class Connection:
    """
    The class Connection captures all the edges between two consecutive layers
    """

    def __init__(self, dictEdges: Dict[Tuple[int, int], Edge]):
        """
        Initialize an object of the class Connection
        :param dictEdges: dictionary of mapping of pairs of source and
        target Node instance's id and respective Edge instances that is,
        dictionary {(sourceId, targetId):objEdge,...}
        :type dictEdges: Dict[Tuple[int, int], Edge]
        """
        # Dictionary of mapping between pairs of source and target Node instance's id
        # and respective Edge instances
        self.dictEdges: Dict[Tuple[int, int], Edge] = dictEdges

