"""
Author: Ratan Lal
Date : January 30, 2025
"""
from abc import ABCMeta, abstractmethod
from typing import Dict
from gurobipy import Var
from src.set.set import Set


class Solver(metaclass=ABCMeta):
    """
    Abstract base class for different solvers
    """
    @abstractmethod
    def satisfy(self) -> bool:
        """
        Check the satisfiability of a set of constraints
        :return: (status -> bool)
        True if satisfies all constraints, False otherwise
        """
        pass

    def getInstance(self) -> Dict[int, Dict[int, float]]:
        """
        Extract a satisfiable instance of the model
        :return: (dictOutput -> Dict[int, Dict[int, float]])
        the satisfiable instance of the model
        """
        pass

    @abstractmethod
    def outputRange(self) -> Set:
        """
        Compute range of a set of constraints
        :return: (objSet -> Set)
        """
        pass

