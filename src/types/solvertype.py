"""
Author: Ratan Lal
Date : November 18, 2024
"""
import enum


class SolverType(enum.Enum):
    """
     It captures different types of solvers
    """
    # Type 1
    Gurobi = 1

    # Type 2
    SMT = 2

    # Otherwise
    UNKNOWN = -1

