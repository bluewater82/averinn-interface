"""
Author: Ratan Lal
Date : November 18, 2024
"""
import enum


class TechniqueType(enum.Enum):
    """
     It captures different types of abstraction
    """
    # Type 1
    MILP = 1
    #Type 2
    PROPAGATION = 2
    # Otherwise
    UNKNOWN = -1

