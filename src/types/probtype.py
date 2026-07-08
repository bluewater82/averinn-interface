"""
Author: Ratan Lal
Date : November 4, 2023
"""
import enum


class ProbType(enum.Enum):
    """
     It captures different types of abstraction
    """
    # Reachability
    REACH = 1

    # SAFETY
    SAFETY = 2

    # Otherwise
    UNKNOWN = -1

