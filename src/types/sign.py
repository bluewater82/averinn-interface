"""
Author: Ratan Lal
Date : November 18, 2024
"""
import enum


class Sign(enum.Enum):
    """
    The class Sign is an enumeration
    """
    # For positive values
    POS = 1

    # For negative values
    NEG = -1

    # For both positive and negative
    BOTH = 0

    # Otherwise
    NONE = 2