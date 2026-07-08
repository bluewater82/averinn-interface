"""
Author: Ratan Lal
Date : November 4, 2023
"""
import enum


class ActivationType(enum.Enum):
    """
     It captures different activation function types
    """
    # for relu activation function
    RELU = 1
    # for sigmoid activation function
    SIGMOID = 2
    # for tanh activation function
    TANH = 3
    # for not defined function
    UNKNOWN = 0
