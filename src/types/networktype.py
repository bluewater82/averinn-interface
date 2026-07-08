"""
Author: Ratan Lal
Date : November 4, 2023
"""

import enum


class NetworkType(enum.Enum):
    """
    It captures different network types,
    such as Neural Network, Interval Neural Network, etc.
    """
    # For Neural Network (NN)
    STANDARD = 1
    # For interval neural network (INN)
    INTERVAL = 2
    # For not defined networks
    UNKNOWN = -1