"""
Author: Ratan Lal
Date : November 4, 2023
"""

import enum


class LastRelu(enum.Enum):
    """
    It captures whether relu is applicable at the last layer
    """
    # For Yes
    YES = 1

    # for NO
    NO = 0

    # Otherwise
    UNKNOWN = -1
