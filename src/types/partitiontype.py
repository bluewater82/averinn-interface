"""
Author: Ratan Lal
Date : November 4, 2024
"""

import enum


class PartitionType(enum.Enum):
    """
    It captures different partition types,
    such as fixed, presum, etc.
    """
    # FIXED
    FIXED = 1
    # UNIFORM
    RANDOM = 2
    # PRESUM
    PRESUM = 3
    #UNKNOWN
    UNKNOWN = -1