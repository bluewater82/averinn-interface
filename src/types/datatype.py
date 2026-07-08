from decimal import Decimal
from enum import Enum

import numpy as np


class DataType(Enum):
    """
    Data type for real value
    """
    #RealType = Decimal
    RealType = np.float64

    def __call__(self, value):
        """
        This method allows the enum member (which holds the Decimal class) to be callable.
        It creates and returns a Decimal instance from the given value.
        """
        return self.value(value)
