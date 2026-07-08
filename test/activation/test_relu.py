from unittest import TestCase

from src.activation.relu import Relu
from typing import List

import numpy.typing as npt
import numpy as np
from src.set.set import Set
from src.set.intervalmatrix import IntervalMatrix
from src.set.intervalstarset import IntervalStarSet


class TestRelu(TestCase):
    def test_point(self):
        """
        Test Relu of a point
        """
        point: float = -5.5
        pointRelu = Relu.point(point)
        self.assertEqual(0, pointRelu, "Relu point should be zero")

    def test_any_set(self):
        """
        Test Relu of any set
        """
        # Create a set
        # Create an IntervalStarSet
        matLow: npt.ArrayLike = np.array([[3, 2, 4], [2, 1, 3]], dtype=np.float64)
        matHigh: npt.ArrayLike = np.array([[4, 3, 5], [3, 2, 4]], dtype=np.float64)
        IMBasisV: IntervalMatrix = IntervalMatrix(matLow=matLow, matHigh=matHigh)
        matConstraintC: npt.ArrayLike = np.array([[1, 1], [-1, 0], [0, -1], [1, 0], [0, 1]], dtype=np.float64)
        arrayConstraintd: npt.ArrayLike = np.array([1, 0, 0, 1, 1], dtype=np.float64)

        objSet: Set = IntervalStarSet(IMBasisV, matConstraintC, arrayConstraintd)

        listSet: List[Set] = Relu.anySet(objSet)
        np.testing.assert_array_equal(objSet.getIMatBasisV().getMatLow(), listSet[0].getIMatBasisV().getMatLow(),
                                      "Basic Matrices are not matched")
        np.testing.assert_array_equal(objSet.getIMatBasisV().getMatHigh(), listSet[0].getIMatBasisV().getMatHigh(),
                                      "Basic Matrices are not matched")

