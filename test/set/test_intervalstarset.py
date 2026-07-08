from unittest import TestCase

import numpy as np
import numpy.typing as npt
from src.set.intervalmatrix import IntervalMatrix
from src.set.intervalstarset import IntervalStarSet
from src.set.set import Set


class TestIntervalStarSet(TestCase):
    def test_getMatBasisV(self):
        """
        Test getMatBasisV
        """
        # Create an IntervalStarSet
        matLow: npt.ArrayLike = np.array([[3, 2, 4], [2, 1, 3]], dtype=np.float64)
        matHigh: npt.ArrayLike = np.array([[4, 3, 5], [3, 2, 4]], dtype=np.float64)
        IMBasisV: IntervalMatrix = IntervalMatrix(matLow=matLow, matHigh=matHigh)
        matConstraintC: npt.ArrayLike = np.array([[1, 1], [-1, 0], [0, -1], [1, 0], [0, 1]], dtype=np.float64)
        arrayConstraintd: npt.ArrayLike = np.array([1, 0, 0, 1, 1], dtype=np.float64)

        objSet: Set = IntervalStarSet(IMBasisV, matConstraintC, arrayConstraintd)

        np.testing.assert_array_equal(IMBasisV.getMatLow(), objSet.getMatBasisV().getMatLow(),
                                      "Basic Matrices are not matched")
        np.testing.assert_array_equal(IMBasisV.getMatHigh(), objSet.getMatBasisV().getMatHigh(),
                                      "Basic Matrices are not matched")

    def test_getMatConstraintC(self):
        """
        Test getMatConstraintC
        """
        # Create an IntervalStarSet
        matLow: npt.ArrayLike = np.array([[3, 2, 4], [2, 1, 3]], dtype=np.float64)
        matHigh: npt.ArrayLike = np.array([[4, 3, 5], [3, 2, 4]], dtype=np.float64)
        IMBasisV: IntervalMatrix = IntervalMatrix(matLow=matLow, matHigh=matHigh)
        matConstraintC: npt.ArrayLike = np.array([[1, 1], [-1, 0], [0, -1], [1, 0], [0, 1]], dtype=np.float64)
        arrayConstraintd: npt.ArrayLike = np.array([1, 0, 0, 1, 1], dtype=np.float64)

        objSet: Set = IntervalStarSet(IMBasisV, matConstraintC, arrayConstraintd)

        np.testing.assert_array_equal(matConstraintC, objSet.getMatConstraintC(),
                                      "Basic Matrices are not matched")

    def test_getArrayConstraintd(self):
        """
        Test arrayConstraintd
        """
        # Create an IntervalStarSet
        matLow: npt.ArrayLike = np.array([[3, 2, 4], [2, 1, 3]], dtype=np.float64)
        matHigh: npt.ArrayLike = np.array([[4, 3, 5], [3, 2, 4]], dtype=np.float64)
        IMBasisV: IntervalMatrix = IntervalMatrix(matLow=matLow, matHigh=matHigh)
        matConstraintC: npt.ArrayLike = np.array([[1, 1], [-1, 0], [0, -1], [1, 0], [0, 1]], dtype=np.float64)
        arrayConstraintd: npt.ArrayLike = np.array([1, 0, 0, 1, 1], dtype=np.float64)

        objSet: Set = IntervalStarSet(IMBasisV, matConstraintC, arrayConstraintd)

        np.testing.assert_array_equal(arrayConstraintd, objSet.getArrayConstraintd(),
                                      "Basic Matrices are not matched")

    def test_getNumOfPredVars(self):
        """
        Test number of predicate variables
        """
        # Create an IntervalStarSet
        matLow: npt.ArrayLike = np.array([[3, 2, 4], [2, 1, 3]], dtype=np.float64)
        matHigh: npt.ArrayLike = np.array([[4, 3, 5], [3, 2, 4]], dtype=np.float64)
        IMBasisV: IntervalMatrix = IntervalMatrix(matLow=matLow, matHigh=matHigh)
        matConstraintC: npt.ArrayLike = np.array([[1, 1], [-1, 0], [0, -1], [1, 0], [0, 1]], dtype=np.float64)
        arrayConstraintd: npt.ArrayLike = np.array([1, 0, 0, 1, 1], dtype=np.float64)

        objSet: Set = IntervalStarSet(IMBasisV, matConstraintC, arrayConstraintd)

        self.assertEqual(2, objSet.getNumOfPredVars(), "Number of predicate variables are not matched")

    def test_getNumOfPredicates(self):
        """
        Test number of predicates
        """
        # Create an IntervalStarSet
        matLow: npt.ArrayLike = np.array([[3, 2, 4], [2, 1, 3]], dtype=np.float64)
        matHigh: npt.ArrayLike = np.array([[4, 3, 5], [3, 2, 4]], dtype=np.float64)
        IMBasisV: IntervalMatrix = IntervalMatrix(matLow=matLow, matHigh=matHigh)
        matConstraintC: npt.ArrayLike = np.array([[1, 1], [-1, 0], [0, -1], [1, 0], [0, 1]], dtype=np.float64)
        arrayConstraintd: npt.ArrayLike = np.array([1, 0, 0, 1, 1], dtype=np.float64)

        objSet: Set = IntervalStarSet(IMBasisV, matConstraintC, arrayConstraintd)
        self.assertEqual(5, objSet.getNumOfPredicates(), "Number of predicates are not matched")

    def test_linearMap(self):
        """
        Test linear map
        """
        # Create an IntervalStarSet
        matLow: npt.ArrayLike = np.array([[3, 2, 4], [2, 1, 3]], dtype=np.float64)
        matHigh: npt.ArrayLike = np.array([[4, 3, 5], [3, 2, 4]], dtype=np.float64)
        IMBasisV: IntervalMatrix = IntervalMatrix(matLow=matLow, matHigh=matHigh)
        matConstraintC: npt.ArrayLike = np.array([[1, 1], [-1, 0], [0, -1], [1, 0], [0, 1]], dtype=np.float64)
        arrayConstraintd: npt.ArrayLike = np.array([1, 0, 0, 1, 1], dtype=np.float64)

        objSet: Set = IntervalStarSet(IMBasisV, matConstraintC, arrayConstraintd)

        # Weight Matrix
        WeightLow: npt.ArrayLike = np.array([[1, 0], [0, 1]], dtype=np.float64)
        WeightHigh: npt.ArrayLike = np.array([[1, 0], [0, 1]], dtype=np.float64)

        # Linear map
        objSetLinear: Set = objSet.linearMap(WeightLow, WeightHigh)
        print(objSetLinear.getMatBasisV().getMatLow())
        print(objSetLinear.getMatBasisV().getMatHigh())
        np.testing.assert_array_equal(IMBasisV.getMatLow(), objSetLinear.getMatBasisV().getMatLow(),
                                      "Basic Matrices are not matched")
        np.testing.assert_array_equal(IMBasisV.getMatHigh(), objSetLinear.getMatBasisV().getMatHigh(),
                                      "Basic Matrices are not matched")

        np.testing.assert_array_equal(matConstraintC, objSetLinear.getMatConstraintC(),
                                      "Basic Matrices are not matched")

        np.testing.assert_array_equal(arrayConstraintd, objSetLinear.getArrayConstraintd(),
                                      "Basic Matrices are not matched")

    def test_affineMap(self):
        """
        Test affine map
        """
        # Create an IntervalStarSet
        matLow: npt.ArrayLike = np.array([[3, 2, 4], [2, 1, 3]], dtype=np.float64)
        matHigh: npt.ArrayLike = np.array([[4, 3, 5], [3, 2, 4]], dtype=np.float64)
        IMBasisV: IntervalMatrix = IntervalMatrix(matLow=matLow, matHigh=matHigh)
        matConstraintC: npt.ArrayLike = np.array([[1, 1], [-1, 0], [0, -1], [1, 0], [0, 1]], dtype=np.float64)
        arrayConstraintd: npt.ArrayLike = np.array([1, 0, 0, 1, 1], dtype=np.float64)

        objSet: Set = IntervalStarSet(IMBasisV, matConstraintC, arrayConstraintd)

        # Weight Matrix
        WeightLow: npt.ArrayLike = np.array([[1, 0], [0, 1]], dtype=np.float64)
        WeightHigh: npt.ArrayLike = np.array([[1, 0], [0, 1]], dtype=np.float64)

        # bias array
        arrayLow: npt.ArrayLike = np.array([1, 1], dtype=np.float64)
        arrayHigh: npt.ArrayLike = np.array([1, 1], dtype=np.float64)

        # Affine map
        objSetAffine: Set = objSet.affineMap(WeightLow, arrayLow, WeightHigh, arrayHigh)
        # Updated expected IMBasisV
        matLow: npt.ArrayLike = np.array([[4, 2, 4], [3, 1, 3]], dtype=np.float64)
        matHigh: npt.ArrayLike = np.array([[5, 3, 5], [4, 2, 4]], dtype=np.float64)
        IMBasisV: IntervalMatrix = IntervalMatrix(matLow=matLow, matHigh=matHigh)
        np.testing.assert_array_equal(IMBasisV.getMatLow(), objSetAffine.getMatBasisV().getMatLow(),
                                      "Basic Matrices are not matched")
        np.testing.assert_array_equal(IMBasisV.getMatHigh(), objSetAffine.getMatBasisV().getMatHigh(),
                                      "Basic Matrices are not matched")

        np.testing.assert_array_equal(matConstraintC, objSetAffine.getMatConstraintC(),
                                      "Basic Matrices are not matched")

        np.testing.assert_array_equal(arrayConstraintd, objSetAffine.getArrayConstraintd(),
                                      "Basic Matrices are not matched")

    def test_intersectPHSByIndex(self):
        """
        Test intersection with positive half space
        """
        # Create an IntervalStarSet
        matLow: npt.ArrayLike = np.array([[3, 2, 4], [2, 1, 3]], dtype=np.float64)
        matHigh: npt.ArrayLike = np.array([[4, 3, 5], [3, 2, 4]], dtype=np.float64)
        IMBasisV: IntervalMatrix = IntervalMatrix(matLow=matLow, matHigh=matHigh)
        matConstraintC: npt.ArrayLike = np.array([[1, 1], [-1, 0], [0, -1], [1, 0], [0, 1]], dtype=np.float64)
        arrayConstraintd: npt.ArrayLike = np.array([1, 0, 0, 1, 1], dtype=np.float64)

        objSet: Set = IntervalStarSet(IMBasisV, matConstraintC, arrayConstraintd)

        # Index for intersection with positive half space
        intIndex: int = 0

        # Intersect with half space x[intIndex] >= 0
        objSetIPHS: Set = objSet.intersectPHSByIndex(intIndex)
        # Updated expected matConstraintC and arrayConstraintd
        matConstraintC: npt.ArrayLike = np.array([[1, 1], [-1, 0], [0, -1], [1, 0], [0, 1], [-3, -5]], dtype=np.float64)
        arrayConstraintd: npt.ArrayLike = np.array([1, 0, 0, 1, 1, 4], dtype=np.float64)

        np.testing.assert_array_equal(IMBasisV.getMatLow(), objSetIPHS.getMatBasisV().getMatLow(),
                                      "Basic Matrices are not matched")
        np.testing.assert_array_equal(IMBasisV.getMatHigh(), objSetIPHS.getMatBasisV().getMatHigh(),
                                      "Basic Matrices are not matched")
        np.testing.assert_array_equal(matConstraintC, objSetIPHS.getMatConstraintC(),
                                      "Basic Matrices are not matched")
        np.testing.assert_array_equal(arrayConstraintd, objSetIPHS.getArrayConstraintd(),
                                      "Basic Matrices are not matched")

    def test_intersectNHSByIndex(self):
        """
        Test intersection with negative half space
        """
        # Create an IntervalStarSet
        matLow: npt.ArrayLike = np.array([[3, 2, 4], [2, 1, 3]], dtype=np.float64)
        matHigh: npt.ArrayLike = np.array([[4, 3, 5], [3, 2, 4]], dtype=np.float64)
        IMBasisV: IntervalMatrix = IntervalMatrix(matLow=matLow, matHigh=matHigh)
        matConstraintC: npt.ArrayLike = np.array([[1, 1], [-1, 0], [0, -1], [1, 0], [0, 1]], dtype=np.float64)
        arrayConstraintd: npt.ArrayLike = np.array([1, 0, 0, 1, 1], dtype=np.float64)

        objSet: Set = IntervalStarSet(IMBasisV, matConstraintC, arrayConstraintd)

        # Index for intersection with positive half space
        intIndex: int = 0

        # Intersect with half space x[intIndex] <= 0
        objSetIPHS: Set = objSet.intersectNHSByIndex(intIndex)
        # Updated expected matConstraintC and arrayConstraintd
        matConstraintC: npt.ArrayLike = np.array([[1, 1], [-1, 0], [0, -1], [1, 0], [0, 1], [2, 4]], dtype=np.float64)
        arrayConstraintd: npt.ArrayLike = np.array([1, 0, 0, 1, 1, -3], dtype=np.float64)

        np.testing.assert_array_equal(IMBasisV.getMatLow(), objSetIPHS.getMatBasisV().getMatLow(),
                                      "Basic Matrices are not matched")
        np.testing.assert_array_equal(IMBasisV.getMatHigh(), objSetIPHS.getMatBasisV().getMatHigh(),
                                      "Basic Matrices are not matched")

        np.testing.assert_array_equal(matConstraintC, objSetIPHS.getMatConstraintC(),
                                      "Basic Matrices are not matched")

        np.testing.assert_array_equal(arrayConstraintd, objSetIPHS.getArrayConstraintd(),
                                      "Basic Matrices are not matched")

    def test_minkowskiSum(self):
        """
        Test Minkowski Sum
        """
        # Create first IntervalStarSet
        matLow: npt.ArrayLike = np.array([[3, 2, 4], [2, 1, 3]], dtype=np.float64)
        matHigh: npt.ArrayLike = np.array([[4, 3, 5], [3, 2, 4]], dtype=np.float64)
        IMBasisV: IntervalMatrix = IntervalMatrix(matLow=matLow, matHigh=matHigh)
        matConstraintC: npt.ArrayLike = np.array([[1, 1], [-1, 0], [0, -1], [1, 0], [0, 1]], dtype=np.float64)
        arrayConstraintd: npt.ArrayLike = np.array([1, 0, 0, 1, 1], dtype=np.float64)

        objSetOne: Set = IntervalStarSet(IMBasisV, matConstraintC, arrayConstraintd)

        # Create second IntervalStarSet
        matLow: npt.ArrayLike = np.array([[4, 3, 1], [3, 3, 1]], dtype=np.float64)
        matHigh: npt.ArrayLike = np.array([[8, 5, 4], [5, 5, 4]], dtype=np.float64)
        IMBasisV: IntervalMatrix = IntervalMatrix(matLow=matLow, matHigh=matHigh)
        matConstraintC: npt.ArrayLike = np.array([[1, 1], [-1, 0], [0, -1], [1, 0], [0, 1]], dtype=np.float64)
        arrayConstraintd: npt.ArrayLike = np.array([1, 0, 0, 1, 1], dtype=np.float64)

        objSetTwo: Set = IntervalStarSet(IMBasisV, matConstraintC, arrayConstraintd)
        # Compute Minkowski Sum
        objSetMWSum = objSetOne.minkowskiSum(objSetTwo)
        # Expected ISS
        matLowMWSum: npt.ArrayLike = np.array([[7, 2, 4, 3, 1], [5, 1, 3, 3, 1]], dtype=np.float64)
        matHighMWSum: npt.ArrayLike = np.array([[12, 3, 5, 5, 4], [8, 2, 4, 5, 4]], dtype=np.float64)
        IMBasisVMWSum: IntervalMatrix = IntervalMatrix(matLow=matLowMWSum, matHigh=matHighMWSum)
        matConstraintCMWSum: npt.ArrayLike = np.array([[1, 1, 0, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0],
                                                  [0,0, 1, 1], [0, 0, -1, 0], [0, 0, 0, -1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
        arrayConstraintdMWSum: npt.ArrayLike = np.array([1, 0, 0, 1, 1, 1, 0, 0, 1, 1], dtype=np.float64)

        np.testing.assert_array_equal(IMBasisVMWSum.getMatLow(), objSetMWSum.getMatBasisV().getMatLow(),
                                      "Basic Matrices are not matched")
        np.testing.assert_array_equal(IMBasisVMWSum.getMatHigh(), objSetMWSum.getMatBasisV().getMatHigh(),
                                      "Basic Matrices are not matched")

        np.testing.assert_array_equal(matConstraintCMWSum, objSetMWSum.getMatConstraintC(),
                                      "Basic Matrices are not matched")

        np.testing.assert_array_equal(arrayConstraintdMWSum, objSetMWSum.getArrayConstraintd(),
                                      "Basic Matrices are not matched")

    def test_empty(self):
        """
        Test empty set
        """
        # Create first IntervalStarSet
        matLow: npt.ArrayLike = np.array([[3, 2, 4], [2, 1, 3]], dtype=np.float64)
        matHigh: npt.ArrayLike = np.array([[4, 3, 5], [3, 2, 4]], dtype=np.float64)
        IMBasisV: IntervalMatrix = IntervalMatrix(matLow=matLow, matHigh=matHigh)
        matConstraintC: npt.ArrayLike = np.array([[1, 1], [-1, 0], [0, -1], [1, 0], [0, 1]], dtype=np.float64)
        arrayConstraintd: npt.ArrayLike = np.array([1, 0, 0, 1, 1], dtype=np.float64)

        objSetOne: Set = IntervalStarSet(IMBasisV, matConstraintC, arrayConstraintd)

        isEmpty: bool = objSetOne.isEmpty()

        self.assertEqual(False, isEmpty, "Empty set is empty")

    def test_getRange(self):
        """
        Test getRange
        """
        # Create first IntervalStarSet
        matLow: npt.ArrayLike = np.array([[3, 2, 4], [2, 1, 3]], dtype=np.float64)
        matHigh: npt.ArrayLike = np.array([[4, 3, 5], [3, 2, 4]], dtype=np.float64)
        IMBasisV: IntervalMatrix = IntervalMatrix(matLow=matLow, matHigh=matHigh)
        matConstraintC: npt.ArrayLike = np.array([[1, 1], [-1, 0], [0, -1], [1, 0], [0, 1]], dtype=np.float64)
        arrayConstraintd: npt.ArrayLike = np.array([1, 0, 0, 1, 1], dtype=np.float64)

        objSetOne: Set = IntervalStarSet(IMBasisV, matConstraintC, arrayConstraintd)
        rangeSet = objSetOne.getRange()
        # Expected lower and upper bound
        lowRange: npt.ArrayLike = np.array([3, 2], dtype=np.float64)
        highRange: npt.ArrayLike = np.array([9, 7], dtype=np.float64)
        np.testing.assert_array_equal(lowRange, rangeSet[0], "Lower range are not matched")
        np.testing.assert_array_equal(highRange, rangeSet[1], "Upper range are not matched")