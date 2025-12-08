"""
Author: Ratan Lal
Date : November 18, 2024
"""
from abc import ABC
from typing import List, Tuple, Dict
import numpy as np
import numpy.typing as npt
from gurobipy import Model, Var, GRB

from src.set.intervalmatrix import IntervalMatrix
from src.set.set import Set
from src.types.datatype import DataType
from src.types.sign import Sign
import itertools


class Box(Set, ABC):
    """
    The class Box captures multidimensional rectangular set
    """

    def __init__(self, arrayLow: npt.ArrayLike, arrayHigh: npt.ArrayLike):
        """
        Initialize an instance of the class Box
        :param arrayLow: one dimensional array for the lower end point
        :type arrayLow: npt.ArrayLike
        :param arrayHigh: one dimensional array for the upper end point
        :type arrayHigh: npt.ArrayLike
        """
        self.__arrayLow__: npt.ArrayLike = arrayLow
        self.__arrayHigh__: npt.ArrayLike = arrayHigh

    ############################################
    ########## Method for Attributes ###########
    ############################################
    def getArrayLow(self) -> npt.ArrayLike:
        """
        Returns the lower end point of the Box instance
        :return: (arrayLow -> npt.ArrayLike)
        One dimensional array for the lower end point fo the box
        """
        return self.__arrayLow__

    def getArrayHigh(self) -> npt.ArrayLike:
        """
        Returns the upper end point of the Box instance
        :return: (arrayHigh -> npt.ArrayLike)
        One dimensional array for the upper end point fo the box
        """
        return self.__arrayHigh__

    ############################################
    ##########      Common Methods     #########
    ############################################
    def getDimension(self) -> int:
        """
        Returns the dimension of the Box instance
        :return: (intDim -> int)
        Dimension of the Box instance
        """
        # Compute dimension of the Box instance
        intDim: int = len(self.__arrayLow__)
        return intDim

    def linearMap(self, matLow: npt.ArrayLike, matHigh: npt.ArrayLike = None) -> Set:
        """
        Linear mapping of a set (matHigh will be used for IntervalMatrix)
        :param matLow: a two-dimensional array for a weight matrix
        :type matLow: npt.ArrayLike
        :param matHigh: a two-dimensional array for a weight matrix
        :type matHigh: npt.ArrayLike
        :return: (objSetLinear -> Set) an instance of Set representing WX,
                where X is a self
        """
        # WX
        arrayWXLow: npt.ArrayLike = np.array(np.matmul(matLow, self.__arrayLow__))
        arrayWXHigh: npt.ArrayLike = np.array(np.matmul(matLow, self.__arrayHigh__))

        # Linear map of self
        objSetLinear: Set = Box(arrayWXLow, arrayWXHigh)

        # Return Linear set
        return objSetLinear

    def affineMap(self, matLow: npt.ArrayLike, arrayLow: npt.ArrayLike,
                  matHigh: npt.ArrayLike = None, arrayHigh: npt.ArrayLike = None) -> 'Set':
        """
        Affine mapping of a set (matHigh and arrayHigh will be used for IntervalMatrix)
        :param matLow: a two-dimensional array for a weight matrix
        :type matLow: npt.ArrayLike
        :param matHigh: a two-dimensional array for a weight matrix
        :type matHigh: npt.ArrayLike
        :param arrayLow: a one dimensional array
        :type arrayLow: npt.ArrayLike
        :param arrayHigh: a one dimensional array
        :type arrayHigh: npt.ArrayLike
        :return: (objSetAffine -> Set) an instance of Set representing WX+b,
        """
        # WX
        arrayWXLow: npt.ArrayLike = np.array(np.matmul(matLow, self.__arrayLow__))
        arrayWXHigh: npt.ArrayLike = np.array(np.matmul(matLow, self.__arrayHigh__))

        # WX + b
        arrayAffineLow = arrayWXLow + arrayLow
        arrayAffineHigh = arrayWXHigh + arrayLow

        # Affine map of self
        objSetAffine: Set = Box(arrayAffineLow, arrayAffineHigh)

        # Return Affine set
        return objSetAffine

    def minkowskiSum(self, objSet: Set) -> Set:
        """
        Construct an instance of Box representing minkowski sum of two boxes,
        that is, self and objSet
        :param objSet: an instance of the Set class
        :type objSet: Set
        :return: (objSetMWSum -> Box) an instance of Set for minkowski sum
        """
        intDim: int = self.getDimension()
        MWArrayLow: npt.ArrayLike = np.array([(self.__arrayLow__[i] + objSet.getArrayLow()[i]) for i in range(intDim)])
        MWArrayHigh: npt.ArrayLike = np.array([(self.__arrayHigh__[i] + objSet.getArrayHigh()[i]) for i in range(intDim)])

        # Minkowski sum
        objSetMWSum: Set = Box(MWArrayLow, MWArrayHigh)

        # Return Minkowski sum
        return objSetMWSum

    def convexHull(self, objSet: Set) -> Set:
        """
        Construct an instance of Box representing convex hull of two boxes,
        that is, self and objBox
        :param objSet: an instance of Set class
        :type objSet: Set
        :return: (objSetCH -> Set) an instance of Set for convex hull
        """
        intDim: int = self.getDimension()
        arrayCHLow = np.array([min(self.__arrayLow__[i], objSet.getArrayLow()[i])
                              for i in range(intDim)])
        arrayCHHigh = np.array([max(self.__arrayHigh__[i], objSet.getArrayHigh()[i])
                               for i in range(intDim)])

        # Convex hull
        objSetCH: Set = Box(arrayCHLow, arrayCHHigh)

        # Return convex hull
        return objSetCH

    def getSameSignPartition(self) -> List[Set]:
        """
        Partition a set in such a way that for each sub set,
        values in each dimension will have the same sign
        :return: (listSets -> List[Box]) list of subsets
        """
        # Get sign for each dimension (POS, NEG, BOTH)
        listSign: List[Sign] = self.getSign()
        # Create a List of Lists, where each sublist have either a numpy array for
        # POS and NEG sign or two numpy arrays for BOTH sign
        listOfListofArrays: List[List[npt.ArrayLike]] = []
        for i in range(len(listSign)):
            # Create a list of numpy arrays
            tempListOfArrays: List[npt.ArrayLike] = []
            if listSign[i] == Sign.POS or listSign[i] == Sign.NEG:
                tempListOfArrays.append(np.array([self.__arrayLow__[i], self.__arrayHigh__[i]]))
            else:
                tempListOfArrays.append(np.array([self.__arrayLow__[i], DataType.RealType(0.0)]))
                tempListOfArrays.append(np.array([DataType.RealType(0.0), self.__arrayHigh__[i]]))
            listOfListofArrays.append(tempListOfArrays)

        # Perform cartesian product among each element of the list
        cartesianProduct = itertools.product(*listOfListofArrays)
        listSets: List[Set] = []
        for idx, item in enumerate(cartesianProduct):
            arrayLow: npt.ArrayLike = np.array([DataType.RealType(arr[0]) for arr in item], dtype=object)
            arrayHigh: npt.ArrayLike = np.array([DataType.RealType(arr[1]) for arr in item], dtype=object)
            listSets.append(Box(arrayLow, arrayHigh))

        return listSets

    def getSign(self) -> List[Sign]:
        """
        extract sign of each dimension
        :return: (listSign -> List[Sign])
        """
        intDim: int = self.getDimension()
        listSign: List[Sign] = []
        for i in range(intDim):
            # Find the sign of index i
            enumSign = self.__getSignByIndex__(i)
            listSign.append(enumSign)

        # Return sign for all the dimensions
        return listSign

    def toAbsolute(self) -> npt.ArrayLike:
        """
        Returns the absolute value of the set
        :return: (arrayPoint -> npt.ArrayLike)
        """
        arrayPoint: npt.ArrayLike = np.array([max(abs(self.__arrayLow__[i]), abs(self.__arrayHigh__[i]))
                                              for i in range(self.getDimension())], dtype=object)

        return arrayPoint

    def isEmpty(self) -> bool:
        """
        Checks if the set is empty
        :return: (status -> bool)
        """
        intDim: int = self.getDimension()
        for i in range(intDim):
            if self.__arrayLow__[i] > self.__arrayHigh__[i]:
                return True
        return False

    def getModelVars(self) -> Tuple[Model, Dict[int, Dict[int, Var]]]:
        """"
        Get encoding of a set and dictionary of variables
        :return: (ModelVars -> Tuple[Model, Dict[Dict[int, Var]]])
        """
        # Encode Interval Star Set in Gurobi
        grbModel, dictVars = self.__encode__()

        # Return Model and DictVars
        return grbModel, dictVars

    def getRange(self) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        """
        Return the range of the Box
        """
        rangeBox: Tuple[npt.ArrayLike, npt.ArrayLike] = (self.__arrayLow__, self.__arrayHigh__)
        return rangeBox

    def display(self) -> str:
        """
        Display lower and upper bounds of the Box
        :return: None
        """
        strBox = "L:         "+str(self.__arrayLow__) + "\n"
        strBox += "U:       "+str(self.__arrayHigh__) + "\n"

        return strBox

    ############################################
    ########## Private Methods  ################
    ############################################

    def __getSignByIndex__(self, intIndex: int) -> Sign:
        """
        Find the type of values at intIndex that self has
        :param intIndex: index of the variable
        :type intIndex: int
        :return: (enumSign -> Sign) Sign is an enumeration class
        """
        sign: Sign = Sign.NONE
        if self.__arrayLow__[intIndex] >= 0.0:
            sign = Sign.POS
        elif self.__arrayHigh__[intIndex] <= 0.0:
            sign = Sign.NEG
        else:
            sign = Sign.BOTH

        return sign

    def __encode__(self) -> Tuple[Model, Dict[int, Dict[int, Var]]]:
        """"
        Encode Interval Star Set into Gurobi format
        :return: (grbModel -> Model)
        """
        # Create state variables
        intDim: int = self.getDimension()
        listStateVars: List[str] = ['x_' + str(i) for i in range(intDim)]
        # Create state variables in gurobi
        grbModel: Model = Model()
        grbStateVars = []
        for i in range(intDim):
            grbStateVars.append(grbModel.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name=listStateVars[i]))

        for i in range(intDim):
            grbModel.addConstr(grbStateVars[i] >= np.float64(self.__arrayLow__[i]))
            grbModel.addConstr(grbStateVars[i] <= np.float64(self.__arrayHigh__[i]))

        grbModel.update()
        # Create dictionary for solver
        dictVars: Dict[int, Dict[int, Var]] = dict()
        dictVars[1] = dict()
        for i in range(1, intDim + 1, 1):
            dictVars[1][i] = grbStateVars[i - 1]

        return grbModel, dictVars

    ############################################
    ###### Unused Methods from other sets  #####
    ############################################
    def getIMatBasisV(self) -> IntervalMatrix:
        """
        Return Basis interval matrix
        :return: (objIMBasisV -> IntervalMatrix )
        """
        # The following line are for just returning (not for implementation)
        matLow: npt.ArrayLike = np.array([[]], dtype=object)
        matHigh: npt.ArrayLike = np.array([[]], dtype=object)
        objIMBasisV: IntervalMatrix = IntervalMatrix(matLow, matHigh)
        return objIMBasisV

    def getMatConstraintC(self) -> npt.ArrayLike:
        """
        Return matrix C in the predicate constraints C * alpha <= d
        :return: (matConstraintC -> npt.ArrayLike )
        """
        # The following line are for just returning (not for implementation)
        matConstraintC: npt.ArrayLike = np.array([[]], dtype=object)
        return matConstraintC

    def getArrayConstraintd(self) -> npt.ArrayLike:
        """
        Return array d in the predicate constraints C * alpha <= d
        :return: (arrayConstraintd -> npt.ArrayLike )
        """
        # The following line are for just returning (not for implementation)
        arrayConstraintd: npt.ArrayLike = np.array([], dtype=object)
        return arrayConstraintd

    def getArrayPredicateLow(self) -> npt.ArrayLike:
        """
        Return lower bound  of alpha in the predicate C * alpha <= d
        :return: (arrayPredicateLow -> npt.ArrayLike)
        """
        # The following line are for just returning (not for implementation)
        arrayPredicateLow: npt.ArrayLike = np.array([], dtype=object)
        return arrayPredicateLow

    def getArrayPredicateHigh(self) -> npt.ArrayLike:
        """
        Return upper bound  of alpha in the predicate C * alpha <= d
        :return: (arrayPredicateHigh -> npt.ArrayLike)
        """
        # The following line are for just returning (not for implementation)
        arrayPredicateHigh: npt.ArrayLike = np.array([], dtype=object)
        return arrayPredicateHigh

    def getArrayStateLow(self) -> npt.ArrayLike:
        """
        Return lower bound  of state x in x = c+a[1]*v[1] + a[2]*v[2]+ a[n]*v[n]
        :return: (arrayStateLow -> npt.ArrayLike)
        """
        # The following line are for just returning (not for implementation)
        arrayStateLow: npt.ArrayLike = np.array([], dtype=object)
        return arrayStateLow

    def getArrayStateHigh(self) -> npt.ArrayLike:
        """
        Return upper bound of state x in  x = c+a[1]*v[1] + a[2]*v[2]+ a[n]*v[n]
        :return: (arrayStateHigh -> npt.ArrayLike)
        """
        # The following line are for just returning (not for implementation)
        arrayStateHigh: npt.ArrayLike = np.array([], dtype=object)
        return arrayStateHigh

    def getNumOfPredVars(self) -> int:
        """
        Return the number of predicate variables
        :return: (intPredVars -> int)
        """
        # The following line are for just returning (not for implementation)
        numOfPredicateVars: int = 0
        return numOfPredicateVars

    def getNumOfPredicates(self) -> int:
        """
        Return the number of predicates
        :return: (intPredicates -> int)
        """
        # The following line are for just returning (not for implementation)
        numOfPredicates: int = 0
        return numOfPredicates

    def intersectPHSByIndex(self, intIndex: int) -> Set:
        """
        Compute intersection between self and a specific half space by index
        intersection(self x[index]>=0)
        :param intIndex: index of the state variables
        :type intIndex: int
        :return: (objSetIPHS-> IntervalStarSet) intersection between self and x[intIndex]>=0
        """
        arrayLow: npt.ArrayLike = np.array([[]], dtype=object)
        arrayHigh: npt.ArrayLike = np.array([[]], dtype=object)
        objSetIPHS: Set = Box(arrayLow, arrayHigh)
        return objSetIPHS

    def intersectNHSByIndex(self, intIndex: int) -> Set:
        """
        Compute intersection between self and a specific half space by index
        intersection(self x[index]<=0)
        :param intIndex: index of the state variables
        :type intIndex: int
        :return: (objSetINHS -> IntervalStarSet) intersection between self and x[intIndex]<=0
        """
        arrayLow: npt.ArrayLike = np.array([[]], dtype=object)
        arrayHigh: npt.ArrayLike = np.array([[]], dtype=object)
        objSetINHS: Set = Box(arrayLow, arrayHigh)
        return objSetINHS

    ############################################
    ########## Unused Common Methods ###########
    ############################################
