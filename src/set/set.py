"""
Author: Ratan Lal
Date : January 28, 2025
"""
from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Dict
import numpy.typing as npt
from gurobipy import Model, Var

from src.set.intervalmatrix import IntervalMatrix
from src.types.sign import Sign


class Set(metaclass=ABCMeta):
    """
    Abstract class for capturing different classes of sets
    """

    ###################################################
    ##### Methods related to attribute of the Box #####
    ###################################################
    @abstractmethod
    def getArrayLow(self) -> npt.ArrayLike:
        """
        Returns the lower bound of the set
        :return: (arrayLow -> npt.ArrayLike)
        lower bound of the Box set
        """
        pass

    @abstractmethod
    def getArrayHigh(self) -> npt.ArrayLike:
        """
        Returns the lower bound of the set
        :return: (arrayHigh -> npt.ArrayLike)
        upper bound of the set
        """
        pass

    ###########################################################################
    ##### Methods related to attribute of the IntervalStarSet and StarSet #####
    ###########################################################################
    # Only for ISS
    @abstractmethod
    def getIMatBasisV(self) -> IntervalMatrix:
        """
        Return Basis interval matrix
        :return: (objIMBasisV -> IntervalMatrix )
        """
        pass

    # Only for SS
    def getMatBasisV(self) -> npt.ArrayLike:
        """
        Return Basis interval matrix
        :return: (matBasisV -> npt.ArrayLike)
        """
        pass

    @abstractmethod
    def getMatConstraintC(self) -> npt.ArrayLike:
        """
        Return matrix C in the predicate constraints C * alpha <= d
        :return: (matConstraintC -> npt.ArrayLike )
        """
        pass

    @abstractmethod
    def getArrayConstraintd(self) -> npt.ArrayLike:
        """
        Return array d in the predicate constraints C * alpha <= d
        :return: (arrayConstraintd -> npt.ArrayLike )
        """
        pass

    @abstractmethod
    def getArrayPredicateLow(self) -> npt.ArrayLike:
        """
        Return lower bound  of alpha in the predicate C * alpha <= d
        :return: (arrayPredicateLow -> npt.ArrayLike)
        """
        pass

    @abstractmethod
    def getArrayPredicateHigh(self) -> npt.ArrayLike:
        """
        Return upper bound  of alpha in the predicate C * alpha <= d
        :return: (arrayPredicateHigh -> npt.ArrayLike)
        """
        pass

    @abstractmethod
    def getArrayStateLow(self) -> npt.ArrayLike:
        """
        Return lower bound  of state x in x = c+a[1]*v[1] + a[2]*v[2]+ a[n]*v[n]
        :return: (arrayStateLow -> npt.ArrayLike)
        """
        pass

    @abstractmethod
    def getArrayStateHigh(self) -> npt.ArrayLike:
        """
        Return upper bound of state x in  x = c+a[1]*v[1] + a[2]*v[2]+ a[n]*v[n]
        :return: (arrayStateHigh -> npt.ArrayLike)
        """
        pass

    ###################################################
    #####       Common methods for all sets       #####
    ###################################################
    @abstractmethod
    def getDimension(self) -> int:
        """
        Returns the dimension of the set
        :return: (intDim -> int)
        dimension of the set
        """
        pass

    @abstractmethod
    def linearMap(self, matLow: npt.ArrayLike, matHigh: npt.ArrayLike = None) -> 'Set':
        """
        Linear mapping of a set (matHigh will be used for IntervalMatrix)
        :param matLow: a two-dimensional array for a weight matrix
        :type matLow: npt.ArrayLike
        :param matHigh: a two-dimensional array for a weight matrix
        :type matHigh: npt.ArrayLike
        :return: (objSetLinear -> Set) an instance of Set representing WX,
                where X is a self
        """
        pass

    @abstractmethod
    def affineMap(self, matLow: npt.ArrayLike, arrayLow: npt.ArrayLike,
                  matHigh: npt.ArrayLike = None, arrayHigh: npt.ArrayLike=None) -> 'Set':
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
        pass

    @abstractmethod
    def minkowskiSum(self, objSet: 'Set') -> 'Set':
        """
        Construct an instance of Set representing minkowski sum of two sets,
        that is, self and objSet
        :param objSet: an instance of the Set class
        :type objSet: Set
        :return: (objSetMWSum -> Box) an instance of Set for minkowski sum
        """
        pass

    @abstractmethod
    def convexHull(self, objSet: 'Set') -> 'Set':
        """
        Construct an instance of Set representing convex hull of two sets,
        that is, self and objSet
        :param objSet: an instance of Set class
        :type objSet: Set
        :return: (objSetCH -> Set) an instance of Set for convex hull
        """
        pass

    @abstractmethod
    def intersectPHSByIndex(self, intIndex: int) -> 'Set':
        """
        Compute intersection between self and a specific half space by index
        intersection(self x[index]>=0)
        :param intIndex: index of the state variables
        :type intIndex: int
        :return: (objSetIPHS-> Set) intersection between self and x[intIndex]>=0
        """
        pass

    @abstractmethod
    def intersectNHSByIndex(self, intIndex: int) -> 'Set':
        """
        Compute intersection between self and a specific half space by index
        intersection(self x[index]<=0)
        :param intIndex: index of the state variables
        :type intIndex: int
        :return: (objSetINHS -> IntervalStarSet) intersection between self and x[intIndex]<=0
        """
        pass

    @abstractmethod
    def getSameSignPartition(self) -> List['Set']:
        """
        Partition a set in such a way that for each sub set,
        values in each dimension will have the same sign
        :return: (listSets -> List[Set]) list of subsets
        """
        pass

    @abstractmethod
    def getSign(self) -> List[Sign]:
        """
        Returns the list of Sign instances for all
        the dimensions of the set
        :return: (listSigns -> List[Sign])
        """
        pass

    @abstractmethod
    def toAbsolute(self) -> npt.ArrayLike:
        """
        Returns the absolute value of the set
        :return: (arrayPoint -> npt.ArrayLike)
        """
        pass

    @abstractmethod
    def getModelVars(self, varName: str) -> Tuple[Model, Dict[int, Dict[int, Var]]]:
        """"
        Get encoding of a set and dictionary of variables
        :param varName: first letter of variable names
        :type varName: str
        :return: (ModelVars -> Tuple[Model, Dict[Dict[int, Var]]])
        """
        pass

    @abstractmethod
    def isEmpty(self) -> bool:
        """
        Checks if the set is empty
        :return: (status -> bool)
        """
        pass

    @abstractmethod
    def getRange(self) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        """
        Return the range of the IntervalStarSet
        """
        pass

    @abstractmethod
    def display(self) -> str:
        """
        Display the set
        :return: None
        """
        pass
