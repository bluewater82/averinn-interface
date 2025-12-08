"""
Author: Ratan Lal
Date : November 4, 2023
"""
from abc import ABC
from typing import List, Tuple, Dict
import numpy.typing as npt
import numpy as np
from gurobipy import Model, Var, GRB, quicksum
from src.set.intervalarray import IntervalArray
from src.set.intervalmatrix import IntervalMatrix
from src.set.set import Set
from src.solver.gurobi import Gurobi
from src.solver.solver import Solver
from src.types.datatype import DataType
from src.types.sign import Sign


class IntervalStarSet(Set, ABC):
    """
    The class StarSet capture both convex/non-convex set
    Star set is expressed by x = c+a[1]*v[1] + a[2]*v[2]+ a[n]*v[n] = V*b, where V = [c, v[1], v[2], ..., v[n]]
    and b =[1, a[1], a[2], ... , a[n]]^{T}], C*a <=d, constraints on a[i]
    """

    def __init__(self, objIMBasisV: IntervalMatrix, matConstraintC: npt.ArrayLike,
                 arrayConstraintd: npt.ArrayLike, arrayPredicateLow: npt.ArrayLike = None,
                 arrayPredicateHigh: npt.ArrayLike = None, arrayStateLow: npt.ArrayLike = None,
                 arrayStateHigh: npt.ArrayLike = None):
        """
        Initialize an instance of StarSet
        :param objIMBasisV: an interval matrix where first column interval vector captures center and rest of them are vertices
        :type objIMBasisV: IntervalMatrix
        :param matConstraintC: a matrix for the constraint C * alpha <= d
        :type matConstraintC: npt.ArrayLike
        :param arrayConstraintd: one dimensional array d for the constraint C * alpha <= d
        :type arrayConstraintd: npt.ArrayLike
        :param arrayPredicateLow: one dimensional array for the lower bound on a[i]s
        :type arrayPredicateLow: npt.ArrayLike
        :param arrayPredicateHigh: one dimensional array for the upper bound on a[i]s
        :type arrayPredicateHigh: npt.ArrayLike
        :param arrayStateLow: one dimensional array for the lower bound on the state
        :type arrayStateLow: npt.ArrayLike
        :param arrayStateHigh: one dimensional array  for the upper bound on the state
        :type arrayStateHigh: npt.ArrayLike
        """
        # Interval Matrix for interval center and vertices
        self.__objIMBasisV__ = objIMBasisV

        # Matrix C for constraint Ca<=d
        self.__matConstraintC__: npt.ArrayLike = matConstraintC

        # A dimensional array for d in constraint Ca<=d
        self.__arrayConstraintd__: npt.ArrayLike = arrayConstraintd

        # A one dimensional array for the lower bound on the predicate variables
        self.__arrayPredicateLow__: npt.ArrayLike = arrayPredicateLow

        # A one dimensional array for the upper bound on the predicate variables
        self.__arrayPredicateHigh__: npt.ArrayLike = arrayPredicateHigh

        # A one dimensional array for the lower bound on the state variables
        self.__arrayStateLow__: npt.ArrayLike = arrayStateLow

        # A one dimensional array for the upper bound on the state variables
        self.__arrayStateHigh__: npt.ArrayLike = arrayStateHigh

    ############################################
    ########## Methods for attributes  #########
    ############################################
    def getIMatBasisV(self) -> IntervalMatrix:
        """
        Return Basis interval matrix
        :return: (objIMBasisV -> IntervalMatrix )
        """
        return self.__objIMBasisV__

    def getMatConstraintC(self) -> npt.ArrayLike:
        """
        Return matrix C in the predicate constraints C * alpha <= d
        :return: (matConstraintC -> npt.ArrayLike )
        """
        return self.__matConstraintC__

    def getArrayConstraintd(self) -> npt.ArrayLike:
        """
        Return array d in the predicate constraints C * alpha <= d
        :return: (arrayConstraintd -> npt.ArrayLike )
        """
        return self.__arrayConstraintd__

    def getArrayPredicateLow(self) -> npt.ArrayLike:
        """
        Return lower bound  of alpha in the predicate C * alpha <= d
        :return: (arrayPredicateLow -> npt.ArrayLike)
        """
        return self.__arrayPredicateLow__

    def getArrayPredicateHigh(self) -> npt.ArrayLike:
        """
        Return upper bound  of alpha in the predicate C * alpha <= d
        :return: (arrayPredicateHigh -> npt.ArrayLike)
        """
        return self.__arrayPredicateHigh__

    def getArrayStateLow(self) -> npt.ArrayLike:
        """
        Return lower bound  of state x in x = c+a[1]*v[1] + a[2]*v[2]+ a[n]*v[n]
        :return: (arrayStateLow -> npt.ArrayLike)
        """
        return self.__arrayStateLow__

    def getArrayStateHigh(self) -> npt.ArrayLike:
        """
        Return upper bound of state x in  x = c+a[1]*v[1] + a[2]*v[2]+ a[n]*v[n]
        :return: (arrayStateHigh -> npt.ArrayLike)
        """
        return self.__arrayStateHigh__

    ############################################
    ########## Methods for only ISS    #########
    ############################################
    def getNumOfPredVars(self) -> int:
        """
        Return the number of predicate variables
        :return: (intPredVars -> int)
        """
        return self.__objIMBasisV__.getNumberOfColumns() - 1

    def getNumOfPredicates(self) -> int:
        """
        Return the number of predicates
        :return: (intPredicates -> int)
        """
        return self.__matConstraintC__.shape[0]

    ############################################
    ########## Common Methods  ################
    ############################################
    def linearMap(self, matLow: npt.ArrayLike, matHigh: npt.ArrayLike = None) -> Set:
        """
        Compute affine map without input of an interval star set
        :param matLow: lower matrix of an interval matrix for weight
        :type matLow: npt.ArrayLike
        :param matHigh: upper matrix of an interval matrix for weight
        :type matHigh: npt.ArrayLike
        :return: (objISS -> 'IntervalStarSet') an instance of an interval star set
        """
        objIMWeight: IntervalMatrix = IntervalMatrix(matLow, matHigh)
        # interval matrix multiplication
        objIMBasisnewV: IntervalMatrix = objIMWeight.product(self.__objIMBasisV__)

        # create a new interval star set
        objSet: Set = IntervalStarSet(objIMBasisnewV, self.__matConstraintC__, self.__arrayConstraintd__)

        # return affine map without input of an interval star set
        return objSet

    def affineMap(self, matLow: npt.ArrayLike, arrayLow: npt.ArrayLike, matHigh: npt.ArrayLike=None,
                  arrayHigh: npt.ArrayLike = None) -> Set:
        """
        Affine mapping of the star set
        :param matLow: lower matrix an interval matrix for weight
        :type matLow: npt.ArrayLike
        :param matHigh: upper matrix an interval matrix for weight
        :type matHigh: npt.ArrayLike
        :param arrayLow: lower array of an interval array for bias
        :type arrayLow: npt.ArrayLike
        :param arrayHigh: upper array of an interval array for bias
        :type arrayHigh: npt.ArrayLike
        :return: (objISS -> IntervalStarSet) an instance of IntervalStarSet representing WX+b,
                where X is a self
        """
        objIMWeight: IntervalMatrix = IntervalMatrix(matLow, matHigh)
        objIAb: IntervalArray = IntervalArray(arrayLow, arrayHigh)
        # interval matrix multiplication
        objIMBasisnewV: IntervalMatrix = objIMWeight.product(self.__objIMBasisV__)

        # shift the center, c' = c + b
        objIAC: IntervalArray = objIMBasisnewV.getColumnIAByIndex(0)
        objIACb = objIAC.addition(objIAb)
        objIMBasisnewV.setColumnIAByIndex(objIACb, 0)

        # create a new interval star set
        objSet: Set = IntervalStarSet(objIMBasisnewV, self.__matConstraintC__, self.__arrayConstraintd__)

        # return affine map with input of an interval star set
        return objSet

    def intersectPHSByIndex(self, intIndex: int) -> Set:
        """
        Compute intersection between self and a specific half space by index
        intersection(self x[index]>=0)
        :param intIndex: index of the state variables
        :type intIndex: int
        :return: (objSetIPHS-> IntervalStarSet) intersection between self and x[intIndex]>=0
        """
        # Get the number of columns
        colSize: int = self.__objIMBasisV__.getNumberOfColumns()

        # Compute predicate matrix for the intersection
        matConstraintIPHSC: npt.ArrayLike = np.concatenate((self.__matConstraintC__,
                                                            np.array([[min(-self.__objIMBasisV__.getRowIAByIndex(
                                                                intIndex).getArrayLow()[i],
                                                                           -self.__objIMBasisV__.getRowIAByIndex(
                                                                               intIndex).getArrayHigh()[i])
                                                                       for i in range(1, colSize, 1)]])))

        # Compute array for the predicate constraint
        arrayConstraintIPHSd: npt.ArrayLike = np.concatenate((self.__arrayConstraintd__,
                                                              np.array([self.__objIMBasisV__.getRowIAByIndex(
                                                                  intIndex).getArrayHigh()[0]])), axis=0)

        # Create interval star set for the intersection
        objSetIPHS: IntervalStarSet = IntervalStarSet(self.__objIMBasisV__,
                                                      matConstraintIPHSC,
                                                      arrayConstraintIPHSd)

        # Return the star set for the intersection
        return objSetIPHS

    def intersectNHSByIndex(self, intIndex: int) -> Set:
        """
        Compute intersection between self and a specific half space by index
        intersection(self x[index]<=0)
        :param intIndex: index of the state variables
        :type intIndex: int
        :return: (objSetINHS -> IntervalStarSet) intersection between self and x[intIndex]<=0
        """
        # Get the number of columns
        colSize: int = self.__objIMBasisV__.getNumberOfColumns()
        # Compute predicate matrix for the intersection
        matConstraintINHSC: npt.ArrayLike = np.concatenate((self.__matConstraintC__,
                                                            np.array([[self.__objIMBasisV__.getRowIAByIndex(
                                                                intIndex).getArrayLow()[i]
                                                                       for i in range(1, colSize, 1)]])))

        # Compute array for the predicate constraint
        arrayConstraintINHSd: npt.ArrayLike = np.concatenate((self.__arrayConstraintd__,
                                                              np.array([max(-self.__objIMBasisV__.getRowIAByIndex(
                                                                  intIndex).getArrayLow()[0],
                                                                            -self.__objIMBasisV__.getRowIAByIndex(
                                                                                intIndex).getArrayHigh()[0])])),
                                                             axis=0)

        # Create interval star set for the intersection
        objSetINHS: Set = IntervalStarSet(self.__objIMBasisV__, matConstraintINHSC, arrayConstraintINHSd)

        # Return the star set for the intersection
        return objSetINHS

    def minkowskiSum(self, objISS: Set) -> Set:
        """
        Minkowski sum of two interval star sets
        :param objISS: second interval star set
        :type objISS: IntervalStarSet
        :return: (objSumIntervalStarSet -> IntervalStarSet) minkowski sum of
        two interval star sets
        """
        # Horizontal concatenation of vertices of both interval star sets
        matBasisVLow = np.concatenate((self.__objIMBasisV__.getMatLow(), objISS.getIMatBasisV().getMatLow()[:, 1:]), axis=1)
        matBasisVHigh = np.concatenate((self.__objIMBasisV__.getMatHigh(), objISS.getIMatBasisV().getMatHigh()[:, 1:]), axis=1)

        # Add centers of both interval star sets
        matBasisVLow[:, 0] = matBasisVLow[:, 0] + objISS.getIMatBasisV().getMatLow()[:, 0]
        matBasisVHigh[:, 0] = matBasisVHigh[:, 0] + objISS.getIMatBasisV().getMatHigh()[:, 0]

        # Construct Basis matrix for minkowski sm
        imatBasisSumV: IntervalMatrix = IntervalMatrix(matBasisVLow, matBasisVHigh)
        # Express matConstraintC of both self and objIntervalStarSet as a block diagonal matrix
        r1: int = np.shape(self.__matConstraintC__)[0]
        c1: int = np.shape(self.__matConstraintC__)[1]
        r2: int = np.shape(objISS.getMatConstraintC())[0]
        c2: int = np.shape(objISS.getMatConstraintC())[1]

        matConstraintC: npt.ArrayLike = np.block([[self.__matConstraintC__, np.zeros((r1, c2))],
                                                           [np.zeros((r2, c1)), objISS.getMatConstraintC()]])

        # Vertical concatenation of column vectors arrayConstraintd of both self and objISS
        arrayConstraintd: npt.ArrayLike = np.concatenate((self.__arrayConstraintd__,
                                                          objISS.getArrayConstraintd()), axis=0)

        # Create an instance of StarSet for the minkowski sum
        objSumISS: IntervalStarSet = IntervalStarSet(imatBasisSumV, matConstraintC, arrayConstraintd)

        # Return minkowski sum of two ISSs
        return objSumISS

    def getRange(self) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        """
        Return the range of the IntervalStarSet
        """
        # Encode Interval Star Set in Gurobi
        grbModel, dictVars = self.__encode__()
        objSolver: Solver = Gurobi(grbModel, dictVars)
        objSet: Set = objSolver.outputRange()
        rangeISS: Tuple[npt.ArrayLike, npt.ArrayLike] = (objSet.getArrayLow(), objSet.getArrayHigh())
        return rangeISS

    def isEmpty(self) -> bool:
        """
        Checks if the set is empty
        :return: (status -> bool)
        """
        # Encode Interval Star Set in Gurobi
        grbModel, dictVars = self.__encode__()
        objSolver: Solver = Gurobi(grbModel, dictVars)
        if objSolver.satisfy():
            return False
        else:
            return True

    def getModelVars(self) -> Tuple[Model, Dict[int, Dict[int, Var]]]:
        """"
        Get encoding of a set and dictionary of variables
        :return: (ModelVars -> Tuple[Model, Dict[Dict[int, Var]]])
        """
        # Encode Interval Star Set in Gurobi
        grbModel, dictVars = self.__encode__()

        # Return Model and DictVars
        return grbModel, dictVars

    ############################################
    ############## Private Methods #############
    ############################################
    def __encode__(self) -> Tuple[Model, Dict[int, Dict[int, Var]]]:
        """"
        Encode Interval Star Set into Gurobi format
        :return: (grbModel -> Model)
        """
        # Create state variables
        intDim: int = self.getDimension()
        listStateVars: List[str] = ['x_' + str(i) for i in range(intDim)]

        # Create predicate variables
        intPredVars: int = self.getNumOfPredVars()
        listPredVars: List[str] = ['alpha_' + str(j) for j in range(intPredVars)]

        # Create a gurobi model
        grbModel: Model = Model()

        # Create state variables in gurobi
        grbStateVars = []
        for i in range(intDim):
            grbStateVars.append(grbModel.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name=listStateVars[i]))

        # Create predicate variables in gurobi
        grbPredVars = []
        for j in range(intPredVars):
            grbPredVars.append(grbModel.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name=listPredVars[j]))

        # Create constraints for states x = c + sum (ai vi)
        for i in range(intDim):
            grbModel.addConstr(grbStateVars[i] >= self.__objIMBasisV__.getColumnIAByIndex(0).getArrayLow()[i] +
                               quicksum(grbPredVars[j] * self.__objIMBasisV__.getColumnIAByIndex(j+1).getArrayLow()[i]
                                        for j in range(intPredVars)))

        for i in range(intDim):
            grbModel.addConstr(grbStateVars[i] <= self.__objIMBasisV__.getColumnIAByIndex(0).getArrayHigh()[i] +
                               quicksum(grbPredVars[j] * self.__objIMBasisV__.getColumnIAByIndex(j+1).getArrayHigh()[i]
                                        for j in range(intPredVars)))

        # Create constraints for predicates C alpha <= d
        intPredicates: int = self.getNumOfPredicates()
        intPredVars: int = self.getNumOfPredVars()
        for i in range(intPredicates):
            grbModel.addConstr(quicksum(self.__matConstraintC__[i][j] * grbPredVars[j]
                                 for j in range(intPredVars)) <= self.__arrayConstraintd__[i])

        grbModel.update()

        # Create dictionary for solver
        dictVars: Dict[int, Dict[int, Var]] = dict()
        dictVars[1] = dict()
        for i in range(1, intDim+1, 1):
            dictVars[1][i] = grbStateVars[i-1]

        return grbModel, dictVars

    ############################################
    ### Unused Methods from other sets #########
    ############################################
    def getArrayLow(self) -> npt.ArrayLike:
        """
        Returns the lower end point of the Box instance
        :return: (arrayLow -> npt.ArrayLike)
        One dimensional array for the lower end point fo the box
        """
        # The following line are for just returning (not for implementation)
        arrayLow: npt.ArrayLike = np.array([], dtype=object)
        return arrayLow

    def getArrayHigh(self) -> npt.ArrayLike:
        """
        Returns the upper end point of the Box instance
        :return: (arrayHigh -> npt.ArrayLike)
        One dimensional array for the upper end point fo the box
        """
        # The following line are for just returning (not for implementation)
        arrayHigh: npt.ArrayLike = np.array([], dtype=object)
        return arrayHigh

    ############################################
    ########### Unused Common Methods ##########
    ############################################
    def getDimension(self) -> int:
        """
        Returns the dimension of the IntervalStarSet instance
        :return: (intDim -> int)
        """
        return self.__objIMBasisV__.getNumberOfRows()

    def convexHull(self, objSet: Set) -> Set:
        """
        Construct an instance of Box representing convex hull of two Interval Star Sets,
        that is, self and objSet
        :param objSet: an instance of Set class
        :type objSet: Set
        :return: (objSetCH -> Set) an instance of Set for convex hull
        """
        # The following line are for just returning (not for implementation)
        objSetCH: Set = IntervalStarSet(None, None, None)

        return objSetCH

    def getSameSignPartition(self) -> List[Set]:
        """
        Partition a set in such a way that for each sub set,
        values in each dimension will have the same sign
        :return: (listSets -> List[Set]) list of subsets
        """
        # The following line are for just returning (not for implementation)
        listSets: List[Set] = []

        return listSets

    def getSign(self) -> List[Sign]:
        """
        Returns the list of Sign instances for all
        the dimensions of the set
        :return: (listSigns -> List[Sign])
        """
        # The following line are for just returning (not for implementation)
        listSigns: List[Sign] = []

        return listSigns

    def toAbsolute(self) -> npt.ArrayLike:
        """
        Returns the absolute value of the set
        :return: (arrayPoint -> npt.ArrayLike)
        """
        # The following line are for just returning (not for implementation)
        arrayPoint: npt.ArrayLike = np.array([], dtype=object)

        return arrayPoint

    def display(self) -> str:
        """
        Display the set
        :return: None
        """
        # The following line are for just returning (not for implementation)

        return ""
