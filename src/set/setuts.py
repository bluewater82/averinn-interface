"""
Author: Ratan Lal
Date : January 28, 2025
"""
from typing import List, Dict

from gurobipy import Model, GRB, quicksum, Var

from src.set.box import Box
from src.set.intervalmatrix import IntervalMatrix
from src.set.intervalstarset import IntervalStarSet
from src.set.set import Set
import numpy.typing as npt
import numpy as np

from src.set.starset import StarSet
from src.solver.gurobi import Gurobi
from src.solver.solver import Solver
from src.types.datatype import DataType
from src.types.solvertype import SolverType
from src.utilities.log import Log


class SetUTS:
    """
    Common functionality related to Set
    """

    @staticmethod
    def displayLinConstr(A: npt.ArrayLike, b: npt.ArrayLike):
        """
        Print Ax <= b in the form of A b
        :param A: Two dimensional array
        :type A: npt.ArrayLike
        :param b: One dimensional array
        :type b: npt.ArrayLike
        :return: None
        """
        for row in range(len(A)):
            paddedRows = []
            paddedRows.append("         " + str(A[row]))
            paddedRows.append(str(b[row]))
            Log.message(str('     '.join(paddedRows)) + "\n")

    @staticmethod
    def displayDictOfDictOfSets(dictPartition: Dict[int, Dict[int, set]]):
        """
        Print dictionary of dictionaries of sets
        :param dictPartition: dictionary of dictionaries of sets
        :type dictPartition: Dict[int, Dict[int, Set]]
        :return: None
        """
        for intLayer in dictPartition.keys():
            paddedRows = []
            paddedRows.append("       " + str(intLayer) + " : " + str(dictPartition[intLayer]))
            Log.message(str('     '.join(paddedRows)) + "\n")

    @staticmethod
    def displayDictOfDictOfValues(dictCE: Dict[int, Dict[int, DataType.RealType]]):
        """
        Print dictionary of dictionaries of floats
        :param dictPartition: dictionary of dictionaries of floats
        :type dictPartition: Dict[int, Dict[int, float]]
        :return: None
        """
        for intLayer in dictCE.keys():
            paddedRows = []
            paddedRows.append("         " + str(intLayer) + " : " + str(dictCE[intLayer]))
            Log.message(str('     '.join(paddedRows)) + "\n")

    @staticmethod
    def displayListOfSets(listOfSets: List[Set]):
        """
        Print list of Sets
        :param listOfSets: list of Sets
        :type listOfSets: List[Set]
        :return: None
        """
        i: int = 0
        for objSet in listOfSets:
            Log.message("           Set " + str(i + 1) + "\n")
            i += 1
            Log.message(objSet.display())

    @staticmethod
    def rangeOfSets(listOfSets: List[Set]) -> Set:
        """
        Compute an over-approximation of sets as a set
        :param listOfSets: List[Set]
        :return: (listOfSet -> List[Set])
        """
        intDim: int = listOfSets[0].getDimension()
        arrayLow: npt.ArrayLike = [min([objSet.getRange()[0][i] for objSet in listOfSets]) for i in range(intDim)]
        arrayHigh: npt.ArrayLike = [max([objSet.getRange()[1][i] for objSet in listOfSets]) for i in range(intDim)]

        objSet: Set = Box(arrayLow, arrayHigh)

        return objSet

    @staticmethod
    def toIntervalStarSet(objSet: Set) -> Set:
        """
        Convert a given get into IntervalStarSet as a Set
        """
        rangeSet = objSet.getRange()
        arrayLow = rangeSet[0]
        arrayHigh = rangeSet[1]
        intDim = objSet.getDimension()
        # define matBasisV, that is, center point and vertices
        matBasisVLow = []
        for i in range(intDim):
            temp = [0.0 for j in range(intDim + 1)]
            temp[0] = (arrayLow[i] + arrayHigh[i]) / 2
            temp[i + 1] = (arrayHigh[i] - arrayLow[i]) / 2
            matBasisVLow.append(temp)
        matBasisVLow = np.array(matBasisVLow, dtype=np.float64)
        objIMBasisV: IntervalMatrix = IntervalMatrix(matBasisVLow, matBasisVLow)
        # define constraint matrix for -1 <=a[1]<=1, ....
        matConstraintC = []
        for i in range(intDim):
            temp = [0.0 for j in range(intDim)]
            temp[i] = 1
            matConstraintC.append(temp)
            temp = [0 for j in range(intDim)]
            temp[i] = -1
            matConstraintC.append(temp)
        matConstraintC = np.array(matConstraintC)

        arrayConstraintd = np.array([1 for j in range(2 * intDim)])

        objSet: Set = IntervalStarSet(objIMBasisV, matConstraintC, arrayConstraintd)
        return objSet

    @staticmethod
    def toStarSet(objSet: Set) -> Set:
        """
        Convert a given get into IntervalStarSet as a Set
        """
        rangeSet = objSet.getRange()
        arrayLow = rangeSet[0]
        arrayHigh = rangeSet[1]
        intDim = objSet.getDimension()
        # define matBasisV, that is, center point and vertices
        matBasisV = []
        for i in range(intDim):
            temp = [0.0 for j in range(intDim + 1)]
            temp[0] = (arrayLow[i] + arrayHigh[i]) / 2
            temp[i + 1] = (arrayHigh[i] - arrayLow[i]) / 2
            matBasisV.append(temp)
        matBasisV = np.array(matBasisV, dtype=np.float64)

        # define constraint matrix for -1 <=a[1]<=1, ....
        matConstraintC = []
        for i in range(intDim):
            temp = [0.0 for j in range(intDim)]
            temp[i] = 1
            matConstraintC.append(temp)
            temp = [0 for j in range(intDim)]
            temp[i] = -1
            matConstraintC.append(temp)
        matConstraintC = np.array(matConstraintC)

        arrayConstraintd = np.array([1 for j in range(2 * intDim)])

        objSet: Set = StarSet(matBasisV, matConstraintC, arrayConstraintd)
        return objSet

    @staticmethod
    def displayModel(grbModel: Model) -> str:
        """
        Return constraints related to the model
        """
        strGRBSet: str = ""
        listVars: List[Var] = grbModel.getVars()
        numVars: int = len(listVars)
        for constr in grbModel.getConstrs():
            listCoeff: List[float] = []
            for i in range(numVars):
                listCoeff.append(grbModel.getCoeff(constr, listVars[i]))
            constantTerm = constr.getAttr("rhs")
            isFirst: bool = True
            for i in range(numVars):
                if (listCoeff[i] != 0.0):
                    if isFirst:
                        strGRBSet += "          " + str(listVars[i].VarName) + "*" + str(listCoeff[i])
                        isFirst = False
                    else:
                        strGRBSet += " + " + str(listVars[i].VarName) + "*" + str(listCoeff[i])

            if constr.Sense == GRB.LESS_EQUAL:
                strGRBSet += " <= " + str(constantTerm)
            elif constr.Sense == GRB.GREATER_EQUAL:
                strGRBSet += " >= " + str(constantTerm)
            else:
                strGRBSet += " == " + str(constantTerm)
            strGRBSet += "\n"

        return strGRBSet

    @staticmethod
    def intersectWithUnsafe(objSet: Set, outputConstr: npt.ArrayLike, solverType: SolverType, varName: str) -> bool:
        """
        "param objSet: an instance of Set
        :type objSet: Set
        :param outputConstr: tuple in the form ([A1, A2,...], [b1,b2,...])
        :type outputConstr: npt.ArrayLike
        :param solverType: an instance of SolverType
        :type solverType: SolverType
        :param varName: first letter of variable name
        :type varName: str
        :return: bool (True if intersect or False otherwise)
        """
        # Encode the set
        grbModel, dictVars = objSet.getModelVars(varName)

        # Encode unsafe set
        listA: npt.ArrayLike = outputConstr[0]
        listb = npt.ArrayLike = outputConstr[1]
        numOfAandb: int = len(listA)
        status: List[bool] = []
        for i in range(numOfAandb):
            A: npt.ArrayLike = listA[i]
            b: npt.ArrayLike = listb[i]
            grbModelCC: Model = grbModel.copy()
            # Map original variables to copied model variables using their names
            varMap = {v: grbModelCC.getVarByName(v.varName) for v in grbModel.getVars()}
            numOfRows: int = A.shape[0]
            numOfCols: int = A.shape[1]

            for j in range(numOfRows):
                grbModelCC.addConstr(quicksum(np.float64(A[j][k]) *
                                              varMap[dictVars[1][k + 1]]
                                              for k in range(numOfCols)) <= np.float64(b[j]))
            grbModelCC.update()

            if solverType == SolverType.Gurobi:
                objSolver: Solver = Gurobi(grbModelCC, dictVars)
                Log.message("       MILP Encoding\n")
                Log.message(SetUTS.displayModel(grbModelCC))
                status.append(objSolver.satisfy())

            if np.any(status):
                return True

        return False