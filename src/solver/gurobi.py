"""
Author: Ratan Lal
Date : January 30, 2025
"""
import time
from abc import ABC
from typing import Dict, List
from gurobipy import Model, GRB, Var
from src.set.box import Box
from src.set.set import Set
from src.solver.solver import Solver
import numpy.typing as npt
import numpy as np

from src.types.datatype import DataType


class Gurobi(Solver, ABC):
    """
    Implement all the functions using Gurobi
    """

    def __init__(self, grbModel: Model, dictVarsX: Dict[int, Dict[int, Var]] = None):
        """
        Solver over gurobi constraints
        :param grbModel: an instance of Gurobi Model
        :type grbModel: Gurobi Model
        :param dictVarsX: dictionary of dictionaries of gurobi variables
        :type dictVarsX: Dict[int, Dict[int, Var]]
        """
        self.__model__ = grbModel
        self.__dictVarsX__ = dictVarsX

    def satisfy(self) -> bool:
        """
        Check the satisfiability of the model
        :return: (status -> bool)
        True if the model satisfies the gurobi constraints
        False otherwise
        """
        # disabled log in optimize function
        self.__model__.setParam('OutputFlag', 0)
        # Set the tolerance
        #self.__model__.setParam('FeasibilityTol', 1e-2)
        #self.__model__.setParam('OptimalityTol', 1e-3)
        #self.__model__.setParam('MIPGap', 0.01)
        #self.__model__.setParam('Presolve', 0)
        #self.__model__.setParam('SolutionLimit', 1)
        #self.__model__.setParam('IntFeasTol', 1e-9)
        # check the feasibility of the model
        self.__model__.setObjective(1, GRB.MAXIMIZE)
        #self.__model__.write('./model.lp')
        #print("Checking satisfy start")
        self.__model__.optimize()
        #print("Checking satisfy stop")
        # check the feasibility status

        if self.__model__.Status == GRB.INFEASIBLE:
            # print(self.__model__.Status)
            self.__model__.write('./model.lp')
            self.__model__.computeIIS()
            self.__model__.write("./model.ilp")
            return False
        else:
            self.__model__.write('./model.lp')
            return True

    def getInstance(self) -> Dict[int, Dict[int, DataType.RealType]]:
        """
        Extract a satisfiable instance of the model
        :return: (dictSatInstance -> Dict[int, Dict[int, float]])
        the satisfiable instance of the model
        """
        # Check whether there is an instance
        status: bool = self.satisfy()
        # Initialize a dictionary for a satisfying instance
        dictSatInstance: Dict[int, Dict[int, DataType.RealType]] = dict()
        if status == True:
            # write satisfiable instance into a file
            self.__model__.write('./model.sol')
            # retrieve solution and store in a dictionary
            dictVarsX: Dict[int, Dict[int, Var]] = self.__dictVarsX__

            for intLayerNum in dictVarsX.keys():
                dictTemp: Dict[int, DataType.RealType] = dict()
                for id in dictVarsX[intLayerNum].keys():
                    y = dictVarsX[intLayerNum][id].x
                    #dictTemp[id] = round(y, 7)
                    dictTemp[id] = DataType.RealType(y)
                    # Log.message(str(dictVarsX[inLayer][id].x) + " ")
                dictSatInstance[intLayerNum] = dictTemp

        # return the satisfying instance
        return dictSatInstance

    def __extractOptValue__(self) -> DataType.RealType:
        """
        Extract value of an objective function from a model.sol file
        :return: (intValue -> int) value of a variable
        """
        # open the model file
        f = open('model.sol', 'r')

        # list of strings where each string is a line
        contents: List[str] = f.readlines()
        f.close()
        # extract the value of objective function
        value: DataType.RealType = DataType.RealType(0.0)
        if "Objective" in contents[0]:
            value = DataType.RealType(contents[0].split('=')[1].strip('\n'))
        else:
            value = DataType.RealType(contents[1].split('=')[1].strip('\n'))


        return self.__truncate__(value, 8)

    def __truncate__(self, number, decimals=0):
        """
        Truncates a float to a given number of decimal places without rounding.
        """
        if decimals < 0:
            raise ValueError("Decimal places must be non-negative")

        factor = 10.0 ** decimals
        return int(number * factor) / factor

    def outputRange(self) -> Set:
        """
        Find output range of the model for the output variables
        :param varMap: dictionary between gurobi variables
        :type varMap: Dict[Var, Var]
        :return: (objSet -> Set)
        """
        # get dictionary of output variables
        dictOutputVars: Dict[int, Var] = self.__dictVarsX__[max(self.__dictVarsX__.keys())]
        status: bool = self.satisfy()
        if not status:
            arrayLow: npt.ArrayLike = np.array([2.0 for i in range(len(dictOutputVars))], dtype=np.float64)
            arrayHigh: npt.ArrayLike = np.array([1.0 for i in range(len(dictOutputVars))], dtype=np.float64)
            objSet: Set = Box(arrayLow, arrayHigh)
            return objSet
        # disabled log in optimize function
        self.__model__.setParam('OutputFlag', 0)
        # declare low and high array
        arrayLow: npt.ArrayLike = np.array([0.0 for idName in dictOutputVars.keys()], dtype=np.float64)
        arrayHigh: npt.ArrayLike = np.array([0.0 for idName in dictOutputVars.keys()], dtype=np.float64)
        # set objective function for each variable
        for idName in dictOutputVars.keys():
            self.__model__.setObjective(dictOutputVars[idName])
            # For upper bound of an output variable
            self.__model__.ModelSense = GRB.MAXIMIZE
            #print("Checking feasibility for Maximum start")
            self.__model__.optimize()
            #print("Checking feasibility for Maximum stop")
            #print(self.__model__.Status)
            self.__model__.write('./model.lp')
            self.__model__.write('./model.sol')
            #self.__model__.computeIIS()
            #print("Print unsatisfiable constraints")
            #self.__model__.write("./model.ilp")
            arrayHigh[idName - 1] = self.__extractOptValue__()
            # For lower bound of an output variable
            self.__model__.ModelSense = GRB.MINIMIZE
            #print("Checking feasibility for Minimum start")
            self.__model__.optimize()
            #print("Checking feasibility for Minimum stop")
            self.__model__.write('./model.sol')
            arrayLow[idName - 1] = self.__extractOptValue__()
        objSet: Set = Box(arrayLow, arrayHigh)
        return objSet
