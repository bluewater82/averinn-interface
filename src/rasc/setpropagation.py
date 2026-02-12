from abc import ABC
from typing import Tuple, List, Dict

import numpy as np
import numpy.typing as npt
from gurobipy import quicksum, Var

from src.activation.relu import Relu
from src.gnn.gnn import GNN
from src.rasc.technique import Technique
from src.set.set import Set
from src.set.setuts import SetUTS
from src.solver.gurobi import Gurobi
from src.solver.solver import Solver
from src.types.lastrelu import LastRelu
from src.types.solvertype import SolverType
from src.utilities.log import Log


class SetPropagation(Technique, ABC):

    def __init__(self, objGNN: GNN, objSet: Set,
                 outputConstr: Tuple[npt.ArrayLike, npt.ArrayLike],
                 solverType: SolverType, lastRelu: LastRelu):
        """
        Compute output range set for NeuralNetwork instance
        :param objGNN: an instance of GNN
        :type objGNN: GNN
        :param objSet: a Set instance for an initial states
        :type objSet: Set
        :param outputConstr: a pair of numpy array (A, b) [Ay<=b]
        :type outputConstr: Tuple[npt.ArrayLike, npt.ArrayLike]
        :param solverType: an instance of SolverType enum class
        :type solverType: SolverType
        :param lastRelu: an instance of LastRelu enum class
        :type lastRelu: LastRelu
        """
        self.__objGNN__ = objGNN
        self.__objSet__ = objSet
        self.__outputConstr__ = outputConstr
        self.__solverType__ = solverType
        self.__lastRelu__ = lastRelu

    def reachSet(self) -> List[Set]:
        """
        Return reach set in the form of a list of reach Set instance
        :return: (listReachSets -> List[Set])
        Reach set in the form of a list of Set instance
        """
        # Get number of layer
        numOfLayer: int = self.__objGNN__.getNumOfLayers()
        listSet: List[Set] = [self.__objSet__]
        Log.message("Initial set\n")
        Log.message(self.__objSet__.display())
        numOfStars: List[int] = []
        numOfStars.append(len(listSet))
        for i in range(1, numOfLayer, 1):
            Log.message("       After Layer         Number of Stars\n")
            Log.message("           "+ str(i+1))
            listTempSet = []
            # Find lower and upper weight matrices
            matLow: npt.ArrayLike = self.__objGNN__.getLowerMatrixByLayerWithSize(i)
            matHigh: npt.ArrayLike = self.__objGNN__.getUpperMatrixByLayerWithSize(i)
            arrayLow: npt.ArrayLike = self.__objGNN__.getLowerBiasByLayer(i+1)
            arrayHigh: npt.ArrayLike = self.__objGNN__.getUpperBiasByLayer(i+1)

            for S in listSet:
                # Compute Reach set for the next layer
                objSet = S.affineMap(matLow, arrayLow, matHigh, arrayHigh)
                if i == numOfLayer - 1:
                    if self.__lastRelu__ == LastRelu.YES:
                        listTempSet.extend(Relu.anySet(objSet))
                    else:
                        listTempSet.append(objSet)
                else:
                    listTempSet.extend(Relu.anySet(objSet))

            listSet = listTempSet
            #objSet: Set = SetUTS.rangeOfSets(listSet)
            #listSet = [SetUTS.toIntervalStarSet(objSet)]
            Log.message("                       "+str(len(listSet))+"\n")
            # number of stars
            numOfStars.append(len(listSet))
            # display star sets
            setNumber: int = 1
            for SP in listSet:
                Log.message("Set "+ str(setNumber)+"\n")
                Log.message(SP.display())
                setNumber += 1

        f = open("temp.txt", 'w')
        f.write(str(numOfStars))
        f.close()
        return listSet

    def checkSafety(self) -> bool:
        """
        Return safety of GNN instance against unsafe set
        :return: (status -> bool)
        True if no valuation of GNN satisfies unsafe set
        otherwise false
        """
        listSet = self.reachSet()
        status: List[bool] = []
        for S in listSet:
            listA: npt.ArrayLike = self.__outputConstr__[0]
            listb = npt.ArrayLike = self.__outputConstr__[1]
            numOfAandb: int = len(listA)

            for i in range(numOfAandb):
                A: npt.ArrayLike = listA[i]
                b: npt.ArrayLike = listb[i]
                grbModel, dictVarsX = S.getModelVars()
                # Dictionary of output variables
                dictOutputVars: Dict[int, Var] = dictVarsX[len(dictVarsX)]
                numOfRows: int = A.shape[0]
                numOfCols: int = A.shape[1]
                for j in range(numOfRows):
                    grbModel.addConstr(quicksum(np.float64(A[j][k]) *
                                                  dictOutputVars[j + 1]
                                                  for k in range(numOfCols)) <= np.float64(b[j]))
                grbModel.update()

                if self.__solverType__ == SolverType.Gurobi:
                    objSolver: Solver = Gurobi(grbModel, dictVarsX)
                    Log.message("Solving\n")
                    status.append(objSolver.satisfy())

                if np.any(status):
                    return False
        return True
