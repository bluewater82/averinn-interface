"""
Author: Ratan Lal
Date : November 18, 2024
"""
from abc import ABC
from src.gnn.number import Number
import numpy as np
import numpy.typing as npt
from src.gnn.connection import Connection
from src.gnn.ireal import IReal
from src.gnn.layer import Layer
from src.gnn.gnn import GNN
from src.gnn.gnnuts import GNNUTS
from src.rasc.technique import Technique
from src.set.box import Box
from src.set.set import Set
from src.set.setuts import SetUTS
from src.solver.gurobi import Gurobi
from src.solver.solver import Solver
from src.types.datatype import DataType
from src.types.lastrelu import LastRelu
from src.types.sign import Sign
from src.types.solvertype import SolverType
from typing import Dict, Tuple, List
from gurobipy import GRB, Var, Model, quicksum

from src.utilities.log import Log


class Milp(Technique, ABC):
    """
    The class Milp compute output range set for GNN instance
    """

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
        listReachSets: List[Set] = []
        # Compute dictM
        objAGNN: GNN = GNNUTS.ToAGNN(self.__objGNN__)
        arrayInput: npt.ArrayLike = self.__objSet__.toAbsolute()
        dictM: Dict[int, Dict[int, DataType.RealType]] = GNNUTS.getM(objAGNN, arrayInput, self.__lastRelu__)
        # Display dictM
        Log.message("Upper Bound (big-M value) for each neuron\n")
        SetUTS.displayDictOfDictOfValues(dictM)
        # Reach set
        listSets: List[Set] = self.__objSet__.getSameSignPartition()

        # Initiate dictionaries for variables
        dictVarsX: Dict[int, Dict[int, Var]] = dict()
        dictVarsQ: Dict[int, Dict[int, Var]] = dict()
        # Model is printed only for the first set
        isFirst: bool = True
        for objSet in listSets:
            grbModel: Model = Model()
            for i in range(2):
                dictX, dictQ, grbModel = self.__createVForALayer__(grbModel, i + 1)
                dictVarsX[i + 1] = dictX
                dictVarsQ[i + 1] = dictQ

            # Encode Initial Set
            grbModel = self.__createICForSet__(objSet, dictVarsX[1], grbModel)
            listSign: List[Sign] = objSet.getSign()
            # Encode the first layer
            grbModel = self.__createCForFirstLayer__(dictM[2], dictVarsX[1], dictVarsX[2],
                                                     dictVarsQ[2], grbModel, 1,
                                                     listSign)
            # Encode from the second layer
            grbModel, dictVarsX, dictVarsQ = self.__encodeFromSecondLayer__(dictM, dictVarsX,
                                                                            dictVarsQ, grbModel)

            # Solve gurobi constraints
            if self.__solverType__ == SolverType.Gurobi:
                objSolver: Solver = Gurobi(grbModel, dictVarsX)
                if isFirst:
                    Log.message("MILP Encoding for the first set if the input set is splitted\n")
                    Log.message("Only initial set encoding will be changed for other sets\n")
                    Log.message(SetUTS.displayModel(grbModel))
                    isFirst = False
                listReachSets.append(objSolver.outputRange())

        return listReachSets

    def checkSafety(self) -> bool:
        """
        Return safety of GNN instance against unsafe set
        :return: (status -> bool)
        True if no valuation of GNN satisfies unsafe set
        otherwise false
        """
        # Compute dictM
        objAGNN: GNN = GNNUTS.ToAGNN(self.__objGNN__)
        arrayInput: npt.ArrayLike = self.__objSet__.toAbsolute()
        dictM: Dict[int, Dict[int, float]] = GNNUTS.getM(objAGNN, arrayInput, self.__lastRelu__)
        Log.message("Upper Bound for each neuron\n")
        SetUTS.displayDictOfDictOfValues(dictM)
        # Safety Checking
        listSets: List[Set] = self.__objSet__.getSameSignPartition()
        # Initiate dictionaries for variables
        dictVarsX: Dict[int, Dict[int, Var]] = dict()
        dictVarsQ: Dict[int, Dict[int, Var]] = dict()
        # List of status
        status: List[bool] = []
        # Constraint for only first properties
        isFirst: bool = True
        for objSet in listSets:
            grbModel: Model = Model()
            for i in range(2):
                dictX, dictQ, grbModel = self.__createVForALayer__(grbModel, i + 1)
                dictVarsX[i + 1] = dictX
                dictVarsQ[i + 1] = dictQ
            # Encode Initial Set
            grbModel = self.__createICForSet__(objSet, dictVarsX[1], grbModel)
            listSign: List[Sign] = objSet.getSign()
            # Encode the first layer
            grbModel = self.__createCForFirstLayer__(dictM[2], dictVarsX[1], dictVarsX[2],
                                                     dictVarsQ[2], grbModel, 1,
                                                     listSign)
            # Encode from the second layer
            grbModel, dictVarsX, dictVarsQ = self.__encodeFromSecondLayer__(dictM, dictVarsX,
                                                                            dictVarsQ, grbModel)

            # Dictionary of output variables
            dictOutputVars: Dict[int, Var] = dictVarsX[len(dictVarsX)]
            # Encode unsafe set
            listA: npt.ArrayLike = self.__outputConstr__[0]
            listb = npt.ArrayLike = self.__outputConstr__[1]
            numOfAandb: int = len(listA)
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
                                                  varMap[dictOutputVars[j + 1]]
                                                  for k in range(numOfCols)) <= np.float64(b[j]))
                grbModelCC.update()

                if self.__solverType__ == SolverType.Gurobi:
                    objSolver: Solver = Gurobi(grbModelCC, dictVarsX)
                    if isFirst:
                        Log.message("MILP Encoding\n")
                        Log.message(SetUTS.displayModel(grbModelCC))
                        isFirst = False
                    status.append(objSolver.satisfy())

                if np.any(status):
                    return False
        return True

    def __encodeFromSecondLayer__(self, dictM: Dict[int, Dict[int, DataType.RealType]],
                                  dictVarsX: Dict[int, Dict[int, Var]],
                                  dictVarsQ: Dict[int, Dict[int, Var]],
                                  grbModel: Model) -> \
            Tuple[Model, Dict[int, Dict[int, Var]], Dict[int, Dict[int, Var]]]:
        """
        Encoding of a GNN from the second layer
        :param dictM: Dictionary of dictionaries between Node's id and its value
        :type dictM: Dict[int, Dict[int, float]]
        :return: (grbModel, dictVarsX, dictVarsQ)
        """
        # Create dictionaries for real and boolean variables for each Node instance of
        # NeuralNetwork instance
        # Get number of layer of GNN instance
        intNumLayer: int = self.__objGNN__.getNumOfLayers()
        # Create variables from the first and second layer
        # Add constraints between intLayerNum and intLayerNum +1 from second layer
        for intLayerNum in range(2, intNumLayer, 1):
            dictX, dictQ, grbModel = self.__createVForALayer__(grbModel, intLayerNum + 1)
            dictVarsX[intLayerNum + 1] = dictX
            dictVarsQ[intLayerNum + 1] = dictQ
            if intLayerNum == intNumLayer - 1:
                grbModel = self.__createCForALayer__(dictM[intLayerNum + 1], dictVarsX[intLayerNum], dictX,
                                                     dictQ, grbModel, intLayerNum, self.__lastRelu__)
            else:
                grbModel = self.__createCForALayer__(dictM[intLayerNum + 1], dictVarsX[intLayerNum], dictX,
                                                     dictQ, grbModel, intLayerNum, LastRelu.YES)
        grbModel.update()

        return grbModel, dictVarsX, dictVarsQ

    def __createVForALayer__(self, grbModel: Model, intLayerNum: int) \
            -> Tuple[Dict[int, Var], Dict[int, Var], Model]:
        """
        Create variables for each Node of a layer of NeuralNetwork instance
        :param grbModel: an instance of GRBModel class
        :type grbModel: Model
        :param intLayerNum: a layer number
        :type intLayerNum: int
        :return: (dictX : Dict[int, Var], dictQ : Dict[int, Var], grbModel: Model)
        dictX: dictionary between Node's ids and Gurobi variables for a layer
        dictQ: dictionary between Node's ids and boolean variables for a layer
        grbModel: an instance of Gurobi model
        """
        objLayer: Layer = self.__objGNN__.getDictLayers()[intLayerNum]
        # Creation variables for intLayerNum
        dictX: Dict[int, Var] = dict()
        dictQ: Dict[int, Var] = dict()
        for NodeId in objLayer.dictNodes.keys():
            dictX[NodeId] = grbModel.addVar(lb=-np.float64('inf'), vtype=GRB.CONTINUOUS,
                                            name='x_' + str(intLayerNum) + '_' + str(NodeId))
            dictQ[NodeId] = grbModel.addVar(vtype=GRB.BINARY,
                                            name='q_' + str(intLayerNum) + '_' + str(NodeId))

        # Update the gurobi model
        grbModel.update()

        # Return both dictionaries and gurobi model
        return dictX, dictQ, grbModel

    def __createICForSet__(self, objSet: Set, dictInitX: Dict[int, Var], grbModel: Model) -> Model:
        """
        Encode initial values for the input layer
        :param objSet: a Set instance
        :type objSet: Set
        :param dictInitX: dictionary between input Node's ids and its real gurobi variables
        :type dictInitX: Dict[int, Var]
        :param grbModel: an instance of gurobi Model
        :type grbModel: Model
        :param varMap: dictionary between gurobi variables
        :type varMap: Dict[Var, var]
        :return: (grbModel -> Model)
        updated gurobi model
        """
        if isinstance(objSet, Box):
            # Set lower and upper bound for the initial variables
            for NodeId in dictInitX.keys():
                grbModel.addConstr(dictInitX[NodeId] >= np.float64(objSet.getArrayLow()[NodeId - 1]))
                grbModel.addConstr(dictInitX[NodeId] <= np.float64(objSet.getArrayHigh()[NodeId - 1]))
            grbModel.update()

        # return updated the model
        return grbModel

    def __createCForALayer__(self, dictMTarget: Dict[int, DataType.RealType],
                             dictXSource: Dict[int, Var],
                             dictXTarget: Dict[int, Var],
                             dictQTarget: Dict[int, Var], grbModel: Model,
                             intLayerNum: int, lastRelu: LastRelu) -> Model:
        """
        :param dictMTarget: dictionary between Node ids and its upper bound
        for the target layer
        :type dictMTarget: Dict[int, float]
        :param dictXSource: dictionary between Node ids and real gurobi variables
        for the source layer
        :type dictXSource: Dict[int, Var]
        :param dictXTarget: dictionary between Node ids and real gurobi variables
        for the target layer
        :type dictXTarget: Dict[int, Var]
        :param dictQTarget: dictionary between Node ids and boolean variables
        for the target layer
        :type dictQTarget: Dict[int, Var]
        :param grbModel: an instance of gurobi Model
        :type grbModel: Model
        :param intLayerNum: source layer number
        :type intLayerNum: int
        :param lastRelu: an instance of LastReLu enum class
        :type lastRelu: LastRelu
        :return: (grbModel -> Model)
        updated gurobi model
        """
        # Attributes of GNN
        objGNN: GNN = self.__objGNN__
        intNumOfLayers: int = objGNN.getNumOfLayers()
        dictLayers: Dict[int, Layer] = objGNN.getDictLayers()
        dictConnections: Dict[int, Connection] = objGNN.getDictConnections()

        # Node instances of the target layer
        for objNodeTarget in dictLayers[intLayerNum + 1].dictNodes.values():
            # constraints for lower limit for the Target Nodes
            grbModel.addConstr(quicksum((dictXSource[objNodeSource.intId] *
                                         np.float64(objNodeSource.intSize * dictConnections[
                                             intLayerNum].dictEdges[
                                             (objNodeSource.intId, objNodeTarget.intId)].weight.getLower()))
                                        for objNodeSource in
                                        dictLayers[intLayerNum].dictNodes.values()) +
                               np.float64(objNodeTarget.bias.getLower()) <=
                               dictXTarget[objNodeTarget.intId])

            # constraints for upper limit for the Target Nodes
            if lastRelu == LastRelu.YES:
                grbModel.addConstr(quicksum((dictXSource[objNodeSource.intId] *
                                             objNodeSource.intSize * np.float64(dictConnections[
                                                                                    intLayerNum].dictEdges[
                                                                                    (objNodeSource.intId,
                                                                                     objNodeTarget.intId)].weight.getUpper()))
                                            for objNodeSource in
                                            dictLayers[intLayerNum].dictNodes.values()) +
                                   np.float64(objNodeTarget.bias.getUpper()) +
                                   (np.float64(dictMTarget[objNodeTarget.intId]) *
                                    dictQTarget[objNodeTarget.intId]) >=
                                   dictXTarget[objNodeTarget.intId])

                # constraints ensuring equality
                grbModel.addConstr(dictXTarget[objNodeTarget.intId] >= 0)
                grbModel.addConstr(
                    dictMTarget[objNodeTarget.intId] * (1 - dictQTarget[objNodeTarget.intId]) >=
                    dictXTarget[objNodeTarget.intId])

            else:
                grbModel.addConstr(quicksum((dictXSource[objNodeSource.intId] *
                                             objNodeSource.intSize * np.float64(dictConnections[
                                                                                    intLayerNum].dictEdges[
                                                                                    (objNodeSource.intId,
                                                                                     objNodeTarget.intId)].weight.getUpper()))
                                            for objNodeSource in
                                            dictLayers[intLayerNum].dictNodes.values()) +
                                   np.float64(objNodeTarget.bias.getUpper()) >=
                                   dictXTarget[objNodeTarget.intId])

        # Update the gurobi Model
        grbModel.update()
        # return the updated model
        return grbModel

    def __createCForFirstLayer__(self, dictMTarget: Dict[int, DataType.RealType],
                                 dictXSource: Dict[int, Var],
                                 dictXTarget: Dict[int, Var],
                                 dictQTarget: Dict[int, Var], grbModel: Model,
                                 intLayerNum: int, listSign: List[Sign]) -> Model:
        """
        :param dictMTarget: dictionary between Node ids and its upper bound
        for the target layer
        :type dictMTarget: Dict[int, float]
        :param dictXSource: dictionary between Node ids and real gurobi variables
        for the source layer
        :type dictXSource: Dict[int, Var]
        :param dictXTarget: dictionary between Node ids and real gurobi variables
        for the target layer
        :type dictXTarget: Dict[int, Var]
        :param dictQTarget: dictionary between Node ids and boolean variables
        for the target layer
        :type dictQTarget: Dict[int, Var]
        :param grbModel: an instance of gurobi Model
        :type grbModel: Model
        :param intLayerNum: source layer number
        :type intLayerNum: int
        :param listSign: list of signs
        :type listSign: List[Sign]
        :param varMap: dictionary between gurobi variables
        :type varMap: Dict[Var, Var]
        :return: (grbModel -> Model)
        updated gurobi model
        """
        # Attributes of GNN
        objGNN: GNN = self.__objGNN__
        dictLayers: Dict[int, Layer] = objGNN.getDictLayers()
        dictConnections: Dict[int, Connection] = objGNN.getDictConnections()
        # Node instances of the target layer
        for objNodeTarget in dictLayers[intLayerNum + 1].dictNodes.values():
            # constraints for lower limit for the Target Nodes\
            grbModel.addConstr(quicksum((dictXSource[objNodeSource.intId] *
                                         objNodeSource.intSize *
                                         np.float64(self.__getWeight__(
                                             dictConnections[intLayerNum].dictEdges[(objNodeSource.intId,
                                                                                     objNodeTarget.intId)].weight,
                                             listSign[objNodeSource.intId - 1]).getLower()))
                                        for objNodeSource in dictLayers[intLayerNum].dictNodes.values()) +
                               np.float64(objNodeTarget.bias.getLower()) <= dictXTarget[objNodeTarget.intId])

            grbModel.addConstr(quicksum((dictXSource[objNodeSource.intId] *
                                         objNodeSource.intSize * np.float64(self.__getWeight__(
                        dictConnections[intLayerNum].dictEdges[(objNodeSource.intId,
                                                                objNodeTarget.intId)].weight,
                        listSign[objNodeSource.intId - 1]).getUpper()))
                                        for objNodeSource in dictLayers[intLayerNum].dictNodes.values()) +
                               np.float64(objNodeTarget.bias.getUpper()) + (dictMTarget[objNodeTarget.intId] *
                                                                            dictQTarget[objNodeTarget.intId]) >=

                                   dictXTarget[objNodeTarget.intId])

            # constraints ensuring equality
            grbModel.addConstr(dictXTarget[objNodeTarget.intId] >= 0)
            grbModel.addConstr(
                np.float64(dictMTarget[objNodeTarget.intId]) * (1 - dictQTarget[objNodeTarget.intId]) >=
                dictXTarget[objNodeTarget.intId])

        # Update the gurobi Model
        grbModel.update()

        # return the updated model
        return grbModel

    def __getWeight__(self, objNumber: Number, sign: Sign) -> Number:
        """
            Return weight based on sign on input value
            :param objNumber: an instance of Number
            :type objNumber: Number
            :param sign: Sign of an input node
            :type sign: Sign
            :return: (weight -> Number)
            """
        weight: Number = IReal(0.0, 0.0)
        if sign == Sign.POS:
            weight = objNumber
        elif sign == Sign.NEG:
            weight = IReal(objNumber.getUpper(), objNumber.getLower())
        return weight
