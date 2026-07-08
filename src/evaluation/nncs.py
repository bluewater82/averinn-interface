from typing import List, Dict

import numpy as np
import numpy.typing as npt
from src.abstraction.abstraction import Abstraction
from src.dyn.dtdyn import DtDyn
from src.gnn.gnn import GNN
from src.parser.onnxtonn import ONNX
from src.parser.parser import Parser
from src.parser.sherlock import Sherlock
from src.partition.partition import Partition
from src.rasc.milp import Milp
from src.rasc.setpropagation import SetPropagation
from src.rasc.technique import Technique
from src.set.box import Box
from src.set.set import Set
from src.set.setuts import SetUTS
from src.types.abstype import AbsType
from src.types.lastrelu import LastRelu
from src.types.partitiontype import PartitionType
from src.types.solvertype import SolverType
from src.types.techniquetype import TechniqueType
from src.utilities.log import Log
from src.utilities.spec import Spec
from src.utilities.vnnlib import VNNLib

##################################
########## Parameters  ###########
##################################
solverType: SolverType = SolverType.Gurobi
techniqueType: TechniqueType = TechniqueType.MILP
lastRelu: LastRelu = LastRelu.NO
partitionType: PartitionType = PartitionType.FIXED
absType: AbsType = AbsType.INTERVAL
K: int = 1

##################################
###### Discrete Dynamics  ########
##################################
A : npt.ArrayLike = np.array([[1, 1], [1, -1]], dtype= np.float64)
B : npt.ArrayLike = np.array([[0], [1]], dtype= np.float64)
objDtDyn: DtDyn = DtDyn(2, 1, A, B)

##################################
##### Initiate the log file ######
##################################
f = open("log.txt", "w")
f.close()

##################################
#####   Reading Examples    ######
##################################
filePath: str =  "/Users/ratanlal/Desktop/repositories/github/AVERINN/resources/test/example.txt"
objParser: Parser = Sherlock(filePath)

# create an instance of GNN
objGNN: GNN = objParser.getNetwork()

Log.message("Original Network Structure\n")
objGNN.display()

##################################
#####  Input specification #######
##################################
arrayLow: npt.ArrayLike = np.array([1, 1], dtype= np.float64)
arrayHigh: npt.ArrayLike = np.array([2, 2], dtype= np.float64)
objStateSet: Set = Box(arrayLow, arrayHigh)

####################################
####  Abstraction of GNN ###########
####################################
objPartition: Partition = Partition(objGNN, partitionType, 2, None)
dictPartition: Dict[int, Dict[int, set[int]]] = objPartition.getPartition()
objAbstraction: Abstraction = Abstraction(objGNN, dictPartition, absType)
objGNNAbs: GNN = objAbstraction.getAbstraction()

####################################
########### Test Reach set #########
####################################

for i in range(K):
    objTechnique: Technique = None
    if techniqueType == TechniqueType.MILP:
        objTechnique = Milp(objGNNAbs, objStateSet, None, solverType, lastRelu)
        listSets: List[Set] = objTechnique.reachSet()
        objInputSet = SetUTS.rangeOfSets(listSets)
        objStateSet = objStateSet.linearMap(objDtDyn.A).minkowskiSum(objInputSet.linearMap(objDtDyn.B))
    elif techniqueType == TechniqueType.PROPAGATION:
        objStateSet = SetUTS.toIntervalStarSet(objStateSet)
        objTechnique = SetPropagation(objGNNAbs, objStateSet, None, solverType, lastRelu)
        listSets: List[Set] = objTechnique.reachSet()
        objInputSet = SetUTS.rangeOfSets(listSets)
        objInputSet = SetUTS.toIntervalStarSet(objInputSet)
        objStateSet = objStateSet.linearMap(objDtDyn.A, objDtDyn.A).minkowskiSum(objInputSet.linearMap(objDtDyn.B, objDtDyn.B))


Log.message("Reach Set \n")
rangeSet = objStateSet.getRange()
Log.message(str(rangeSet[0])+"\n")
Log.message(str(rangeSet[1])+"\n")
Log.message("Reach set ends\n")
