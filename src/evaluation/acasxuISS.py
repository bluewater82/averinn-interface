from typing import List, Dict
from pandas import DataFrame
from src.abstraction.abstraction import Abstraction
from src.gnn.gnn import GNN
from src.parser.onnxtonn import ONNX
from src.parser.parser import Parser
from src.partition.partition import Partition
from src.rasc.setpropagation import SetPropagation
from src.rasc.technique import Technique
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
import time

#########################################
### Dictionaries for Time and Result #####
#########################################
dictTime: Dict[int, float] = dict()
dictInputLow: Dict[int, Dict[int, float]] = dict()
dictInputHigh: Dict[int, Dict[int, float]] = dict()
dictOutputLow: Dict[int, Dict[int, float]] = dict()
dictOutputHigh: Dict[int, Dict[int, float]] = dict()
dictConfig: Dict[int, Dict[int, int]] = dict()
start_time = time.time()
##################################
########## Parameters  ###########
##################################
solverType: SolverType = SolverType.Gurobi
techniqueType: TechniqueType = TechniqueType.MILP
lastRelu: LastRelu = LastRelu.NO
partitionType: PartitionType = PartitionType.FIXED
absType: AbsType = AbsType.INTERVAL

##################################
##### Initiate the log file ######
##################################
f = open("log.txt", "w")
f.close()

##################################
##### Reading ONNX and spec ######
##################################
# read acasxu benchmark files and specification
#listPartition = [5, 10, 15, 20, 25]
listPartition = [5]
itr = 1
for size in listPartition:
    for j in range(1, 2, 1):
        for k in range(1, 2, 1):
            filePath: str = \
                "/Users/ratanlal/Desktop/repositories/github/AVERINN/resources/acasxuonnx/ACASXU_run2a_1_1_batch_2000.onnx"

            specPath: str = (
                    "/Users/ratanlal/Desktop/repositories/github/AVERINN/resources/acasxuvnnlib/prop_" +
                    str(1) + ".vnnlib")


            objParser: Parser = ONNX(filePath)

            # create an instance of GNN
            objGNN: GNN = objParser.getNetwork()

            i, o, d = VNNLib.get_num_inputs_outputs(filePath)
            ioSpec = VNNLib.read_vnnlib_simple(specPath, i, o)
            objInputSet = Spec.getInput(ioSpec)
            outputConstr = Spec.getOutput(ioSpec)
            Log.message("Input Set\n")
            Log.message(str(objInputSet.getArrayLow())+"\n")
            Log.message(str(objInputSet.getArrayHigh())+"\n")
            dictInputLow[itr] = dict()
            dictInputHigh[itr] = dict()
            for p in range(objInputSet.getDimension()):
                dictInputLow[itr][p] = objInputSet.getArrayLow()[p]
                dictInputHigh[itr][p] = objInputSet.getArrayHigh()[p]
            Log.message("Specification\n")
            Log.message(str(outputConstr[0])+"\n")
            Log.message(str(outputConstr[1])+"\n")

            ####################################
            ####  Abstraction of GNN ###########
            ####################################
            objPartition: Partition = Partition(objGNN, partitionType, size, None)
            dictPartition: Dict[int, Dict[int, set[int]]] = objPartition.getPartition()
            objAbstraction: Abstraction = Abstraction(objGNN, dictPartition, absType)
            objGNNAbs: GNN = objAbstraction.getAbstraction()
            Log.message("Network Configuration\n")
            dictConfig[itr] = objGNNAbs.getDictNumNeurons()
            Log.message(str(objGNNAbs.getDictNumNeurons())+"\n")
            ####################################
            ########### Test Reach set #########
            ####################################
            objInputSet = SetUTS.toIntervalStarSet(objInputSet)
            objTechnique: Technique = SetPropagation(objGNNAbs, objInputSet, outputConstr, solverType, lastRelu)
            Log.message("Reach Set \n")
            listSets: List[Set] = objTechnique.reachSet()
            rangeSet: Set = SetUTS.rangeOfSets(listSets)
            dictOutputLow[itr] = dict()
            dictOutputHigh[itr] = dict()
            for p in range(rangeSet.getDimension()):
                dictOutputLow[itr][p] = rangeSet.getArrayLow()[p]
                dictOutputHigh[itr][p] = rangeSet.getArrayHigh()[p]
            Log.message(str(rangeSet.getArrayLow())+"\n")
            Log.message(str(rangeSet.getArrayHigh())+"\n")
            Log.message("Reach set ends\n")
            end_time = time.time()
            dictTime[itr] = end_time - start_time
            itr += 1
            Log.message("Total Time: "+ str(end_time - start_time) + "\n")

lstOutputLow1 = []
lstOutputHigh1 = []
for key in dictOutputLow.keys():
        lstOutputLow1.append(dictOutputLow[key][0])
        lstOutputHigh1.append(dictOutputHigh[key][0])

lstOutputLow2 = []
lstOutputHigh2 = []
for key in dictOutputLow.keys():
        lstOutputLow2.append(dictOutputLow[key][1])
        lstOutputHigh2.append(dictOutputHigh[key][1])

lstOutputLow3 = []
lstOutputHigh3 = []
for key in dictOutputLow.keys():
        lstOutputLow3.append(dictOutputLow[key][2])
        lstOutputHigh3.append(dictOutputHigh[key][2])

lstOutputLow4 = []
lstOutputHigh4 = []
for key in dictOutputLow.keys():
        lstOutputLow4.append(dictOutputLow[key][3])
        lstOutputHigh4.append(dictOutputHigh[key][3])

lstOutputLow5 = []
lstOutputHigh5 = []
for key in dictOutputLow.keys():
        lstOutputLow5.append(dictOutputLow[key][4])
        lstOutputHigh5.append(dictOutputHigh[key][4])

lstTime = []
for key in dictTime.keys():
    lstTime.append(dictTime[key])

lstConfig = []
for key in dictConfig.keys():
    strConfig = ""
    for key1 in dictConfig[key].keys():
        strConfig += str(dictConfig[key][key1])+", "
    lstConfig.append(strConfig)

df = DataFrame({'Config': lstConfig, 'Time': lstTime, 'Y1_Low':lstOutputLow1, 'Y1_High':lstOutputHigh1,
                'Y2_Low':lstOutputLow2, 'Y2_High':lstOutputHigh2,
                'Y3_Low':lstOutputLow3, 'Y3_High':lstOutputHigh3,
                'Y4_Low': lstOutputLow4, 'Y3_High': lstOutputHigh4,
                'Y5_Low': lstOutputLow5, 'Y5_High': lstOutputHigh5})
df.to_csv("data.csv")

