import sys
from typing import List, Dict, Tuple

import numpy as np
import numpy.typing as npt
from src.abstraction.abstraction import Abstraction
from src.dyn.dtdyn import DtDyn
from src.gnn.gnn import GNN
from src.parser.isherlock import ISherlock
from src.parser.nnet import Nnet
from src.parser.onnxtonn import ONNX
from src.parser.parser import Parser
from src.parser.parseruts import ParserUTS
from src.parser.sherlock import Sherlock
from src.partition.partition import Partition
from src.rasc.milp import Milp
from src.rasc.setpropagation import SetPropagation
from src.rasc.technique import Technique
from src.set.box import Box
from src.set.set import Set
from src.set.setuts import SetUTS
from src.solver.gurobi import Gurobi
from src.types.abstype import AbsType
from src.types.lastrelu import LastRelu
from src.types.partitiontype import PartitionType
from src.types.probtype import ProbType
from src.types.solvertype import SolverType
from src.types.techniquetype import TechniqueType
from src.utilities.log import Log
import configparser
import json
import time

from src.utilities.spec import Spec
from src.utilities.vnnlib import VNNLib

stime = time.time()
##################################
##### Parse Config.ini File ######
##################################
if len(sys.argv) < 2 or len(sys.argv) > 2:
    print("Terminated")
    sys.exit(0)
inifile = configparser.ConfigParser()
inifile.read(sys.argv[1], 'UTF-8')
# In nncs-config.ini, please specify type of neural network file format
# "SHERLOCK"/"ONNX"/"NNET"/"ISHERLOCK"
nnformat: str = json.loads(inifile.get('settings', 'nnformat'))
# In nncs-config.ini, please specify correct path of neural network
nnpath: str = json.loads(inifile.get('settings', 'nnpath'))
# path of the dynamics
dynpath: str = json.loads(inifile.get('settings', 'dynpath'))
# In nncs-config.ini, please specify correct path of spec if needed, "NONE" otherwise
specpath: str = json.loads(inifile.get('settings', 'specpath'))
# Problem type "REACH" or "SAFETY"
probTypeC: str = json.loads(inifile.get('settings', 'probType'))
# In nncs-config.ini, please specify problem type ("REACH"/"SAFETY")
# In nncs-config.ini, please use "YES" if you want to abstract, "NO" otherwise
absRequired: str = json.loads(inifile.get('settings', 'absRequired'))
# In nncs-config.ini, specify  abstraction type. Since the tool supports only
# interval abstraction, use "INTERVAL"
absTypeC: str = json.loads(inifile.get('settings', 'absType'))
# In nncs-config.ini, specify number of abstract nodes needed at hidden layers
numOfAbsNodes: int = json.loads(inifile.get('settings', 'numOfAbsNodes'))
# In nncs-config.ini, specify solver. Since the tool uses only
# Gurobi solver, use "Gurobi"
solverTypeC: str = json.loads(inifile.get('settings', 'solverType'))
# In nncs-config.ini, specify technique. The tool supports two techniques MILP/PROPAGATION.
# Use one of them like "MILP" or "PROPAGATION"
techniqueC: str = json.loads(inifile.get('settings', 'technique'))
# In nncs-config.ini, specify whether relu will be applied on the last layer
# Use "YES" or "NO"
lastReluC: str = json.loads(inifile.get('settings', 'lastRelu'))
# In config,ini, specify partition strategy, since only one strategy is
# is implemented which is fixed. Use "FIXED"
partitionTypeC: str = json.loads(inifile.get('settings', 'partitionType'))
# Number of time steps
K: int = json.loads(inifile.get('settings', 'K'))

# Reading matrics corresponding to dynamics
dynfile = configparser.ConfigParser()
dynfile.read(dynpath, 'UTF-8')

# Dimension of the dynamical system
sysDim: int = json.loads(dynfile.get('settings', 'sysDim'))
# Dimension of the input in the dynamical system
inputDim: int = json.loads(dynfile.get('settings', 'inputDim'))
# Matrix A for linear discrete time system AX+BU+C
matAC = json.loads(dynfile.get('settings', 'matA'))
# Matrix B for linear discrete time system AX+BU+C
# If B is not in the dynamics, use NONE
matBC = json.loads(dynfile.get('settings', 'matB'))
# Vector C for linear discrete time system AX+BU+C
# If C is not in the dynamics, use NONE
arrayCC = json.loads(dynfile.get('settings', 'arrayC'))
# listA for unsafe set
listA = json.loads(dynfile.get('settings', 'listA'))
# listB for unsafe set
listB = json.loads(dynfile.get('settings', 'listB'))
##################################
########## Parameters  ###########
##################################
probType = ProbType.UNKNOWN
if probTypeC == "REACH":
    probType = ProbType.REACH
elif probTypeC == "SAFETY":
    probType = ProbType.SAFETY

absType = AbsType.UNKNOWN
partitionType = PartitionType.UNKNOWN
if absRequired == "YES":
    if absTypeC == "INTERVAL":
        absType = AbsType.INTERVAL
    if partitionTypeC == "FIXED":
        partitionType = PartitionType.FIXED

solverType = SolverType.UNKNOWN
if solverTypeC == "Gurobi":
    solverType = SolverType.Gurobi

techniqueType = TechniqueType.UNKNOWN
if techniqueC == "MILP":
    techniqueType = TechniqueType.MILP
elif techniqueC == "PROPAGATION":
    techniqueType = TechniqueType.PROPAGATION

lastRelu = LastRelu.YES
if lastReluC == "NO":
    lastRelu = LastRelu.NO

matA = None
if matAC != "NONE":
    matA = np.array(matAC, dtype=np.float64)

matB = None
if matBC != "NONE":
    matB = np.array(matBC, dtype=np.float64)

arrayC = None
if arrayCC != "NONE":
    arrayC = np.array(arrayCC, dtype=np.float64)

objDtDyn: DtDyn = DtDyn(sysDim, inputDim, matA, matB, arrayC)

##################################
##### Initiate the log file ######
##################################
f = open("log.txt", "w")
f.close()

##################################
##### Read and Create Network ####
##################################
objParser: Parser = None
if nnformat == "SHERLOCK":
    objParser = Sherlock(nnpath)
elif nnformat == "ONNX":
    objParser = ONNX(nnpath)
elif nnformat == "NNET":
    objParser = Nnet(nnpath)
elif nnformat == "ISHERLOCK":
    objParser = ISherlock(nnpath)

# create an instance of GNN
objGNN = objParser.getNetwork()
print(objGNN.getDictNumNeurons())
# Convert into robust GNN
#robustValue: np.float64 = np.float64(0.0001)
#objGNN = ParserUTS.toRobustGNN(objGNN, robustValue)

Log.message("Neural Network\n")
objGNN.display()
##############################################
##### Read input and output specification ####
##############################################
dictNeurons = objGNN.getDictNumNeurons()
ioSpec = VNNLib.read_vnnlib_simple(specpath, dictNeurons[1], dictNeurons[1])
objStateSet = Spec.getInput(ioSpec)
outputConstr = Spec.getOutput(ioSpec)
#outputConstr = (np.array(listA, dtype=np.float64), np.array(listB, dtype=np.float64))
Log.message("Specification\n")
Log.message("       Input Set\n")
Log.message("       Lower: "+str(objStateSet.getArrayLow())+"\n")
Log.message("       Upper: "+str(objStateSet.getArrayHigh())+"\n")
Log.message("       Unsafe Property (A1X<=b1 or A2X<=b1 or ...)\n")
Log.message(Spec.display(outputConstr))
####################################
####  Abstraction of GNN ###########
####################################
objGNNAbs: GNN = None
if absRequired == "YES":
    objPartition: Partition = Partition(objGNN, partitionType, numOfAbsNodes, None)
    dictPartition: Dict[int, Dict[int, set[int]]] = objPartition.getPartition()
    objAbstraction: Abstraction = Abstraction(objGNN, dictPartition, absType)
    objGNNAbs = objAbstraction.getAbstraction()

####################################
#### Solution based on problems ####
####################################
objGNNUse: GNN = None
if absRequired == "YES":
    objGNNUse = objGNNAbs
    try:
        objGNNUse.display()
    except:
        Log.message("For Large Network no display")
else:
    objGNNUse = objGNN


for i in range(K):
    Log.message("Interation"+str(i)+"\n")
    if techniqueType == TechniqueType.MILP:
        objTechnique = Milp(objGNNUse, objStateSet, outputConstr, solverType, lastRelu)
        listSets: List[Set] = objTechnique.reachSet()
        objInputSet = SetUTS.rangeOfSets(listSets)
        rangeSet = objInputSet.getRange()
        Log.message("       Lower: " + str(rangeSet[0]) + "\n")
        Log.message("       Upper: " + str(rangeSet[1]) + "\n")
        if (objDtDyn.B is None) and (objDtDyn.C is None):
            objStateSet = objStateSet.linearMap(objDtDyn.A, objDtDyn.A)
        elif (objDtDyn.B is None) and (objDtDyn.C is not None):
            objStateSet = objStateSet.affineMap(objDtDyn.A, objDtDyn.C, objDtDyn.A, objDtDyn.C)
        elif (objDtDyn.B is not None) and (objDtDyn.C is None):
            objStateSet = objStateSet.linearMap(objDtDyn.A, objDtDyn.A)
            linearInput = objInputSet.linearMap(objDtDyn.B, objDtDyn.B)
            objStateSet = objStateSet.minkowskiSum(linearInput)
        elif (objDtDyn.B is not None) and (objDtDyn.C is not None):
            objStateSet = objStateSet.linearMap(objDtDyn.A, objDtDyn.A)
            affineInput = objInputSet.affineMap(objDtDyn.B, objDtDyn.C, objDtDyn.B, objDtDyn.C)
            objStateSet = objStateSet.minkowskiSum(affineInput)

    elif techniqueType == TechniqueType.PROPAGATION:
        if absRequired == "YES":
            objStateSet = SetUTS.toIntervalStarSet(objStateSet)
        else:
            objStateSet = SetUTS.toStarSet(objStateSet)
        Log.message("Number of Interval Stars after each layer\n")
        objTechnique = SetPropagation(objGNNUse, objStateSet, outputConstr, solverType, lastRelu)
        listSets: List[Set] = objTechnique.reachSet()
        objInputSet = SetUTS.rangeOfSets(listSets)
        rangeSet = objInputSet.getRange()
        Log.message("       Lower: " + str(rangeSet[0]) + "\n")
        Log.message("       Upper: " + str(rangeSet[1]) + "\n")
        if absRequired == "YES":
            objInputSet = SetUTS.toIntervalStarSet(objInputSet)
        else:
            objInputSet = SetUTS.toStarSet(objInputSet)
        if (objDtDyn.B is None) and (objDtDyn.C is None):
            objStateSet = objStateSet.linearMap(objDtDyn.A, objDtDyn.A)
        elif (objDtDyn.B is None) and (objDtDyn.C is not None):
            objStateSet = objStateSet.affineMap(objDtDyn.A, objDtDyn.C, objDtDyn.A, objDtDyn.C)
        elif (objDtDyn.B is not None) and (objDtDyn.C is None):
            objStateSet = objStateSet.linearMap(objDtDyn.A, objDtDyn.A)
            objInputSet = objInputSet.linearMap(objDtDyn.B, objDtDyn.B)
            objStateSet = objStateSet.minkowskiSum(objInputSet)
        elif (objDtDyn.B is not None) and (objDtDyn.C is not None):
            objStateSet = objStateSet.linearMap(objDtDyn.A, objDtDyn.A)
            objStateSet = objStateSet.minkowskiSum(objInputSet.affineMap(objDtDyn.B, objDtDyn.C,
                                                                         objDtDyn.B, objDtDyn.C))
    Log.message("Reach Set \n")
    rangeSet = objStateSet.getRange()
    Log.message("       Lower: " + str(rangeSet[0]) + "\n")
    Log.message("       Upper: " + str(rangeSet[1]) + "\n")


if probType == ProbType.REACH:
    Log.message("Reach Set \n")
    rangeSet = objStateSet.getRange()
    Log.message("       Lower: "+str(rangeSet[0])+"\n")
    Log.message("       Upper: "+str(rangeSet[1])+"\n")
elif probType == ProbType.SAFETY:
    Log.message("Reach Set \n")
    rangeSet = objStateSet.getRange()
    Log.message("       Lower: " + str(rangeSet[0]) + "\n")
    Log.message("       Upper: " + str(rangeSet[1]) + "\n")
    Log.message("Safety Checking \n")
    isIntersect: bool = SetUTS.intersectWithUnsafe(objStateSet, outputConstr, solverType, 'X')
    if isIntersect:
        Log.message("Safety Status: Unsafe \n")
    else:
        Log.message("Safety Status: Safe \n")

etime = time.time()
print(etime-stime)