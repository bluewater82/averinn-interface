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

from src.utilities.spec import Spec
from src.utilities.vnnlib import VNNLib

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
Log.message("Neural Network\n")
try:
    objGNN.display()
except:
    Log.message("For Large Network no display")
##############################################
##### Read input and output specification ####
##############################################
dictNeurons = objGNN.getDictNumNeurons()
ioSpec = VNNLib.read_vnnlib_simple(specpath, dictNeurons[1], dictNeurons[1])
objStateSet = Spec.getInput(ioSpec)
outputConstr = Spec.getOutput(ioSpec)
Log.message("Specification\n")
Log.message("       Input Set\n")
Log.message("       Lower: " + str(objStateSet.getArrayLow()) + "\n")
Log.message("       Upper: " + str(objStateSet.getArrayHigh()) + "\n")
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


if techniqueType == TechniqueType.MILP:
    objTechnique = Milp(objGNNUse, objStateSet, outputConstr, solverType, lastRelu)
    listSets: List[Set] = objTechnique.reachSet()
    objStateSet = SetUTS.rangeOfSets(listSets)

elif techniqueType == TechniqueType.PROPAGATION:
    if absRequired == "YES":
        objStateSet = SetUTS.toIntervalStarSet(objStateSet)
    else:
        objStateSet = SetUTS.toStarSet(objStateSet)
    Log.message("Number of Interval Stars after each layer\n")
    objTechnique = SetPropagation(objGNNUse, objStateSet, outputConstr, solverType, lastRelu)
    listSets: List[Set] = objTechnique.reachSet()
    objStateSet = SetUTS.rangeOfSets(listSets)

if probType == ProbType.REACH:
    Log.message("Reach Set \n")
    rangeSet = objStateSet.getRange()
    Log.message("       Lower: " + str(rangeSet[0]) + "\n")
    Log.message("       Upper: " + str(rangeSet[1]) + "\n")
elif probType == ProbType.SAFETY:
    Log.message("Reach Set \n")
    rangeSet = objStateSet.getRange()
    Log.message("       Lower: " + str(rangeSet[0]) + "\n")
    Log.message("       Upper: " + str(rangeSet[1]) + "\n")
    Log.message("Safety Checking \n")
    isIntersect: bool = SetUTS.intersectWithUnsafe(objStateSet, outputConstr, solverType, 'Y')
    if isIntersect:
        Log.message("Safety Status: Unsafe \n")
    else:
        Log.message("Safety Status: Safe \n")
