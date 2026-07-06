from hybrid.parser import *
import sys

from hybrid.reach import hybrid_k_step_reach
from src.dyn.dtdyn import DtDyn
from src.parser.onnxtonn import ONNX
from src.types.lastrelu import LastRelu
from src.types.solvertype import SolverType
from src.types.techniquetype import TechniqueType
from src.utilities.log import Log
from src.utilities.spec import Spec
from src.utilities.vnnlib import VNNLib
from src.gnn.gnn import GNN
from src.set.set import Set
from src.set.setuts import SetUTS

import configparser
import json
import time

stime = time.time()

def read_spec_input_output(objGNN, specpath, num_input, listA, listB):
    dictNeurons = objGNN.getDictNumNeurons()
    ioSpec = VNNLib.read_vnnlib_simple(specpath, num_input, dictNeurons[1])
    print(ioSpec)
    objStateSet = Spec.getInput(ioSpec)
    # outputConstr = Spec.getOutput(ioSpec)
    outputConstr = (np.array(listA, dtype=np.float64), np.array(listB, dtype=np.float64))
    print("Specification\n")
    print("       Input Set\n")
    print("       Lower: " + str(objStateSet.getArrayLow()) + "\n")
    print("       Upper: " + str(objStateSet.getArrayHigh()) + "\n")
    print("       Unsafe Property (A1X<=b1 or A2X<=b1 or ...)\n")
    Log.message(Spec.display(outputConstr))
    print(outputConstr)

    return objStateSet, outputConstr

def extract_hybrid_system(hybrid_sys_conf):
    hybrid_systems = []
    num_modes = hybrid_sys_conf["num_modes"]
    sysDim = hybrid_sys_conf["num_states"]
    inputDim = hybrid_sys_conf["num_inputs"]

    for i in range(1, num_modes + 1):
        mode_key = f'mode{i}'
        mode_dynamic_conf = hybrid_sys_conf["modes"][mode_key]
        print()
        C = np.array(mode_dynamic_conf["C"])
        # TODO : fix it in parser
        is_none_like = C is None or (isinstance(C, str) and C.lower() in {"none", "null"})

        if C is None or C.ndim == 0:
            mode_dynamic = DtDyn(sysDim, inputDim, mode_dynamic_conf["A"], mode_dynamic_conf["B"])
        else:
            mode_dynamic = DtDyn(sysDim, inputDim, mode_dynamic_conf["A"], mode_dynamic_conf["B"],
                                 mode_dynamic_conf["C"])
        hybrid_systems.append(mode_dynamic)
    return hybrid_systems, sysDim, inputDim

if __name__ == '__main__':
    inifile = configparser.ConfigParser()
    inifile.read("hybrid-nncs-config.ini", 'UTF-8')
    FILE_PATH = "./resources/hybrid-nncs-benchmarks/acc/acc_2region_onedelay.yaml"
    hybrid_sys_conf = load_hybrid_system(FILE_PATH)

    # 2. Load Reach Settings
    reach_set = load_controller_settings(FILE_PATH)

    # 3. Create NN Parser
    # Assuming format is ONNX based on your YAML key
    nn_parser = create_network_parser("ONNX", reach_set['onnx_path'])
    reach_set = load_controller_settings(FILE_PATH)

    specpath: str = json.loads(inifile.get('settings', 'specpath'))
    lastReluC: str = json.loads(inifile.get('settings', 'lastRelu'))
    K: int = json.loads(inifile.get('settings', 'K'))

    # 3. Create NN Parser
    # Assuming format is ONNX based on your YAML key
    solverType = SolverType.Gurobi
    techniqueType = TechniqueType.PROPAGATION
    lastRelu = LastRelu.YES
    if lastReluC == "NO":
        lastRelu = LastRelu.NO
    objParser = ONNX(nn_parser["path"])

    objGNNUse: GNN = None
    objGNN = objParser.getNetwork()
    objGNNUse = objGNN
    print(objGNN.getDictNumNeurons())

    # 4. Extract HS
    # Extract and transform HS for verification
    hybrid_systems, state_num, input_num = extract_hybrid_system(hybrid_sys_conf)
    m = hybrid_sys_conf["M"]
    b = hybrid_sys_conf["B"]

    listA = hybrid_sys_conf["listA"]
    listB = hybrid_sys_conf["listB"]
    # 5.Read input and output specification
    objStateSet, outputConstr = read_spec_input_output(objGNNUse, specpath, state_num, listA, listB)

    hybrid_k_step_reach(K, objGNNUse, hybrid_sys_conf, objStateSet,  hybrid_systems, solverType, lastRelu, outputConstr, m, b )
    stime = time.time()

    Log.message("Reach Set \n")
    rangeSet = objStateSet.getRange()
    Log.message("       Lower: " + str(rangeSet[0]) + "\n")
    Log.message("       Upper: " + str(rangeSet[1]) + "\n")
    Log.message("Safety Checking \n")

    etime = time.time()
    print(etime - stime)