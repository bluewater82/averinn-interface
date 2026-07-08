from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import simulinkengine as sl
import numpy as np
import onnx
from pathlib import Path
from configparser import ConfigParser
import subprocess
import csv
import sys

simulink = None
nnpath = None
app = FastAPI()
app.add_middleware(CORSMiddleware,
                   allow_origins="http://localhost:5173",
                   allow_methods=["*"],
                   allow_headers=["*"],
                   expose_headers=["*"])

class slxfileOpts(BaseModel):
    stepSize: float
    timePeriod: float
    initialState: list[float] = "[]"

"""
Takes in a simulink file and initializes the simulinkengine to be ready to simulate that file.
"""
@app.post("/slxfile")
async def read_file(file: UploadFile = File(...)):
    global simulink
    global nnpath
    try:
        if simulink is not None:
            simulink.close()
        with open("./ri_simulink.slx", 'wb') as f:
            while chunk := await file.read(65536):
                f.write(chunk)
        simulink = sl.sl("./ri_simulink.slx")
        if nnpath is not None:
            simulink.changeModelNN(nnpath)
        return {"message": f"Success: Simulink File Opened", "numIn": simulink.getNumIn(), "numOut": simulink.getNumOut()}
    except Exception as e:
        return {"err": "Could not launch simulink file", "errmessage": f"{e}"}

"""
/slxFileOpts
Takes in file options and sets the simulink object to use those options
"""
@app.post("/slxfileOpts")
async def read_json(opts: slxfileOpts):
    global simulink
    global nnpath
    try:
        assert simulink is not None
        simulink.params(opts.stepSize, opts.timePeriod, opts.initialState)
        if nnpath is not None:
            simulink.changeModelNN(nnpath)
        out = simulink.sim()
        di = {}
        for i in range(len(out[0])):
            di[np.array2string(out[0][i], formatter={'float_kind': lambda x: f"{x:.1f}"})] = str(out[1][i])
        return di
    except Exception as e:
        return {"err": "Server cannot accept request", "errmessage": f"{e}"}


"""
/slxnnfile
Read_file reads in a neural network file to be processed by the simulink engine.
This currently only supports ONNX Files, so that is what is checked for.
"""
@app.post("/slxnnfile")
async def read_file(file: UploadFile = File(...)):
    try:
        global nnpath
        with open("./ri_frontendfile.onnx", 'wb') as f:
           while chunk := await file.read(65536):
               f.write(chunk)
        global simulink
        if simulink is not None:
            simulink.changeModelNN("./ri_frontendfile.onnx")
        onnx_model = onnx.load("ri_frontendfile.onnx")
        onnx.checker.check_model(onnx_model)
        nnpath = "./ri_frontendfile.onnx"
        return {"message": f"Success! {file.filename}"}
    except Exception as e:
        return {"err" : "Could Not verify ONNX model", "errmessage": f"{e}"}


def summarize_csv(csv_path: Path):
    if not csv_path.exists():
        return None

    with open(csv_path, newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        return None

    return {
        "row_count": len(rows),
        "columns": reader.fieldnames,
        "initial_set": rows[0],
        "final_set": rows[-1]
    }

class VerificationSettings(BaseModel):

    probType: str
    absRequired: str
    numOfAbsNodes: int
    technique: str
    lastRelu: str
    K: int
    absType: str
    partitionType: str
    solverType: str



def build_config(settings: VerificationSettings):

    config = ConfigParser()
    config.optionxform = str

    config["settings"] = {
    "nnformat": '"ONNX"',
    "nnpath": '"./resources/nncs-benchmarks/acc/controller.onnx"',
    "dynpath": '"./resources/nncs-benchmarks/acc/dynamics.ini"',
    "specpath": '"./resources/nncs-benchmarks/acc/spec.vnnlib"',

    "probType": f'"{settings.probType}"',
    "absRequired": f'"{settings.absRequired}"',
    "numOfAbsNodes": str(settings.numOfAbsNodes),
    "technique": f'"{settings.technique}"',
    "lastRelu": f'"{settings.lastRelu}"',
    "K": str(settings.K),
    "absType": f'"{settings.absType}"',
    "partitionType": f'"{settings.partitionType}"',
    "solverType": f'"{settings.solverType}"'
}

    return config

@app.post("/run-nncs-averinn")
def run_nncs_averinn(settings: VerificationSettings):
    project_root = Path(__file__).resolve().parent.parent

    script_path_tool = project_root / "nncs-averinn.py"
    generated_config_path = project_root / "generated-nncs-config.ini"

    print("main.py is located at:", Path(__file__).resolve())
    print("project_root is:", project_root)
    print("generated config path is:", generated_config_path)
    
    generated_config = build_config(settings)
    with open(generated_config_path, "w") as config_file:
        generated_config.write(config_file)

    completed = subprocess.run(
        [sys.executable, str(script_path_tool), str(generated_config_path)],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=60
    )

    csv_path = project_root / "data.csv"
    log_path = project_root / "log.txt"
    lp_path = project_root / "model.lp"
    sol_path = project_root / "model.sol"

    csv_summary = summarize_csv(csv_path)

    safety_result = None

    if log_path.exists():
        with open(log_path, "r", encoding="utf-8", errors="replace") as file:
            lines = [line.strip() for line in file.readlines() if line.strip()]

        if lines:
            safety_result = lines[-1]

    return {
        "returncode": completed.returncode,
        "success": completed.returncode == 0,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "csv_summary": csv_summary,
        "safety_result": safety_result,
        "files_created": {
            "data_csv": csv_path.exists(),
            "log_txt": log_path.exists(),
            "model_lp": lp_path.exists(),
            "model_sol": sol_path.exists()
        }
    }