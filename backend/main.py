from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import json
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from configparser import ConfigParser
import subprocess
import csv
import sys
from tempfile import TemporaryDirectory


app = FastAPI()

ALLOWED_EXTENSIONS = {
    "network": {
        ".onnx",
        ".nnet",
        ".sherlock",
        ".isherlock"
    },
    "property": {".vnnlib"},
    "dynamics": {".py"}
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



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

class NNVerificationSettings(BaseModel):

    probType: str
    absRequired: str
    numOfAbsNodes: int
    technique: str
    lastRelu: str
    absType: str
    partitionType: str
    solverType: str



class NNCSVerificationSettings(BaseModel):

    probType: str
    absRequired: str
    numOfAbsNodes: int
    technique: str
    lastRelu: str
    K: int
    absType: str
    partitionType: str
    solverType: str


def build_nn_config(
        settings: NNVerificationSettings,
        network_path: Path,
        property_path: Path
):

    config = ConfigParser()
    config.optionxform = str

    config["settings"] = {
    "nnformat": '"ONNX"',
    "nnpath": f'"{network_path}"',
    "specpath": f'"{property_path}"',

    "probType": f'"{settings.probType}"',
    "absRequired": f'"{settings.absRequired}"',
    "numOfAbsNodes": str(settings.numOfAbsNodes),
    "technique": f'"{settings.technique}"',
    "lastRelu": f'"{settings.lastRelu}"',
    "absType": f'"{settings.absType}"',
    "partitionType": f'"{settings.partitionType}"',
    "solverType": f'"{settings.solverType}"'
}

    return config

def build_nncs_config(
        settings: NNCSVerificationSettings,
        network_path: Path,
        dyn_path: Path,
        property_path: Path
):

    config = ConfigParser()
    config.optionxform = str

    config["settings"] = {
    "nnformat": '"ONNX"',
    "nnpath": f'"{network_path}"',
    "dynpath": f'"{dyn_path}"',
    "specpath": f'"{property_path}"',

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


def validate_extension(file: UploadFile, allowed_extensions: set[str]):
    filename = file.filename or ""
    filename = filename.lower()

    if not any(filename.endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type for {filename}. Allowed: {', '.join(allowed_extensions)}"
        )
    

@app.post("/run-nncs-averinn")
async def run_nncs_averinn(
    settings: str = Form(...),
    network_file: UploadFile = File(...),
    property_file: UploadFile = File(...),
    dynamics_file: UploadFile = File(...)
):
    settings_dict = json.loads(settings)
    settings_obj = NNCSVerificationSettings(**settings_dict)

    project_root = Path(__file__).resolve().parent.parent
    script_path_tool = project_root / "nncs-averinn.py"

    validate_extension(network_file, ALLOWED_EXTENSIONS["network"])
    validate_extension(property_file, ALLOWED_EXTENSIONS["property"])
    validate_extension(dynamics_file, ALLOWED_EXTENSIONS["dynamics"])

    with TemporaryDirectory(prefix="averinn_run_") as temp_dir:
        run_dir = Path(temp_dir)

        network_path = run_dir / network_file.filename
        property_path = run_dir / property_file.filename
        dyn_path = run_dir / dynamics_file.filename

        generated_config_path = run_dir / "generated-nncs-config.ini"

        network_path.write_bytes(await network_file.read())
        dyn_path.write_bytes(await dynamics_file.read())
        property_path.write_bytes(await property_file.read())


        generated_config = build_nncs_config(
            settings_obj,
            network_path=network_path,
            dyn_path=dyn_path,
            property_path=property_path
        )




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


@app.post("/run-nn-averinn")
async def run_nn_averinn(
    settings: str = Form(...),
    network_file: UploadFile = File(...),
    property_file: UploadFile = File(...)
):
    settings_dict = json.loads(settings)
    settings_obj = NNVerificationSettings(**settings_dict)

    project_root = Path(__file__).resolve().parent.parent
    script_path_tool = project_root / "nn-averinn.py"

    validate_extension(network_file, ALLOWED_EXTENSIONS["network"])
    validate_extension(property_file, ALLOWED_EXTENSIONS["property"])

    with TemporaryDirectory(prefix="averinn_run_") as temp_dir:
        run_dir = Path(temp_dir)

        network_path = run_dir / network_file.filename
        property_path = run_dir / property_file.filename

        generated_config_path = run_dir / "generated-nn-config.ini"

        network_path.write_bytes(await network_file.read())
        property_path.write_bytes(await property_file.read())


        generated_config = build_nn_config(
            settings_obj,
            network_path=network_path,
            property_path=property_path
        )

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

