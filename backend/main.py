"""
main.py

FastAPI backend for the AVERINN web interface.

Responsibilities:
- Receives uploaded benchmark files from the React frontend.
- Validates uploaded file types.
- Dynamically generates AVERINN configuration files.
- Executes either nn-averinn.py or nncs-averinn.py.
- Collects generated output files.
- Returns verification results to the frontend as JSON.
"""


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


# Allowed file extensions for each upload category.
# Files are validated before being written to disk.
ALLOWED_EXTENSIONS = {
    "network": {
        ".onnx",
        ".nnet",
        ".sherlock",
        ".isherlock"
    },
    "property": {".vnnlib"},
    "dynamics": {".ini"}
}


# Allow requests from local React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# Helper Functions
# ==========================================================

# Reads the CSV file generated from AVERINN and creates a summary that the
# frontend will use for displaying results to the user.
#
# Only the initial and final sets for each run/loop are returned rather than
# sending the entire CSV contents over the API.
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


# Ensures that uploaded files have the expected extensions before saving them
# to disk or passing to AVERINN.
#
# Additional layer of protection against accidental uploads of unsupported
# files.
def validate_extension(file: UploadFile, allowed_extensions: set[str]):
    filename = file.filename or ""
    filename = filename.lower()

    if not any(filename.endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type for {filename}. Allowed: {', '.join(allowed_extensions)}"
        )


# ==========================================================
# Request Models
# ==========================================================

# Settings expected from the frontend when running a standard NN verification
class NNVerificationSettings(BaseModel):
    probType: str
    absRequired: str
    numOfAbsNodes: int
    technique: str
    lastRelu: str
    absType: str
    partitionType: str
    solverType: str


# Settings expected from the frontend when running NNCS verification
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


# ==========================================================
# Configuration Builders
# ==========================================================

# Constructs a temporary nn-config.ini file based on the user's selected
# verification settings and uploaded files.
#
# The resulting ConfigParser object is written to disk immediately before
# launching AVERINN.
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


# Constructs a temporary nncs-config.ini file based on the user's selected
# verification settings and uploaded files.
#
# The resulting ConfigParser object is written to disk immediately before
# launching AVERINN.
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


# ==========================================================
# API Endpoints
# ==========================================================

# ------------------------------------------------------------------
# NNCS Verification Endpoint
#
# Workflow:
#   1. Receive uploaded files and verification settings.
#   2. Validate file types.
#   3. Create a temporary execution directory.
#   4. Generate an AVERINN configuration file.
#   5. Launch nncs-averinn.py.
#   6. Collect output artifacts.
#   7. Return results as JSON.
# ------------------------------------------------------------------
@app.post("/run-nncs-averinn")
async def run_nncs_averinn(
    settings: str = Form(...),
    network_file: UploadFile = File(...),
    property_file: UploadFile = File(...),
    dynamics_file: UploadFile = File(...)
):
    
    # Convert submitted JSON setting into object
    settings_dict = json.loads(settings)
    settings_obj = NNCSVerificationSettings(**settings_dict)

    # Locate the project root and tool script
    project_root = Path(__file__).resolve().parent.parent
    script_path_tool = project_root / "nncs-averinn.py"

    # Rejects attempt to write to disk if uploaded files do not meet
    # required extension types.
    validate_extension(network_file, ALLOWED_EXTENSIONS["network"])
    validate_extension(property_file, ALLOWED_EXTENSIONS["property"])
    validate_extension(dynamics_file, ALLOWED_EXTENSIONS["dynamics"])

    # Each verification run executes inside an isolated temp directory.
    # Prevents concurrent users from overwriting files.
    # TODO: Include a way to let users download results
    with TemporaryDirectory(prefix="averinn_run_") as temp_dir:
        run_dir = Path(temp_dir)

        network_path = run_dir / network_file.filename
        property_path = run_dir / property_file.filename
        dyn_path = run_dir / dynamics_file.filename

        generated_config_path = run_dir / "generated-nncs-config.ini"

        # Save uploaded files into temp workspace
        network_path.write_bytes(await network_file.read())
        dyn_path.write_bytes(await dynamics_file.read())
        property_path.write_bytes(await property_file.read())

        # Build the custom configuration file expected by AVERINN
        generated_config = build_nncs_config(
            settings_obj,
            network_path=network_path,
            dyn_path=dyn_path,
            property_path=property_path
        )

        with open(generated_config_path, "w") as config_file:
            generated_config.write(config_file)

        # Execute AVERINN and capture console output
        completed = subprocess.run(
            [sys.executable, str(script_path_tool), str(generated_config_path)],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=60
        )

    # Collect all output artifacts generated by AVERINN
    csv_path = project_root / "data.csv"
    log_path = project_root / "log.txt"
    lp_path = project_root / "model.lp"
    sol_path = project_root / "model.sol"

    csv_summary = summarize_csv(csv_path)

    safety_result = None

    # Looks at end of generated log.txt file to extract safety verdict
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8", errors="replace") as file:
            lines = [line.strip() for line in file.readlines() if line.strip()]

        if lines:
            safety_result = lines[-1]

    # Return execution information for frontend display
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


# ------------------------------------------------------------------
# Standard Neural Network Verification Endpoint
#
# Workflow:
#   1. Receive uploaded files.
#   2. Validate extensions.
#   3. Generate a temporary configuration file.
#   4. Execute nn-averinn.py.
#   5. Collect generated artifacts.
#   6. Return verification results.
# ------------------------------------------------------------------
@app.post("/run-nn-averinn")
async def run_nn_averinn(
    settings: str = Form(...),
    network_file: UploadFile = File(...),
    property_file: UploadFile = File(...)
):
    # Convert submitted json setting into object
    settings_dict = json.loads(settings)
    settings_obj = NNVerificationSettings(**settings_dict)

    # Locate the project root and tool script
    project_root = Path(__file__).resolve().parent.parent
    script_path_tool = project_root / "nn-averinn.py"

    # Rejects attempt to write to disk if uploaded files do not meet
    # required extension types.
    validate_extension(network_file, ALLOWED_EXTENSIONS["network"])
    validate_extension(property_file, ALLOWED_EXTENSIONS["property"])

    # Each verification run executes inside an isolated temp directory.
    # Prevents concurrent users from overwriting files.
    with TemporaryDirectory(prefix="averinn_run_") as temp_dir:
        run_dir = Path(temp_dir)

        network_path = run_dir / network_file.filename
        property_path = run_dir / property_file.filename

        generated_config_path = run_dir / "generated-nn-config.ini"

        # Save uploaded files into temp workspace
        network_path.write_bytes(await network_file.read())
        property_path.write_bytes(await property_file.read())

        # Build the custom configuration file expected by AVERINN
        generated_config = build_nn_config(
            settings_obj,
            network_path=network_path,
            property_path=property_path
        )

        with open(generated_config_path, "w") as config_file:
            generated_config.write(config_file)

        # Execute AVERINN and capture console output
        completed = subprocess.run(
            [sys.executable, str(script_path_tool), str(generated_config_path)],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=60
        )

    # Collect all output artifacts generated by AVERINN
    csv_path = project_root / "data.csv"
    log_path = project_root / "log.txt"
    lp_path = project_root / "model.lp"
    sol_path = project_root / "model.sol"

    csv_summary = summarize_csv(csv_path)

    safety_result = None

    # Looks at end of generated log.txt file to extract safety verdict
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8", errors="replace") as file:
            lines = [line.strip() for line in file.readlines() if line.strip()]

        if lines:
            safety_result = lines[-1]

    # Return execution information for frontend display
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

