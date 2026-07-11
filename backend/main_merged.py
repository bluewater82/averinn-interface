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
from fastapi.responses import StreamingResponse
import json
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from configparser import ConfigParser
import subprocess
import csv
import sys
from tempfile import TemporaryDirectory
import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import onnx
    import onnxruntime
    import simulinkengine as sl
    onnxruntime.set_default_logger_severity(3)
except Exception:
    onnx = None
    onnxruntime = None
    sl = None


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
    "dynamics": {".ini"},
    "simulink": {".slx"}
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

def summarize_csv(csv_path: Path):
    """
    Convert an AVERINN CSV into a semantic results model.

    Expected CSV structure:

        "", 0Low, 0High, 1Low, 1High, ...

    Each row represents one state variable.
    Each numbered Low/High pair represents one reachable set.
    """

    if not csv_path.exists():
        return None

    with open(csv_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        columns = reader.fieldnames or []

    if not rows or not columns:
        return None

    variable_column = columns[0]

    # Discover all numbered Low/High column pairs.
    set_indices = sorted({
        int(column.removesuffix("Low"))
        for column in columns
        if (
            column.endswith("Low")
            and column.removesuffix("Low").isdigit()
            and f"{column.removesuffix('Low')}High" in columns
        )
    })

    if not set_indices:
        return None

    variables = []

    for row_position, row in enumerate(rows):
        raw_variable_index = row.get(variable_column, row_position)

        try:
            variable_index = int(float(raw_variable_index))
        except (TypeError, ValueError):
            variable_index = row_position

        history = []

        for set_position, set_index in enumerate(set_indices):
            low_column = f"{set_index}Low"
            high_column = f"{set_index}High"

            try:
                low = float(row[low_column])
                high = float(row[high_column])
            except (TypeError, ValueError, KeyError):
                continue

            if set_position == 0:
                label = "Initial Set"
                short_label = "Initial"
            elif set_position == len(set_indices) - 1:
                label = "Final Set"
                short_label = "Final"
            else:
                label = f"Intermediate Set {set_position}"
                short_label = f"Step {set_position}"

            history.append({
                "set_index": set_index,
                "position": set_position,
                "label": label,
                "short_label": short_label,
                "low": low,
                "high": high,
                "width": high - low,
                "center": (low + high) / 2
            })

        if history:
            initial_interval = history[0]
            final_interval = history[-1]

            initial_width = initial_interval["width"]
            final_width = final_interval["width"]
            width_change = final_width - initial_width

            width_change_percent = (
                (width_change / initial_width) * 100
                if initial_width != 0
                else None
            )

            variables.append({
                "name": f"x{variable_index}",
                "index": variable_index,
                "history": history,
                "initial_width": initial_width,
                "final_width": final_width,
                "width_change": width_change,
                "width_change_percent": width_change_percent,
                "center_shift": (
                    final_interval["center"]
                    - initial_interval["center"]
                )
            })

    variables.sort(key=lambda variable: variable["index"])

    if not variables:
        return None

    final_widths = [
        variable["final_width"]
        for variable in variables
    ]

    widest_final_variable = max(
        variables,
        key=lambda variable: variable["final_width"]
    )

    narrowest_final_variable = min(
        variables,
        key=lambda variable: variable["final_width"]
    )

    all_widths = [
        interval["width"]
        for variable in variables
        for interval in variable["history"]
    ]

    return {
        "variable_count": len(variables),
        "set_count": len(set_indices),
        "variables": variables,
        "statistics": {
            "largest_final_width": max(final_widths),
            "average_final_width": (
                sum(final_widths) / len(final_widths)
            ),
            "smallest_final_width": min(final_widths),
            "widest_final_variable": (
                widest_final_variable["name"]
            ),
            "narrowest_final_variable": (
                narrowest_final_variable["name"]
            ),
            "largest_width_overall": max(all_widths),
            "average_width_overall": (
                sum(all_widths) / len(all_widths)
            )
        }
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
    nnFormat: str
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
    nnFormat: str
    probType: str
    absRequired: str
    numOfAbsNodes: int
    technique: str
    lastRelu: str
    K: int
    absType: str
    partitionType: str
    solverType: str


# Settings expected from the Simulink simulation form
class SLXFileOptions(BaseModel):
    stepSize: float
    timePeriod: float
    initialState: list[float] | None = None
    initialBlock: list[float] | None = None


# ==========================================================
# Simulink Runtime State
# ==========================================================

# The frontend uploads the SLX and ONNX files in separate requests, then submits simulation options later.
# These globals preserve that temporary state for the local development server.
simulink_model = None
simulink_nn_path = None
simulink_nn_inputs = 0
simulink_engine_holder = None
simulink_engine = None


def get_simulink_engine():

    global simulink_engine_holder
    global simulink_engine

    if sl is None:
        raise HTTPException(
            status_code=455,
            detail="Simulink support is unavailable. Verify matlab.engine and simulinkengine.py are installed."
        )

    if simulink_engine is None:
        simulink_engine_holder = sl.slEngineHolder()
        simulink_engine = simulink_engine_holder.geteng()

    return simulink_engine


def finish_slx_init(slx_was_uploaded: bool):

    global simulink_model
    global simulink_nn_path
    global simulink_nn_inputs

    if simulink_model is None:
        return {
            "message": "Success: Awaiting Simulink File",
            "numSIn": -1,
            "numBIn": -1,
            "numOut": -1
        }

    if simulink_nn_path is None:
        return {
            "message": "Success: Awaiting NN File",
            "numSIn": -1,
            "numBIn": -1,
            "numOut": simulink_model.getNumOut()
        }

    try:
        slx_nn_inputs = simulink_model.getnnInputDims()

        if simulink_nn_inputs != slx_nn_inputs:
            if slx_was_uploaded:
                simulink_model.close()
                simulink_model = None
            else:
                simulink_nn_path = None

            raise HTTPException(
                status_code=400,
                detail=f"Incorrect inputs ({simulink_nn_inputs}, {slx_nn_inputs})"
            )

        simulink_model.changeModelNN(simulink_nn_path)

    except HTTPException:
        raise
    except Exception:
        if slx_was_uploaded and simulink_model is not None:
            simulink_model.close()
            simulink_model = None
        else:
            simulink_nn_path = None

        raise HTTPException(
            status_code=400,
            detail="Verify matching input and output dimensions."
        )

    return {
        "message": "Success: Simulink File Fully Initialized",
        "numSIn": simulink_model.getNumIn(),
        "numBIn": simulink_model.getNumBIn(),
        "numOut": simulink_model.getNumOut()
    }


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
        "nnformat": f'"{settings.nnFormat}"',
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
    "nnformat": f'"{settings.nnFormat}"',
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



# ------------------------------------------------------------------
# Simulink Endpoint Helpers
# ------------------------------------------------------------------
@app.get("/continueok")
async def continueok():
    return {"message": simulink_model is not None and simulink_nn_path is not None}


@app.get("/reset")
async def reset_simulink_state():
    global simulink_model
    global simulink_nn_path
    global simulink_nn_inputs

    if simulink_model is not None:
        simulink_model.close()

    simulink_model = None
    simulink_nn_path = None
    simulink_nn_inputs = 0

    return {"message": "Simulink state reset"}


@app.post("/slxfile")
async def upload_slx_file(file: UploadFile = File(...)):
    global simulink_model

    validate_extension(file, ALLOWED_EXTENSIONS["simulink"])

    try:
        if simulink_model is not None:
            simulink_model.close()
            simulink_model = None

        slx_path = Path("./ri_simulink.slx")
        slx_path.write_bytes(await file.read())

        simulink_model = sl.sl(str(slx_path), get_simulink_engine())
        return finish_slx_init(slx_was_uploaded=True)

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=455,
            detail="Error initializing SLX file."
        )


@app.post("/nnfile")
async def upload_slx_nn_file(file: UploadFile = File(...)):
    global simulink_nn_path
    global simulink_nn_inputs

    if onnx is None or onnxruntime is None:
        raise HTTPException(
            status_code=422,
            detail="ONNX support is unavailable. Verify onnx and onnxruntime are installed."
        )

    validate_extension(file, {".onnx"})

    try:
        nn_path = Path("./ri_frontendfile.onnx")
        nn_path.write_bytes(await file.read())

        onnx_model = onnx.load(str(nn_path))
        onnx.checker.check_model(onnx_model)

        ort = onnxruntime.InferenceSession(str(nn_path))
        input_shape = ort.get_inputs()[0].shape

        # Richard's current code assumes image-style ONNX input and reads index 3.
        # This fallback makes the route less brittle for flat controller vectors.
        simulink_nn_inputs = int(input_shape[3] if len(input_shape) > 3 else input_shape[-1])
        simulink_nn_path = str(nn_path)

        return {"message": "Model Accepted"}

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="Cannot accept neural network."
        )


@app.post("/slxnnfile")
async def finalize_slx_nn_file():
    return finish_slx_init(slx_was_uploaded=False)


@app.post("/slxfileOpts")
async def run_slx_simulation(opts: SLXFileOptions):
    if simulink_model is None:
        raise HTTPException(
            status_code=456,
            detail="Uninitialized Simulink object is attempting to submit."
        )

    try:
        simulink_model.params(
            opts.stepSize,
            opts.timePeriod,
            opts.initialState,
            opts.initialBlock
        )

        out = simulink_model.sim()
        results = {}
        legend_labels = []

        for i in range(len(out[1])):
            for j in range(len(out[1][i][0])):
                legend_labels.append(f"x{i}_{j}")

            plt.plot(np.transpose(out[0])[0], out[1][i])
            plt.xlabel("Time (s)")
            plt.ylabel("Output")
            plt.title("Simulink Output")

        plt.legend(legend_labels, bbox_to_anchor=(1.05, 1))

        for i in range(len(out[0])):
            row_string = ""

            for j in range(len(out[1])):
                separator = "-=-" if j != simulink_model.getNumOut() - 1 else ""
                row_string += np.array2string(
                    out[1][j][i],
                    formatter={"float_kind": lambda x: f"{x:.3f}"}
                ) + separator

            results[np.array2string(
                out[0][i],
                formatter={"float_kind": lambda x: f"{x:.1f}"}
            )] = row_string

        return results

    except HTTPException:
        raise
    except Exception as error:
        print(error)
        raise HTTPException(
            status_code=457,
            detail="Error simulating SLX file."
        )


@app.get("/slxplt")
def get_slx_plot():
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    plt.clf()

    return StreamingResponse(buffer, media_type="image/png")
