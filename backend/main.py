"""
FastAPI backend for the AVERINN web interface.

Responsibilities:
- Receives uploaded benchmark files from the React frontend.
- Validates uploaded file types.
- Dynamically generates AVERINN configuration files.
- Executes NN, NNCS, and Hybrid NNCS verification tools.
- Collects generated output files.
- Returns normalized verification results to the frontend.
- Manages Richard's Simulink upload and simulation workflow.
"""

from contextlib import asynccontextmanager
from configparser import ConfigParser
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from tempfile import TemporaryDirectory
from typing import Any


from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# ==========================================================
# Optional ONNX / Simulink Dependencies
# ==========================================================

try:
    import onnx
    from onnx import numpy_helper
    import onnxruntime

    onnxruntime.set_default_logger_severity(3)
except Exception:
    onnx = None
    numpy_helper = None
    onnxruntime = None


try:
    import simulinkengine as sl
except Exception:
    sl = None


# ==========================================================
# Paths and Constants
# ==========================================================

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
SIMULINK_TEMP_DIR = BACKEND_DIR / "tmp"

SIMULINK_TEMP_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ALLOWED_EXTENSIONS = {
    "network": {
        ".onnx",
        ".nnet",
        ".sherlock",
        ".isherlock",
    },
    "hybrid_network": {".onnx"},
    "property": {".vnnlib"},
    "dynamics": {".ini"},
    "hybrid_dynamics": {".yaml", ".yml"},
    "simulink": {".slx"},
}


# ==========================================================
# Simulink Runtime State
# ==========================================================

simulink_model = None
simulink_slx_path: Path | None = None
simulink_nn_path: Path | None = None
simulink_nn_inputs = 0

simulink_engine_holder = None
simulink_engine = None


def remove_file_if_present(path: Path | None) -> None:
    """Delete a temporary file when it exists."""

    if path is None:
        return

    try:
        path.unlink(missing_ok=True)
    except OSError:
        # Cleanup should not prevent the application from shutting down.
        pass


def clear_simulink_model() -> None:
    """Close the current Simulink model and delete its uploaded SLX file."""

    global simulink_model
    global simulink_slx_path

    if simulink_model is not None:
        try:
            simulink_model.close()
        except Exception:
            pass

    simulink_model = None

    remove_file_if_present(simulink_slx_path)
    simulink_slx_path = None


def clear_simulink_network() -> None:
    """Delete the current temporary ONNX file and reset network state."""

    global simulink_nn_path
    global simulink_nn_inputs

    remove_file_if_present(simulink_nn_path)

    simulink_nn_path = None
    simulink_nn_inputs = 0


def reset_simulink_runtime() -> None:
    """Reset all uploaded Simulink runtime state."""

    clear_simulink_model()
    clear_simulink_network()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Clean up Simulink resources when FastAPI shuts down."""

    yield

    reset_simulink_runtime()


app = FastAPI(
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

class NetworkLayerSummary(BaseModel):
    """Structural and parameter info for one neural network layer."""

    id: int
    name: str
    role: str
    size: int
    activation: str | None = None
    weight_shape: list[int] | None = None
    bias_shape: list[int] | None = None
    parameter_count: int = 0

class NetworkVisualizationResponse(BaseModel):
    """Normalized network description consumed by React visualizer."""
    filename: str
    network_type: str
    layer_count: int
    parameter_count: int
    layers: list[NetworkLayerSummary]

class WeightStatistics(BaseModel):
    """Summary statistics for one connection's weight matrix"""
    minimum: float
    maximum: float
    mean: float
    mean_absolute: float
    maximum_absolute: float

class NetworkConnectionResponse(BaseModel):
    """Weight and bias data for the connection between two layers"""
    source_layer_id: int
    source_layer_name: str
    destination_layer_id: int
    destination_layer_name: str
    activation: str
    weight_shape: list[int]
    weights: list[list[float]]
    biases: list[float]
    statistics: WeightStatistics


# ==========================================================
# General Helper Functions
# ==========================================================

def validate_extension(
    file: UploadFile,
    allowed_extensions: set[str],
) -> None:
    """
    Ensure an uploaded file has one of the expected extensions.
    """

    filename = (file.filename or "").lower()

    if not any(
        filename.endswith(extension)
        for extension in allowed_extensions
    ):
        allowed = ", ".join(
            sorted(allowed_extensions)
        )

        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid file type for {filename or 'uploaded file'}. "
                f"Allowed: {allowed}"
            ),
        )


async def save_upload_in_chunks(
    upload: UploadFile,
    suffix: str,
) -> Path:
    """
    Save an uploaded file to the backend temporary directory in chunks.
    """

    temporary_file = tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=SIMULINK_TEMP_DIR,
        suffix=suffix,
        delete=False,
    )

    temporary_path = Path(
        temporary_file.name
    )

    try:
        with temporary_file:
            while chunk := await upload.read(65536):
                temporary_file.write(chunk)

        return temporary_path

    except Exception:
        remove_file_if_present(
            temporary_path
        )
        raise


def resolve_tool_script(filename: str) -> Path:
    """
    Locate an AVERINN entry-point script.

    The project has used both of these layouts:
        project_root/<script>
        project_root/AVERINN/<script>
    """

    candidates = [
        PROJECT_ROOT / filename,
        PROJECT_ROOT / "AVERINN" / filename,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = ", ".join(
        str(candidate)
        for candidate in candidates
    )

    raise HTTPException(
        status_code=500,
        detail=(
            f"Could not locate {filename}. "
            f"Searched: {searched}"
        ),
    )


def extract_safety_result(
    log_path: Path,
) -> str | None:
    """
    Read the final non-empty line of AVERINN's log file.
    """

    if not log_path.exists():
        return None

    with log_path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as log_file:
        lines = [
            line.strip()
            for line in log_file
            if line.strip()
        ]

    if not lines:
        return None

    return lines[-1]


def summarize_csv(
    csv_path: Path,
    includes_initial_set: bool,
) -> dict[str, Any] | None:
    """
    Convert an AVERINN CSV into the semantic results model used by React.

    Expected structure:

        "", 0Low, 0High, 1Low, 1High, ...

    Each numbered Low/High pair represents one reachable set.
    Each row represents one state variable. The existing AVERINN format may
    include a final timing row; this function currently preserves that row
    for compatibility with the existing Results components.
    """
    
    if not csv_path.exists():
        return None

    with csv_path.open(
        newline="",
        encoding="utf-8",
    ) as csv_file:
        reader = csv.DictReader(
            csv_file
        )
        rows = list(
            reader
        )
        columns = reader.fieldnames or []

    if not rows or not columns:
        return None

    variable_column = columns[0]

    set_indices = sorted({
        int(
            column.removesuffix("Low")
        )
        for column in columns
        if (
            column.endswith("Low")
            and column.removesuffix("Low").isdigit()
            and (
                f"{column.removesuffix('Low')}High"
                in columns
            )
        )
    })

    if not set_indices:
        return None

    initial_set_count = (
        1
        if includes_initial_set and set_indices
        else 0
    )

    computed_reachable_set_count = (
        len(set_indices) - initial_set_count
    )

    total_set_count = len(set_indices)

    if not set_indices:
        return None

    variables = []

    for row_position, row in enumerate(rows):
        raw_row_label = str(
            row.get(variable_column, "")
        ).strip().lower()

        # Newer CSV files explicitly label the timing row.
        if raw_row_label in {
            "elapsed_time",
            "elapsed time",
            "runtime",
            "time",
        }:
            continue

        # Backward compatibility for older CSV files:
        # the timing row was written as the final numerically indexed row.
        # It can be recognized because every Low/High pair is identical,
        # the initial value is zero, and all values are nonnegative.
        #
        # Restricting this check to the final CSV row avoids removing a
        # legitimate point-valued state variable elsewhere in the file.
        if row_position == len(rows) - 1:
            possible_times = []
            is_point_row = True

            for set_index in set_indices:
                low_column = f"{set_index}Low"
                high_column = f"{set_index}High"

                try:
                    low_value = float(row[low_column])
                    high_value = float(row[high_column])
                except (TypeError, ValueError, KeyError):
                    is_point_row = False
                    break

                if low_value != high_value:
                    is_point_row = False
                    break

                possible_times.append(low_value)

            if (
                is_point_row
                and possible_times
                and possible_times[0] == 0.0
                and all(value >= 0.0 for value in possible_times)
            ):
                continue

        raw_variable_index = row.get(
            variable_column,
            row_position,
        )

        try:
            variable_index = int(
                float(raw_variable_index)
            )
        except (TypeError, ValueError):
            variable_index = row_position

        history = []

        for set_position, set_index in enumerate(
            set_indices
        ):
            low_column = f"{set_index}Low"
            high_column = f"{set_index}High"

            try:
                low = float(
                    row[low_column]
                )
                high = float(
                    row[high_column]
                )
            except (
                TypeError,
                ValueError,
                KeyError,
            ):
                continue

            if set_position == 0:
                label = "Initial Set"
                short_label = "Initial"

            elif set_position == len(set_indices) - 1:
                label = "Final Set"
                short_label = "Final"

            else:
                label = (
                    f"Intermediate Set {set_position}"
                )
                short_label = (
                    f"Step {set_position}"
                )

            history.append({
                "set_index": set_index,
                "position": set_position,
                "label": label,
                "short_label": short_label,
                "low": low,
                "high": high,
                "width": high - low,
                "center": (low + high) / 2,
            })

        if not history:
            continue

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
            ),
        })

    variables.sort(
        key=lambda variable: variable["index"]
    )

    if not variables:
        return None

    final_widths = [
        variable["final_width"]
        for variable in variables
    ]

    all_widths = [
        interval["width"]
        for variable in variables
        for interval in variable["history"]
    ]

    widest_final_variable = max(
        variables,
        key=lambda variable: variable["final_width"],
    )

    narrowest_final_variable = min(
        variables,
        key=lambda variable: variable["final_width"],
    )

    return {
        "variable_count": len(variables),
        "initial_set_count": initial_set_count,
        "computed_reachable_set_count": (computed_reachable_set_count),
        "total_set_count": total_set_count,
        "variables": variables,
        "statistics": {
            "largest_final_width": max(
                final_widths
            ),
            "average_final_width": (
                sum(final_widths)
                / len(final_widths)
            ),
            "smallest_final_width": min(
                final_widths
            ),
            "widest_final_variable": (
                widest_final_variable["name"]
            ),
            "narrowest_final_variable": (
                narrowest_final_variable["name"]
            ),
            "largest_width_overall": max(
                all_widths
            ),
            "average_width_overall": (
                sum(all_widths)
                / len(all_widths)
            ),
        },
    }

def get_onnx_attribute(
    node: Any,
    attribute_name: str,
    default: int,
) -> int:
    """Return an integer ONNX node attribute or a default value."""

    for attribute in node.attribute:
        if attribute.name == attribute_name:
            return int(attribute.i)

    return default


def get_onnx_input_size(model: Any) -> int:
    """Return the feature count for the model's primary input tensor."""

    initializer_names = {
        initializer.name
        for initializer in model.graph.initializer
    }

    network_inputs = [
        graph_input
        for graph_input in model.graph.input
        if graph_input.name not in initializer_names
    ]

    if not network_inputs:
        raise ValueError(
            "The ONNX model does not define a network input."
        )

    dimensions = [
        dimension.dim_value
        for dimension in (
            network_inputs[0]
            .type
            .tensor_type
            .shape
            .dim
        )
        if dimension.dim_value > 0
    ]

    if not dimensions:
        raise ValueError(
            "The ONNX input does not contain a fixed feature dimension."
        )

    # Feed-forward models commonly use [batch_size, feature_count].
    return int(dimensions[-1])


def extract_feedforward_layers(
    model: Any,
) -> list[dict[str, Any]]:
    """
    Extract normalized dense-layer data from a sequential ONNX network.

    Every returned weight matrix uses the visualization convention:

        [destination neurons, source neurons]

    Keeping the complete NumPy arrays here allows both the lightweight
    architecture summary and future connection-detail endpoints to consume
    the same parsed representation.
    """

    initializer_map = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }

    dense_layers: list[dict[str, Any]] = []
    last_dense_layer: dict[str, Any] | None = None

    for node in model.graph.node:
        if node.op_type == "Gemm":
            if len(node.input) < 2:
                raise ValueError(
                    "Encountered a Gemm operation without a weight matrix."
                )

            stored_weight = initializer_map.get(node.input[1])

            if stored_weight is None or stored_weight.ndim != 2:
                raise ValueError(
                    "Unable to read a two-dimensional Gemm weight matrix."
                )

            transposed_weight = get_onnx_attribute(
                node,
                "transB",
                0,
            )

            # Gemm computes A × B when transB=0 and A × Bᵀ when transB=1.
            # Normalize both forms to [destination, source].
            weights = (
                stored_weight
                if transposed_weight
                else stored_weight.T
            )

            biases = None

            if len(node.input) >= 3:
                stored_bias = initializer_map.get(node.input[2])

                if stored_bias is not None:
                    biases = stored_bias.reshape(-1)

            last_dense_layer = {
                "input_size": int(weights.shape[1]),
                "output_size": int(weights.shape[0]),
                "weights": weights,
                "biases": biases,
                "activation": "Linear",
            }

            dense_layers.append(last_dense_layer)

        elif node.op_type == "MatMul":
            if len(node.input) < 2:
                raise ValueError(
                    "Encountered a MatMul operation without a weight matrix."
                )

            stored_weight = initializer_map.get(node.input[1])

            if stored_weight is None or stored_weight.ndim != 2:
                raise ValueError(
                    "Unable to read a two-dimensional MatMul weight matrix."
                )

            # MatMul commonly stores B as [source, destination].
            weights = stored_weight.T

            last_dense_layer = {
                "input_size": int(weights.shape[1]),
                "output_size": int(weights.shape[0]),
                "weights": weights,
                "biases": None,
                "activation": "Linear",
            }

            dense_layers.append(last_dense_layer)

        elif (
            node.op_type == "Add"
            and last_dense_layer is not None
        ):
            initializer_inputs = [
                initializer_map[input_name]
                for input_name in node.input
                if input_name in initializer_map
            ]

            if initializer_inputs:
                last_dense_layer["biases"] = (
                    initializer_inputs[0].reshape(-1)
                )

        elif (
            node.op_type == "Relu"
            and last_dense_layer is not None
        ):
            last_dense_layer["activation"] = "ReLU"

    if not dense_layers:
        raise ValueError(
            "No supported feed-forward layers were found. "
            "The prototype currently supports Gemm and MatMul layers."
        )

    return dense_layers


def inspect_network_connection(
        model: Any,
        destination_layer_id: int,
) -> NetworkConnectionResponse:
    """
    Build detailed weight and bias data for the one network connection.
    """

    dense_layers = extract_feedforward_layers(model)
    dense_layer_count = len(dense_layers)

    if (
        destination_layer_id < 1
        or destination_layer_id > dense_layer_count
    ):
        raise ValueError(
            "Destination layer ID must be between "
            f"1 and {dense_layer_count}."
        )
    
    dense_layer = dense_layers[
        destination_layer_id - 1
    ]

    weights = dense_layer["weights"]
    biases = dense_layer["biases"]

    if biases is not None:
        biases = biases.reshape(-1)

        if int(biases.size) != int(weights.shape[0]):
            raise ValueError(
                "The bias-vector length does not match the "
                "number of destination neurons."
            )
        
    is_output_layer = (
        destination_layer_id == dense_layer_count
    )

    source_layer_id = destination_layer_id - 1

    source_layer_name = (
        "Input Layer"
        if source_layer_id == 0
        else f"Hidden Layer {source_layer_id}"
    )

    destination_layer_name = (
        "Output Layer"
        if is_output_layer
        else f"Hidden Layer {destination_layer_id}"
    )

    return NetworkConnectionResponse(
        source_layer_id=source_layer_id,
        source_layer_name=source_layer_name,
        destination_layer_id=destination_layer_id,
        destination_layer_name=destination_layer_name,
        activation=dense_layer["activation"],
        weight_shape=[
            int(dimension)
            for dimension in weights.shape
        ],
        weights=[
            [
                float(value)
                for value in row
            ]
            for row in weights
        ],
        biases=(
            [
                float(value)
                for value in biases
            ]
            if biases is not None
            else []
        ),
        statistics=WeightStatistics(
            minimum=float(weights.min()),
            maximum=float(weights.max()),
            mean=float(weights.mean()),
            mean_absolute=float(abs(weights).mean()),
            maximum_absolute=float(abs(weights).max()),
        ),
    )

def inspect_feedforward_onnx(
    model: Any,
    filename: str,
) -> NetworkVisualizationResponse:
    """Convert a sequential ONNX graph into visualization metadata."""

    dense_layers = extract_feedforward_layers(model)
    input_size = get_onnx_input_size(model)

    layers = [
        NetworkLayerSummary(
            id=0,
            name="Input Layer",
            role="input",
            size=input_size,
        )
    ]

    total_dense_layers = len(dense_layers)
    total_parameters = 0

    for position, dense_layer in enumerate(
        dense_layers,
        start=1,
    ):
        is_output_layer = position == total_dense_layers
        weights = dense_layer["weights"]
        biases = dense_layer["biases"]

        weight_shape = [
            int(dimension)
            for dimension in weights.shape
        ]

        bias_size = (
            int(biases.size)
            if biases is not None
            else 0
        )

        parameter_count = int(weights.size) + bias_size
        total_parameters += parameter_count

        layers.append(
            NetworkLayerSummary(
                id=position,
                name=(
                    "Output Layer"
                    if is_output_layer
                    else f"Hidden Layer {position}"
                ),
                role=(
                    "output"
                    if is_output_layer
                    else "hidden"
                ),
                size=dense_layer["output_size"],
                activation=dense_layer["activation"],
                weight_shape=weight_shape,
                bias_shape=(
                    [bias_size]
                    if bias_size > 0
                    else None
                ),
                parameter_count=parameter_count,
            )
        )

    return NetworkVisualizationResponse(
        filename=filename,
        network_type="feedforward",
        layer_count=len(layers),
        parameter_count=total_parameters,
        layers=layers,
    )



def build_verification_response(
    completed: subprocess.CompletedProcess[str],
    includes_initial_set: bool,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Normalize the result returned by NN, NNCS, and Hybrid verification.
    """

    csv_path = output_dir / "data.csv"
    log_path = output_dir / "log.txt"
    lp_path = output_dir / "model.lp"
    sol_path = output_dir / "model.sol"

    csv_summary = summarize_csv(
        csv_path,
        includes_initial_set,
    )

    safety_result = extract_safety_result(
        log_path
    )

    return {
        "returncode": completed.returncode,
        "success": (
            completed.returncode == 0
            and csv_summary is not None
        ),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "csv_summary": csv_summary,
        "safety_result": safety_result,
        "files_created": {
            "data_csv": csv_path.exists(),
            "log_txt": log_path.exists(),
            "model_lp": lp_path.exists(),
            "model_sol": sol_path.exists(),
        },
    }


# ==========================================================
# Request Models
# ==========================================================

class NNVerificationSettings(BaseModel):
    """Settings for standard neural-network verification."""

    nnFormat: str
    probType: str
    absRequired: str
    numOfAbsNodes: int
    technique: str
    lastRelu: str
    absType: str
    partitionType: str
    solverType: str


class NNCSVerificationSettings(BaseModel):
    """Settings for NN controller-system verification."""

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


class HybridVerificationSettings(BaseModel):
    """Settings consumed by hybrid-nncs-averinn.py."""

    lastRelu: str
    K: int


class SLXFileOptions(BaseModel):
    """Simulation options submitted after SLX/ONNX initialization."""

    stepSize: float
    timePeriod: float
    initialState: list[float] | None = None
    initialBlock: list[float] | None = None


# ==========================================================
# Configuration Builders
# ==========================================================

def build_nn_config(
    settings: NNVerificationSettings,
    network_path: Path,
    property_path: Path,
) -> ConfigParser:
    """Build an NN AVERINN configuration."""

    config = ConfigParser()
    config.optionxform = str

    config["settings"] = {
        "nnformat": f'"{settings.nnFormat}"',
        "nnpath": f'"{network_path}"',
        "specpath": f'"{property_path}"',
        "probType": f'"{settings.probType}"',
        "absRequired": f'"{settings.absRequired}"',
        "numOfAbsNodes": str(
            settings.numOfAbsNodes
        ),
        "technique": f'"{settings.technique}"',
        "lastRelu": f'"{settings.lastRelu}"',
        "absType": f'"{settings.absType}"',
        "partitionType": (
            f'"{settings.partitionType}"'
        ),
        "solverType": f'"{settings.solverType}"',
    }

    return config


def build_nncs_config(
    settings: NNCSVerificationSettings,
    network_path: Path,
    dynamics_path: Path,
    property_path: Path,
) -> ConfigParser:
    """Build an NNCS AVERINN configuration."""

    config = ConfigParser()
    config.optionxform = str

    config["settings"] = {
        "nnformat": f'"{settings.nnFormat}"',
        "nnpath": f'"{network_path}"',
        "dynpath": f'"{dynamics_path}"',
        "specpath": f'"{property_path}"',
        "probType": f'"{settings.probType}"',
        "absRequired": f'"{settings.absRequired}"',
        "numOfAbsNodes": str(
            settings.numOfAbsNodes
        ),
        "technique": f'"{settings.technique}"',
        "lastRelu": f'"{settings.lastRelu}"',
        "K": str(
            settings.K
        ),
        "absType": f'"{settings.absType}"',
        "partitionType": (
            f'"{settings.partitionType}"'
        ),
        "solverType": f'"{settings.solverType}"',
    }

    return config


def build_hybrid_config(
    settings: HybridVerificationSettings,
) -> ConfigParser:
    """Build the small settings file consumed by Hybrid AVERINN."""

    config = ConfigParser()
    config.optionxform = str

    config["settings"] = {
        "lastRelu": f'"{settings.lastRelu}"',
        "K": str(
            settings.K
        ),
    }

    return config


# ==========================================================
# NN Verification Endpoints
# ==========================================================

@app.post("/run-nn-averinn")
async def run_nn_averinn(
    settings: str = Form(...),
    network_file: UploadFile = File(...),
    property_file: UploadFile = File(...),
):
    """Run standard neural-network verification."""

    try:
        settings_dict = json.loads(
            settings
        )
        settings_obj = NNVerificationSettings(
            **settings_dict
        )
    except (
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as error:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid NN settings: {error}",
        ) from error

    validate_extension(
        network_file,
        ALLOWED_EXTENSIONS["network"],
    )
    validate_extension(
        property_file,
        ALLOWED_EXTENSIONS["property"],
    )

    script_path = resolve_tool_script(
        "nn-averinn.py"
    )

    with TemporaryDirectory(
        prefix="averinn_nn_"
    ) as temporary_directory:
        run_dir = Path(
            temporary_directory
        )

        network_path = (
            run_dir
            / (network_file.filename or "network")
        )
        property_path = (
            run_dir
            / (property_file.filename or "property.vnnlib")
        )
        config_path = (
            run_dir
            / "generated-nn-config.ini"
        )

        network_path.write_bytes(
            await network_file.read()
        )
        property_path.write_bytes(
            await property_file.read()
        )

        config = build_nn_config(
            settings_obj,
            network_path,
            property_path,
        )

        with config_path.open(
            "w",
            encoding="utf-8",
        ) as config_file:
            config.write(
                config_file
            )

        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    str(config_path),
                ],
                cwd=run_dir,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired as error:
            raise HTTPException(
                status_code=504,
                detail="NN verification exceeded the 300-second timeout.",
            ) from error

        return build_verification_response(
            completed,
            includes_initial_set=False,
            output_dir=run_dir,
        )


@app.post("/run-nncs-averinn")
async def run_nncs_averinn(
    settings: str = Form(...),
    network_file: UploadFile = File(...),
    property_file: UploadFile = File(...),
    dynamics_file: UploadFile = File(...),
):
    """Run neural-network controller-system verification."""

    try:
        settings_dict = json.loads(
            settings
        )
        settings_obj = NNCSVerificationSettings(
            **settings_dict
        )
    except (
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as error:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid NNCS settings: {error}",
        ) from error

    validate_extension(
        network_file,
        ALLOWED_EXTENSIONS["network"],
    )
    validate_extension(
        property_file,
        ALLOWED_EXTENSIONS["property"],
    )
    validate_extension(
        dynamics_file,
        ALLOWED_EXTENSIONS["dynamics"],
    )

    script_path = resolve_tool_script(
        "nncs-averinn.py"
    )

    with TemporaryDirectory(
        prefix="averinn_nncs_"
    ) as temporary_directory:
        run_dir = Path(
            temporary_directory
        )

        network_path = (
            run_dir
            / (network_file.filename or "network")
        )
        property_path = (
            run_dir
            / (property_file.filename or "property.vnnlib")
        )
        dynamics_path = (
            run_dir
            / (dynamics_file.filename or "dynamics.ini")
        )
        config_path = (
            run_dir
            / "generated-nncs-config.ini"
        )

        network_path.write_bytes(
            await network_file.read()
        )
        property_path.write_bytes(
            await property_file.read()
        )
        dynamics_path.write_bytes(
            await dynamics_file.read()
        )

        config = build_nncs_config(
            settings_obj,
            network_path,
            dynamics_path,
            property_path,
        )

        with config_path.open(
            "w",
            encoding="utf-8",
        ) as config_file:
            config.write(
                config_file
            )

        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    str(config_path),
                ],
                cwd=run_dir,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired as error:
            raise HTTPException(
                status_code=504,
                detail="NNCS verification exceeded the 300-second timeout.",
            ) from error

        return build_verification_response(
            completed,
            includes_initial_set=True,
            output_dir=run_dir,
        )


@app.post("/run-hybrid-averinn")
async def run_hybrid_averinn(
    settings: str = Form(...),
    network_file: UploadFile = File(...),
    property_file: UploadFile = File(...),
    dynamics_file: UploadFile = File(...),
):
    """
    Run Hybrid NNCS verification.

    hybrid-nncs-averinn.py receives:
        1. generated INI settings
        2. uploaded YAML hybrid-system definition
        3. uploaded ONNX controller
        4. uploaded VNNLIB specification

    The modified Hybrid tool writes the same data.csv schema used by NNCS.
    """

    try:
        settings_dict = json.loads(
            settings
        )
        settings_obj = HybridVerificationSettings(
            **settings_dict
        )
    except (
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as error:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Hybrid settings: {error}",
        ) from error

    validate_extension(
        network_file,
        ALLOWED_EXTENSIONS["hybrid_network"],
    )
    validate_extension(
        property_file,
        ALLOWED_EXTENSIONS["property"],
    )
    validate_extension(
        dynamics_file,
        ALLOWED_EXTENSIONS["hybrid_dynamics"],
    )

    script_path = resolve_tool_script(
        "hybrid-nncs-averinn.py"
    )

    with TemporaryDirectory(
        prefix="averinn_hybrid_"
    ) as temporary_directory:
        run_dir = Path(
            temporary_directory
        )

        network_path = (
            run_dir
            / (network_file.filename or "network.onnx")
        )
        property_path = (
            run_dir
            / (property_file.filename or "property.vnnlib")
        )
        dynamics_path = (
            run_dir
            / (dynamics_file.filename or "hybrid.yaml")
        )
        config_path = (
            run_dir
            / "generated-hybrid-config.ini"
        )

        network_path.write_bytes(
            await network_file.read()
        )
        property_path.write_bytes(
            await property_file.read()
        )
        dynamics_path.write_bytes(
            await dynamics_file.read()
        )

        config = build_hybrid_config(
            settings_obj
        )

        with config_path.open(
            "w",
            encoding="utf-8",
        ) as config_file:
            config.write(
                config_file
            )

        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    str(config_path),
                    str(dynamics_path),
                    str(network_path),
                    str(property_path),
                ],
                cwd=run_dir,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired as error:
            raise HTTPException(
                status_code=504,
                detail="Hybrid verification exceeded the 300-second timeout.",
            ) from error

        return build_verification_response(
            completed,
            includes_initial_set=True,
            output_dir=run_dir,
        )


# ==========================================================
# Simulink Helpers
# ==========================================================

def get_simulink_engine():
    """Start the MATLAB/Simulink engine only when it is first needed."""

    global simulink_engine_holder
    global simulink_engine

    if sl is None:
        raise HTTPException(
            status_code=455,
            detail=(
                "Simulink support is unavailable. Verify that "
                "MATLAB Engine and simulinkengine.py are installed."
            ),
        )

    if simulink_engine is None:
        try:
            simulink_engine_holder = (
                sl.slEngineHolder()
            )
            simulink_engine = (
                simulink_engine_holder.geteng()
            )
        except Exception as error:
            raise HTTPException(
                status_code=455,
                detail="Unable to initialize the Simulink engine.",
            ) from error

    return simulink_engine


def get_onnx_input_count(
    model_path: Path,
) -> int:
    """
    Read the controller input dimension from an ONNX model.
    """

    if onnxruntime is None:
        raise HTTPException(
            status_code=422,
            detail="ONNX Runtime is unavailable.",
        )

    session = onnxruntime.InferenceSession(
        str(model_path)
    )
    input_shape = session.get_inputs()[0].shape

    if not input_shape:
        raise HTTPException(
            status_code=422,
            detail="The ONNX model has no readable input shape.",
        )

    preferred_dimension = (
        input_shape[3]
        if len(input_shape) > 3
        else input_shape[-1]
    )

    try:
        return int(
            preferred_dimension
        )
    except (
        TypeError,
        ValueError,
    ) as error:
        raise HTTPException(
            status_code=422,
            detail=(
                "The ONNX input dimension is dynamic or unsupported: "
                f"{preferred_dimension}"
            ),
        ) from error


def finish_slx_init(
    slx_was_uploaded: bool,
):
    """
    Complete SLX/ONNX initialization once both uploads are available.

    If the two files are incompatible, the most recently uploaded file is
    cleared while the previously accepted file remains available.
    """

    global simulink_model
    global simulink_nn_path
    global simulink_nn_inputs

    if simulink_model is None:
        return {
            "message": "Success: Awaiting Simulink File",
            "numSIn": -1,
            "numBIn": -1,
            "numOut": -1,
        }

    if simulink_nn_path is None:
        return {
            "message": "Success: Awaiting NN File",
            "numSIn": -1,
            "numBIn": -1,
            "numOut": simulink_model.getNumOut(),
        }

    try:
        slx_nn_inputs = (
            simulink_model.getnnInputDims()
        )

        if simulink_nn_inputs != slx_nn_inputs:
            if slx_was_uploaded:
                clear_simulink_model()
            else:
                clear_simulink_network()

            raise HTTPException(
                status_code=400,
                detail=(
                    "Incorrect input dimensions "
                    f"({simulink_nn_inputs}, {slx_nn_inputs})."
                ),
            )

        simulink_model.changeModelNN(
            str(simulink_nn_path)
        )

    except HTTPException:
        raise

    except Exception as error:
        if slx_was_uploaded:
            clear_simulink_model()
        else:
            clear_simulink_network()

        raise HTTPException(
            status_code=400,
            detail=(
                "Verify that the Simulink model and neural network "
                "have matching input and output dimensions."
            ),
        ) from error

    return {
        "message": "Success: Simulink File Fully Initialized",
        "numSIn": simulink_model.getNumIn(),
        "numBIn": simulink_model.getNumBIn(),
        "numOut": simulink_model.getNumOut(),
    }


# ==========================================================
# Simulink Endpoints
# ==========================================================

@app.get("/continueok")
async def continueok():
    """Tell the frontend whether both Simulink uploads are ready."""

    return {
        "message": (
            simulink_model is not None
            and simulink_nn_path is not None
        )
    }


@app.get("/reset")
async def reset():
    """Clear all current Simulink upload and model state."""

    reset_simulink_runtime()

    return {
        "message": "Simulink state reset"
    }


@app.post("/slxfile")
async def upload_slx_file(
    file: UploadFile = File(...),
):
    """Upload and initialize a Simulink SLX model."""

    global simulink_model
    global simulink_slx_path

    validate_extension(
        file,
        ALLOWED_EXTENSIONS["simulink"],
    )

    clear_simulink_model()

    try:
        simulink_slx_path = (
            await save_upload_in_chunks(
                file,
                suffix=".slx",
            )
        )

        simulink_model = sl.sl(
            str(simulink_slx_path),
            get_simulink_engine(),
        )

        return finish_slx_init(
            slx_was_uploaded=True
        )

    except HTTPException:
        clear_simulink_model()
        raise

    except Exception as error:
        clear_simulink_model()

        raise HTTPException(
            status_code=455,
            detail="Error initializing SLX file.",
        ) from error


@app.post("/nnfile")
async def upload_simulink_nn_file(
    file: UploadFile = File(...),
):
    """Upload and validate the ONNX controller used by Simulink."""

    global simulink_nn_path
    global simulink_nn_inputs

    if onnx is None or onnxruntime is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "ONNX support is unavailable. Verify that onnx "
                "and onnxruntime are installed."
            ),
        )

    validate_extension(
        file,
        ALLOWED_EXTENSIONS["hybrid_network"],
    )

    clear_simulink_network()

    try:
        simulink_nn_path = (
            await save_upload_in_chunks(
                file,
                suffix=".onnx",
            )
        )

        onnx_model = onnx.load(
            str(simulink_nn_path)
        )
        onnx.checker.check_model(
            onnx_model
        )

        simulink_nn_inputs = get_onnx_input_count(
            simulink_nn_path
        )

        return finish_slx_init(
            slx_was_uploaded=False
        )

    except HTTPException:
        clear_simulink_network()
        raise

    except Exception as error:
        clear_simulink_network()

        raise HTTPException(
            status_code=422,
            detail="Cannot accept neural network.",
        ) from error


@app.post("/slxnnfile")
async def finalize_slx_nn_file():
    """
    Compatibility endpoint retained for Richard's earlier frontend flow.

    The /slxfile and /nnfile endpoints now finalize automatically, but this
    endpoint safely repeats the finalization check.
    """

    return finish_slx_init(
        slx_was_uploaded=False
    )


@app.post("/slxfileOpts")
async def run_slx_simulation(
    opts: SLXFileOptions,
):
    """Run the initialized Simulink model with submitted options."""

    if simulink_model is None:
        raise HTTPException(
            status_code=456,
            detail=(
                "Uninitialized Simulink object is attempting to submit."
            ),
        )

    try:
        simulink_model.params(
            opts.stepSize,
            opts.timePeriod,
            opts.initialState,
            opts.initialBlock,
        )

        output = simulink_model.sim()

        results = {}
        legend_labels = []

        plt.clf()

        for output_index in range(
            len(output[1])
        ):
            for variable_index in range(
                len(output[1][output_index][0])
            ):
                legend_labels.append(
                    f"x{output_index}_{variable_index}"
                )

            plt.plot(
                np.transpose(output[0])[0],
                output[1][output_index],
            )

        plt.xlabel(
            "Time (s)"
        )
        plt.ylabel(
            "Output"
        )
        plt.title(
            "Simulink Output"
        )

        if legend_labels:
            plt.legend(
                legend_labels,
                bbox_to_anchor=(1.05, 1),
            )

        for time_index in range(
            len(output[0])
        ):
            row_parts = []

            for output_index in range(
                len(output[1])
            ):
                row_parts.append(
                    np.array2string(
                        output[1][output_index][time_index],
                        formatter={
                            "float_kind": (
                                lambda value: f"{value:.3f}"
                            )
                        },
                    )
                )

            time_key = np.array2string(
                output[0][time_index],
                formatter={
                    "float_kind": (
                        lambda value: f"{value:.1f}"
                    )
                },
            )

            results[time_key] = "-=-".join(
                row_parts
            )

        return results

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=457,
            detail="Error simulating SLX file.",
        ) from error
    

@app.post(
    "/visualize-network",
    response_model=NetworkVisualizationResponse,
)
async def visualize_network(
    network_file: UploadFile = File(...),
):
    """Inspect an uploaded ONNX network for the visualization interface."""

    if onnx is None or numpy_helper is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "ONNX visualization support is unavailable. "
                "Verify that the onnx package is installed."
            ),
        )

    validate_extension(
        network_file,
        {".onnx"},
    )

    temporary_path: Path | None = None

    try:
        temporary_path = await save_upload_in_chunks(
            network_file,
            suffix=".onnx",
        )

        model = onnx.load(
            str(temporary_path)
        )

        onnx.checker.check_model(
            model
        )

        return inspect_feedforward_onnx(
            model,
            filename=(
                network_file.filename
                or "uploaded-network.onnx"
            ),
        )

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=422,
            detail=(
                "Unable to visualize this ONNX network: "
                f"{error}"
            ),
        ) from error

    finally:
        remove_file_if_present(
            temporary_path
        )


@app.post(
        "/visualize-network/connection",
        response_model=NetworkConnectionResponse,
)
async def visualize_network_connection(
    network_file: UploadFile = File(...),
    destination_layer_id: int = Form(...),
) -> NetworkConnectionResponse:
    """
    Return the weights, biases, and statistics for on conneciton
    in an uploaded feed-forward ONNX network.
    """

    if onnx is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "ONNX support is unavailable because the required "
                "dependencies are not installed."
            ),
        )
    
    filename = network_file.filename or ""

    if not filename.lower().endswith(".onnx"):
        raise HTTPException(
            status_code=400,
            detail="The uploaded network must be an ONNX file."
        )
    
    temporary_path: Path | None = None

    try:
        with NamedTemporaryFile(
            suffix=".onnx",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

            while chunk := await network_file.read(1024 * 1024):
                temporary_file.write(chunk)

        model = onnx.load(str(temporary_path))
        onnx.checker.check_model(model)

        return inspect_network_connection(
            model=model,
            destination_layer_id=destination_layer_id,
        )
    
    except ValueError as error:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=422,
            detail=(
                "Unable to inspect this network connection: "
                f"{error}"
            ),
        ) from error
    
    finally:
        await network_file.close()

        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()


@app.get("/slxplt")
def get_slx_plot():
    """Return the most recently generated Simulink plot as PNG."""

    buffer = io.BytesIO()

    plt.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
    )

    buffer.seek(0)
    plt.clf()

    return StreamingResponse(
        buffer,
        media_type="image/png",
    )
