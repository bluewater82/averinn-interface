import csv
import math
import random
from pathlib import Path

import numpy as np

from src.parser.onnxtonn import ONNX
from src.parser.newonnx import NEWONNX


STEPS = 100
NUM_TRAJECTORIES = 10


NN_PATH = "./resources/nncs-benchmarks/tora/controller.onnx"
OUTPUT_CSV = "tora_simulation_results.csv"

INIT_BOUNDS = [
    (0.6, 0.7),     # x1
    (-0.7, -0.6),   # x2
    (-0.4, -0.3),   # x3
    (0.5, 0.6),     # x4
]


def load_controller(nn_path):


    try:
        parser = ONNX(nn_path)
    except Exception:
        parser = NEWONNX(nn_path)

    return parser.getNetwork()


def relu(z):
    return np.maximum(z, 0.0)


def controller_forward(controller_net, x):

    a = np.array(x, dtype=np.float64).reshape(-1, 1)

    num_layers = controller_net.getNumOfLayers()

    for layer in range(1, num_layers):
        W = np.array(
            controller_net.getLowerMatrixByLayer(layer),
            dtype=np.float64
        )

        b = np.array(
            controller_net.getLowerBiasByLayer(layer + 1),
            dtype=np.float64
        ).reshape(-1, 1)

        z = W @ a + b

        if layer < num_layers - 1:
            a = relu(z)
        else:
            a = z

    return float(a.flatten()[0])

# arrays from dynamics file
A = np.array([
    [ 0.5403,  0.8415,  0.0450,  0.0083],
    [-0.8415,  0.5403,  0.0841,  0.0045],
    [ 0.0000,  0.0000,  1.0000,  1.0000],
    [ 0.0000,  0.0000,  0.0000,  1.0000],
], dtype=np.float64)

B = np.array([
    [0.0014],
    [0.0045],
    [0.5],
    [1.0],
], dtype=np.float64)


def tora_dynamics(x, nn_output):


    u_actual = nn_output - 10.0

    x_col = np.array(x, dtype=np.float64).reshape(4, 1)
    u_col = np.array([[u_actual]], dtype=np.float64)

    x_next = A @ x_col + B @ u_col

    return x_next.flatten(), u_actual


def sample_initial_state():
    return np.array([
        random.uniform(low, high)
        for low, high in INIT_BOUNDS
    ], dtype=np.float64)


def simulate_trajectory(traj_id, x0, controller_net):
    rows = []
    x = x0.copy()

    for step in range(STEPS):
        nn_output = controller_forward(controller_net, x)
        x_next, u_actual = tora_dynamics(x, nn_output)

        rows.append({
            "Trajectory": traj_id,
            "Step": step,

            "Position": round(x[0], 4),
            "Velocity": round(x[1], 4),
            "Rotor Angle": round(x[2], 4),
            "Rotor Speed": round(x[3], 4),

            "Raw Output": round(nn_output, 4),
            "Normalized Output": round(u_actual, 4),

            "next_Position": round(x_next[0], 4),
            "next_Velocity": round(x_next[1], 4),
            "next_Rotor Angle": round(x_next[2], 4),
            "next_Rotor Speed": round(x_next[3], 4),
        })

        x = x_next

    return rows


def write_csv(rows, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not Path(NN_PATH).exists():
        raise FileNotFoundError(f"Could not find controller file: {NN_PATH}")

    controller_net = load_controller(NN_PATH)

    all_rows = []

    for traj_id in range(NUM_TRAJECTORIES):
        x0 = sample_initial_state()
        rows = simulate_trajectory(traj_id, x0, controller_net)
        all_rows.extend(rows)

    write_csv(all_rows, OUTPUT_CSV)

    print(f"Completed {NUM_TRAJECTORIES} trajectories for {STEPS} steps each.")
    print(f"Saved results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()