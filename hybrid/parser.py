import yaml
import numpy as np
import os


def load_hybrid_system(yaml_path):
    """
    Parses the hybrid-system YAML, handles nested lin_exp, 
    and validates matrix/vector dimensions for each mode.
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Hybrid system file not found: {yaml_path}")

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # Convert float metadata to integers for indexing
    num_modes = int(data.get('modes_num', 0))
    num_states = int(data.get('state_dim', 0))
    num_inputs = int(data.get('inputs_dim', 0))
    listA = np.array(data['listA'])
    listB = np.array(data['listB'])
    M = np.array(data['M']) if data.get('M') is not None else np.eye(num_states)
    B = np.array(data['B'])if data.get('M') is not None else np.zeros((num_states, 1))
    system_info = {
        "listA": listA,
        "listB": listB,
        "num_modes": num_modes,
        "num_states": num_states,
        "num_inputs": num_inputs,
        "lin_exp": np.array(data['lin_exp']),
        "initial_states": np.array(data.get('initial_states', [])),
        "M": M,
        "B": B,
        "modes": {}
    }

    # Iterate through modes and validate dimensions
    for i in range(1, num_modes+1):
        mode_key = f'mode{i}'
        if mode_key in data:
            mode_raw = data[mode_key]

            # Mapping based on your file structure: 
            # [0]=A matrix, [1]=B matrix, [2]=offset, [3]=constraints
            A = np.array(mode_raw[0])
            B = np.array(mode_raw[1]) if mode_raw[1] is not None else None
            C = np.array(mode_raw[2]) if mode_raw[2] is not None else None
            U = np.array(mode_raw[3])

            # Validation
            if A.shape != (num_states, num_states):
                raise ValueError(f"{mode_key}: A-matrix must be {num_states}x{num_states}")
            if B is not None and B.shape != (num_states, num_inputs):
                raise ValueError(f"{mode_key}: B-matrix must be {num_states}x{num_inputs}")

            system_info["modes"][mode_key] = {"A": A, "B": B, "C": C, "U": U}

    return system_info


def load_controller_settings(yaml_path):
    """
    Loads general reachability settings and extracts 
    the controller path.
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    settings = {
        "onnx_path": data.get('ONNX_path_mode', "")
    }
    return settings


def create_network_parser(nn_format, nn_path):
    """
    Creates the correct neural-network parser based on the 
    specified format (e.g., 'ONNX', 'ReluVal', 'Sherlock').
    """
    if not os.path.exists(nn_path):
        print(f"Warning: Controller file not found at {nn_path}")
        return None

    nn_format = nn_format.upper()
    print(f"Initializing {nn_format} parser for: {os.path.basename(nn_path)}")

    # Selection logic for different backend parsers
    if nn_format == "ONNX":
        # Placeholder for actual ONNX parsing logic
        # return ONNXParser(nn_path)
        return {"format": "ONNX", "path": nn_path}
    elif nn_format == "VNNLIB":
        return {"format": "VNNLIB", "path": nn_path}
    else:
        raise ValueError(f"Unsupported neural network format: {nn_format}")


# --- Example Usage ---
if __name__ == "__main__":
    FILE_PATH = "../resources/test-hybrid-nncs/test_hybrid.yaml"  # Replace with your actual filename

    # 1. Load System
    hybrid_sys = load_hybrid_system(FILE_PATH)

    # 2. Load Reach Settings
    reach_set = load_controller_settings(FILE_PATH)

    # 3. Create NN Parser
    # Assuming format is ONNX based on your YAML key
    nn_parser = create_network_parser("ONNX", reach_set['onnx_path'])

    print("Parsing Complete.")