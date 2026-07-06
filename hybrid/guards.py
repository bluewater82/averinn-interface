import numpy as np


def build_guard_constraints(lin_exp, signature, epsilon=1e-9):
    """
    Constructs the guard constraints for a specific mode.

    Args:
        lin_exp (np.array): The 2D array of linear expression coefficients.
        signature (list/np.array): Entries of -1, 1, or 0.
        epsilon (float): Small offset for strict inequalities (> 0 becomes >= epsilon).

    Returns:
        dict: A dictionary containing the A_g and b_g matrices for the 
              polyhedral constraint (A_g * x <= b_g).
    """
    lin_exp = np.array(lin_exp)
    signature = np.array(signature)

    if len(lin_exp) != len(signature):
        raise ValueError("lin_exp and signature must have the same number of rows.")

    A_list = []
    b_list = []

    for i, val in enumerate(signature):
        # Extract coefficients (first n-1 elements) and constant (last element)
        coeffs = lin_exp[i, :-1]
        constant = lin_exp[i, -1]

        if val == -1:
            # Condition: expr <= 0  => (ax + b) <= 0 => ax <= -b
            A_list.append(coeffs)
            b_list.append(-constant)

        elif val == 1:
            # Condition: expr > 0   => (ax + b) >= epsilon 
            # Multiply by -1 to flip to <= format: -ax - b <= -epsilon => -ax <= b - epsilon
            A_list.append(-coeffs)
            b_list.append(constant - epsilon)

        elif val == 0:
            # Inactive expression: skip
            continue

    if not A_list:
        return None

    return {
        "A_g": np.array(A_list),
        "b_g": np.array(b_list)
    }


def intersect_with_guard(current_region, guard_constraints):
    """
    Intersects the current reachable region with the mode guard.

    Args:
        current_region: The object representing the current reachable set (Polytope/Zonotope).
        guard_constraints (dict): Output from build_guard_constraints.

    Returns:
        The intersected region, or None if the intersection is empty.
    """
    if guard_constraints is None:
        return current_region

    # Note: The implementation of 'intersect' depends on your reachability library
    # (e.g., CORA, HyPRO, or custom Polytope library).
    try:
        # Example pseudo-logic:
        # intersected = current_region.intersect(guard_constraints['A_g'], guard_constraints['b_g'])

        # if intersected.is_empty():
        #     return None

        # For now, we return a conceptual placeholder
        return "intersected_region_object"

    except Exception as e:
        print(f"Intersection failed or resulted in empty set: {e}")
        return None


# --- Example Usage ---
if __name__ == "__main__":
    # From your YAML
    lin_exp = [
        [0.0, 0.0, 0.0, -1.0, -6.0],  # expr 0
        [0.0, 0.0, 0.0, 1.0, 3.0],  # expr 1
        [0.0, 0.0, 0.0, -1.0, -3.0],  # expr 2
        [0.0, 0.0, 0.0, 1.0, 0.0]  # ...
    ]
    # Example: Mode 1 signature from your YAML [-1, -1, 0, 0, ...]
    sig_mode1 = [-1, -1, 0, 0]

    guard = build_guard_constraints(lin_exp[:4], sig_mode1, epsilon=1e-5)

    if guard:
        print("Guard A matrix:\n", guard['A_g'])
        print("Guard b vector:\n", guard['b_g'])