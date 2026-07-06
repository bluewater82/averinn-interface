from datetime import datetime
import os
from sympy.physics.units import length

from src.rasc.setpropagation import SetPropagation
from src.rasc.technique import Technique
from src.set.box import Box
from src.set.set import Set
from src.set.setuts import SetUTS
from src.utilities.log import Log
from typing import List, Dict, Tuple

import numpy as np

np.set_printoptions(suppress=True, precision=4)


def save_bounds_data(lower_bounds_list, upper_bounds_list):
    """
    Saves the iteration results to a .npz file.
    lower_bounds_list: List of arrays [pos_l, spd_l, acc_l, pos_e, spd_e, acc_e]
    upper_bounds_list: List of arrays [pos_l, spd_l, acc_l, pos_e, spd_e, acc_e]
    """
    folder_path = os.path.join("outputs", "reachability", "data")
    data_lower = np.array(lower_bounds_list)
    data_upper = np.array(upper_bounds_list)
    num_states = data_lower.shape[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reachability_{num_states}_step_{timestamp}"
    full_path = os.path.join(folder_path, filename)
    # Save multiple arrays into one file
    np.savez(
        full_path,
        lower=np.array(data_lower),
        upper=np.array(data_upper),
        # total_time=duration,
        total_steps=num_states
    )
    print(f"Data saved to {filename} with {num_states} states.")

def intersect_constraints(objStateSet, constraints, indicators):
    """
    Intersects the objStateSet with the active mode constraints.

    :param objStateSet: The current Set (Box or Star)
    :param constraints: List of [a1, a2, a3, a4, b]
    :param indicators: List of -1, 1, or 0
    :return: The constrained Set
    """
    active_constraints_A = []
    active_constraints_b = []

    for i in range(len(indicators)):
        mode = indicators[i]
        if mode == 0:
            continue  # Ignore this constraint

        # Extract A (first 4 dims) and b (5th element)
        row = np.array(constraints[i])
        A_vec = row[:-1]
        b_val = row[-1]

        if mode == -1:
            # Constraint is Ax + b <= 0 -> Ax <= -b
            active_constraints_A.append(A_vec)
            active_constraints_b.append(-b_val)
        elif mode == 1:
            # Constraint is Ax + b > 0 -> -Ax - b < 0 -> -Ax < b
            # (Using <= for the solver limit)
            active_constraints_A.append(-A_vec)
            active_constraints_b.append(b_val)

    # Convert to numpy arrays for the solver
    A_matrix = np.array(active_constraints_A)
    b_vector = np.array(active_constraints_b)

    # Perform the intersection
    # Note: Assuming your Set class has an 'intersectHalfSpace' or 'addConstraints' method
    # If it's a Star Set, we typically use:
    constrained_set = objStateSet.intersectHalfSpace(A_matrix, b_vector, 'X')

    return constrained_set


# Example Usage:
# my_constraints = [[0.0, 0.0, 0.0, -1.0, -6.0], ...]
# my_indicators = [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# new_set = intersect_constraints(objStateSet, my_constraints, my_indicators)
def one_step_affine_post(objInputSet, objStateSet, objDtDyn):
    if objInputSet is not None:
        objInputSet = SetUTS.toStarSet(objInputSet)
    objStateSet = SetUTS.toStarSet(objStateSet)
    # rangeSet = objInputSet.getRange()
    # print("\t\tBefore One Step Affine with starset U")
    # print("\t\tobjInputSet  Lower: " + str(rangeSet[0]) + "\n")
    # print("\t\tobjInputSet Upper: " + str(rangeSet[1]) + "\n")
    # rangeSet = objStateSet.getRange()
    # print("\t\t\Before One Step Affine with starset X")
    # print("\t\tobjInputSet  Lower: " + str(rangeSet[0]) + "\n")
    # print("\t\tobjInputSet Upper: " + str(rangeSet[1]) + "\n")
    if (objDtDyn.B is None) and (objDtDyn.C is None):
        objStateSet = objStateSet.linearMap(objDtDyn.A, objDtDyn.A)
    elif (objDtDyn.B is None) and (objDtDyn.C is not None):
        objStateSet = objStateSet.affineMap(objDtDyn.A, objDtDyn.C, objDtDyn.A, objDtDyn.C)
    elif (objDtDyn.B is not None) and (objDtDyn.C is None):
        objStateSet = objStateSet.linearMap(objDtDyn.A, objDtDyn.A)
        mapped_input = objInputSet.linearMap(objDtDyn.B, objDtDyn.B)
        objStateSet = objStateSet.minkowskiSum(mapped_input)
    elif (objDtDyn.B is not None) and (objDtDyn.C is not None):
        objStateSet = objStateSet.linearMap(objDtDyn.A, objDtDyn.A)
        objStateSet = objStateSet.minkowskiSum(objInputSet.affineMap(objDtDyn.B, objDtDyn.C,
                                                                     objDtDyn.B, objDtDyn.C))
    # rangeSet = objInputSet.getRange()
    # print("\t\t\After One Step Affine with starset U")
    # print("\t\tobjInputSet  Lower: " + str(rangeSet[0]) + "\n")
    # print("\t\tobjInputSet Upper: " + str(rangeSet[1]) + "\n")
    # rangeSet = objStateSet.getRange()
    # print("\t\t\After One Step Affine with starset X")
    # print("\t\tobjInputSet  Lower: " + str(rangeSet[0]) + "\n")
    # print("\t\tobjInputSet Upper: " + str(rangeSet[1]) + "\n")
    return objStateSet, objInputSet


def compute_nn_output_box(state_box, obj_gnn, output_constr, solverType, lastRelu, TranM, TranB):
    # Log.message("Interation" + str(i) + "\n")
    # lower_0 = np.array([90.0, 32.0, 0.0, 10.0, 30.0])
    # upper_0 = np.array([110.0, 32.2, 0.0, 11.0, 30.2])
    # objSet: Set = Box(lower_0, upper_0)
    print("--------------------------------------------" + "\n")
    objStateSet = SetUTS.toStarSet(state_box)
    # objSet = SetUTS.toStarSet(objSet)

    objStateSet = objStateSet.affineMap(TranM, TranB, TranM, TranB)
    print("State after transform")
    rangeSet = objStateSet.getRange()
    print("Lower: " + str(rangeSet[0]))
    print("Upper: " + str(rangeSet[1]))
    print("--------------------------------------------" + "\n")

    objTechnique = SetPropagation(obj_gnn, objStateSet, output_constr, solverType, lastRelu)
    listSets: List[Set] = objTechnique.reachSet()
    objInputSet = SetUTS.rangeOfSets(listSets)

    # objTechnique = SetPropagation(objGNNUse, objStateSet, outputConstr, solverType, lastRelu)
    return objInputSet

def hybrid_k_step_reach(K, objGNN, hs_cong, objStateSet,  hybrid_systems, solverType, lastRelu, outputConstr, TranM, TranB):
    Q = len(hybrid_systems)
    lower_bounds_list = []
    upper_bounds_list = []
    X_k = objStateSet
    print("X_k start regions (initial regions):")
    rangeSet = X_k.getRange()
    print("Lower: " + str(rangeSet[0]) )
    print("Upper: " + str(rangeSet[1]) )
    print("--------------------------------------------" + "\n")
    for i in range(K):
        print("\tIter K = " + str(i) + "\n")
        S = []
        objInputSet = None
        if int(hs_cong["num_inputs"]) > 0 :
            objInputSet = compute_nn_output_box(X_k, objGNN, outputConstr, solverType, lastRelu, TranM, TranB)
            rangeSet = objInputSet.getRange()
            clean_list = [float(i) for z in rangeSet]
            print("\tControl U after Propagate/.." )
            print(f"\tU Lower: {rangeSet[0]}")
            print(f"\tU Upper: {rangeSet[1]} \n")
        S_k = None
        for j in (range(Q)):
            mode_key = f'mode{j+1}'
            X_k_i = intersect_constraints(X_k, hs_cong["lin_exp"], hs_cong["modes"][mode_key]["U"])
            # X_k_i = objStateSet

            if X_k_i is not None:
                print("\t\tX_K intersection with Mode Q=" + str(j) + " (if any)")
                rangeSet = X_k_i.getRange()
                print("\t\tX_k_i Lower: " + str(rangeSet[0]) )
                print("\t\tX_k_i Upper: " + str(rangeSet[1]) + "\n")
                Post_k_i, objInputSet = one_step_affine_post(objInputSet, X_k_i, hybrid_systems[j])
                if S_k is None:
                    S_k = Post_k_i
                else:
                    # Use the convexHull
                    S_k = S_k.convexHull(Post_k_i)
                S.append(S_k)
                print("\t\tNew Region Post_k_i")
                rangeSet = Post_k_i.getRange()
                print("\t\tPost_k_i Lower: " + str(rangeSet[0]))
                print("\t\tPost_k_i Upper: " + str(rangeSet[1]) + "\n")
                # np.array2string(upper, suppress=True, precision=4)
                print("\t\tconvexHull S_k")
                rangeSet = S_k.getRange()
                print(f"\t\tconvexHull S_k Lower: {rangeSet[0]}")
                print("\t\tconvexHull S_k Upper: " + str(rangeSet[1]) + "\n")
        if len(S) > 0:
            X_k = S[0]
            for k in range(1, len(S)):
                X_k = X_k.convexHull(S[k])
            rangeSet = X_k.getRange()
            print("\tX_k+1 is:")
            print("\tX_k+1 Lower: " + str(rangeSet[0]))
            print("\tX_k+1 Upper: " + str(rangeSet[1]) + "\n")
            lower_bounds_list.append(rangeSet[0])
            upper_bounds_list.append(rangeSet[1])
    # if S_k is None:
    #     Log.message("Reach is un Safe")
    #     print("\tS_k is null so reach Set Safe")
    #     rangeSet = X_k.getRange()
    #     print("\tLower: " + str(rangeSet[0]))
    #     print("\tUpper: " + str(rangeSet[1]) + "\n")
    #     break
    print("Final X_k:")
    rangeSet = X_k.getRange()
    print("Lower: " + str(rangeSet[0]))
    print("Upper: " + str(rangeSet[1]) + "\n")
    isIntersect: bool = SetUTS.intersectWithUnsafe(X_k, outputConstr, solverType, 'X')

    if isIntersect:
        print("Safety Status: Unsafe \n")
        Log.message("Safety Status: Unsafe \n")
    else:
        print("Safety Status: Safe \n")
        Log.message("Safety Status: Safe \n")

    save_bounds_data(lower_bounds_list, upper_bounds_list)

# if __name__ == '__main__':
#     reachability(K)