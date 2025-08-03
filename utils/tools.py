import ast
from itertools import combinations, product

import numpy as np
from collections.abc import Iterable

import gurobipy as gp
from gurobipy import GRB


def numpy_shift(arr, num_places, fill_na=0):
    """Shifts the elements of the array to the right by num_places, filling with 0."""
    if num_places == 0:
        return arr
    elif num_places > 0:
        result = np.empty_like(arr, dtype=float)
        result[:num_places] = fill_na
        result[num_places:] = arr[:-num_places]
    else:
        result = np.empty_like(arr, dtype=float)
        result[num_places:] = fill_na
        result[:num_places] = arr[-num_places:]

    return result

def iter_to_tuple(lst):
    if isinstance(lst, Iterable):
        return tuple(iter_to_tuple(sub) for sub in lst)
    return lst

def iter_to_list(obj):
    """Recursively convert tuples back to lists (inverse of iter_to_tuple)."""
    if isinstance(obj, (str, bytes)):
        return obj
    if isinstance(obj, tuple):
        return [iter_to_list(sub) for sub in obj]
    # Keep other iterables (e.g. set, dict) unchanged unless you want to handle them too
    if isinstance(obj, Iterable):
        return type(obj)(iter_to_list(sub) for sub in obj)
    return obj

def convert_tuple_keys_to_str(d):
    """ Recursively convert tuple keys to strings. """
    if isinstance(d, dict):
        return {str(k): convert_tuple_keys_to_str(v) for k, v in d.items()}
    else:
        return d

def convert_str_keys_to_tuple(d):
    """ Recursively convert string keys back to tuples. """
    if isinstance(d, dict):
        return {ast.literal_eval(k): convert_str_keys_to_tuple(v) for k, v in d.items()}
    else:
        return d

def keep_significant_digits(number, significant: int) -> float:
    return float(("{:." + str(significant) + "g}").format(number))

def get_solution_value(var_array, use_xn=False):
    """
    Extracts numerical solution values from a numpy array of Gurobi variables.

    Args:
        var_array (np.ndarray): A numpy array containing gurobipy.Var objects.
        use_xn (bool): If True, extracts values from the solution pool (.Xn).
                       If False, extracts the primary solution value (.X).

    Returns:
        np.ndarray: A numpy array with the same shape as the input,
                    containing the floating-point solution values.
    """
    # Determine which Gurobi attribute to get: 'X' for the main solution
    # or 'Xn' for solutions from the pool.
    attribute = "Xn" if use_xn else "X"

    # Define a simple function that gets the desired attribute from a single variable.
    getter = lambda v: v.getAttr(attribute)

    # Use np.vectorize to apply the getter function to each element of the input array.
    return np.vectorize(getter)(var_array)

def integer_partitions_fixed_bins(total, bins):
    """
    Generate all integer partitions of `total` into `bins` non-negative integers.
    """
    for bars in combinations(range(total + bins - 1), bins - 1):
        result = []
        prev = -1
        for b in bars:
            result.append(b - prev - 1)
            prev = b
        result.append(total + bins - 1 - prev - 1)
        yield result

def generate_arrivals(maximum_arrival, num_type):
    """
    Enumerate all possible arrival vectors (x_1, ..., x_I) and their
    associated probability. Returns a list of (prob, counts_vector).
    """
    # For each possible total arrival from 0..maximum_arrival
    for N in range(maximum_arrival + 1):
        for arrivals in integer_partitions_fixed_bins(total=N, bins=num_type):
            yield np.array(arrivals)

def generate_bookings(maximum_slots, planning_horizon):
    for p in product(range(maximum_slots + 1), repeat=planning_horizon):
        yield np.array(p)

def generate_states(maximum_slots, planning_horizon, maximum_arrival, num_type):
    for bookings in generate_bookings(maximum_slots, planning_horizon):
        for arrivals in generate_arrivals(maximum_arrival, num_type):
            yield (bookings, arrivals)

def generate_advance_actions(waitlist, number_days):
    if number_days < 1:
        raise ValueError('number_days must be at least 1')

    # Step 1: Precompute valid partitions for each class
    valid_partitions_by_class = [
        list(integer_partitions_fixed_bins(w, number_days)) for w in waitlist
    ]  # len = num_classes, each element is a list of (number_days,) vectors

    # Step 2: Take Cartesian product over classes
    for action in product(*valid_partitions_by_class):
        yield np.array(action).T

def generate_state_action_pairs(maximum_slots, maximum_num_sessions, maximum_arrival, num_type, period_to_go):
    for N in range(period_to_go, 0, -1):
        t = period_to_go - N + 1
        for state in generate_states(maximum_slots, N+maximum_num_sessions-1, maximum_arrival, num_type):
            for action in generate_advance_actions(state[1], N):
                yield (state, action, t)


def get_status_string(status_code):
    """
    Converts a Gurobi status code into a human-readable string.

    Args:
        status_code (int): The status code from a Gurobi model.

    Returns:
        str: A string representation of the status.
    """
    status_map = {
        GRB.LOADED: "LOADED",
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INPROGRESS: "INPROGRESS",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
    }
    return status_map.get(status_code, "UNKNOWN_STATUS")


def solve_and_handle_errors(model: gp.Model):
    """
    Optimizes a Gurobi model and handles non-optimal statuses by saving the model.

    Args:
        model (gp.Model): The Gurobi model to be solved.
    """
    try:
        # Optimize the model
        model.optimize()

        # Check the final optimization status
        if model.Status == GRB.OPTIMAL:
            print(f"Model '{model.ModelName}' solved to optimality.")
            print(f"Objective value: {model.ObjVal}")
            return True

        else:
            # --- Handle non-optimal cases ---
            status_string = get_status_string(model.Status)
            model_name = model.ModelName if model.ModelName.strip() else "unnamed_model"

            print(f"\n--- Optimization Failed for model '{model_name}' ---")
            print(f"Status: {status_string} ({model.Status})")

            # Construct filename and save the model as an LP file
            filename = f"{model_name}_{status_string}.lp"
            print(f"Saving model to file: {filename}")
            model.write(filename)

            # If the model is infeasible, compute and save the IIS
            if model.Status == GRB.INFEASIBLE:
                print("Model is infeasible. Computing Irreducible Inconsistent Subsystem (IIS)...")
                model.computeIIS()

                # The IIS is a subset of the original model's constraints and bounds
                # that is still infeasible, but becomes feasible if any single one
                # of its constraints or bounds is removed.
                iis_filename = f"{model_name}_infeasible_iis.ilp"
                print(f"Saving IIS to file: {iis_filename}")
                model.write(iis_filename)
                print("Use the .ilp file to identify the conflicting constraints.")

            return False

    except gp.GurobiError as e:
        print(f"A Gurobi error occurred: {e.message} (error code {e.errno})")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False
if __name__ == '__main__':
    state_action_pairs = list(generate_state_action_pairs(maximum_slots=3,
                                              maximum_num_sessions=2,
                                              maximum_arrival=3,
                                              num_type=2,
                                              period_to_go=3))
    print(len(state_action_pairs))

