from tensor_data_handler import TensorDataHandler
from tucker_decomposition_solver import *
from tucker_decomposition_utils import *

import copy
import numpy as np
import tensorly
import tensorly.decomposition
import matplotlib.pyplot as plt

# TODO(fahrbach): Add trials to output paths so we can average running times.

@dataclasses.dataclass
class CoreShapeSolveResult:
    date: str = datetime.datetime.now().isoformat()

    # Inputs
    input_shape: tuple = None
    algorithm: str = None
    budget: int = None
    
    # Outputs
    core_shape: tuple = None
    num_tucker_params: int = None
    compression_ratio: float = None
    hosvd_time_seconds: float = 0.0
    # `solve_time_seconds` is the sum of HOSVD and core shape search time.
    solve_time_seconds: float = None

    # `hosvd_prefix_sum` is the sum of packed singular values across all modes.
    hosvd_prefix_sum: float = None
    # `hosvd_suffix_sum` is the sum of unpacked singular values, i.e.,
    # N * |X|_{F}^2 - `hosvd_prefix_sum`.
    hosvd_suffix_sum: float = None

def write_core_shape_solve_result_to_file(core_shape_solve_result, filename):
    """
    Writes `core_shape_solve_result` (key, values) to `filename`.
    """
    with open(filename, 'w') as f:
        for field in dataclasses.fields(core_shape_solve_result):
            f.write(str(field.name) + ' ')
            f.write(str(getattr(core_shape_solve_result, field.name)) + '\n')

def read_core_shape_solve_result_from_file(filename):
    """
    Reads `filename` and constructs the `CoreShapeSolveResult` data.
    """
    assert os.path.exists(filename)
    solve_result = CoreShapeSolveResult()
    with open(filename, 'r') as f:
        lines = f.readlines()
        assert len(lines) == len(dataclasses.fields(solve_result))
        for line in lines:
            line = line.strip()
            tokens = line.split()
            assert len(tokens) >= 2
            key = tokens[0]
            value_str = ' '.join(tokens[1:])
            value = value_str
            if key not in ['date', 'algorithm']:
                value = eval(value_str)
            setattr(solve_result, key, value)
    return solve_result

def get_rre_from_core_shape(X, core_shape):
    core, factors = tensorly.decomposition.tucker(X,
                                                  rank=core_shape,
                                                  init='random',
                                                  n_iter_max=20,
                                                  tol=1e-20,
                                                  random_state=0)
    X_hat = tensorly.tucker_to_tensor((core, factors))
    return compute_rre(X, X_hat)

def compute_core_shape_greedy(X, budget):
    N = len(X.shape)
    core_shape = [1] * N
    cur_rre = get_rre_from_core_shape(X, core_shape)
    for step in range(10**10):
        print('step:', step, '\t', 'core_shape:', core_shape, '\t', 'rre:', cur_rre)
        best_rre = -1
        best_core_shape = []
        for n in range(N):
            if core_shape[n] == X.shape[n]:
                continue
            new_core_shape = copy.deepcopy(core_shape)
            new_core_shape[n] += 1
            new_num_tucker_params = get_num_tucker_params(X, new_core_shape)
            print(' - new_core_shape:', new_core_shape, '\t', 'budget:', new_num_tucker_params, budget)
            if new_num_tucker_params <= budget:
                new_rre = get_rre_from_core_shape(X, new_core_shape)
                print(' * rre: ', new_rre)
                if best_rre == -1 or new_rre < best_rre:
                    best_rre = new_rre
                    best_core_shape = new_core_shape
        if best_rre == -1:  # all new core shapes are infeasible
            break
        if best_rre > cur_rre + 1e9:
            break
        core_shape = best_core_shape
        cur_rre = best_rre
        if cur_rre < 1e-15:
            break
    return tuple(core_shape), cur_rre

def compute_core_shape_bang_for_buck(X, budget):
    N = len(X.shape)
    core_shape = [1] * N
    rre = get_rre_from_core_shape(X, core_shape)
    num_tucker_params = get_num_tucker_params(X, core_shape)
    for step in range(10**10):
        print('step:', step, '\t', 'core_shape:', core_shape, '\t', 'rre:', rre)
        best_core_shape = []
        best_rre = -1
        best_avg_gain = -1
        best_num_tucker_params = -1
        for n in range(N):
            if core_shape[n] == X.shape[n]:
                continue
            new_core_shape = copy.deepcopy(core_shape)
            new_core_shape[n] += 1
            new_num_tucker_params = get_num_tucker_params(X, new_core_shape)
            print(' - new_core_shape:', new_core_shape, '\t', 'budget:', new_num_tucker_params, budget)
            if new_num_tucker_params <= budget:
                new_rre = get_rre_from_core_shape(X, new_core_shape)
                rre_diff = rre - new_rre
                avg_gain = rre_diff / (new_num_tucker_params - num_tucker_params)
                print(' * rre:', new_rre)
                print(' - rre_diff:', rre_diff)
                print(' - avg_gain:', avg_gain)
                if avg_gain <= 0:
                    continue
                if best_avg_gain == -1 or avg_gain > best_avg_gain:
                    best_avg_gain = avg_gain
                    best_core_shape = new_core_shape
                    best_rre = new_rre
                    best_num_tucker_params = new_num_tucker_params
        if best_avg_gain == -1:  # all new core shapes are infeasible
            break
        core_shape = best_core_shape
        rre = best_rre
        num_tucker_params = best_num_tucker_params
        if best_rre < 1e-15:
            break
    return tuple(core_shape), rre

def get_all_unfolded_squared_singular_values(X):
    """
    Returns a list of each n-mode unfolding squared singular values as tuples.
    """
    N = len(X.shape)
    singular_values = []
    for n in range(N):
        X_n = tensorly.base.unfold(X, n)
        Sigma_n = np.linalg.svd(X_n, full_matrices=False, compute_uv=False)
        Sigma_n = Sigma_n**2
        singular_values.append(tuple(Sigma_n))
    return singular_values

# TODO: Rename `cache_path` as output_path and end with '/'.
def compute_core_shapes(X, budgets, algorithm, cache_path, trial=0):
    """
    Computes budget-constrained core shapes for `budgets` using `algorithm`.

    Args:
        - X: Input tensor.
        - budgets: List of budgets that all share the same HOSVD computation.
        - algorithm: string specifying algorith (e.g., 'hosvd-greedy').
        - trial: Marks separate runs.
    """

    assert algorithm in ['hosvd-greedy', 'hosvd-bang-for-buck', 'hosvd-brute-force']
    assert '_' not in cache_path
    assert cache_path[-1] != '/'
    cache_path += '/core-shape-solve-results/'
    
    core_shape_solve_results = [None] * len(budgets)
    need_to_compute_unfolded_singular_values = True
    print('running:', algorithm)
    for i in range(len(budgets)):
        budget = budgets[i]

        filename = cache_path
        filename += 'algorithm-' + algorithm + '_'
        filename += 'budget-' + str(budget)
        filename += 'trial-' + str(trial)
        filename += '.txt'

        if os.path.exists(filename):
            core_shape_solve_results[i] = read_core_shape_solve_result_from_file(filename)
            continue

        if need_to_compute_unfolded_singular_values:
            print(' - get_all_unfolded_squared_singular_values()')
            # Squared singular values of the mode-n unfoldings are shared.
            hosvd_start_time = time.time()
            unfolded_squared_singular_values = get_all_unfolded_squared_singular_values(X)
            hosvd_end_time = time.time()
            need_to_compute_unfolded_singular_values = False

        print(' - budget:', budget)
        if algorithm == 'hosvd-greedy':
            solve_result = compute_core_shape_hosvd_greedy(X, unfolded_squared_singular_values, budget)
        elif algorithm == 'hosvd-bang-for-buck':
            solve_result = compute_core_shape_hosvd_bang_for_buck(X, unfolded_squared_singular_values, budget)
        elif algorithm == 'hosvd-brute-force':
            solve_result = compute_core_shape_hosvd_brute_force(X, unfolded_squared_singular_values, budget)
        else:
            assert False

        if 'hosvd' in algorithm:
            solve_result.hosvd_time_seconds = hosvd_end_time - hosvd_start_time
            solve_result.solve_time_seconds += solve_result.hosvd_time_seconds

        # Cache results
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        write_core_shape_solve_result_to_file(solve_result, filename)

        core_shape_solve_results[i] = solve_result
    return core_shape_solve_results

def compute_core_shape_hosvd_greedy(X, unfolded_squared_singular_values, budget):
    start_time = time.time()

    N = len(X.shape)
    core_shape = [1] * N
    singular_sum = sum([s[0] for s in unfolded_squared_singular_values])
    for step in range(10**10):
        #print('step:', step, '\t', 'core_shape:', core_shape, '\t', 'singular_sum:', singular_sum)
        best_singular_sum = -1
        best_core_shape = []
        for n in range(N):
            if core_shape[n] == X.shape[n]:
                continue
            new_core_shape = copy.deepcopy(core_shape)
            new_core_shape[n] += 1
            new_num_tucker_params = get_num_tucker_params(X, new_core_shape)
            #print(' - new_core_shape:', new_core_shape, '\t', 'budget:', new_num_tucker_params, budget)
            if new_num_tucker_params <= budget:
                new_singular_sum = singular_sum + unfolded_squared_singular_values[n][new_core_shape[n] - 1]
                #print(' * singular_sum: ', new_singular_sum)
                if new_singular_sum > best_singular_sum:
                    best_singular_sum = new_singular_sum
                    best_core_shape = new_core_shape
        if best_singular_sum == -1:  # all new core shapes are infeasible
            break
        core_shape = best_core_shape
        singular_sum = best_singular_sum

    end_time = time.time()

    solve_result = CoreShapeSolveResult()
    solve_result.input_shape = X.shape
    solve_result.algorithm = 'hosvd-greedy'
    solve_result.budget = budget
    solve_result.core_shape = core_shape
    solve_result.num_tucker_params = get_num_tucker_params(X, core_shape)
    solve_result.compression_ratio = solve_result.num_tucker_params / X.size

    solve_result.solve_time_seconds = end_time - start_time

    solve_result.hosvd_prefix_sum = singular_sum
    solve_result.hosvd_suffix_sum = 0.0
    for n in range(N):
        solve_result.hosvd_suffix_sum += sum(unfolded_squared_singular_values[n])
    solve_result.hosvd_suffix_sum -= singular_sum
    return solve_result

def compute_core_shape_hosvd_bang_for_buck(X, unfolded_squared_singular_values, budget):
    start_time = time.time()

    N = len(X.shape)
    core_shape = [1] * N
    singular_sum = sum([s[0] for s in unfolded_squared_singular_values])
    num_tucker_params = get_num_tucker_params(X, core_shape)
    for step in range(10**10):
        best_singular_sum = -1
        best_core_shape = []
        best_avg_gain = -1
        for n in range(N):
            if core_shape[n] == X.shape[n]:
                continue
            new_core_shape = copy.deepcopy(core_shape)
            new_core_shape[n] += 1
            new_num_tucker_params = get_num_tucker_params(X, new_core_shape)
            if new_num_tucker_params <= budget:
                new_singular_sum = singular_sum + unfolded_squared_singular_values[n][new_core_shape[n] - 1]
                new_avg_gain = unfolded_squared_singular_values[n][new_core_shape[n] - 1]
                new_avg_gain /= (new_num_tucker_params - num_tucker_params)
                if new_avg_gain > best_avg_gain:
                    best_singular_sum = new_singular_sum
                    best_core_shape = new_core_shape
                    best_avg_gain = new_avg_gain
        if best_singular_sum == -1:  # all new core shapes are infeasible
            break
        core_shape = best_core_shape
        singular_sum = best_singular_sum

    end_time = time.time()

    solve_result = CoreShapeSolveResult()
    solve_result.input_shape = X.shape
    solve_result.algorithm = 'hosvd-bang-for-buck'
    solve_result.budget = budget
    solve_result.core_shape = core_shape
    solve_result.num_tucker_params = get_num_tucker_params(X, core_shape)
    solve_result.compression_ratio = solve_result.num_tucker_params / X.size

    solve_result.solve_time_seconds = end_time - start_time

    solve_result.hosvd_prefix_sum = singular_sum
    solve_result.hosvd_suffix_sum = 0.0
    for n in range(N):
        solve_result.hosvd_suffix_sum += sum(unfolded_squared_singular_values[n])
    solve_result.hosvd_suffix_sum -= singular_sum
    return solve_result

# TODO(fahrbach): Is there a simple DP / brute force to maximize singular sum?
def compute_core_shape_hosvd_brute_force(X, unfolded_squared_singular_values, budget):
    start_time = time.time()

    N = len(X.shape)
    core_shape = [1] * N
    best_singular_sum = -1
    best_core_shape = []
    if N == 3:
        for i0 in range(X.shape[0]):
            if get_num_tucker_params(X, [i0, 1, 1]) > budget: break
            for i1 in range(X.shape[1]):
                if get_num_tucker_params(X, [i0, i1, 1]) > budget: break
                for i2 in range(X.shape[2]):
                    if get_num_tucker_params(X, [i0, i1, i2]) > budget: break
                    s = 0.0
                    s += sum(unfolded_squared_singular_values[0][:i0 + 1])
                    s += sum(unfolded_squared_singular_values[1][:i1 + 1])
                    s += sum(unfolded_squared_singular_values[2][:i2 + 1])
                    if s > best_singular_sum:
                        best_singular_sum = s
                        best_core_shape = [i0, i1, i2]
    elif N == 4:
        for i0 in range(X.shape[0]):
            if get_num_tucker_params(X, [i0, 1, 1, 1]) > budget: break
            for i1 in range(X.shape[1]):
                if get_num_tucker_params(X, [i0, i1, 1, 1]) > budget: break
                for i2 in range(X.shape[2]):
                    if get_num_tucker_params(X, [i0, i1, i2, 1]) > budget: break
                    for i3 in range(X.shape[3]):
                        if get_num_tucker_params(X, [i0, i1, i2, i3]) > budget: break
                        s = 0.0
                        s += sum(unfolded_squared_singular_values[0][:i0 + 1])
                        s += sum(unfolded_squared_singular_values[1][:i1 + 1])
                        s += sum(unfolded_squared_singular_values[2][:i2 + 1])
                        s += sum(unfolded_squared_singular_values[3][:i3 + 1])
                        if s > best_singular_sum:
                            best_singular_sum = s
                            best_core_shape = [i0, i1, i2, i3]
    else:
        assert False
    end_time = time.time()

    solve_result = CoreShapeSolveResult()
    solve_result.input_shape = X.shape
    solve_result.algorithm = 'hosvd-brute-force'
    solve_result.budget = budget
    solve_result.core_shape = core_shape
    solve_result.num_tucker_params = get_num_tucker_params(X, core_shape)
    solve_result.compression_ratio = solve_result.num_tucker_params / X.size

    solve_result.solve_time_seconds = end_time - start_time

    solve_result.hosvd_prefix_sum = best_singular_sum
    solve_result.hosvd_suffix_sum = 0.0
    for n in range(N):
        solve_result.hosvd_suffix_sum += sum(unfolded_squared_singular_values[n])
    solve_result.hosvd_suffix_sum -= best_singular_sum
    return solve_result

"""
Note: Very good experimental setup:
  - dataset: hyperspectral
  - budget: ~0.001 * X.size (~0.1% compression of original tensor)
"""
def main():
    handler = TensorDataHandler()
    input_shape = [20, 100, 200]
    core_shape = [3, 7, 13]
    random_seed = 123
    #X = handler.generate_random_tucker(input_shape, core_shape, random_seed)
    #X = handler.load_image('data/images/cat.jpg', resize_shape=(100, 200))
    X = handler.load_hyperspectral()
    #X = handler.load_cardiac_mri_data()
    #X = handler.load_coil_100()
    print(X.shape)
    print(X.size)

    budget = int(0.0001 * X.size)
    print('budget:', budget)

    core_shape, rre = compute_core_shape(X, budget, 'greedy-approx')
    print('greedy_approx_core_shape:', core_shape)
    print('greedy_approx_rre:', rre)
    print()

    """
    core_shape, rre = compute_core_shape(X, budget, 'brute-force-approx')
    print('brute_force_approx_core_shape:', core_shape)
    print('brute_force_approx_rre:', rre)
    print()

    core_shape, rre = compute_core_shape(X, budget, 'greedy')
    print('greedy_core_shape:', core_shape)
    print('greedy_rre:', rre)
    print()

    core_shape, rre = compute_core_shape(X, budget, 'bang-for-buck')
    print('bfb_core_shape:', core_shape)
    print('bfb_rre:', rre)
    """

#main()

