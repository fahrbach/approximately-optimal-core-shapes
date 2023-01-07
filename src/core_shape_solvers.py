from tensor_data_handler import TensorDataHandler
from tucker_decomposition_solver import *
from tucker_decomposition_utils import *
from scipy.optimize import milp
from scipy.optimize import LinearConstraint

import copy
import numpy as np
import scipy
import tensorly
import sparse
import tensorly.decomposition
import matplotlib.pyplot as plt

@dataclasses.dataclass
class CoreShapeSolveResult:
    date: str = datetime.datetime.now().isoformat()

    # Inputs
    input_shape: tuple = None
    algorithm: str = None
    budget: int = None
    trial: int = None
    
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

def compute_core_shape_rre_greedy(X, unfolded_squared_singular_values, budget, output_path, trial):
    """
    Construct core shape greedily using the true RRE via Tucker decomposition.
    """
    start_time = time.time()
    tucker_decomposition_subroutine_total_time = 0.0

    N = len(X.shape)
    core_shape = [1] * N
    tmp_tucker_solve_result = compute_tucker_decomposition(X, core_shape, output_path, trial)
    cur_rre = tmp_tucker_solve_result.rre
    tucker_decomposition_subroutine_total_time += tmp_tucker_solve_result.solve_time_seconds
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
                tmp_tucker_solve_result = compute_tucker_decomposition(X, new_core_shape, output_path, trial)
                new_rre = tmp_tucker_solve_result.rre
                tucker_decomposition_subroutine_total_time += tmp_tucker_solve_result.solve_time_seconds
                print(' * rre: ', new_rre)
                if best_rre == -1 or new_rre < best_rre:
                    best_rre = new_rre
                    best_core_shape = new_core_shape
        if best_rre == -1:  # all new core shapes are infeasible
            break
        if best_rre > cur_rre + 1e-9:
            break
        core_shape = best_core_shape
        cur_rre = best_rre
        if cur_rre < 1e-15:
            break
    end_time = time.time()

    singular_sum = 0.0
    for n in range(N):
        singular_sum += sum(unfolded_squared_singular_values[n][:core_shape[n]])

    # Construct RRE-greedy solve result
    solve_result = CoreShapeSolveResult()
    solve_result.input_shape = X.shape
    solve_result.algorithm = 'rre-greedy'
    solve_result.budget = budget
    solve_result.core_shape = core_shape
    solve_result.num_tucker_params = get_num_tucker_params(X, core_shape)
    solve_result.compression_ratio = solve_result.num_tucker_params / X.size

    solve_result.solve_time_seconds = (end_time - start_time) + tucker_decomposition_subroutine_total_time

    solve_result.hosvd_prefix_sum = singular_sum
    solve_result.hosvd_suffix_sum = 0.0
    for n in range(N):
        solve_result.hosvd_suffix_sum += sum(unfolded_squared_singular_values[n])
    solve_result.hosvd_suffix_sum -= singular_sum
    return solve_result

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
        if isinstance(X, sparse.COO):
            X_n = sparse_unfold(X, n)
        else:
            X_n = tensorly.base.unfold(X, n)
        Sigma_n = np.linalg.svd(X_n, full_matrices=False, compute_uv=False)
        Sigma_n = Sigma_n**2
        singular_values.append(tuple(Sigma_n))
    return singular_values

def compute_core_shapes(X, budgets, algorithm, output_path, trial=0):
    """
    Computes budget-constrained core shapes for `budgets` using `algorithm`.

    Args:
        - X: Input tensor.
        - budgets: List of budgets that all share the same HOSVD computation.
        - algorithm: string specifying algorith (e.g., 'hosvd-greedy').
        - trial: Used to mark separate runs and averaging.
    """

    assert algorithm in ['hosvd-greedy', 'hosvd-bang-for-buck', 'hosvd-brute-force', 'hosvd-ip', 'rre-greedy']
    assert output_path[-1] == '/'
    core_shape_output_path = output_path + 'core-shape-solve-results/'
    
    core_shape_solve_results = [None] * len(budgets)
    need_to_compute_unfolded_singular_values = True
    print('running:', algorithm)
    for i in range(len(budgets)):
        budget = budgets[i]

        filename = core_shape_output_path
        filename += 'algorithm-' + algorithm + '_'
        filename += 'budget-' + str(budget) + '_'
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
        elif algorithm == 'hosvd-ip':
            solve_result = compute_core_shape_hosvd_integer_program(X, unfolded_squared_singular_values, budget, 0.5)
        elif algorithm == 'rre-greedy':
            solve_result = compute_core_shape_rre_greedy(X, unfolded_squared_singular_values, budget, output_path, trial)
        else:
            assert False

        if 'hosvd' in algorithm:
            solve_result.hosvd_time_seconds = hosvd_end_time - hosvd_start_time
            solve_result.solve_time_seconds += solve_result.hosvd_time_seconds

        solve_result.trial = trial
        # Cache results
        if not os.path.exists(core_shape_output_path):
            os.makedirs(core_shape_output_path)
        write_core_shape_solve_result_to_file(solve_result, filename)

        core_shape_solve_results[i] = solve_result
    return core_shape_solve_results

def compute_core_shape_hosvd_greedy(X, unfolded_squared_singular_values, budget):
    """
    Greedily construct core tensor by augmenting dims until budget is exhausted.
    """
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
    """
    Constuct core tensor using bang-for-buck algorithm w.r.t. remaining budget.
    """
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

def recurse(X, unfolded_squared_singular_values, budget, core_shape):
    N = len(X.shape)
    best_s = -1
    best_core_shape = None

    if len(core_shape) == N:
        s = 0.0
        for n in range(N):
            s += sum(unfolded_squared_singular_values[n][:core_shape[n]])
        return s, tuple(core_shape)

    n = len(core_shape)
    for i in range(1, X.shape[n] + 1):
        new_core_shape = copy.deepcopy(core_shape)
        new_core_shape.append(i)
        new_core_shape_tmp = copy.deepcopy(new_core_shape)
        while len(new_core_shape_tmp) < N:
            new_core_shape_tmp.append(1)
        num_params = get_num_tucker_params(X, new_core_shape_tmp)
        if num_params > budget:
            break
        # Process good core shape candidate
        tmp_best_s, tmp_best_core_shape = recurse(X, unfolded_squared_singular_values, budget, new_core_shape)
        if tmp_best_s > best_s:
            best_s = tmp_best_s
            best_core_shape = tmp_best_core_shape
    return best_s, best_core_shape

# TODO(fahrbach): Write O(B^2) DP algorithm for this packing problem?
def compute_core_shape_hosvd_brute_force(X, unfolded_squared_singular_values, budget):
    """
    Loop over all feasible core shapes and take the best singular sum.
    """
    start_time = time.time()

    N = len(X.shape)
    core_shape = []
    best_singular_sum, best_core_shape = recurse(X, unfolded_squared_singular_values, budget, core_shape)
    print(budget, '-->', best_singular_sum, best_core_shape)

    end_time = time.time()

    solve_result = CoreShapeSolveResult()
    solve_result.input_shape = X.shape
    solve_result.algorithm = 'hosvd-brute-force'
    solve_result.budget = budget
    solve_result.core_shape = best_core_shape
    solve_result.num_tucker_params = get_num_tucker_params(X, best_core_shape)
    solve_result.compression_ratio = solve_result.num_tucker_params / X.size

    solve_result.solve_time_seconds = end_time - start_time

    solve_result.hosvd_prefix_sum = best_singular_sum
    solve_result.hosvd_suffix_sum = 0.0
    for n in range(N):
        solve_result.hosvd_suffix_sum += sum(unfolded_squared_singular_values[n])
    solve_result.hosvd_suffix_sum -= best_singular_sum
    return solve_result


def compute_core_shape_hosvd_integer_program(X, unfolded_squared_singular_values, budget, eps=0.5):
    start_time = time.time()

    N = len(X.shape)

    temp_shape = np.zeros(N,dtype=np.int)

    sq_sing_vals = [[] for i in range(N)]

    for i in range(N):
        temp_shape[i] = np.minimum(np.ceil(1/eps),X.shape[i],dtype=np.int,casting="unsafe")
        sq_sing_vals[i] = unfolded_squared_singular_values[i][:temp_shape[i]]

    best_result = compute_core_shape_hosvd_brute_force(sparse.zeros(tuple(temp_shape), format='coo'), sq_sing_vals, budget)
    best_singular_sum = best_result.hosvd_prefix_sum
    besti = -1
    bestj = -1

    for k in range(int(np.floor(np.log(budget)/np.log(1+eps)))):
        for i in range(N):
            Xtemp_shape = np.zeros(N-1,dtype=np.int)
            counter = 0
            sq_sing_vals = [[] for i in range(len(Xtemp_shape))]
            for l in range(N):
                if i == l:
                    continue
                Xtemp_shape[counter] = X.shape[l]
                sq_sing_vals[counter] = unfolded_squared_singular_values[l]
                counter += 1
            for j in range(np.maximum(int(np.floor(np.ceil(1/eps) / (1+eps))),1,dtype=int,casting="unsafe"), X.shape[i] + 1):
                b_prod = np.power(1+eps, k)
                b_sum = budget - b_prod - X.shape[i] * j
                b_prod /= j

                temp_result = compute_core_shape_hosvd_ip_double_budget(sparse.zeros(tuple(Xtemp_shape)), sq_sing_vals, b_prod, b_sum)
                if temp_result.hosvd_prefix_sum + np.sum(unfolded_squared_singular_values[i][:(j+1)]) > best_singular_sum:
                    best_result = temp_result
                    besti = i
                    bestj = j
                    best_singular_sum = temp_result.hosvd_prefix_sum + np.sum(unfolded_squared_singular_values[i][:(j+1)])

    for k in range(int(np.floor(np.log(budget)/np.log(1+eps)))):
        for i in range(N):
            Xtemp_shape = np.zeros(N-1,dtype=np.int)
            counter = 0
            sq_sing_vals = [[] for i in range(len(Xtemp_shape))]
            for l in range(N):
                if i == l:
                    continue
                Xtemp_shape[counter] = X.shape[l]
                sq_sing_vals[counter] = unfolded_squared_singular_values[l]
                counter += 1
            for j in range(np.maximum(int(np.floor(np.ceil(1/eps) / (1+eps))),1,dtype=int,casting="unsafe"), X.shape[i] + 1):
                b_prod = (budget - np.power(1+eps, k)) / j
                b_sum = np.power(1+eps, k) - X.shape[i] * j

                temp_result = compute_core_shape_hosvd_ip_double_budget(sparse.zeros(tuple(Xtemp_shape)), sq_sing_vals, b_prod, b_sum)
                if temp_result.hosvd_prefix_sum + np.sum(unfolded_squared_singular_values[i][:(j+1)]) > best_singular_sum:
                    best_result = temp_result
                    besti = i
                    bestj = j
                    best_singular_sum = temp_result.hosvd_prefix_sum + np.sum(unfolded_squared_singular_values[i][:(j+1)])

    if besti == -1:
        best_core_shape = best_result.core_shape
    else:
        best_singular_sum += np.sum(unfolded_squared_singular_values[besti][:(bestj+1)])
        best_core_shape = np.zeros(N,dtype=np.int)
        best_core_shape[besti] = bestj
        if besti > 0:
            best_core_shape[:besti] = best_result.core_shape[:besti]
        if besti < N - 1:
            best_core_shape[(besti+1):] = best_result.core_shape[besti:]

    best_core_shape = tuple(best_core_shape)
    print(budget, '-->', best_singular_sum, best_core_shape)

    end_time = time.time()

    solve_result = CoreShapeSolveResult()
    solve_result.input_shape = X.shape
    solve_result.algorithm = 'hosvd-ip'
    solve_result.budget = budget
    solve_result.core_shape = best_core_shape
    solve_result.num_tucker_params = get_num_tucker_params(X, best_core_shape)
    solve_result.compression_ratio = solve_result.num_tucker_params / X.size

    solve_result.solve_time_seconds = end_time - start_time

    solve_result.hosvd_prefix_sum = best_singular_sum
    solve_result.hosvd_suffix_sum = 0.0
    for n in range(N):
        solve_result.hosvd_suffix_sum += sum(unfolded_squared_singular_values[n])
    solve_result.hosvd_suffix_sum -= best_singular_sum
    return solve_result

def compute_core_shape_hosvd_ip_double_budget(X, unfolded_squared_singular_values, budget_prod, budget_sum):
    """
    Solve an integer-linear program to find the best shape acccording to HOSVD
    """
    start_time = time.time()

    N = len(X.shape)
    
    col_num = 0;
    for i in range(N):
        col_num += X.shape[i]

    A = np.zeros((N+2, col_num))
    c = np.zeros(col_num)
    bu = np.zeros(N+2)
    bl = np.zeros(N+2)
    best_core_shape = np.zeros(N, dtype=np.int)

    col_count = 0
    bu[0] = np.log(budget_prod)
    bu[1] = budget_sum

    for i in range(N):
        prefix_sum = 0
        bu[2+i] = 1
        bl[2+i] = 1
        for j in range(X.shape[i]):
            c[col_count+j] = prefix_sum + unfolded_squared_singular_values[i][j]
            prefix_sum = c[col_count+j]
            A[0, col_count+j] = np.log(j+1)
            A[1, col_count+j] = X.shape[i] * (j+1)
            A[2+i, col_count+j] = 1

        col_count += X.shape[i]

    constraints = LinearConstraint(A, bl, bu)
    res = milp(c=-c, constraints=constraints, integrality=np.ones_like(c))

    if res.x is None:
        solve_result = CoreShapeSolveResult()
        solve_result.hosvd_prefix_sum = 0
        return solve_result

    best_singular_sum = 0
    col_count = 0
    for i in range(N):
        for j in range(X.shape[i]):
            if res.x[col_count+j] > 0.9:
                best_singular_sum += np.sum(unfolded_squared_singular_values[i][:(j+1)])
                best_core_shape[i] = int(j + 1)

        col_count += X.shape[i]

    print(budget_prod + budget_sum, '-->', best_singular_sum, tuple(best_core_shape))

    end_time = time.time()

    solve_result = CoreShapeSolveResult()
    solve_result.input_shape = X.shape
    solve_result.algorithm = 'hosvd-ip-double-budget'
    solve_result.budget = budget_prod + budget_sum
    solve_result.core_shape = best_core_shape
    solve_result.num_tucker_params = get_num_tucker_params(X, best_core_shape)
    solve_result.compression_ratio = solve_result.num_tucker_params / X.size

    solve_result.solve_time_seconds = end_time - start_time

    solve_result.hosvd_prefix_sum = best_singular_sum
    solve_result.hosvd_suffix_sum = 0.0
    for n in range(N):
        solve_result.hosvd_suffix_sum += sum(unfolded_squared_singular_values[n])
    solve_result.hosvd_suffix_sum -= best_singular_sum
    return solve_result


def sparse_unfold(X, n):
    coords = X.coords.T

    temp_coords = np.zeros((coords.shape[0], coords.shape[1] - 1))
    if n > 0:
        temp_coords[:, 0:n] = coords[:, 0:n]
    if n < len(X.shape) - 1:
        temp_coords[:, n:] = coords[:, (n+1):]

    rows = np.unique(temp_coords, axis=0)
    mat = np.zeros((rows.shape[0], X.shape[n]))

    for i in range(coords.shape[0]):
        row_ind = np.where(np.logical_and.reduce(np.equal(rows,temp_coords[i]),axis=1)==True)[0][0]
        mat[row_ind, coords[i, n]] = X.data[i]

    return mat

