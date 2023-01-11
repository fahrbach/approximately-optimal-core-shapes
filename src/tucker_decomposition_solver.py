from tucker_decomposition_utils import get_num_tucker_params

import dataclasses
import datetime
import numpy as np
import os
import tensorly
import tensorly.decomposition
import time
import sparse
from tensorly.contrib.sparse.decomposition import tucker as sparse_tucker

@dataclasses.dataclass
class TuckerDecompositionSolveResult:
    date: str = datetime.datetime.now().isoformat()

    input_shape: tuple = None
    core_shape: tuple = None
    
    num_tucker_params: int = None
    compression_ratio: float = None

    loss: float = None
    rre: float = None
    solve_time_seconds: float = None

    # Tensorly Tucker decomposition params
    init: str = None
    n_iter_max: int = None
    tol: float = None
    random_state: int = None

def write_dataclass_to_file(dataclass, filename):
    """
    Writes `dataclass` (key, values) to `filename`.
    """
    with open(filename, 'w') as f:
        for field in dataclasses.fields(dataclass):
            f.write(str(field.name) + ' ')
            f.write(str(getattr(dataclass, field.name)) + '\n')

def read_tucker_decomposition_solve_result_from_file(filename):
    """
    Reads `filename` and constructs the `TuckerDecompositionSolveResult` data.
    """
    assert os.path.exists(filename)
    solve_result = TuckerDecompositionSolveResult()
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
            if key not in ['date', 'init']:
                value = eval(value_str)
            setattr(solve_result, key, value)
    return solve_result

def compute_tucker_decomposition(X, core_shape, output_path, trial=0):
    """
    Computes Tucker decomposition of `X` with `core_shape`, and caches result.

    Args:
        X: Arbitrary tensor of type `np.ndarray`.
        core_shape: List of integers that is the same dimension as `X`.
        output_path: Path where Tucker decompositions for `X` are cached.
    """

    assert output_path[-1] == '/'
    output_path += 'tucker-decomposition-solve-results/'

    filename = output_path
    filename += 'rank-' + '-'.join([str(_) for _ in core_shape]) + '_'
    filename += 'trial-' + str(trial)
    filename += '.txt'

    if os.path.exists(filename):
        solve_result = read_tucker_decomposition_solve_result_from_file(filename)
        return solve_result

    solve_result = TuckerDecompositionSolveResult()
    solve_result.input_shape = X.shape
    solve_result.core_shape = tuple(core_shape)

    solve_result.num_tucker_params = get_num_tucker_params(X, core_shape)
    solve_result.compression_ratio = solve_result.num_tucker_params / X.size

    solve_result.init = 'random'
    solve_result.n_iter_max = 20
    solve_result.tol = 1e-20
    solve_result.random_state = trial
    start_time = time.time()
    if isinstance(X, sparse.COO):
        core, factors = sparse_tucker(X, rank=core_shape,
                                      init=solve_result.init,
                                      n_iter_max=solve_result.n_iter_max,
                                      tol=solve_result.tol,
                                      random_state=solve_result.random_state)
    else:
        core, factors = tensorly.decomposition.tucker(X, rank=core_shape,
                                                      init=solve_result.init,
                                                      n_iter_max=solve_result.n_iter_max,
                                                      tol=solve_result.tol,
                                                      random_state=solve_result.random_state)
    end_time = time.time()
    X_hat = tensorly.tucker_to_tensor((core, factors))
    loss = np.linalg.norm(X - X_hat)**2
    rre = loss / np.linalg.norm(X)**2
    solve_result.loss = loss
    solve_result.rre = rre
    solve_result.solve_time_seconds = end_time - start_time

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    write_dataclass_to_file(solve_result, filename)
    return solve_result
