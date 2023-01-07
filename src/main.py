from core_shape_solvers import *
from tensor_data_handler import TensorDataHandler
from tucker_decomposition_solver import *

import numpy as np
import tensorly
import scipy
import sparse
import tensorly.decomposition
import matplotlib.pyplot as plt
import os

def create_pareto_curve():
    """
    Goal: Brute force a simple tensor to illustrate core shape problem (Pareto).
    """
    handler = TensorDataHandler()

    """
    # Budget range [400, 2000] is reasonable.
    input_shape = [100, 100, 100]
    core_shape = [4, 8, 16]
    random_seed = 123

    X = handler.generate_random_tucker(input_shape, core_shape, random_seed)
    cache_path = 'output/random-tucker'
    cache_path += '-input-' + '-'.join([str(_) for _ in input_shape])
    cache_path += '-rank-' + '-'.join([str(_) for _ in core_shape])
    cache_path += '-seed-' + str(random_seed)
    """

    # Budget range [4000, 16000] is good
    X = handler.load_hyperspectral()
    cache_path = 'output/hyperspectral'

    print('X.shape:', X.shape)
    print('X.size:', X.size)
    print('cache_path:', cache_path)

    BUDGET = 16000

    N = 8
    compression_ratios = []
    rres = []
    for r1 in range(1, N + 1):
        for r2 in range(1, N + 1):
            for r3 in range(1, N + 1):
                core_shape = [r1, r2, r3]
                print('solving:', core_shape)
                solve_result = compute_tucker_decomposition(X, core_shape, cache_path)
                if solve_result.num_tucker_params > BUDGET:
                    continue
                print(core_shape, solve_result.rre, solve_result.num_tucker_params,
                      solve_result.compression_ratio)
                compression_ratios.append(solve_result.compression_ratio)
                rres.append(solve_result.rre)

    alg_ratios = []
    alg_rres = []
    for budget in range(4000, BUDGET + 1, 100):
        print(budget)
        greedy_core_shape = compute_core_shape(X, budget, 'greedy-approx')
        print(greedy_core_shape)
        solve_result = compute_tucker_decomposition(X, greedy_core_shape, cache_path)
        print(solve_result.rre)
        alg_ratios.append(budget / X.size)
        alg_rres.append(solve_result.rre)

    plt.scatter(compression_ratios, rres, marker='.', label='all-core-shapes')
    plt.scatter(alg_ratios, alg_rres, marker='+', label='greedy-approx')
    plt.legend()
    plt.show()

def compare_core_shape_solvers():
    handler = TensorDataHandler()

    # Traffic data
    #budgets = list(range(4000, 20000 + 1, 1000))
    #X = handler.load_traffic()
    #output_path = handler.output_path

    # Good range: [4000, 100000]
    #budgets = list(range(4000, 64000 + 1, 2000))
    #X = handler.load_cardiac_mri()
    #output_path = handler.output_path

    #budgets = list(range(4000, 64000 + 1, 2000))
    #budgets = list(range(4000, 20000 + 1, 1000))
    #X = handler.load_hyperspectral()
    #output_path = handler.output_path

    # Good range: [8000, 64k, by 1000]
    #budgets = list(range(8000, 100000 + 1, 2000))
    #X = handler.load_coil_100()
    #output_path = handler.output_path

    #budgets = list(range(1000, 10000 + 1, 1000))
    #X = handler.generate_random_tucker([100, 100, 100], [16, 32, 64], 1234)
    #X = handler.generate_random_tucker([10, 10, 10, 10], [2, 3, 4, 5], 1234)
    #output_path = handler.output_path

    budgets = list(range(1000, 10000 + 1, 1000))
    X = handler.make_random_tensor((100, 100, 100), seed=0)
    output_path = handler.output_path

    # Note: Not very interesting. Too easy to solve.
    #budgets = list(range(3000, 100000, 1000))
    #X = handler.load_image('data/images/cat.jpg', resize_shape=[1000, 1000])
    #output_path = handler.output_path

    print('X.shape:', X.shape)
    print('X.size:', X.size)
    print('output_path:', output_path)

    algorithms = ['hosvd-greedy', 'hosvd-bang-for-buck', 'hosvd-brute-force', 'rre-greedy']
    #algorithms = ['hosvd-greedy', 'hosvd-bang-for-buck', 'hosvd-brute-force']
    markers = ['x', '+', '.', '.']
    linestyles = ['solid', 'solid', 'solid', 'dashed']
    core_shape_solve_results = []
    tucker_decomposition_solve_results = []
    for algorithm in algorithms:
        print('algorithm:', algorithm)
        core_shape_solve_results.append(compute_core_shapes(X, budgets, algorithm, output_path))
        # Run Tucker decomposition for each (budget, core_shape) found by this algorithm.
        tmp = []
        for core_shape_result in core_shape_solve_results[-1]:
            core_shape = core_shape_result.core_shape
            print(' - core_shape:', core_shape)
            tmp.append(compute_tucker_decomposition(X, core_shape, output_path))
        tucker_decomposition_solve_results.append(tmp)

    # --------------
    figure_directory = output_path + 'figures/'
    if not os.path.exists(figure_directory):
        os.makedirs(figure_directory)

    # Plot prefix sums of two methods (squared singular values)
    for i in range(len(algorithms)):
        prefix_sums = np.array([result.hosvd_prefix_sum for result in core_shape_solve_results[i]])
        plt.plot(budgets, prefix_sums, marker=markers[i], label=algorithms[i], linestyle=linestyles[i])
    plt.grid()
    plt.legend()
    plt.title(output_path)
    plt.xlabel('budget')
    plt.ylabel('HOSVD packing')
    plt.savefig(figure_directory + 'hosvd-packing.png', transparent=True, bbox_inches='tight', dpi=256)
    plt.show()

    # Plot prefix sums of two methods (squared singular values)
    for i in range(len(algorithms)):
        rres = np.array([result.rre for result in tucker_decomposition_solve_results[i]])
        plt.plot(budgets, rres, marker=markers[i], label=algorithms[i], linestyle=linestyles[i])
    plt.grid()
    plt.legend()
    plt.title(output_path)
    plt.xlabel('budget')
    plt.ylabel('RRE')
    plt.savefig(figure_directory + 'rre.png', transparent=True, bbox_inches='tight', dpi=256)
    plt.show()

    # Plot running times (core shape solve)
    for i in range(len(algorithms)):
        solve_times = np.array([(result.solve_time_seconds - result.hosvd_time_seconds) for result in core_shape_solve_results[i]])
        plt.plot(budgets, solve_times, marker=markers[i], label=algorithms[i], linestyle=linestyles[i])
    plt.grid()
    plt.legend()
    plt.title(output_path)
    plt.xlabel('budget')
    plt.ylabel('core shape solve time (s)')
    plt.savefig(figure_directory + 'core-shape-solve-time.png', transparent=True, bbox_inches='tight', dpi=256)
    plt.show()

def main():
    #create_pareto_curve()
    compare_core_shape_solvers()

main()

