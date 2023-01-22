from core_shape_solvers import *
from tensor_data_handler import TensorDataHandler
from tucker_decomposition_solver import *

import numpy as np
import tensorly
import tensorly.decomposition
import matplotlib.pyplot as plt
import os

import seaborn as sns
#sns.set_theme(style="white")

#plt.style.use('fivethirtyeight')

def main():
    algorithms = ['hosvd-greedy', 'hosvd-bang-for-buck', 'hosvd-brute-force', 'hosvd-ip', 'rre-greedy']

    #markers = ['x', '+', '.', '.', None]
    #linestyles = ['solid', 'solid', 'solid', 'dashed', 'dashed']

    markers = ['.', 'x', '.', '+', None]
    linestyles = ['solid', 'solid', 'solid', 'dashed', 'dashed']

    fig, axs = plt.subplots(3, 4, figsize=(16, 7), sharex=True, constrained_layout=True)

    # --------------------------------------------------------------------------
    # Tensor 0: Cardiac MRI
    # --------------------------------------------------------------------------
    handler = TensorDataHandler()
    budgets = list(range(4000, 100000 + 1, 2000))
    X = handler.load_cardiac_mri()
    output_path = handler.output_path

    core_shape_solve_results = []
    tucker_decomposition_solve_results = []
    for algorithm in algorithms:
        print('algorithm:', algorithm)
        core_shape_solve_results.append(compute_core_shapes(X, budgets, algorithm, output_path))
        # Run Tucker decomposition for each (budget, core_shape) found by this algorithm.
        tmp = []
        for core_shape_result in core_shape_solve_results[-1]:
            core_shape = core_shape_result.core_shape
            print(' - shape:', core_shape)
            tmp.append(compute_tucker_decomposition(X, core_shape, output_path))
        tucker_decomposition_solve_results.append(tmp)

    # --------------
    # HOSVD packing
    # --------------
    for i in range(len(algorithms)):
        prefix_sums = np.array([result.hosvd_prefix_sum for result in core_shape_solve_results[i]])
        axs[0, 0].plot(budgets, prefix_sums, marker=markers[i], label=algorithms[i], linestyle=linestyles[i])
    axs[0, 0].grid()
    axs[0, 0].set_title('Cardiac MRI')
    axs[0, 0].set_ylabel('HOSVD packing')
    
    # --------------
    # RRE
    # --------------
    for i in range(len(algorithms)):
        rres = np.array([result.rre for result in tucker_decomposition_solve_results[i]])
        axs[1, 0].plot(budgets, rres, marker=markers[i], label=algorithms[i], linestyle=linestyles[i])
    axs[1, 0].grid()
    axs[1, 0].set_ylabel('RRE')

    # --------------
    # Running time
    # --------------
    for i in range(len(algorithms)):
        solve_times = np.array([result.solve_time_seconds for result in core_shape_solve_results[i]])
        axs[2, 0].plot(budgets, solve_times, marker=markers[i], label=algorithms[i], linestyle=linestyles[i])
    axs[2, 0].grid()
    axs[2, 0].set_ylabel('core shape solve time (s)')
    axs[2, 0].set_ylim(10, 10**5)
    axs[2, 0].set_yscale('log')
    axs[2, 0].set_xlabel('budget')

    # --------------------------------------------------------------------------
    # Tensor 1: Hyperspectral
    # --------------------------------------------------------------------------
    handler = TensorDataHandler()
    budgets = list(range(4000, 100000 + 1, 2000))
    X = handler.load_hyperspectral()
    output_path = handler.output_path

    core_shape_solve_results = []
    tucker_decomposition_solve_results = []
    for algorithm in algorithms:
        print('algorithm:', algorithm)
        core_shape_solve_results.append(compute_core_shapes(X, budgets, algorithm, output_path))
        # Run Tucker decomposition for each (budget, core_shape) found by this algorithm.
        tmp = []
        for core_shape_result in core_shape_solve_results[-1]:
            core_shape = core_shape_result.core_shape
            tmp.append(compute_tucker_decomposition(X, core_shape, output_path))
        tucker_decomposition_solve_results.append(tmp)

    # --------------
    # HOSVD packing
    # --------------
    for i in range(len(algorithms)):
        prefix_sums = np.array([result.hosvd_prefix_sum for result in core_shape_solve_results[i]])
        axs[0, 1].plot(budgets, prefix_sums, marker=markers[i], label=algorithms[i], linestyle=linestyles[i])
    axs[0, 1].grid()
    axs[0, 1].set_title('Hyperspectral')
    
    # --------------
    # RRE
    # --------------
    for i in range(len(algorithms)):
        rres = np.array([result.rre for result in tucker_decomposition_solve_results[i]])
        axs[1, 1].plot(budgets, rres, marker=markers[i], label=algorithms[i], linestyle=linestyles[i])
    axs[1, 1].grid()

    # --------------
    # Running time
    # --------------
    for i in range(len(algorithms)):
        solve_times = np.array([result.solve_time_seconds for result in core_shape_solve_results[i]])
        axs[2, 1].plot(budgets, solve_times, marker=markers[i], label=algorithms[i], linestyle=linestyles[i])
    axs[2, 1].grid()
    axs[2, 1].set_yscale('log')
    axs[2, 1].set_ylim(10, 10**5)
    axs[2, 1].set_xlabel('budget')

    # --------------------------------------------------------------------------
    # Tensor 2: Traffic
    # --------------------------------------------------------------------------
    handler = TensorDataHandler()
    budgets = list(range(4000, 100000 + 1, 2000))
    X = handler.load_traffic()
    output_path = handler.output_path

    core_shape_solve_results = []
    tucker_decomposition_solve_results = []
    for algorithm in algorithms:
        print('algorithm:', algorithm)
        core_shape_solve_results.append(compute_core_shapes(X, budgets, algorithm, output_path))
        # Run Tucker decomposition for each (budget, core_shape) found by this algorithm.
        tmp = []
        for core_shape_result in core_shape_solve_results[-1]:
            core_shape = core_shape_result.core_shape
            tmp.append(compute_tucker_decomposition(X, core_shape, output_path))
        tucker_decomposition_solve_results.append(tmp)

    # --------------
    # HOSVD packing
    # --------------
    for i in range(len(algorithms)):
        prefix_sums = np.array([result.hosvd_prefix_sum for result in core_shape_solve_results[i]])
        axs[0, 2].plot(budgets, prefix_sums, marker=markers[i], label=algorithms[i], linestyle=linestyles[i])
    axs[0, 2].grid()
    axs[0, 2].set_title('VicRoads')
    
    # --------------
    # RRE
    # --------------
    for i in range(len(algorithms)):
        rres = np.array([result.rre for result in tucker_decomposition_solve_results[i]])
        axs[1, 2].plot(budgets, rres, marker=markers[i], label=algorithms[i], linestyle=linestyles[i])
    axs[1, 2].grid()

    # --------------
    # Running time
    # --------------
    for i in range(len(algorithms)):
        solve_times = np.array([result.solve_time_seconds for result in core_shape_solve_results[i]])
        axs[2, 2].plot(budgets, solve_times, marker=markers[i], label=algorithms[i], linestyle=linestyles[i])
    axs[2, 2].grid()
    axs[2, 2].set_ylim(10, 10**5)
    axs[2, 2].set_yscale('log')
    axs[2, 2].set_xlabel('budget')

    # --------------------------------------------------------------------------
    # Tensor 3: COIL-100
    # --------------------------------------------------------------------------
    handler = TensorDataHandler()
    budgets = list(range(8000, 100000 + 1, 2000))
    X = handler.load_coil_100()
    output_path = handler.output_path

    core_shape_solve_results = []
    tucker_decomposition_solve_results = []
    for algorithm in algorithms:
        print('algorithm:', algorithm)
        core_shape_solve_results.append(compute_core_shapes(X, budgets, algorithm, output_path))
        # Run Tucker decomposition for each (budget, core_shape) found by this algorithm.
        tmp = []
        for core_shape_result in core_shape_solve_results[-1]:
            core_shape = core_shape_result.core_shape
            print(' -', core_shape)
            tmp.append(compute_tucker_decomposition(X, core_shape, output_path))
        tucker_decomposition_solve_results.append(tmp)

    # --------------
    # HOSVD packing
    # --------------
    for i in range(len(algorithms)):
        prefix_sums = np.array([result.hosvd_prefix_sum for result in core_shape_solve_results[i]])
        axs[0, 3].plot(budgets, prefix_sums, marker=markers[i], label=algorithms[i], linestyle=linestyles[i])
    axs[0, 3].grid()
    axs[0, 3].set_title('COIL-100')
    
    # --------------
    # RRE
    # --------------
    for i in range(len(algorithms)):
        rres = np.array([result.rre for result in tucker_decomposition_solve_results[i]])
        axs[1, 3].plot(budgets, rres, marker=markers[i], label=algorithms[i], linestyle=linestyles[i])
    axs[1, 3].grid()

    # --------------
    # Running time
    # --------------
    for i in range(len(algorithms)):
        solve_times = np.array([result.solve_time_seconds for result in core_shape_solve_results[i]])
        axs[2, 3].plot(budgets, solve_times, marker=markers[i], label=algorithms[i], linestyle=linestyles[i])
    axs[2, 3].grid()
    axs[2, 3].set_ylim(10, 10**5)
    axs[2, 3].set_yscale('log')
    axs[2, 3].set_xlabel('budget')

    fig.legend(labels=algorithms, loc='lower center', ncol=5, fancybox=False, shadow=False, bbox_to_anchor=(0.515, -0.05))

    plt.savefig('plots/compare-core-shape-solvers.png', transparent=True, bbox_inches='tight', dpi=256)
    plt.show()

main()

