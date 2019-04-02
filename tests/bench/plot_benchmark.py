#!/usr/bin/env python3
import os
import re
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools

# -------------------------------------------------------------------------------------
def main(result_file, node_to_plot, nrep_to_plot, matrix_size_to_plot, print_raw):
    print("Program parameters:", result_file, node_to_plot, nrep_to_plot, matrix_size_to_plot)

    # Read and parse file into a pandas Dataframe
    results = parse_file(result_file)

    # Print raw data
    if print_raw:
        print("Raw data:")
        print(np.unique(results.columns.values))
        print(results.to_string(index=False))

    # Plot results
    plot_benchmark(results, node_to_plot, nrep_to_plot)


# -------------------------------------------------------------------------------------
numnodes_descr = re.compile(
    r"""perf-H2O.n(?P<nnodes>\d+)(.*)?.\d+.o:  # file name
        \s+                                    # space
        numnodes\s+(?P<numnodes>\d+)           # number of nodes
    """, re.X)
nrep_descr = re.compile(
    r"""perf-H2O.n(?P<nnodes>\d+)(.*)?.\d+.o:  # file name
        \s+                                    # space
        nrep\s+(?P<nrep>\d+)                   # number of repetitions
    """, re.X)
matrix_size_descr = re.compile(
    r"""perf-H2O.n(?P<nnodes>\d+)?.\d+.o:  # file name
        \s+                                # space
        matrix\ sizes
        \ A\(\s+(?P<A_size_x>\d+)\ x\s+(?P<A_size_y>\d+)\), # A
        \ B\(\s+(?P<B_size_x>\d+)\ x\s+(?P<B_size_y>\d+)\)  # B
        \ and
        \ C\(\s+(?P<C_size_x>\d+)\ x\s+(?P<C_size_y>\d+)\)  # C
    """, re.VERBOSE)
time_descr = re.compile(
    r"""perf-H2O.n(?P<nnodes>\d+)?.\d+.o:  # file name
        \s+                                # space
        time\s+=\s+                        # time
        (?P<time_mean_sec>
            \d+.\d+
            E(?:\+|-)\d\d
        )\s+                               # mean
        (?P<time_std_sec>
            \d+.\d+
            E(?:\+|-)\d\d
        )\s+                               # std
        (?P<time_minmin_sec>
            \d+.\d+
            E(?:\+|-)\d\d
        )\s+                               # minmin
        (?P<time_maxmax_sec>
            \d+.\d+
            E(?:\+|-)\d\d
        )\s+                               # maxmax
    """, re.X)
perf_total_descr = re.compile(
    r"""perf-H2O.n(?P<nnodes>\d+)?.\d+.o:  # file name
        \s+                                # space
        perf\ total\s+=\s+                 # perf total
        (?P<perf_total_mean_flops>
            \d+.\d+
            E(?:\+|-)\d\d
        )\s+                               # mean
        (?P<perf_total_std_flops>
            \d+.\d+
            E(?:\+|-)\d\d
        )\s+                               # std
        (?P<perf_total_minmin_flops>
            \d+.\d+
            E(?:\+|-)\d\d
        )\s+                               # minmin
        (?P<perf_total_maxmax_flops>
            \d+.\d+
            E(?:\+|-)\d\d
        )\s+                               # maxmax
    """, re.X)
perf_per_node_descr = re.compile(
    r"""perf-H2O.n(?P<nnodes>\d+)?.\d+.o:  # file name
        \s+                                # space
        perf\ per\ node\s+=\s+             # perf per node
        (?P<perf_per_node_mean_flops>
            \d+.\d+
            E(?:\+|-)\d\d
        )\s+                               # mean
        (?P<perf_per_node_std_flops>
            \d+.\d+
            E(?:\+|-)\d\d
        )\s+                               # std
        (?P<perf_per_node_minmin_flops>
            \d+.\d+
            E(?:\+|-)\d\d
        )\s+                               # minmin
        (?P<perf_per_node_maxmax_flops>
            \d+.\d+
            E(?:\+|-)\d\d
        )\s+                               # maxmax
    """, re.X)
perf_per_thread_descr = re.compile(
    r"""perf-H2O.n(?P<nnodes>\d+)?.\d+.o:  # file name
        \s+                                # space
        perf\ per\ thread\s+=\s+           # perf per thread
        (?P<perf_per_thread_mean_flops>
            \d+.\d+
            E(?:\+|-)\d\d
        )\s+                               # mean
        (?P<perf_per_thread_std_flops>
            \d+.\d+
            E(?:\+|-)\d\d
        )\s+                               # std
        (?P<perf_per_thread_minmin_flops>
            \d+.\d+
            E(?:\+|-)\d\d
        )\s+                               # minmin
        (?P<perf_per_thread_maxmax_flops>
            \d+.\d+
            E(?:\+|-)\d\d
        )\s+                               # maxmax
    """, re.X)
benchmark_descrs = [nrep_descr, matrix_size_descr, time_descr,
                    perf_total_descr, perf_per_node_descr, perf_per_thread_descr]
def parse_file(file):
    """
    Typical use case: on daint in `~/dbcsr/tests/bench`: 
    Run `grep -P '(numnodes    |matrix sizes| time| perf )' perf-H2O.n*.*.o | tee results.txt`
    and use this as input file
    """
    with open(file) as f:
        file_content = f.readlines()

    results = list()  # of dictionaries
    benchmark_numbers = dict()
    for line in file_content:
        for benchmark_descr in benchmark_descrs:
            if benchmark_descr.match(line):
                benchmark_numbers.update(benchmark_descr.match(line).groupdict())

                if benchmark_descr == perf_per_thread_descr:  # i.e., the last one
                    results.append(benchmark_numbers)
                    benchmark_numbers = dict()

    ret = pd.DataFrame(results)
    for col in ret.columns:
        ret[col] = pd.to_numeric(ret[col])
    return ret


def plot_benchmark(results, node_to_plot, nrep_to_plot):

    # Plot benchmarks
    # Loop on problem sizes
    for nrep, mat_size in itertools.product(set(results['nrep'].values), set(results['A_size_x'].values)):
        # Loop on performance quantity to plot
        for perf_quantity in ['time_minmin_sec', 'perf_total_minmin_flops', 'perf_per_node_minmin_flops']:
            # Plot scaling over number of nodes
            to_plot = results[results['nrep'] == nrep]
            to_plot = to_plot[results['A_size_x'] == mat_size]
            plot_single_benchmark(to_plot, '(number of repetitions, matrix size)', (nrep, mat_size), 
                                  'nnodes', perf_quantity)

def plot_single_benchmark(df, fixed_quantity, fixed_quantity_value, col_x, col_y):
    df.plot(x=col_x, y=col_y, kind='scatter', title=fixed_quantity + " = " + str(fixed_quantity_value))
    plt.show()


# -------------------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""
        Plot pretty CP2K water benchmarks 
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--file', metavar="results_file.txt", default="results.txt", type=str, help="results file to parse")
    parser.add_argument('-n', '--nodes', metavar="NUM_NODES", default=0, type=int, help="Plot only this number of nodes")
    parser.add_argument('-r', '--nrep', metavar="NREP", default=0, type=int, help="Plot only this nrep")
    parser.add_argument('-s', '--size', metavar="MATRIX_SIZE", default=0, type=int, help="Plot only this matrix size")

    args = parser.parse_args()
    main(args.file, args.nodes, args.nrep, args.size, True) 
