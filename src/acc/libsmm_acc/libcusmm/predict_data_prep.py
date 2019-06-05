#!/usr/bin/env python3
# -*- coding: utf-8 -*-
####################################################################################################
# Copyright (C) by the DBCSR developers group - All rights reserved                                #
# This file is part of the DBCSR library.                                                          #
#                                                                                                  #
# For information on the license, see the LICENSE file.                                            #
# For further information please visit https://dbcsr.cp2k.org                                      #
# SPDX-License-Identifier: GPL-2.0+                                                                #
####################################################################################################


import os, re, json, argparse
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from joblib import Parallel, delayed
from kernels.cusmm_predict import kernel_algorithm, mnk_pattern


# ===============================================================================
def process_chunk(data_chunk, algorithm, gpu_properties, autotuning_properties):
    """

    """
    # Add "mnk" column
    data_chunk["mnk"] = (
        data_chunk["m"].astype(str) + "x" + data_chunk["n"].astype(str) + "x" + data_chunk["k"].astype(str)
    )
    # Get mnks
    mnks = data_chunk["mnk"].unique()
    nmnks = len(mnks)
    baseline_performances = dict()
    max_performances = dict()
    for i, mnk in enumerate(mnks):

        data_mnk = data_chunk[data_chunk["mnk"] == mnk]
        data_mnk_len = len(data_mnk)
        print("Processing mnk = {}, nrows = {}, ({}/{})".format(mnk, data_mnk_len, i, nmnks))
        m, n, k = mnk_pattern.match(mnk).groups()
        m, n, k = int(m), int(n), int(k)

        # Get baseline configuration for this algorithm & this mnk:
        baseline_pars = kernel_algorithm[algorithm].baseline(m, n, k, gpu_properties, autotuning_properties)

        # Get performance of baseline parameters for this algorithm & this mnk:
        if np.isnan(baseline_pars["tile_m"]):
            idx_baseline = data_mnk[
                (data_mnk.m == baseline_pars["m"])
                & (data_mnk.n == baseline_pars["n"])
                & (data_mnk.k == baseline_pars["k"])
                & (data_mnk.threads == baseline_pars["threads"])
                & (data_mnk.grouping == baseline_pars["grouping"])
                & (data_mnk.minblocks == baseline_pars["minblocks"])
            ].index.tolist()
        elif np.isnan(baseline_pars["w"]):
            idx_baseline = data_mnk[
                (data_mnk.m == baseline_pars["m"])
                & (data_mnk.n == baseline_pars["n"])
                & (data_mnk.k == baseline_pars["k"])
                & (data_mnk.threads == baseline_pars["threads"])
                & (data_mnk.grouping == baseline_pars["grouping"])
                & (data_mnk.minblocks == baseline_pars["minblocks"])
                & (data_mnk.tile_m == baseline_pars["tile_m"])
                & (data_mnk.tile_n == baseline_pars["tile_n"])
            ].index.tolist()
        else:
            idx_baseline = data_mnk[
                (data_mnk.m == baseline_pars["m"])
                & (data_mnk.n == baseline_pars["n"])
                & (data_mnk.k == baseline_pars["k"])
                & (data_mnk.threads == baseline_pars["threads"])
                & (data_mnk.minblocks == baseline_pars["minblocks"])
                & (data_mnk.tile_m == baseline_pars["tile_m"])
                & (data_mnk.tile_n == baseline_pars["tile_n"])
                & (data_mnk.w == baseline_pars["w"])
                & (data_mnk.v == baseline_pars["v"])
            ].index.tolist()

        if len(idx_baseline) < 1:
            baseline_perf = 0
        else:
            idx_baseline = idx_baseline[0]
            baseline_perf = data_mnk["perf (Gflop/s)"][idx_baseline]

        baseline_performances[mnk] = baseline_perf

        # Get max performance for this algorithm & this mnk
        max_perf = data_mnk["perf (Gflop/s)"].max()
        max_performances[mnk] = max_perf

    return baseline_performances, max_performances


# ===============================================================================
data_types_raw = {'m': 'int64', 'n': 'int64', 'k': 'int64', 'tile_m': 'int64', 'tile_n': 'int64', 'threads': 'int64', 'grouping': 'int64', 'minblocks': 'int64', 'perf (Gflop/s)': 'float64'}
data_types_derived = {'perf_squared': 'float64', 'perf_scaled': 'float64', 'perf_scaled_by_algo': 'float64', 'mxnxk': 'int64', 'size_a': 'int64', 'size_b': 'int64', 'size_c': 'int64', 'sm_desired': 'int64', 'grouping': 'int64', 'nblks': 'int64', 'nthreads': 'int64', 'ru_tinysmallmed_unroll_factor_a': 'int64', 'ru_tinysmallmed_unroll_factor_a_total': 'int64', 'ru_tinysmallmed_unroll_factor_b': 'int64', 'ru_tinysmallmed_unroll_factor_b_total': 'int64', 'ru_tinysmallmed_unroll_factor_c_total': 'int64', 'ru_smallmed_unroll_factor_c': 'int64', 'ru_smallmed_loop_matmul': 'int64', 'ru_smallmed_max_parallel_work': 'int64', 'ru_smallmed_smem_per_block': 'int64', 'ru_smallmed_regs_per_thread': 'int64', 'ru_smallmedlarge_cmax': 'int64', 'ru_smallmedlarge_rmax': 'int64', 'ru_smallmedlarge_T': 'int64', 'ru_smallmedlarge_min_threads': 'int64', 'Koth_small_Nmem': 'int64', 'Koth_small_perf_K': 'float64'}


def write_to_parquet(data_path, algorithm):
    """
    Compress CSV files to parquet
    """
    # Check whether the files corresponding to this algorithm have been compressed to parquet already
    parquet_file = os.path.join(data_path, "training_data_" + algorithm + ".parquet")
    parquet_file_done = os.path.join(data_path, "training_data_" + algorithm + ".parquet.done")
    if os.path.exists(parquet_file_done):
        print("Found {:40}, skipping".format(parquet_file_done))

    else:

        # [RAW] Read CSV files into Pandas dataframes
        data_file_raw = os.path.join(data_path, "raw_training_data_" + algorithm + ".csv")
        print('\nRead raw data from: {}'.format(data_file_raw))
        data_raw = pd.read_csv(data_file_raw, dtype=data_types_raw)
        raw_data_nrows = len(data_raw)
        print("Raw data head:\n", data_raw.head())

        # [DERIVED] Read CSV files into Pandas dataframes
        data_file_derived = os.path.join(data_path, "training_data_" + algorithm + ".csv")
        print('\nRead derived data from: {}'.format(data_file_derived))
        data_derived = pd.read_csv(data_file_derived, dtype=data_types_derived)
        derived_data_nrows = len(data_derived)
        print("Derived data head:\n", data_derived.head())

        # Merge raw/derived data together
        data = pd.merge(data_raw, data_derived, left_index=True, right_index=True)

        # Add "mnk" column
        data["mnk"] = (
            data["m"].astype(str) + "x" + data["n"].astype(str) + "x" + data["k"].astype(str)
        )

        # If there are unique-valued columns, drop these since they do not contribute to any prediction
        for col in data.columns.values.tolist():
            if len(data[col].unique())==1:
                data = data.drop(col, axis=1)

        # Print info on merged dataset
        print("\nMerged data head:", data.head())
        data_nrows = len(data)
        nrows_message = """
Data        : {:15,},
Raw data    : {:15,},
Derived data: {:15,}""".format(data_nrows, raw_data_nrows, derived_data_nrows)
        assert data_nrows == raw_data_nrows, "Mismatch in number of rows\n" + nrows_message
        assert data_nrows == derived_data_nrows, "Mismatch in number of rows\n" + nrows_message
        print(nrows_message)

        # Compress files to Parquet
        print("Compress and write to {}".format(parquet_file))
        data.to_parquet(parquet_file, engine='fastparquet', compression='snappy')
        open(parquet_file_done, 'w').close() # touch a file to mark that parquet is done


# ===============================================================================
def get_non_null(l):
    """
    Return .... 
    """
    for e in l:
        if e > 0: return e
    return 0


def get_max(l):
    """
    Return the largest element of a list of numbers
    """
    return np.array(l).max()


def list_of_dics_to_dic_of_lists(list_of_dics):
    dic_of_lists = dict()
    for dic in list_of_dics:
        for k, v in dic.items():
            if k not in dic_of_lists.keys():
                dic_of_lists[k] = list()
            dic_of_lists[k].append(v)
    return dic_of_lists


def write_baseline_and_max_records_per_algorithm(data_path, algorithm, arch, n_jobs, chunk_size):
    """

    """
    # Read GPU properties and autotuning properties
    with open("kernels/gpu_properties.json") as f:
        gpu_properties = json.load(f)["sm_" + str(arch)]
    with open("kernels/autotuning_properties.json") as f:
        autotuning_properties = json.load(f)

    # Check whether record of baseline exists
    baseline_performances_per_algo_file = os.path.join(data_path, "baseline_performances_" + algorithm + ".json")
    max_performances_per_algo_file = os.path.join(data_path, "max_performances_" + algorithm + ".json")
    if os.path.exists(baseline_performances_per_algo_file) and os.path.exists(max_performances_per_algo_file):
        print("Found {:40}, skipping".format(baseline_performances_per_algo_file))
        print("Found {:40}, skipping".format(max_performances_per_algo_file))

    else:

        raw_pars_cols = kernel_algorithm[algorithm].launch_parameters
        if algorithm in ["largeDB1", "largeDB2"]:
            raw_pars_cols.remove("grouping")

        data_file_raw = os.path.join(data_path, "raw_training_data_" + algorithm + ".csv")
        baseline_and_maximums_performance_dictionnaries = Parallel(n_jobs=n_jobs, verbose=2)(delayed(
            process_chunk, check_pickle=True)(
                data_chunk, algorithm, gpu_properties, autotuning_properties,
            )
            for data_chunk in pd.read_csv(data_file_raw, chunksize=chunk_size)
        )

        baseline_performance_dictionnaries, maximums_performance_dictionnaries = zip(*baseline_and_maximums_performance_dictionnaries)
        baseline_performance_dictionnary = list_of_dics_to_dic_of_lists(baseline_performance_dictionnaries)
        maximums_performance_dictionnary = list_of_dics_to_dic_of_lists(maximums_performance_dictionnaries)

        # Print max performances
        max_performances = dict()
        for mnk, max_list in maximums_performance_dictionnary.items():
            perf = get_max(max_list)
            max_performances[mnk] = perf
        with open(max_performances_per_algo_file, "w") as f:
            	json.dump(max_performances, f, indent='\t')
        print("\nWrote maximum performances to:\n", max_performances_per_algo_file)

        # Print baseline
        baseline_performances = dict()
        for mnk, base_list in baseline_performance_dictionnary.items():
            perf = get_non_null(base_list)
            baseline_performances[mnk] = perf
        with open(baseline_performances_per_algo_file, "w") as f:
            json.dump(baseline_performances, f, indent='\t')
        print("\nWrote baseline performances to:\n", baseline_performances_per_algo_file,)


# ===============================================================================
def write_baseline_record(data_path, algorithms):
    baseline_performances_by_algo_file = os.path.join(data_path, "baseline_performances_by_algo.json")
    if os.path.exists(baseline_performances_by_algo_file):
        print("Found {:40}, skipping".format(baseline_performances_by_algo_file))

    else:
        # Get baseline performances by algorithm
        baseline_performances_by_algo = dict()
        for algorithm in algorithms:
            # Read baseline parameters
            baseline_performances_per_algo_file = os.path.join(data_path, "baseline_performances_" + algorithm + ".json")
            with open(baseline_performances_per_algo_file, "r") as f:
                baseline_algorithm = json.load(f)
            # Add to dictionary
            baseline_performances_by_algo[algorithm] = baseline_algorithm

        # Write to file
        with open(baseline_performances_by_algo_file, "w") as f:
            json.dump(baseline_performances_by_algo, f, indent='\t')
        print("\nWrote baseline performances to:\n", baseline_performances_by_algo_file,)


def write_max_by_algo_record(data_path, algorithms):
    max_performances_by_algo_file = os.path.join(data_path, "max_performances_by_algo.json")
    if os.path.exists(max_performances_by_algo_file):
        print("Found {:40}, skipping".format(max_performances_by_algo_file))

    else:
        # Get max performances by algorithm
        max_performances_by_algo = dict()
        for algorithm in algorithms:
            # Read max parameters
            max_performances_per_algo_file = os.path.join(data_path, "max_performances_" + algorithm + ".json")
            with open(max_performances_per_algo_file, "r") as f:
                max_algorithm = json.load(f)
            # Add to dictionary
            max_performances_by_algo[algorithm] = max_algorithm

        # Write to file
        with open(max_performances_by_algo_file, "w") as f:
            json.dump(max_performances_by_algo, f, indent='\t')
        print("\nWrote max performances by algorithm to:\n", max_performances_by_algo_file,)


def dic_of_dics_to_dic_of_lists(dic_of_dics):
    dic_of_lists = dict()
    for _, dic in dic_of_dics.items():
        for k, v in dic.items():
            if k not in dic_of_lists.keys():
                dic_of_lists[k] = list()
            dic_of_lists[k].append(v)
    return dic_of_lists


def write_max_record(data_path, algorithms):
    max_performances_file = os.path.join(data_path, "max_performances.json")
    if os.path.exists(max_performances_file):
        print("Found {:40}, skipping".format(max_performances_file))

    else:
        # Get max performances
        max_performances_by_algo = dict()
        for algorithm in algorithms:
            # Read max parameters
            max_performances_per_algo_file = os.path.join(data_path, "max_performances_" + algorithm + ".json")
            with open(max_performances_per_algo_file, "r") as f:
                max_algorithm = json.load(f)
            # Add to dictionary
            max_performances_by_algo[algorithm] = max_algorithm

        # Reduce along max
        max_performances_list = dic_of_dics_to_dic_of_lists(max_performances_by_algo)
        max_performances = dict()
        for mnk, max_list in max_performances_list.items():
            max_performances[mnk] = get_max(max_list)

        # Write to file
        with open(max_performances_file, "w") as f:
            json.dump(max_performances, f, indent='\t')
        print("\nWrote max performances to:\n", max_performances_file,)


# ===============================================================================
def main(data_path, algorithms_to_prep, arch, n_jobs, chunk_size):
    """
    This script is part of the workflow for predictive modelling of optimal libcusmm parameters.
    For more details, see predict.md
    """
    for algorithm in algorithms_to_prep:
        write_to_parquet(data_path, algorithm)
        write_baseline_and_max_records_per_algorithm(data_path, algorithm, arch, n_jobs, chunk_size)

    write_baseline_record(data_path, algorithms_to_prep)
    write_max_by_algo_record(data_path, algorithms_to_prep)
    write_max_record(data_path, algorithms_to_prep)


# ===============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Prepare the data collected with autotuning for training, 
        - Convert from CSV -> Parquet
        - 
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--data_path",
        metavar="FOLDER",
        type=str,
        default=".",
        help="Path to the data to be converted to parquet.",
    )
    parser.add_argument(
        "-l",
        "--algorithm",
        metavar="ALGORITHM",
        default="",
        help="Algorithms to prepare",
    )
    parser.add_argument(
        "-a",
        "--arch",
        metavar="ARCHITECTURE_NUMBER",
        type=int,
        default="60",
        help="CUDA architecture number. Options: 35, 37, 60, 70",
            )
    parser.add_argument(
        "-j",
        "--njobs",
        default=-1,
        metavar="NUMBER",
        type=int,
        help="Number of parallel jobs that Joblib will launch",
    )
    parser.add_argument(
        "-c",
        "--chunk_size",
        type=int,
        default=20000,
        help="Chunk size for dispatching joblib jobs. If memory errors are experienced, reduce this number",
    )

    args = parser.parse_args()
    algorithms_to_prep = kernel_algorithm.keys() if args.algorithm == "" else [args.algorithm]
    main(args.data_path, algorithms_to_prep, args.arch, args.njobs, args.chunk_size)

