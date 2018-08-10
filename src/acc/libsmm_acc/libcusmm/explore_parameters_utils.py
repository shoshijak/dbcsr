import os
import re
import pickle
import sys
import numpy as np
import pandas as pd  # cheat sheet: https://pandas.pydata.org/pandas-docs/stable/10min.html


def ceiling(x, step):
    return np.where(x % step == 0, x, x + step - x % step)


def flooring(x, step):
    return np.where(x % step == 0, x, x - x % step)


def ceil_division(a, b):
    return (a + b - 1) // b


def quick_inspect(df):
    """
    Print out basic information on a pandas dataframe in order to quickly inspect it
    """
    print('\n---------------------------')
    print('---- Columns:')
    print('---------------------------\n')
    print(df.columns)
    print('\n---------------------------')
    print('---- Head:')
    print('---------------------------\n')
    print(df.head())
    print('\n---------------------------')
    print('---- Tail:')
    print('---------------------------\n')
    print(df.tail())
    print('\n---------------------------')
    print('---- Describe:')
    print('---------------------------\n')
    if df.shape[0] > 5:
        print(df.describe())


# Maximum number of bytes to pickle in one chunk
max_bytes = 2**31 - 1


def safe_pickle(data, file):
    """
    Pickle big files safely, processing them in chunks
    :param data: data to be pickled
    :param file: file to pickle it into
    """
    pickle_out = pickle.dumps(data)
    n_bytes = sys.getsizeof(pickle_out)
    with open(file, 'wb') as f:
        count = 0
        for i in range(0, n_bytes, max_bytes):
            f.write(pickle_out[i:min(n_bytes, i + max_bytes)])
            count += 1


tiny_mangle = re.compile('_Z\d+cusmm_dnt_tiny[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]+PKdS3_Pd')
small_medium_mangle = re.compile('_Z\d+cusmm_dnt_(small|medium)[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]+PKdS3_Pd')
largeDB_mangle = re.compile('_Z\d+cusmm_dnt_largeDB(1|2)[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]+PKdS3_Pd')
def quick_demangle(mangled_name):
    """
    Basic function name demangling, examples:
        _Z14cusmm_dnt_tinyILi4ELi4ELi8ELi32ELi15ELi30EEvPKiiPKdS3_Pd -> void cusmm_dnt_tiny<4, 4, 8, 32, 15, 30>(int const*, int, double const*, double const*, double*)
        _Z15cusmm_dnt_smallILi16ELi16ELi16ELi2ELi2ELi64ELi29ELi8EEvPKiiPKdS3_Pd  -> void cusmm_dnt_small<16, 16, 16, 2, 2, 64, 29, 8>(int const*, int, double const*, double const*, double*)
        _Z16cusmm_dnt_mediumILi25ELi32ELi5ELi5ELi1ELi192ELi16ELi7EEvPKiiPKdS3_Pd -> void cusmm_dnt_medium<25, 32, 5, 5, 1, 192, 16, 7>(int const*, int, double const*, double const*, double*)
    :return: dictionary of algorithm and parameters
    """
    if 'tiny' in mangled_name:
        match = tiny_mangle.match(mangled_name)
        assert match is not None, "Cannot match name:\n" + mangled_name
        return {'algo': 'tiny',
                'm': int(match.group(1)), 'n': int(match.group(2)), 'k': int(match.group(3)),
                'tile_m': None, 'tile_n': None, 'w': None, 'v': None,
                'threads': int(match.group(4)), 'grouping': int(match.group(5)), 'minblocks': int(match.group(6))}
    elif 'small' in mangled_name or 'medium' in mangled_name:
        match = small_medium_mangle.match(mangled_name)
        assert match is not None, "Cannot match name:\n" + mangled_name
        return {'algo': match.group(1),
                'm': int(match.group(2)), 'n': int(match.group(3)), 'k': int(match.group(4)),
                'tile_m': int(match.group(5)), 'tile_n': int(match.group(6)),
                'w': None, 'v': None,
                'threads': int(match.group(7)), 'grouping': int(match.group(8)), 'minblocks': int(match.group(9))}
    elif 'largeDB' in mangled_name:
        match = largeDB_mangle.match(mangled_name)
        assert match is not None, "Cannot match name:\n" + mangled_name
        return {'algo': 'largeDB' + match.group(1),
                'm': int(match.group(2)), 'n': int(match.group(3)), 'k': int(match.group(4)),
                'tile_m': int(match.group(5)), 'tile_n': int(match.group(6)),
                'w': int(match.group(7)), 'v': int(match.group(8)),
                'threads': int(match.group(9)), 'grouping': int(match.group(10)), 'minblocks': int(match.group(11))}
    else:
        assert False, "Cannot find base name in:\n" + mangled_name


ptxas_intro = re.compile('ptxas info\s+: Function properties for (_Z\d+.*)$')
ptxas_values = re.compile('ptxas info\s+: Used (\d+) registers, (\d+) bytes smem, (\d+) bytes cmem\[0\]')
autotuning_line = re.compile(
    'OK Kernel_dnt_(\w+) m (\d+)\s+n (\d+)\s+k (\d+)\s+(?:tile_m (\d+)\s+tile_n (\d+)\s+(?:w (\d+)\s+v (\d+)\s+)?)?threads (\d+)\s+grouping (\d+)\s+minblocks (\d+)\s+GFlop/s (\d+\.?\d+)')


def read_kernel_autotuning(log_folder):
    """
    Given a folder of kernel autotuning, read in compilation information and autotuning information
    :param log_folder: folder of kernel autotuning
    :return: pandas Dataframe containing above information
    """
    pickle_file = os.path.join(log_folder, "data_dump")
    if os.path.exists(pickle_file):
        print("Loading data from", pickle_file)
        data = pickle.load(open(pickle_file, "rb"))
        return pd.DataFrame.from_dict(data)
    # Find relevant files
    in_log_folder_files = os.listdir(log_folder)
    slurm_file = ''
    log_files = list()
    for f in in_log_folder_files:
        if f[:5] == 'slurm': slurm_file = f
        if f[-4:] == '.log': log_files.append(f)
    print('Found slurm file:', slurm_file)
    log_files = sorted(log_files)
    print('Found log files:', log_files)
    del in_log_folder_files

    # Compilation data from slurm.out
    data = {'algo': [], 'm': [], 'n': [], 'k': [],
            'tile_m': [], 'tile_n': [], 'w': [], 'v': [],
            'threads': [], 'grouping': [], 'minblocks': [], 'perf (Gflop/s)': [],
            'nregs': [], 'nbytes_smem': [], 'nbytes_cmem': []}
    with open(os.path.join(log_folder, slurm_file), 'r') as f:
        slurm_file_content = f.read().splitlines()
    for i, l in enumerate(slurm_file_content):
        if ptxas_intro.match(l):
            kernel = quick_demangle(ptxas_intro.match(l).group(1))
            for k, v in kernel.items():
                data[k].append(v)
        if 'Used' in l:
            if ptxas_values.match(l):
                m = ptxas_values.match(l)
                data['nregs'].append(int(m.group(1)))
                data['nbytes_smem'].append(int(m.group(2)))
                data['nbytes_cmem'].append(int(m.group(3)))
            else:
                assert False, "No match at line " + str(i) + ":\n" + l
    assert np.all([len(val) == len(data['algo']) for key, val in data.items() if key != 'perf (Gflop/s)']), \
        "Found varying lengths:\n" + str([key + ': ' + str(len(val)) + '\n' for key, val in data.items()])

    # Autotuning information (performance)
    num_items = len(data['algo'])
    data['perf (Gflop/s)'] = [0.0] * num_items
    autotuning_line_counter = 0
    for log_file in log_files:
        with open(os.path.join(log_folder, log_file), 'r') as f:
            log_file_content = f.read().splitlines()
        for l in log_file_content:
            if autotuning_line.match(l):

                # Get algo, parameters, and performance
                autotuning_line_counter += 1
                match = autotuning_line.match(l)
                algo = match.group(1)
                tile_m = int(match.group(5)) if match.group(5) is not None else None
                tile_n = int(match.group(6)) if match.group(6) is not None else None
                w = int(match.group(7)) if match.group(7) is not None else None
                v = int(match.group(8)) if match.group(8) is not None else None
                threads = int(match.group(9))
                grouping = int(match.group(10))
                minblocks = int(match.group(11))
                perf = float(match.group(12))

                # Add to data base
                indices_algo = [i for i, x in enumerate(data['algo']) if x == algo]
                indices_tile_m = [i for i, x in enumerate(data['tile_m']) if x == tile_m]
                indices_tile_n = [i for i, x in enumerate(data['tile_n']) if x == tile_n]
                indices_w = [i for i, x in enumerate(data['w']) if x == w]
                indices_v = [i for i, x in enumerate(data['v']) if x == v]
                indices_threads = [i for i, x in enumerate(data['threads']) if x == threads]
                indices_grouping = [i for i, x in enumerate(data['grouping']) if x == grouping]
                indices_minblocks = [i for i, x in enumerate(data['minblocks']) if x == minblocks]
                indices = set.intersection(set(indices_algo), set(indices_tile_m), set(indices_tile_n), set(indices_w),
                                           set(indices_v), set(indices_threads), set(indices_grouping), set(indices_minblocks))
                assert len(indices) == 1, "Found multiple corresponding indices: " + str(indices) + ', lenght = ' + str(len(indices))
                index = list(indices)[0]
                assert index < num_items, "Found an index longer than the data frame: " + str(index) + ", dataframe length = " + str(num_items)
                assert data['perf (Gflop/s)'][index] == 0.0, "This performance was already assigned, index = " + str(index)
                data['perf (Gflop/s)'][index] = perf

    # Assemble and return pandas dataframe
    print('Autotuning lines found: ', autotuning_line_counter, ', numlines: ', num_items)
    assert 0.0 not in data['perf (Gflop/s)'], "Found a kernel with 0 performance"

    # Dump to pickle and return
    print("Writing data to", pickle_file)
    safe_pickle(data, pickle_file)
    return pd.DataFrame.from_dict(data)
