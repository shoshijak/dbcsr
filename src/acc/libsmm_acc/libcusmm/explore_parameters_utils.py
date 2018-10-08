import os
import re
import pickle
import sys
import math
import numpy as np
import pandas as pd

#################################################################################################################################
# Helper variables and functions (formatting & writing) 
#################################################################################################################################

algo_dict = {'tiny': 0, 'small': 1, 'medium': 2, 'largeDB1': 3, 'largeDB2': 4}  # category-encoder 
algo_dict_rev = {0: 'tiny', 1: 'small', 2: 'medium', 3: 'largeDB1', 4: 'largeDB2'}
npars = 3   # number of parameters in stack list
max_bytes = 2**31 - 1  # Maximum number of bytes to pickle in one chunk


def print_dic(dic):
    for k, v in dic.items():
        if isinstance(v, str):
            print('{:<40}: {:>8}'.format(k, v))
        else:
            print('{:<40}: {:>8,}'.format(k, v))

            
def format_pars(df):
    #df.replace(to_replace=algo_dict, inplace=True)
    df.fillna(value=0, inplace=True)
    df = df.rename(columns={'threads': 'threads_per_blk', 'nregs': 'regs_per_thread'})
    return df 


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

            
#################################################################################################################################
# Helper variables and functions (computing helpers) 
#################################################################################################################################

def ceiling(x, step):
    return np.where(x % step == 0, x, x + step - x % step)


def flooring(x, step):
    return np.where(x % step == 0, x, x - x % step)


def ceil_division(a, b):
    return (a + b - 1) // b


#################################################################################################################################
# Read data
#################################################################################################################################

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
    #print('\t\t', mangled_name)
    if 'tiny' in mangled_name:
        match = tiny_mangle.match(mangled_name)
        assert match is not None, "Cannot match name:\n" + mangled_name
        return {'algo': algo_dict['tiny'],
                'm': int(match.group(1)), 'n': int(match.group(2)), 'k': int(match.group(3)),
                'tile_m': None, 'tile_n': None, 'w': None, 'v': None,
                'threads': int(match.group(4)), 'grouping': int(match.group(5)), 'minblocks': int(match.group(6))}
    elif 'small' in mangled_name or 'medium' in mangled_name:
        match = small_medium_mangle.match(mangled_name)
        assert match is not None, "Cannot match name:\n" + mangled_name
        return {'algo': algo_dict[match.group(1)],
                'm': int(match.group(2)), 'n': int(match.group(3)), 'k': int(match.group(4)),
                'tile_m': int(match.group(5)), 'tile_n': int(match.group(6)),
                'w': None, 'v': None,
                'threads': int(match.group(7)), 'grouping': int(match.group(8)), 'minblocks': int(match.group(9))}
    elif 'largeDB' in mangled_name:
        match = largeDB_mangle.match(mangled_name)
        assert match is not None, "Cannot match name:\n" + mangled_name
        return {'algo': algo_dict['largeDB' + match.group(1)],
                'm': int(match.group(2)), 'n': int(match.group(3)), 'k': int(match.group(4)),
                'tile_m': int(match.group(5)), 'tile_n': int(match.group(6)),
                'w': int(match.group(7)), 'v': int(match.group(8)),
                'threads': int(match.group(9)), 'grouping': int(match.group(10)), 'minblocks': int(match.group(11))}
    else:
        assert False, "Cannot find base name in:\n" + mangled_name


log_file_ = re.compile('tune_(\d+)x(\d+)x(\d+)_exe')
ptxas_intro = re.compile('ptxas info\s+: Function properties for (_Z\d+.*)$')
ptxas_values = re.compile('ptxas info\s+: Used (\d+) registers, (\d+) bytes smem, (\d+) bytes cmem\[0\]')
autotuning_line = re.compile(
    'OK Kernel_dnt_(\w+) m (\d+)\s+n (\d+)\s+k (\d+)\s+(?:tile_m (\d+)\s+tile_n (\d+)\s+(?:w (\d+)\s+v (\d+)\s+)?)?threads (\d+)\s+grouping (\d+)\s+minblocks (\d+)\s+GFlop/s (\d+(?:\.\d+)?)')
def read_kernel_autotuning(log_folder, algo_selected=''):
    """
    Given a folder of kernel autotuning, read in compilation information and autotuning information
    :param log_folder: folder of kernel autotuning
    :return: pandas Dataframe containing above information
    """
    pickle_file = os.path.join(log_folder, "data_dump")
    if os.path.exists(pickle_file):
        print("Loading data from", pickle_file)
        data = pickle.load(open(pickle_file, "rb"))
        algo_ = list()
        mat_m_ = list()
        mat_n_ = list()
        mat_k_ = list()
        tile_m_ = list()
        tile_n_ = list()
        w_ = list()
        v_ = list()
        threads_ = list()
        grouping_ = list()
        minblocks_ = list()
        perf_ = list()
        nregs_ = list()
        nbytes_smem_ = list()
        nbytes_cmem_ = list()
        for kernel_pars, perf_vals in data.items():
            algo_.append(kernel_pars[0])
            mat_m_.append(kernel_pars[1])
            mat_n_.append(kernel_pars[2])
            mat_k_.append(kernel_pars[3])
            tile_m_.append(kernel_pars[4])
            tile_n_.append(kernel_pars[5])
            w_.append(kernel_pars[6])
            v_.append(kernel_pars[7])
            threads_.append(kernel_pars[8])
            grouping_.append(kernel_pars[9])
            minblocks_.append(kernel_pars[10])
            perf_.append(perf_vals['perf (Gflop/s)'])
            nregs_.append(perf_vals['nregs'])
            nbytes_smem_.append(perf_vals['nbytes_smem'])
            nbytes_cmem_.append(perf_vals['nbytes_cmem'])
        new_dat_ = dict({'algo': algo_,
                         'mat_m': mat_m_, 'mat_n': mat_n_, 'mat_k': mat_k_,
                         'tile_m': tile_m_, 'tile_n': tile_n_,
                         'w': w_, 'v': v_,
                         'threads': threads_, 'grouping': grouping_, 'minblocks': minblocks_,
                         'perf': perf_,
                         'nregs': nregs_, 'nbytes_smem': nbytes_smem_, 'nbytes_cmem': nbytes_cmem_
                        })
        #safe_pickle(data, pickle_file)
        data_pd = pd.DataFrame.from_dict(new_dat_)
        del data, new_dat_, algo_, mat_m_, mat_n_, mat_k_, tile_m_, tile_n_, \
            w_, v_, threads_, grouping_, minblocks_, perf_, \
            nregs_ , nbytes_smem_, nbytes_cmem_
        if algo_selected != '':
            data_pd = data_pd.groupby('algo').get_group(algo_dict[algo_selected])
        return data_pd

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
    match_ = log_file_.match(log_files[0])
    mat_m = int(match_.group(1))
    mat_n = int(match_.group(2))
    mat_k = int(match_.group(3))
    print('Processing data of matrix sizes (', mat_m, 'x', mat_n, 'x', mat_k, ')')

    # Compilation data from slurm.out
    data = dict()
    with open(os.path.join(log_folder, slurm_file), 'r') as f:
        slurm_file_content = f.read().splitlines()
    print('\tprocessing slurm file', slurm_file)
    for i, l in enumerate(slurm_file_content):
        if ptxas_intro.match(l):
            kernel_ = quick_demangle(ptxas_intro.match(l).group(1))
        if 'Used' in l:
            if ptxas_values.search(l):
                m = ptxas_values.search(l)
                data[tuple(kernel_.values())] = {'perf (Gflop/s)': 0,
                                                 'nregs': int(m.group(1)),
                                                 'nbytes_smem': int(m.group(2)),
                                                 'nbytes_cmem': int(m.group(3))}
            else:
                assert False, "No match at line " + str(i) + ":\n" + l
    #assert np.all([len(val) == len(data['algo']) for key, val in data.items() if key != 'perf (Gflop/s)']), \
    #    "Found varying lengths:\n" + str([key + ': ' + str(len(val)) + '\n' for key, val in data.items()])
    print("\tData items found in slurm files:", len(data))

    # Autotuning information (performance)
    autotuning_line_counter = 0
    for log_file in log_files:
        print('\t\tprocessing log file', log_file)
        with open(os.path.join(log_folder, log_file), 'r') as f:
            log_file_content = f.read().splitlines()
        for l in log_file_content:
            if 'OK' in l:
            #if autotuning_line.match(l):

                # Get algo, parameters, and performance
                autotuning_line_counter += 1
                match = autotuning_line.match(l)
                assert match is not None, "Found null match: " + l
                algo = algo_dict[match.group(1)]
                tile_m = int(match.group(5)) if match.group(5) is not None else None
                tile_n = int(match.group(6)) if match.group(6) is not None else None
                w = int(match.group(7)) if match.group(7) is not None else None
                v = int(match.group(8)) if match.group(8) is not None else None
                threads = int(match.group(9))
                grouping = int(match.group(10))
                minblocks = int(match.group(11))
                perf = float(match.group(12))

                # Add to data base
                key = (algo, mat_m, mat_n, mat_k, tile_m, tile_n, w, v,
                      threads, grouping, minblocks)
                if key in data:
                    data[key]['perf (Gflop/s)'] = perf
                else:
                    data[key] = {'perf (Gflop/s)': perf,
                                 'nregs': None,
                                 'nbytes_smem': None,
                                 'nbytes_cmem': None}

    # Assemble and return pandas dataframe
    print('\tAutotuning lines found: ', autotuning_line_counter)
    #assert 0.0 not in data['perf (Gflop/s)'], "Found a kernel with 0 performance"

    # Dump to pickle and return
    print("Writing data to", pickle_file)
    safe_pickle(data, pickle_file)
    return pd.DataFrame.from_dict(data)


def read_kernel_autotuning_old(log_folder, algo_selected=''):
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
    match_ = log_file_.match(log_files[0])
    mat_m = int(match_.group(1))
    mat_n = int(match_.group(2))
    mat_k = int(match_.group(3))
    print('Processing data of matrix sizes (', mat_m, 'x', mat_n, 'x', mat_k, ')')

    # Compilation data from slurm.out
    data = dict()
    with open(os.path.join(log_folder, slurm_file), 'r') as f:
        slurm_file_content = f.read().splitlines()
    print('\tprocessing slurm file', slurm_file)
    for i, l in enumerate(slurm_file_content):
        if ptxas_intro.match(l):
            kernel_ = quick_demangle(ptxas_intro.match(l).group(1))
        if 'Used' in l:
            if ptxas_values.search(l):
                m = ptxas_values.search(l)
                data[tuple(kernel_.values())] = {'perf (Gflop/s)': 0,
                                                 'nregs': int(m.group(1)),
                                                 'nbytes_smem': int(m.group(2)),
                                                 'nbytes_cmem': int(m.group(3))}
            else:
                assert False, "No match at line " + str(i) + ":\n" + l
    #assert np.all([len(val) == len(data['algo']) for key, val in data.items() if key != 'perf (Gflop/s)']), \
    #    "Found varying lengths:\n" + str([key + ': ' + str(len(val)) + '\n' for key, val in data.items()])
    print("\tData items found in slurm files:", len(data))

    # Autotuning information (performance)
    autotuning_line_counter = 0
    for log_file in log_files:
        print('\t\tprocessing log file', log_file)
        with open(os.path.join(log_folder, log_file), 'r') as f:
            log_file_content = f.read().splitlines()
        for l in log_file_content:
            if 'OK' in l:
            #if autotuning_line.match(l):

                # Get algo, parameters, and performance
                autotuning_line_counter += 1
                match = autotuning_line.match(l)
                assert match is not None, "Found null match: " + l
                algo = algo_dict[match.group(1)]
                tile_m = int(match.group(5)) if match.group(5) is not None else None
                tile_n = int(match.group(6)) if match.group(6) is not None else None
                w = int(match.group(7)) if match.group(7) is not None else None
                v = int(match.group(8)) if match.group(8) is not None else None
                threads = int(match.group(9))
                grouping = int(match.group(10))
                minblocks = int(match.group(11))
                perf = float(match.group(12))

                # Add to data base
                key = (algo, mat_m, mat_n, mat_k, tile_m, tile_n, w, v,
                      threads, grouping, minblocks)
                if key in data:
                    data[key]['perf (Gflop/s)'] = perf
                else:
                    data[key] = {'perf (Gflop/s)': perf,
                                 'nregs': None,
                                 'nbytes_smem': None,
                                 'nbytes_cmem': None}

    # Assemble and return pandas dataframe
    print('\tAutotuning lines found: ', autotuning_line_counter)
    #assert 0.0 not in data['perf (Gflop/s)'], "Found a kernel with 0 performance"

    # Dump to pickle and return
    print("Writing data to", pickle_file)
    safe_pickle(data, pickle_file)
    return pd.DataFrame.from_dict(data)


def read_autotuning_data(read_from, algo_to_read, parse_mode, num_kernels=0): 
    import os, sys, re
    pars_autotuning = pd.DataFrame()
    mnks = list()
    all_kernels = [ak for ak in os.listdir(read_from) if ak[:5] == 'tune_']
    mnk_kernel_pattern = 'tune_(\d+)x(\d+)x(\d+)'
    n_kernels = len(all_kernels)
    file_counter = 0 
    line_counter = 0
    size_counter = 0
    for i, mnk_kernel_ in enumerate(all_kernels):
        
        if num_kernels > 0 and i > num_kernels:
            break

        print(mnk_kernel_, '(', i, '/', n_kernels, ')')
        mnk_kernel_folder = os.path.join(read_from, mnk_kernel_)
        csv_file = os.path.join(mnk_kernel_folder, os.path.split(mnk_kernel_folder)[-1].replace('tune', 'train') + \
                                '_' + algo_to_read + '.csv')
        pars_algo_ = pd.DataFrame()

        # Parse or read file
        if not os.path.exists(csv_file):
            if parse_mode: 
                print('Start parsing kernel', mnk_kernel_folder)
                if os.path.exists(os.path.join(mnk_kernel_folder, 'data_dump')):
                    pars_ = explore_utils.read_kernel_autotuning(mnk_kernel_folder)
                    pars_ = explore_utils.format_pars(pars_)
                    pars_.rename(columns={'mat_k': 'k', 'mat_m': 'm', 'mat_n': 'n'}, inplace=True)
                    for col in pars_: 
                        if col not in ['perf', 'need_sync', 'occupancy', 'Koth_med_perf_K']: 
                            pars_[col] = pars_[col].astype(int)
                    pars_.describe()

                    if algo_to_read != 'all':
                        algo_num = explore_utils.algo_dict[algo_to_read]
                        algos = [algo_to_read]
                    else: 
                        algos = list(explore_utils.algo_dict.keys())

                    for algo in algos: 
                        algo_num = explore_utils.algo_dict[algo]
                        if algo_num in pars_.algo.unique():
                            print('Parsing for algorithm:', algo)
                            pars_algo_ = pars_.groupby('algo').get_group(algo_num).copy()
                            explore_utils.add_derived_parameter(pars_algo_, gpu, autotuning, explore_utils.npars, algo)
                            pars_algo_ = explore_utils.remove_intermediate_columns(pars_algo_, algo)
                            print('\t{:<6} :       lines: {:>8,}      size: {:>10,}'.format(
                                algo, pars_algo_.shape[0], sys.getsizeof(pars_algo_)))
                            pars_algo_.to_csv(csv_file)

                    else: 
                        continue
                else: 
                    continue
            else: 
                continue 

        else: 
            pars_algo_ = pd.read_csv(csv_file, index_col=0)
            print('Found {:<45}  lines: {:>8,}      size: {:>10,}'.format(
                csv_file, pars_algo_.shape[0], sys.getsizeof(pars_algo_)))

        # Increment counters and add dataframe to list 
        file_counter += 1
        line_counter += pars_algo_.shape[0]
        size_counter += sys.getsizeof(pars_algo_)
        pars_algo_['mnk'] = pars_algo_['mnk'].astype(str)
        pars_autotuning = pd.concat([pars_autotuning, pars_algo_])
        del pars_algo_
        m_ = re.match(mnk_kernel_pattern, mnk_kernel_)
        mnks.append((int(m_.group(1)), int(m_.group(2)), int(m_.group(3))))

    print('\n\nFound', file_counter, 'training data files, with', line_counter, 
          'rows/observations, and size: ', sys.getsizeof(pars_autotuning)/10**6, 'MB')
    
    return pars_autotuning, mnks


#################################################################################################################################
# Feature predictors 
#################################################################################################################################

def add_matrix_sizes(df):
    df['size_a'] = df['m'] * df['k']
    df['size_b'] = df['k'] * df['n']
    df['size_c'] = df['m'] * df['n']
    df['mnk'] = df['m'].astype(str) + 'x' + df['n'].astype(str) + 'x' + df['k'].astype(str)
    
def add_launch_pars(df, gpu, autotuning): 
    # Need sync? (relevant for tiny, small, medium) 
    # (mn > warp_size || mk > warp_size || kn > warp_size || threads > warp_size);
    df['need_sync'] = (np.where(df['size_c'] > gpu['Threads per Warp'], True, False)
                       | np.where(df['size_a'] > gpu['Threads per Warp'], True, False) 
                       | np.where(df['size_b'] > gpu['Threads per Warp'], True, False)
                       | np.where(df['threads_per_blk'] > gpu['Threads per Warp'], True, False)).tolist()
    
    # Launching parameters
    df['nblks'] = ceil_division(autotuning['stack_size'], df['grouping'])
    df['warps_per_blk'] = ceil_division(df['threads_per_blk'], gpu['Threads per Warp'])
    df['nwarps'] = df['warps_per_blk'] * df['nblks']
    df['nwarps_inv'] = ceil_division(1., df['nwarps'])
    df['nthreads'] = df['threads_per_blk'] * df['nblks']
    df['sm_desired'] = ceil_division(df['nblks'], df['minblocks'])

def add_occupancy_estimation(df, gpu, one_col):
    # Resource occupations (warps, blocks), (Follows CUDA calculator sheet)
    df['nblocks_per_sm_lim_blks_warps'] = np.minimum(one_col * gpu['Max Thread Blocks per Multiprocessor'], \
                                                     np.floor(gpu['Max Warps per Multiprocessor'] / df['warps_per_blk']))

    # Resource occupations (registers)
    if 'regs_per_thread' in df.columns:
        df['i1'] = flooring(
            gpu['Max Registers per Thread Block'] * one_col / ceiling(df['regs_per_thread'] * gpu['Threads per Warp'],
                                                                      gpu['Register allocation unit size'] * one_col), 
            gpu['Warp allocation granularity'] * one_col)
        df['nblocks_per_sm_lim_reg_'] = np.floor(df['i1'] / df['warps_per_blk']) * \
                                        math.floor(gpu['Registers per Multiprocessor'] / gpu['Max Registers per Thread Block'])
        df['nblocks_per_sm_lim_reg'] = np.where(df['nblocks_per_sm_lim_reg_'] == 0, 1, df['nblocks_per_sm_lim_reg_'])
        
    # Resource occupations (shared memory)
    if 'nbytes_smem' in df.columns:
        df['smem_per_block'] = ceiling(df['nbytes_smem'], gpu['Shared Memory allocation unit size'])
        df['nblocks_per_sm_lim_smem'] = np.floor(one_col * gpu['Shared Memory per Multiprocessor (bytes)'] / df['smem_per_block'], one_col)
        
    # Aggregate 
    if 'nblocks_per_sm_lim_blks_warps' in df.columns and 'nblocks_per_sm_lim_reg' in df.columns and 'nblocks_per_sm_lim_smem' in df.columns :
        df['nblks_per_sm'] = df[['nblocks_per_sm_lim_blks_warps', 'nblocks_per_sm_lim_reg', 'nblocks_per_sm_lim_smem']].min(axis=1)
        df['nwarps_per_sm'] = df['nblks_per_sm'] * df['warps_per_blk']
        df['nsm'] = ceiling(df['nblks'], df['nblks_per_sm'])
        df['ngpu'] = ceiling(df['nsm'], gpu['Multiprocessors'])
        df['occupancy'] = df['nwarps_per_sm'] / gpu['Max Warps per Multiprocessor']
        
    
def add_ru_common(df, autotuning, one_col): 
    # Loop counts
    df['ru_param_stack_unroll_factor'] = ceil_division(df['grouping'], df['threads_per_blk'])
    df['ru_param_stack_unroll_factor_inv'] = ceil_division(1., df['ru_param_stack_unroll_factor'])
    df['n_iter'] = np.maximum(3 * one_col, 12500 * (one_col // (df['m'].values * df['n'].values * df['k'].values)))
    df['Gflops'] = df['n_iter'] * autotuning['stack_size'] * df['m'] * df['n'] * df['k'] * 2 * 10**(-9)

atomicAdd_factor = 5
def perf_Kothapalli(N_mem, nblks, threads_per_blk, Gflops): 
    c_K = nblks * threads_per_blk * N_mem  # ignore number of threads per warp 
    return Gflops / c_K # ignore clock rate 

def add_Kothapalli(df, gpu, Nmem_glob, Nmem_shared, Nmem, perf_K): 
    df[Nmem] = gpu['Global memory access latency'] * df[Nmem_glob] + gpu['Shared memory access latency'] * df[Nmem_shared]
    df[perf_K] = np.vectorize(perf_Kothapalli)(df[Nmem], df['nblks'], df['threads_per_blk'], df['Gflops'])

def add_ru_tinysmallmed(df):
    # Resource usage estimation and loop counts (tiny)
    df['ru_tinysmallmed_unroll_factor_a'] = ceil_division(df['size_a'], df['threads_per_blk'])
    df['ru_tinysmallmed_unroll_factor_a_inv'] = ceil_division(1., df['ru_tinysmallmed_unroll_factor_a'])
    df['ru_tinysmallmed_unroll_factor_b'] = ceil_division(df['size_b'], df['threads_per_blk'])
    df['ru_tinysmallmed_unroll_factor_b_inv'] = ceil_division(1., df['ru_tinysmallmed_unroll_factor_b'])
    df['ru_tinysmallmed_unroll_factor_a_total'] = df['ru_tinysmallmed_unroll_factor_a'] * df['grouping']
    df['ru_tinysmallmed_unroll_factor_b_total'] = df['ru_tinysmallmed_unroll_factor_b'] * df['grouping']
    df['ru_tinysmallmed_unroll_factor_c_total'] = df['k'] * df['grouping']
    
def add_ru_tiny(df, gpu, autotuning, npars):
    # Resource usage estimation and loop counts (tiny)
    add_ru_tinysmallmed(df)
    df['ru_tiny_max_parallel_work'] = df[['grouping', 'size_a', 'size_b', 'size_c']].max(axis=1)
    df['ru_tiny_min_threads'] = df['size_c']
    df['ru_tiny_max_threads'] = ceiling(df['ru_tiny_max_parallel_work'], gpu['Threads per Warp'])
    df['ru_tiny_buf_size'] = df['k'] * (df['m'] + df['n'])
    df['ru_tiny_smem_per_block'] = (df['ru_tiny_buf_size'] * autotuning['sizeof double']) + (npars * df['grouping'] * autotuning['sizeof int']) 
    # Occupancy estimation: 
    # Assumption (verified on a sample of mnks): nblks is always limited by number of threads
    df['ru_tiny_nblks_per_sm'] = df['nblocks_per_sm_lim_blks_warps']
    df['ru_tiny_nwarps_per_sm'] = df['nblks_per_sm'] * df['warps_per_blk']
    df['ru_tiny_nsm'] = ceiling(df['nblks'], df['nblks_per_sm'])
    df['ru_tiny_ngpu'] = ceiling(df['nsm'], gpu['Multiprocessors'])
    df['ru_tiny_occupancy'] = df['nwarps_per_sm'] / gpu['Max Warps per Multiprocessor']
    # Kothapalli et al. modelling, communication-bound
    df['Koth_tiny_Nmem_shared'] = 3*df['grouping'] + df['grouping']*(3 + df['ru_tinysmallmed_unroll_factor_a'] + df['ru_tinysmallmed_unroll_factor_b'] + 2*df['k'])
    df['Koth_tiny_Nmem_glob'] = 3*df['grouping'] + df['grouping']*(df['ru_tinysmallmed_unroll_factor_a'] + df['ru_tinysmallmed_unroll_factor_b'])
    add_Kothapalli(df, gpu, 'Koth_tiny_Nmem_glob', 'Koth_tiny_Nmem_shared', 'Koth_tiny_Nmem', 'Koth_tiny_perf_K')


def add_ru_smallmedlarge(df):
    df['ru_smallmedlarge_cmax'] = ceil_division(df['n'], df['tile_n'])
    df['ru_smallmedlarge_rmax'] = ceil_division(df['m'], df['tile_m'])
    df['ru_smallmedlarge_T'] = df['tile_m'] * df['tile_n']
    df['ru_smallmedlarge_min_threads'] = df['ru_smallmedlarge_cmax'] * df['ru_smallmedlarge_rmax']

    
def add_ru_smallmed(df, gpu, autotuning, npars):
    # Resource usage estimation and loop counts (small, medium) 
    add_ru_tinysmallmed(df)
    add_ru_smallmedlarge(df)
    df['ru_smallmed_tm_max'] = df['m']
    df['ru_smallmed_tn_max'] = df['n']
    df['ru_smallmed_unroll_factor_c'] = ceil_division(df['size_c'], df['threads_per_blk'])
    df['ru_smallmed_loop_matmul'] = df['k'] * df['tile_m'] * df['tile_n']

    df['ru_smallmed_max_parallel_work'] = df[['grouping', 'size_a', 'size_b', 'size_c', 'ru_smallmedlarge_min_threads']].max(axis=1)        
    df['ru_smallmed_max_threads'] = ceiling(df['ru_smallmed_max_parallel_work'], gpu['Threads per Warp'])

    df['intermediate1'] = df['size_a'] + df['k'] * df['tile_n'] * df['ru_smallmedlarge_cmax']
    df['intermediate2'] = df['tile_m'] * df['ru_smallmedlarge_rmax'] * df['k'] + 1
    df['ru_smallmed_buf_size'] = df[['size_c', 'intermediate1', 'intermediate2']].max(axis=1)
    df['ru_smallmed_smem_per_block'] = (df['ru_smallmed_buf_size'] * autotuning['sizeof double']) + (npars * df['grouping'] * autotuning['sizeof int'])

    df['ru_smallmed_regs_per_thread'] = df['tile_m'] * df['tile_n'] + (df['m'] * df['k'] + df['threads_per_blk'] - 1) // df['threads_per_blk'] + (df['k'] * df['n'] + df['threads_per_blk'] - 1) // df['threads_per_blk']

        
def add_ru_small(df, gpu, autotuning, npars):
    add_ru_smallmed(df, gpu, autotuning, npars)
    # Kothapalli et al. modelling, communication-bound
    df['Koth_small_Nmem_shared'] = 3*df['grouping'] + df['grouping']*(
        3 + df['ru_tinysmallmed_unroll_factor_a'] + df['ru_tinysmallmed_unroll_factor_b'] + 2*df['k']*df['tile_m']*df['tile_n'] + df['tile_m']*df['tile_n'])
    df['Koth_small_Nmem_shared'] = df.apply(
        lambda row: row['Koth_small_Nmem_shared'] + atomicAdd_factor*row['ru_smallmed_unroll_factor_c']
        if row['tile_m'] > 1 and row['tile_n'] > 1 
        else row['Koth_small_Nmem_shared'], axis=1)
    df['Koth_small_Nmem_glob'] =   3*df['grouping'] + df['grouping']*(
            df['ru_tinysmallmed_unroll_factor_a'] + df['ru_tinysmallmed_unroll_factor_b'] + atomicAdd_factor*df['ru_smallmed_unroll_factor_c'])
    add_Kothapalli(df, gpu, 'Koth_small_Nmem_glob', 'Koth_small_Nmem_shared', 'Koth_small_Nmem', 'Koth_small_perf_K')


def add_ru_med(df, gpu, autotuning, npars):
    add_ru_smallmed(df, gpu, autotuning, npars)
    
    # Loop bounds
    df['load_unroll_factor_1'] = ceil_division(df['size_a'], df['threads_per_blk']) + 1;
    df['load_unroll_factor_2'] = ceil_division(df['size_b'], df['threads_per_blk']) + 1;
    df['n_mkloads'] = ceil_division(df['size_a'], (df['load_unroll_factor_1'] * df['threads_per_blk']));
    df['n_knloads'] = ceil_division(df['size_b'], (df['load_unroll_factor_2'] * df['threads_per_blk']));

    # Kothapalli et al. modelling, communication-bound
    df['Koth_med_Nmem_shared'] = 3*df['grouping'] + df['grouping']*(
        2 + df['n_mkloads'] * df['load_unroll_factor_1'] + 
            df['n_knloads'] * df['load_unroll_factor_2'] + 
        2 + 2*df['k']*df['tile_m']*df['tile_n'] + 1)
    df['Koth_med_Nmem_shared'] = df.apply(
        lambda row: row['Koth_med_Nmem_shared'] + atomicAdd_factor*row['ru_smallmed_unroll_factor_c']
        if row['tile_m'] > 1 and row['tile_n'] > 1 
        else row['Koth_med_Nmem_shared'], axis=1)
    df['Koth_med_Nmem_glob'] =   3*df['grouping'] + df['grouping']*(
            2*df['n_mkloads']*df['load_unroll_factor_1'] + df['n_knloads'] * df['load_unroll_factor_2'] + atomicAdd_factor*df['ru_smallmed_unroll_factor_c'])
    add_Kothapalli(df, gpu, 'Koth_med_Nmem_glob', 'Koth_med_Nmem_shared', 'Koth_med_Nmem', 'Koth_med_perf_K')


def add_ru_large(df, gpu, autotuning, npars):
    # Resource usage estimation and loop counts (largeDB1, largeDB2) 
    add_ru_smallmedlarge(df)
    df['ru_large_Pa'] = df['m'] * df['w']  # Input slab sizes 
    df['ru_large_Pb'] = df['w'] * df['n']
    df['ru_large_Pc'] = df['m'] * df['v']  # Output slabs
    df['ru_large_unroll_factor_a'] = ceil_division(df['ru_large_Pa'], df['threads_per_blk'])
    df['ru_large_unroll_factor_b'] = ceil_division(df['ru_large_Pb'], df['threads_per_blk'])
    df['ru_large_unroll_factor_c'] = ceil_division(df['ru_large_Pc'], df['threads_per_blk'])
    df['ru_large_loop_matmul'] = df['w'] * df['tile_m'] * df['tile_n']
    df['ru_large_max_concurrent_work'] = df[['grouping', 'ru_large_Pa', 'ru_large_Pb', 'ru_large_Pc', 'ru_smallmedlarge_T']].max(axis=1)
    
    df['ru_large_regs_per_thread'] = df['tile_m'] * df['tile_n'] + \
                                    (df['w'] * df['m'] + df['threads_per_blk'] - 1) // df['threads_per_blk'] + \
                                    (df['w'] * df['n'] + df['threads_per_blk'] - 1) // df['threads_per_blk']
    df['ru_large_n_DB_iter'] = df['k'] // (2*df['w'])
    df['intermediate1'] = (df['w'] - 1) * df['m'] + df['ru_smallmedlarge_rmax'] * df['tile_m']
    df['intermediate2'] = df['m'] * df['w'] + (df['w'] - 1) * df['n'] + df['ru_smallmedlarge_cmax'] * df['tile_n']
    df['ru_large_buf_size'] = df[['ru_large_Pc', 'intermediate1', 'intermediate2']].max(axis=1)
    df['ru_large_smem_per_block'] = (df['ru_large_buf_size'] * autotuning['sizeof double']) + (npars * df['grouping'] * autotuning['sizeof int'])

    # Kothapalli et al. modelling, communication-bound
    df['Koth_large_Nmem_glob'] = 3*df['grouping'] + df['grouping']*(
        2 + 
        (df['m']*df['w']) // df['threads_per_blk'] + (df['n']*df['w']) // df['threads_per_blk'] + # load_gmem_into_smem
        (df['k'] // df['w'])*(
            (df['m']*df['w']) // df['threads_per_blk'] + (df['n']*df['w']) // df['threads_per_blk'] # load_gmem_into_regs
        ) + # double-buffering
        (df['n'] // df['v'])*(
            atomicAdd_factor * (df['ru_large_Pc'] // df['threads_per_blk']) # result accumulation
        ) # write out
    )
    df['Koth_large_Nmem_shared'] = 3*df['grouping'] + df['grouping']*(
        (df['m']*df['w']) // df['threads_per_blk'] + (df['n']*df['w']) // df['threads_per_blk'] + # load_gmem_into_smem
        (df['k'] // df['w'])*(
            (df['m']*df['w']) // df['threads_per_blk'] + (df['n']*df['w']) // df['threads_per_blk'] +   # load_regs_into_smem
            3*df['w']*df['tile_m']*df['tile_n'] # multiply
        ) + # double-buffering
        (df['n'] // df['v'])*(
            df['tile_m'] * df['tile_n'] + # store_results_into_smem
            df['ru_large_Pc'] // df['threads_per_blk'] # result accumulation
        ) # write out
    )
    add_Kothapalli(df, gpu, 'Koth_large_Nmem_glob', 'Koth_large_Nmem_shared', 'Koth_large_Nmem', 'Koth_large_perf_K')


to_remove_ = {
    'tiny': ['tile_m', 'tile_n', 'v', 'w'], 
    'small': ['v', 'w'], 'medium': ['v', 'w'], 
    'largeDB1': [], 'largeDB2': [] 
}
def remove_intermediate_columns(df, algo):
    to_remove = ['i1', 'nblocks_per_sm_lim_reg_', 'intermediate1', 'intermediate2', 'n_iter']
    to_remove += to_remove_[algo]
    for col in to_remove: 
        if col in list(df.columns): 
            df = df.drop(col, axis=1)
    return df 


def add_derived_parameter(df, gpu, autotuning, npars, algo):
    one_col = np.ones(len(df['algo']))
    add_matrix_sizes(df)
    add_launch_pars(df, gpu, autotuning)
    add_occupancy_estimation(df, gpu, one_col)
    add_ru_common(df, autotuning, one_col)
    if algo == 'tiny': 
        add_ru_tiny(df, gpu, autotuning, npars)
    elif algo == 'small': 
        add_ru_small(df, gpu, autotuning, npars)
    elif algo == 'medium': 
        add_ru_med(df, gpu, autotuning, npars)
    elif algo == 'largeDB1' or algo == 'largeDB2' : 
        add_ru_large(df, gpu, autotuning, npars)
    else: 
        print("Cannot recognize algorithm:", algo)
        

#################################################################################################################################
# Feature plotting
#################################################################################################################################

def make_range(ar, step):
    _min = np.amin(ar)
    _max = np.amax(ar)
    _range = list(range(_min, _max+1, step))
    _vals = dict(zip(_range, list(range(0, len(_range)))))
    return _range, _vals 
    
def perf_heatmap(Xph, to_set, x_axis, y_axis):
    for k, v in to_set.items():
        Xph = pd.concat([X, Y_perf], axis=1).groupby(k).get_group(v)
    display(Xph.describe())
    
    th_range, th_vals = make_range(Xph[x_axis].values, 32)
    gp_range, gp_vals = make_range(Xph[y_axis].values, 1)

    # Create heatmap
    hm_Z = np.zeros((len(th_range), len(gp_range)), dtype=float)
    for i, p in enumerate(Xph['perf (Gflop/s)'].tolist()): 
        hm_Z[th_vals[Xph[x_axis].tolist()[i]], 
             gp_vals[Xph[y_axis].tolist()[i]]] = p 

    plt.figure()
    plt.title('performance for ' + str(to_set))
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    hm = plt.pcolor(hm_Z)
    plt.colorbar(hm)
    plt.show()

#################################################################################################################################
# Predictive modelling (fitting utilities)
#################################################################################################################################
from sklearn import metrics
from bokeh.layouts import column 

def print_error(y_true, y_pred):
    print('Mean Absolute Error:     {:>7.4f}'.format(metrics.mean_absolute_error(y_true, y_pred)))
    print('Mean Squared Error:      {:>7.4f}'.format(metrics.mean_squared_error(y_true, y_pred)))
    print('Root Mean Squared Error: {:>7.4f}\n'.format(np.sqrt(metrics.mean_squared_error(y_true, y_pred))))


def goodness_plot(mnks, X_mnk, y, y_pred, top_k, title):

    y = np.sqrt(y)
    y_pred = np.sqrt(y_pred)
    perf_losses = list()
    
    for m, n, k in mnks: 

        idx_mnk = np.where(X_mnk == (str(m) + 'x' + str(n) + 'x' + str(k)))[0].tolist()
        assert(len(idx_mnk) > 0), "idx_mnk is empty" 
        y_mnk = y.iloc[idx_mnk]
        y_pred_mnk = y_pred[idx_mnk]

        y_predmax = -np.partition(-y_pred_mnk, top_k)[:top_k]     # top-k predicted maxima
        top_k_idx = np.argpartition(-y_pred_mnk, top_k)[:top_k]
        y_correspmax = y_mnk.iloc[top_k_idx]                      # corresponding true maxima

        maxperf = float(y_mnk.max(axis=0))      # true max. performance
        assert maxperf > 0, "Found non-positive value for maxperf: " + str(maxperf)
        maxperf_chosen = np.amax(y_correspmax)  # chosen max perf. among predicted max performances

        # perf. loss incurred by using model-predicted parameters instead of autotuned ones 
        perf_loss = 100*(maxperf - maxperf_chosen) / maxperf
        perf_losses.append(perf_loss)

    # Goodness plot 
    p = figure(plot_width=800, plot_height=800, title=title + " top " + str(top_k))
    x = np.arange(len(mnks))
    y = np.array(perf_losses)
    pl_max = float(y.max(axis=0))
    pl_min = float(y.min(axis=0))
    pl_mean = float(y.mean(axis=0))
    pl_rms = np.sqrt(np.square(y).mean(axis=0))
    p.circle(x=x, y=y, size=4, color='blue',  legend="relative perf. loss [%] (top-" + str(top_k) + ")")
    p.line(x=[x[0], x[-1]], y=[pl_min, pl_min],   legend="relative perf loss: min  ({:0.2f} %)".format(pl_min), color='red')
    p.line(x=[x[0], x[-1]], y=[pl_max, pl_max],   legend="relative perf loss: max  ({:0.2f} %)".format(pl_max), color='blue')
    p.line(x=[x[0], x[-1]], y=[pl_mean, pl_mean], legend="relative perf loss: mean ({:0.2f} %)".format(pl_mean), line_dash='dashed')
    p.line(x=[x[0], x[-1]], y=[pl_rms, pl_rms],   legend="relative perf loss: rms  ({:0.2f} %)".format(pl_rms), color='green', line_dash='dashed')
    p.yaxis.axis_label = 'Relative performance loss compared to autotuned max. perf [%]'
    p.xaxis.axis_label = 'Test-kernels (arbitrary order)'
    p.legend.location = "top_right"

    # Losses-histogram 
    num_bins = 50 
    hist, edges = np.histogram(y, bins=num_bins)
    df_hist = pd.DataFrame({'hist': hist, 'left': edges[:-1], 'right': edges[1:]})
    source = ColumnDataSource(df_hist)
    q = figure(plot_width=800, plot_height=800, title="Histogram of relative performance losses")
    q.xgrid.grid_line_color = None
    q.xaxis.axis_label = "relative performance loss [%]"
    q.xaxis.major_label_orientation = 1.2
    q.yaxis.axis_label = "# occurrences"
    q.quad(source=source, bottom=0, top='hist', left='left', right='right', fill_color='blue')
    
    return column([p, q])


#################################################################################################################################
# Predictive modelling (displaying, plotting)
#################################################################################################################################

from sklearn import metrics
from sklearn.model_selection import train_test_split  
from bokeh.plotting import figure 
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.io import output_notebook, show
output_notebook()

def print_error(y_true, y_pred):
    print('Mean Absolute Error:     {:>7.4f}'.format(metrics.mean_absolute_error(y_true, y_pred)))
    print('Mean Squared Error:      {:>7.4f}'.format(metrics.mean_squared_error(y_true, y_pred)))
    print('Root Mean Squared Error: {:>7.4f}\n'.format(np.sqrt(metrics.mean_squared_error(y_true, y_pred))))

def display_tree(x, model, model_name):
    from sklearn import tree
    import graphviz
    dot_data = tree.export_graphviz(model, out_file=None, filled=True, 
                                    leaves_parallel=True, feature_names=list(x.columns))
    graph = graphviz.Source(dot_data)
    graph.render(model_name)
    return graph

def perf_pred_hist(y):
    # Create histogram and hover tool 
    num_bins = 200
    hist, edges = np.histogram(y, bins=num_bins)
    df_hist = pd.DataFrame({'hist': hist, 'left': edges[:-1], 'right': edges[1:]})
    source = ColumnDataSource(df_hist)
    hover = HoverTool(tooltips=[('# occurences', '@hist'), ('low', '@left'), ('high', '@right')])

    # Create the figure
    p = figure(plot_width=800, plot_height=800, title="Predicted performance histogram",
               toolbar_location=None, tools="")
    p.xgrid.grid_line_color = None
    p.xaxis.axis_label = "predicted perf (GFlop/s)"
    p.xaxis.major_label_orientation = 1.2
    p.yaxis.axis_label = "# occurrences"
    p.quad(source=source, bottom=0, top='hist', left='left', right='right', fill_color='blue')
    p.add_tools(hover)
    return p

def perf_pred_scatter(y, y_pred, top_k, title): 
    y = np.sqrt(y)
    y_pred = np.sqrt(y_pred)
    y_predmax = -np.partition(-y_pred, top_k)[:top_k]
    top_k_idx = np.argpartition(-y_pred, top_k)[:top_k]
    y_correspmax = y.iloc[top_k_idx]
    
    maxperf = float(y.max(axis=0))
    maxperf_pred = float(y_pred.max(axis=0))
    maxperf_chosen = np.amax(y_correspmax)
    
    p = figure(plot_width=800, plot_height=800, title=title + " top " + str(top_k))
    x = np.arange(top_k)
    p.circle(x=x, y=y_predmax, size=4, color='blue', legend="predicted top " + str(top_k) + " perf")
    p.circle(x=x, y=y_correspmax, size=4, color='red', legend="corresponding true perf.")
    p.line(x=[0, top_k], y=[maxperf_pred, maxperf_pred], legend="predicted max. perf (" + str(maxperf_pred) + ")", color='blue')
    p.line(x=[0, top_k], y=[maxperf, maxperf], legend="true max. perf (" + str(maxperf) + ")", color='red')
    p.line(x=[0, top_k], y=[maxperf_chosen, maxperf_chosen], legend="chosen max. perf (" + str(maxperf_chosen) + "), diff = " + str(maxperf-maxperf_chosen), color='green', line_dash='dashed')
    p.legend.location = "bottom_right"
    return p 

def accuracy_scatter(y, y_pred, title, top_k=10): 
  
    p = figure(plot_width=800, plot_height=800, title="Actual VS predicted execution time (" + title + ")")

    y = np.sqrt(y)
    y_pred = np.sqrt(y_pred)
    p.circle(x=y, y=y_pred, size=4, color='black', legend="actual VS predicted")

    top_k = 10
    y_predmax = -np.partition(-y_pred, top_k)[:top_k]
    top_k_idx = np.argpartition(-y_pred, top_k)[:top_k]
    y_correspmax = y.iloc[top_k_idx]
    p.square(x=y_correspmax, y=y_predmax, size=8, color='green', legend="top " + str(top_k) + " predicted")

    top_k = 1
    y_correspmax = -np.partition(-y, top_k)[:top_k]
    top_k_idx = np.argpartition(-y, top_k)[:top_k]
    y_predmax = y_pred[top_k_idx]
    p.asterisk(x=y_correspmax, y=y_predmax, size=10, color='red', legend="true max")

    max_ = max(float(y.max(axis=0)), float(y_pred.max(axis=0)))
    min_ = min(float(y.min(axis=0)), float(y_pred.min(axis=0)))
    p.line(x=[min_, max_], y=[min_, max_], legend="y=x", line_width=2, color='red')
    
    p.xaxis.axis_label = "Actual performance (GFlop/s)"
    p.yaxis.axis_label = "Predicted performance (GFlop/s)"
    
    p.legend.location = "bottom_right"
    return p 


#################################################################################################################################
# Predictive modelling (fitting)
#################################################################################################################################

def fit_model(model, x, y, test_size=None, idx_train=None, idx_test=None): 
    
    if idx_train==None and idx_test==None:         
        # X, Y, split, model
        print('splitting into train/test on the spot')
        X_train, X_test, y_train, y_test = train_test_split(x, y.values.ravel(), test_size=test_size, random_state=0)  
    elif test_size==None: 
        print('Found train/test split')
        X_train = x.loc[idx_train] 
        X_test  = x.loc[idx_test]
        y_train = y.loc[idx_train]
        y_test  = y.loc[idx_test]
    else: 
        print('INVALID INPUT PARAMETERS')

    # Fit
    model.fit(X_train, y_train)
    print('\n-------------------')
    print(model, '\n')
    #print('Feature importance:')
    #print(model.feature_importances_, '\n')

    # Training error
    y_train_pred = model.predict(X_train)
    print('Training error:')
    print_error(y_train, y_train_pred)

    # Test
    y_pred = model.predict(X_test)  
    print('Testing error:')
    print_error(y_test, y_pred)
    
    return model.predict(x)

#################################################################################################################################
# Predictive modelling (feature selection)
#################################################################################################################################

def plot_feat_importance(X, model): 
    import matplotlib.pyplot as plt
    importances = model.feature_importances_
    names = X.columns
    #std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    print("Feature ranking:")    
    for f in range(X.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, names[indices[f]], importances[indices[f]]))

    plt.rcdefaults()
    fig, ax = plt.subplots()

    ax.set_title("Feature importances")
    ax.barh(range(X.shape[1]), importances[indices],
             color="g", 
             #yerr=std[indices], 
             align="center")
    ax.set_yticks( np.arange(len(importances)))
    ax.set_yticklabels(names[indices])
    ax.invert_yaxis() 
    #plt.set_ylim([-1, X.shape[1]])
    plt.show()
    
def RFECV_selection(model, x, y, cv=3):
    from sklearn.feature_selection import RFECV
    from sklearn.model_selection import KFold
    import matplotlib.pyplot as plt

    rfecv = RFECV(estimator=model, step=1, verbose=2, n_jobs=-1, cv=cv)
    fit = rfecv.fit(x, y)

    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()

    # Get features
    print("Num Features to select:", fit.n_features_)
    print("All Features:\n", list(x.columns))
    print("Feature Ranking:\n", fit.ranking_)
    selected_RFECV = list() 
    for i, f in enumerate(list(x.columns)):
        if fit.support_[i]: 
            selected_RFECV.append(f)
    print("Selected Features:\n", selected_RFECV)
    
    return  x[selected_RFECV]

def RFE_selection(model, x, y, num_features):
    from sklearn.feature_selection import RFE

    rfe = RFE(model, num_features)
    fit = rfe.fit(x, y)
    print("Num Features to select:", fit.n_features_)
    print("All Features:\n", list(x.columns))
    print("Feature Ranking:\n", fit.ranking_)
    selected_RFE = list() 
    for i, f in enumerate(list(x.columns)):
        if fit.support_[i]: 
            selected_RFE.append(f)
    print("\n\nSelected Features:\n", selected_RFE)
    
    return x[selected_RFE]

# source: http://blog.datadive.net/selecting-good-features-part-ii-linear-models-and-regularization/
def pretty_print_linear(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)
                                   for coef, name in lst)