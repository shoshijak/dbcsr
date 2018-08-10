import explore_parameters_utils as explore_utils


def format_pars(df):
    df.replace(to_replace=algo_dict, inplace=True)
    df.fillna(value=0, inplace=True)
    df = df.rename(columns={'threads': 'threads_per_blk', 'nregs': 'regs_per_thread'})
    return df

# Helper variables and functions
algo_dict = {'tiny': 0, 'small': 1, 'medium': 2, 'largeDB1': 3, 'largeDB2': 4}  # category-encoder
algo_dict_rev = {0: 'tiny', 1: 'small', 2: 'medium', 3: 'largeDB1', 4: 'largeDB2'}
npars = 3   # number of parameters in stack list

##mnk_kernel = 'tune_4x4x4_test-stacksize_1605'
##mnk_kernel = 'tune_4x4x5_test-stacksize_1605'
##mnk_kernel = 'tune_4x4x8_test-stacksize_1605'
##mnk_kernel = 'tune_4x4x32_test-stacksize_1605'
mnk_kernel = 'tune_13x28x45'
pars_autotuning = explore_utils.read_kernel_autotuning(mnk_kernel)
pars_autotuning = format_pars(pars_autotuning)