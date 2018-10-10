########################################################################################################################
# DBCSR optimal parameters prediction
# Shoshana Jakobovits
# August-September 2018
########################################################################################################################

import numpy as np
import pandas as pd
import os
import sys
import pickle
import datetime


########################################################################################################################
# Flags
########################################################################################################################
from absl import flags, app
flags.DEFINE_string('in_folder', 'tune_big/', 'Folder from which to read data')
flags.DEFINE_string('algo', 'tiny', 'Algorithm to train on')
flags.DEFINE_string('tune', 'DT', 'Model to tune (Options: DT, RF, all)')
flags.DEFINE_integer('nruns', '10', '#times to run train-test split, variable selection and GridSearch on model')
flags.DEFINE_integer('splits', '5', 'number of cross-validation splits used in GridSearchCV')
FLAGS = flags.FLAGS


########################################################################################################################
# Optimized Hyperparameters
########################################################################################################################
optimized_hyperparameters = {
    'tiny': {'max_depth': np.NaN, 'min_samples_split': np.NaN, 'min_samples_leaf': np.NaN},
    'small': {'max_depth': np.NaN, 'min_samples_split': np.NaN, 'min_samples_leaf': np.NaN},
    'medium': {'max_depth': np.NaN, 'min_samples_split': np.NaN, 'min_samples_leaf': np.NaN},
    'largeDB1': {'max_depth': np.NaN, 'min_samples_split': np.NaN, 'min_samples_leaf': np.NaN},
    'largeDB2': {'max_depth': np.NaN, 'min_samples_split': np.NaN, 'min_samples_leaf': np.NaN}
}


########################################################################################################################
# Formatting and printing helpers
########################################################################################################################
algo_dict = {'tiny': 0, 'small': 1, 'medium': 2, 'largeDB1': 3, 'largeDB2': 4}  # category-encoder
algo_dict_rev = {0: 'tiny', 1: 'small', 2: 'medium', 3: 'largeDB1', 4: 'largeDB2'}
npars = 3   # number of parameters in stack list
max_bytes = 2**31 - 1  # Maximum number of bytes to pickle in one chunk


def safe_pickle(data, file):
    """
    Pickle big files safely by processing them in chunks
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


def print_and_log(msg, log):
    log += '\n' + msg
    print(msg)


########################################################################################################################
# Custom loss functions
########################################################################################################################
def _num_samples(x):
    """
    Return number of samples in array-like x.
    TAKEN VERBATIM FROM SKLEARN code !!
    """
    if hasattr(x, 'fit') and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError('Expected sequence or array-like, got '
                        'estimator %s' % x)
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError("Expected sequence or array-like, got %s" %
                            type(x))
    if hasattr(x, 'shape'):
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        return x.shape[0]
    else:
        return len(x)


def check_consistent_length(*arrays):
    """Check that all arrays have consistent first dimensions.

    Checks whether all objects in arrays have the same shape or length.

    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.

    TAKEN VERBATIM FROM SKLEARN code !!
    """

    lengths = [_num_samples(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError("Found input variables with inconsistent numbers of"
                         " samples: %r" % [int(l) for l in lengths])


def perf_loss(y_true, y_pred, top_k, X_mnk):
    """
    Compute the relative performance losses per mnk if one were to
    :param y_true: ground truth
    :param y_pred: estimated performances
    :param top_k: #top performances to consider
    :param X_mnk: corresponding mnks
    :return: perf_losses: array of relative performance losses (in %), one element per mnk
    """
    check_consistent_length(y_true, y_pred, X_mnk)
    y_true = np.sqrt(y_true)
    y_pred = np.sqrt(y_pred)
    perf_losses = list()

    mnks = np.unique(X_mnk['mnk'].values)
    for mnk in mnks:

        # Get performances per mnk
        idx_mnk = np.where(X_mnk == mnk)[0].tolist()
        assert (len(idx_mnk) > 0), "idx_mnk is empty"
        y_true_mnk = y_true.iloc[idx_mnk]
        y_pred_mnk = y_pred[idx_mnk]
        top_k_idx = np.argpartition(-y_pred_mnk, top_k)[:top_k]
        y_correspmax = y_true_mnk.iloc[top_k_idx]

        # Max. performances
        maxperf = float(y_true_mnk.max(axis=0))  # true max. performance
        assert maxperf > 0, "Found non-positive value for maxperf: " + str(maxperf)
        maxperf_chosen = np.amax(y_correspmax)  # chosen max perf. among predicted max performances

        # perf. loss incurred by using model-predicted parameters instead of autotuned ones
        perf_loss = 100 * (maxperf - maxperf_chosen) / maxperf
        perf_losses.append(perf_loss)

    return perf_losses


def worse_rel_perf_loss_of_k(y_true, y_pred, top_k, X_mnk):
    y = np.array(perf_loss(y_true, y_pred, top_k, X_mnk))
    return float(y.max(axis=0))


def mean_rel_perf_loss_of_k(y_true, y_pred, top_k, X_mnk):
    y = np.array(perf_loss(y_true, y_pred, top_k, X_mnk))
    return float(y.mean(axis=0))


########################################################################################################################
# Custom Scorers
# Ref: http://scikit-learn.org/stable/modules/model_evaluation.html#implementing-your-own-scoring-object
########################################################################################################################
def worse_case_scorer(estimator, X, y, top_k):
    """
    :param estimator: the model that should be evaluated
    :param X: validation data
    :param y: ground truth target for X
    :return: score: a floating point number that quantifies the estimator prediction quality on X, with reference to y
    """
    mnk = pd.DataFrame()
    mnk['mnk'] = X['mnk'].copy()
    y_pred = estimator.predict(X.drop(['mnk'], axis=1))
    score = worse_rel_perf_loss_of_k(y, y_pred, top_k, mnk)
    return -score  # by scikit-learn convention, higher numbers are better, so the value should be negated


def worse_case_scorer_top1(estimator, X, y):
    return worse_case_scorer(estimator, X, y, 1)


def worse_case_scorer_top3(estimator, X, y):
    return worse_case_scorer(estimator, X, y, 3)


def worse_case_scorer_top5(estimator, X, y):
    return worse_case_scorer(estimator, X, y, 5)


def mean_scorer(estimator, X, y, top_k):
    """
    :param estimator: the model that should be evaluated
    :param X: validation data
    :param y: ground truth target for X
    :return: score: a floating point number that quantifies the estimator prediction quality on X, with reference to y
    """
    mnk = pd.DataFrame()
    mnk['mnk'] = X['mnk'].copy()
    y_pred = estimator.predict(X.drop(['mnk'], axis=1))
    score = mean_rel_perf_loss_of_k(y, y_pred, top_k, mnk)
    return -score  # by scikit-learn convention, higher numbers are better, so the value should be negated


def mean_scorer_top1(estimator, X, y):
    return mean_scorer(estimator, X, y, 1)


def mean_scorer_top3(estimator, X, y):
    return mean_scorer(estimator, X, y, 3)


def mean_scorer_top5(estimator, X, y):
    return mean_scorer(estimator, X, y, 5)


########################################################################################################################
# Helper functions for displaying errors
########################################################################################################################
def print_error(y_true, y_pred, X_mnk):
    result_line = "top-{}: worse: {:>.3f} mean: {:>.3f}"
    for top_k in [1, 3, 5]:
        print(result_line.format(top_k,
                                 worse_rel_perf_loss_of_k(y_true, y_pred, top_k, X_mnk),
                                 mean_rel_perf_loss_of_k(y_true, y_pred, top_k, X_mnk)))


def plot_loss_histogram(y_true, y_pred, X_mnk, folder, model_name, decisive_score):
    import matplotlib.pyplot as plt

    # Get losses
    top_k = 3
    y = np.array(perf_loss(y_true, y_pred, top_k, X_mnk))

    # Losses-histogram
    num_bins = 50
    plt.hist(y, num_bins, facecolor='green', alpha=0.75)
    plt.xlabel("relative performance loss [%]")
    plt.ylabel("# occurrences")
    plt.title("Performance losses for top-k=" + str(top_k) + "decisive metric: " + decisive_score)
    plt.grid(True)
    plt.savefig(os.path.join(folder, model_name + "_result_losses.png"))


def plot_cv_scores(param_grid, scoring, results, folder, algo, model_name):
    # Inspired by http://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html
    import matplotlib.pyplot as plt
    plt.figure()
    plt.title("CV scores (" + algo + " , " + model_name + " )")
    for p in param_grid.keys():

        plt.xlabel("parameter: " + p)
        plt.ylabel("cv-score: relative perf loss [%] (mean over 5 folds)")
        ax = plt.gca()

        # Get the regular numpy array from the MaskedArray
        X_axis = np.array(results['param_' + p].values, dtype=float)

        for scorer, color in zip(scoring, ['b']*2 + ['g']*2 + ['k']*2):
            #for sample, style in (('train', '--'), ('test', '-')):
            sample = 'test'
            style = '-'
            sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
            sample_score_std = results['std_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                    alpha=1 if sample == 'test' else 0.7,
                    label="%s (%s)" % (scorer, sample))

            best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
            best_score = results['mean_test_%s' % scorer][best_index]

            # Plot a dotted vertical line at the best score for that scorer marked by x
            ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                    linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

            # Annotate the best score for that scorer
            ax.annotate("%0.2f" % best_score,
                        (X_axis[best_index], best_score + 0.005))

        plt.legend(loc="best")
        plt.grid(False)

        plt.savefig(os.path.join(folder, "cv_results_" + algo + "_" + model_name + "_" + p))


########################################################################################################################
# Main
########################################################################################################################
def main(argv):
    del argv  # unused

    ####################################################################################################################
    # Create folder to store results of this program
    read_from = FLAGS.in_folder
    algo = FLAGS.algo
    file_signature = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M")
    log = ''
    folder_ = os.path.join("model_selection", os.path.join(algo, file_signature))
    log_file = os.path.join(folder_, "log.txt")
    if not os.path.exists(folder_):
        os.makedirs(folder_)

    ####################################################################################################################
    # Read data
    print_and_log('Read training data X ...', log)
    X = pd.read_csv(os.path.join(read_from, 'train_all_' + algo + '_X.csv'), index_col=0)
    print_and_log('Data size X    : ' + str(sys.getsizeof(X)/10**6) + ' MB', log)

    print_and_log('Read training data Y ...', log)
    Y = pd.read_csv(os.path.join(read_from, 'train_all_' + algo + '_Y.csv'), index_col=0)
    print_and_log('Data size Y    : ' + str(sys.getsizeof(Y)/10**6) + ' MB', log)

    print_and_log('Read training data X_mnk ...', log)
    X_mnk = pd.read_csv(os.path.join(read_from, 'train_all_' + algo + '_X_mnk.csv'), index_col=0)
    print_and_log('Data size X_mnk: ' + str(sys.getsizeof(X_mnk)/10**6) + ' MB', log)

    n_features = len(list(X.columns))
    predictor_names = X.columns.values
    print_and_log('Predictor variables: (' + str(n_features) + ')', log)
    for i, p in enumerate(predictor_names):
        print_and_log("\t{:2}) {}".format(i+1, p), log)

    with open(log_file, 'w') as f:
        f.write(log)

    ####################################################################################################################
    # Decision tree
    from itertools import chain
    from sklearn.tree import DecisionTreeRegressor

    # Fixed parameters
    model_name_DT = "Decision_Tree"
    splitting_criterion = "mse"
    splitter = "best"
    max_features = None
    max_leaf_nodes = None

    # Parameters to optimize
    step_small = 1
    step_med = 3
    max_depth = chain(range(4, n_features, step_small), range(n_features, n_features*3, step_med))
    min_samples_split = chain(range(2, 5, step_small), range(8, n_features, step_med))
    min_samples_leaf = chain(range(2, 5, step_small), range(8, n_features, step_med))
    param_grid_DT = {
        'max_depth': list(max_depth),
        'min_samples_split': list(min_samples_split),
        'min_samples_leaf': list(min_samples_leaf)
    }

    # Tree model
    model_DT = DecisionTreeRegressor(
        criterion=splitting_criterion,
        splitter=splitter,
        min_samples_split=3,
        min_samples_leaf=1,
        max_depth=n_features,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes
    )

    ####################################################################################################################
    # Random Forest
    from sklearn.ensemble import RandomForestRegressor

    # Optimal parameters found from optimizing the DT's hyperparameters
    max_depth_from_DT_hpo = optimized_hyperparameters[algo]['max_depth']
    min_samples_split_from_DT_hpo = optimized_hyperparameters[algo]['min_samples_split']
    min_samples_leaf_from_DT_hpo = optimized_hyperparameters[algo]['min_samples_leaf']

    # Fixed parameters
    model_name_RF = "Random Forest"
    bootstrap = True
    max_depth = max_depth_from_DT_hpo
    min_samples_split = min_samples_split_from_DT_hpo
    min_samples_leaf = min_samples_leaf_from_DT_hpo
    max_features = 'sqrt'

    # Parameters to optimize
    step_big = 25
    n_estimators = chain(range(1, 10, step_small), range(25, 200, step_big))
    param_grid_RF = {'n_estimators': list(n_estimators)}

    # Random Forest model
    model_RF = RandomForestRegressor(
        criterion=splitting_criterion,
        n_estimators=30,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        bootstrap=bootstrap,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=-1
    )

    ####################################################################################################################
    # Tune hyperparameters
    tune = FLAGS.tune
    if tune == "DT":
        models_to_tune = [(model_DT, model_name_DT, param_grid_DT)]
    elif tune == "RF":
        models_to_tune = [(model_RF, model_name_RF, param_grid_RF)]
    elif tune == "all":
        models_to_tune = [(model_DT, model_name_DT, param_grid_DT), (model_RF, model_name_RF, param_grid_RF)]
    else:
        assert False, "Unrecognized option for tune:" + str(tune)

    for model, model_name, param_grid in models_to_tune:

        N_RUNS = FLAGS.nruns
        for i in range(N_RUNS):

            print('\n')
            print('###################################################################################################')
            print("Start hyperparameter optimization for model", model_name, ", run number", i+1, "/", N_RUNS)
            print('###################################################################################################')
            folder = os.path.join(folder_, model_name + "_" + str(i))
            log_file = os.path.join(folder, 'log.txt')
            log = ''
            if not os.path.exists(folder):
                os.makedirs(folder)

            ############################################################################################################
            # Train/test split
            from sklearn.model_selection import GroupShuffleSplit
            cv = GroupShuffleSplit(n_splits=2, test_size=0.2)
            train, test = cv.split(X, Y, groups=X_mnk['mnk'])
            train = train[0]
            test = test[0]
            X_train = X.iloc[train, :]  # train: use for hyperparameter optimization (via CV) and training
            X_mnk_train = X_mnk.iloc[train, :]
            Y_train = Y.iloc[train, :]
            X_test  = X.iloc[test,  :]  # test : use for evaluation of 'selected/final' model
            X_mnk_test = X_mnk.iloc[test, :]
            Y_test  = Y.iloc[test,  :]

            ############################################################################################################
            # Cross-validation splitter
            n_splits = FLAGS.splits
            test_size = 0.3
            cv = GroupShuffleSplit(n_splits=n_splits, test_size=test_size)

            ############################################################################################################
            # Feature selection
            from sklearn.feature_selection import RFECV
            predictor_names = X_train.columns.values
            print_and_log("Selecting optimal features among:\n" + str(predictor_names), log)
            if 'mnk' in X_train.columns.values:
                X_train = X_train.drop(["mnk"], axis=1)  # leftover from previous iteration (?)
            rfecv = RFECV(estimator=model, step=1, n_jobs=-1, cv=cv)
            fit = rfecv.fit(X_train, Y_train, X_mnk_train['mnk'])
            print_and_log("Optimal number of features : %d" % rfecv.n_features_, log)
            #print("Feature Ranking:\n", fit.ranking)
            selected_features_ = list()
            for i, f in enumerate(predictor_names):
                if fit.support_[i]:
                    selected_features_.append(f)
            features_by_ranking = sorted(zip(map(lambda x: round(x, 4), fit.ranking_), predictor_names), reverse=False)
            print_and_log("Selected Features:", log)
            for feature in selected_features_:
                print_and_log("\t{}".format(feature), log)

            #print("Features sorted by ranking:")
            #for f in features_by_ranking:
            #    print(f)

            features_to_drop = [f for f in predictor_names if f not in selected_features_]
            #print('Dropping features:\n', features_to_drop)
            for f in features_to_drop:
                X_train = X_train.drop([f], axis=1)
                X_test = X_test.drop([f], axis=1)

            ############################################################################################################
            # Hyperparameter optimization

            # Grid search
            from sklearn.model_selection import GridSearchCV
            print_and_log('--------------------------------------------------------------------------------------', log)
            print_and_log('Parameter grid:\n' + str(param_grid), log)
            X_train["mnk"] = X_mnk_train['mnk']  # add to X-DataFrame (needed for scoring function)
            scoring = {
                'worse_top-1': worse_case_scorer_top1, 'mean_top-1': mean_scorer_top1,
                'worse_top-3': worse_case_scorer_top3, 'mean_top-3': mean_scorer_top3,
                'worse_top-5': worse_case_scorer_top5, 'mean_top-5': mean_scorer_top5
            }
            decisive_score = 'worse_top-3'  # REALLY?? Or should I choose a different one?
            gs = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                pre_dispatch=8,
                n_jobs=-1,
                verbose=1,
                refit=decisive_score,
                return_train_score=False  # incompatible with ignore_in_fit
            )
            print_and_log('--------------------------------------------------------------------------------------', log)
            gs.fit(X_train, Y_train, X_mnk_train['mnk'], ignore_in_fit=["mnk"])

            # Get results, pickle them
            cv_results = pd.DataFrame(gs.cv_results_)
            safe_pickle(cv_results, os.path.join(folder, "cv_rests.p"))

            ############################################################################################################
            # Model evaluation
            print_and_log('--------------------------------------------------------------------------------------', log)
            predictor_names = X_train.columns.values
            print_and_log('Predictor variables:', log)
            for p in predictor_names:
                print_and_log("\t{}".format(p), log)

            print_and_log("\nDecisive metric:", decisive_score)

            print_and_log("\nBest parameters set found on development set:", log)
            print_and_log(gs.best_params_, log)

            print_and_log("\nBest estimator:", log)
            best_estimator = gs.best_estimator_
            print_and_log(best_estimator, log)
            print_and_log('--------------------------------------------------------------------------------------', log)

            # Training error
            X_train = X_train.drop(['mnk'], axis=1)
            best_estimator.fit(X_train, Y_train)
            y_train_pred = best_estimator.predict(X_train)
            print_and_log('\nTraining error: (train&val)', log)
            print_error(Y_train, y_train_pred, X_mnk_train)

            # Test error
            y_test_pred = best_estimator.predict(X_test)
            print_and_log('\nTesting error:', log)
            print_error(Y_test, y_test_pred, X_mnk_test)

            # Print histogram for "best" estimator
            print_and_log('Plot result histogram:', log)
            plot_loss_histogram(Y_test, y_test_pred, X_mnk_test, folder, model_name, decisive_score)

            # Plot CV results by evaluation metric
            print_and_log('Plot CV scores:', log)
            plot_cv_scores(param_grid, scoring, cv_results, folder, algo, model_name)

            ############################################################################################################
            # Print log
            with open(log_file, 'w') as f:
                f.write(log)

########################################################################################################################
# Run
########################################################################################################################
if __name__ == '__main__':
    app.run(main)
