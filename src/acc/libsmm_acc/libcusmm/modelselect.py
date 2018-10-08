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

from absl import flags, app
flags.DEFINE_string('in_folder', 'tune_big/', 'Folder from which to read data')
flags.DEFINE_string('algo', 'tiny', 'Algorithm to train on')
flags.DEFINE_string('tune', 'DT', 'Model to tune (Options: DT, RF, all)')
FLAGS = flags.FLAGS

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


########################################################################################################################
# Fitting (helper function)
########################################################################################################################
def plot_feat_importance(X, model):
    import matplotlib.pyplot as plt
    importances = model.feature_importances_
    names = X.columns
    # std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    print("Feature ranking:")
    for f in range(X.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, names[indices[f]], importances[indices[f]]))

    plt.rcdefaults()
    fig, ax = plt.subplots()

    ax.set_title("Feature importances")
    ax.barh(range(X.shape[1]), importances[indices],
            color="g",
            # yerr=std[indices],
            align="center")
    ax.set_yticks(np.arange(len(importances)))
    ax.set_yticklabels(names[indices])
    ax.invert_yaxis()
    # plt.set_ylim([-1, X.shape[1]])
    plt.show()


def n_features_vs_cv_score(rfecv):
    import matplotlib.pyplot as plt
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()


########################################################################################################################
# Custom loss functions
########################################################################################################################
def perf_loss(y_true, y_pred, top_k, X_mnk):
    y_true_squared = np.copy(y_true)
    y_true = np.sqrt(y_true)
    y_pred = np.sqrt(y_pred)
    perf_losses = list()

    mnks = np.unique(X_mnk['mnk'].values)
    for mnk in mnks:
        idx_mnk = np.where(X_mnk == mnk)[0].tolist()
        assert (len(idx_mnk) > 0), "idx_mnk is empty"
        #idx_mnk = list(set(range(len(y_true))).intersection(set(idx_mnk)))
        #if len(idx_mnk) <= 0:
        #    continue
        #else:

        # Find rows in y_true where element = X_mnk['perf_squared'] elements -> keep those indices
        idx_mnk_for_y = list()
        for idx in idx_mnk:
            found = np.where(y_true_squared == X_mnk['perf_squared'].iloc[idx])[0].tolist()
            if len(found) != 1:
                print('shit')
                assert len(found) == 1, "Found indices:\n" + str(found)
            idx_mnk_for_y.append(found[0])

        idx_mnk = np.array(idx_mnk_for_y)
        y_mnk = y_true.iloc[idx_mnk]
        y_pred_mnk = y_pred[idx_mnk]
        top_k_idx = np.argpartition(-y_pred_mnk, top_k)[:top_k]
        y_correspmax = y_mnk.iloc[top_k_idx]

        # Max. performances
        maxperf = float(y_mnk.max(axis=0))  # true max. performance
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


def print_error(y_true, y_pred, X_mnk):
    result_line = "top-{}: worse: {0:>.3f} mean: {0:>.3f}"
    for top_k in [1, 3, 5]:
        print(result_line.format(top_k,
                                 worse_rel_perf_loss_of_k(y_true, y_pred, top_k, X_mnk),
                                 mean_rel_perf_loss_of_k(y_true, y_pred, top_k, X_mnk)))


########################################################################################################################
# Main
########################################################################################################################
def main(argv):
    del argv  # unused

    ####################################################################################################################
    # Read data
    read_from = FLAGS.in_folder
    algo = FLAGS.algo
    out_folder = 'model_selection_' + algo
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    print('Read training data X ...')
    X = pd.read_csv(os.path.join(read_from, 'train_all_' + algo + '_X.csv'), index_col=0)
    print('Data size X    :', sys.getsizeof(X)/10**6, 'MB')

    print('Read training data Y ...')
    Y = pd.read_csv(os.path.join(read_from, 'train_all_' + algo + '_Y.csv'), index_col=0)
    #Y = Y['perf_squared']
    print('Data size Y    :', sys.getsizeof(Y)/10**6, 'MB')

    print('Read training data X_mnk ...')
    X_mnk = pd.read_csv(os.path.join(read_from, 'train_all_' + algo + '_X_mnk.csv'), index_col=0)
    X_mnk['perf_squared'] = Y['perf_squared']
    print('Data size X_mnk:', sys.getsizeof(X_mnk)/10**6, 'MB')

    mnks = pickle.load(open(os.path.join(read_from, 'train_all_' + algo + '_mnks.csv'), 'rb'))
    n_features = len(list(X.columns))
    predictor_names = X.columns.values
    maxperf = float(Y.max(axis=0))
    print('Predictor variables: (', n_features, ')\n', X.columns)
    print("\nOutcome variable:\n with max. value =", maxperf)


    ####################################################################################################################
    # Decision tree
    from itertools import chain
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import GridSearchCV

    # Parameters
    step = 2
    model_name = "Decision_Tree"
    splitting_criterion = "mse"  # the performances are squared already, so we use mean absolute error (prev: "mse")
    splitter = "best"            # other option: "random"
    max_depth = chain(range(12, n_features, step), range(n_features, n_features*2, 10))
    min_samples_split = 3        # [to tune by CV?] (prev: min_samples_split=20)
    min_samples_leaf = 1         # [to tune by CV?]

    # Tree model
    model_DT = DecisionTreeRegressor(
        criterion=splitting_criterion,
        splitter=splitter,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=None
    )

    ####################################################################################################################
    # Random Forest
    from sklearn.ensemble import RandomForestRegressor

    # Parameters
    n_estimators = 50
    bootstrap = True

    model_name = "Random Forest"
    model_RF = RandomForestRegressor(
        n_estimators=n_estimators,
        criterion=splitting_criterion,
        #splitter=splitter,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        max_features='sqrt',
        bootstrap=bootstrap,
        n_jobs=-1
    )
    print(model_RF)

    for model in [model_DT]: # [model_DT, model_RF]:

        N_RUNS = 1
        for i in range(N_RUNS):

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
            n_splits = 5
            test_size = 0.3
            cv = GroupShuffleSplit(n_splits=n_splits, test_size=test_size)

            ############################################################################################################
            # Feature selection
            from sklearn.feature_selection import RFECV
            rfecv = RFECV(estimator=model, step=1, verbose=2, n_jobs=-1, cv=cv, scoring=mean_scorer_top3)
            fit = rfecv.fit(X_train, Y_train, X_mnk_train['mnk'])
            print("Optimal number of features : %d" % rfecv.n_features_)
            print("Num Features to select:", fit.n_features_)
            print("All Features:\n", predictor_names)
            print("Feature Ranking:\n", fit.ranking_)
            selected_features = list()
            for i, f in enumerate(predictor_names):
                if fit.support_[i]:
                    selected_features.append(f)
            print("Selected Features:\n", selected_features)

            features_by_ranking = sorted(zip(map(lambda x: round(x, 4), fit.ranking_), predictor_names), reverse=False)
            print("Features sorted by ranking:")
            for f in features_by_ranking:
                print(f)

            features_to_drop = [f for f in predictor_names if f not in selected_features]
            print('Dropping features:\n', features_to_drop)
            for f in features_to_drop:
                X_train = X_train.drop([f], axis=1)
                X_test  = X_test.drop([f], axis=1)

            ############################################################################################################
            # Hyperparameter optimization

            # Parameter search space
            param_grid = [{'max_depth': list(max_depth)}]

            # Grid search
            from sklearn.metrics import mean_absolute_error
            from sklearn.metrics import mean_squared_error
            model_DT_hpo = GridSearchCV(
                estimator=model_DT,
                param_grid=param_grid,
                cv=cv,
                scoring={
                    'mean-absolute': mean_absolute_error, 'mean-squared': mean_squared_error
                },
                pre_dispatch=8,
                n_jobs=-1,
                verbose=2,
                refit=False
            )
            model_DT_hpo.fit(X_train, Y_train, X_mnk_train)

            # Get results
            print("Grid scores on development set:")
            cv_results = pd.DataFrame(model_DT_hpo.cv_results_)
            #cv_results = pd.DataFrame(model_DT_hpo.cv_results_.items())
            print(cv_results)
            print("Best estimator:")
            print(model_DT_hpo.best_estimator_)
            print("Best parameters set found on development set:")
            print(model_DT_hpo.best_params_)
            print("Best score found on development set:")
            print(model_DT_hpo.best_score_)

            ############################################################################################################
            # Model evaluation
            print("Start evaluating", model_name, "run #", i)
            print(model)
            model_trained_file = os.path.join(read_from, model_name + "_" + algo + "_" + str(i) + ".p")
            model.fit(X_train, Y_train)
            safe_pickle(model, model_trained_file)

            # Training error
            y_train_pred = model.predict(X_train)
            print('Training error:')
            print_error(Y_train, y_train_pred, X_mnk)

            # Test error
            y_test_pred = model.predict(X_test)
            print('Testing error:')
            print_error(Y_test, y_test_pred, X_mnk)


########################################################################################################################
# Run
########################################################################################################################
if __name__ == '__main__':
    app.run(main)
