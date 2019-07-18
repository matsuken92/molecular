#!/usr/bin/env python
# coding: utf-8

import os
import sys
import os
import datetime
import json
import gc
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from numba import jit

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold
import warnings
warnings.filterwarnings("ignore")

sys.path.append('..')
from lib.line_notif import send_message
from lib.utils import current_time, unpickle, to_pickle, reduce_mem_usage

import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

DATA_VERSION = __file__.split("_")[-2]
TRIAL_NO = __file__.split("_")[-1].replace(".py", "")

sys.path.append(".")
import importlib

use_cols = importlib.import_module(f'use_cols_{DATA_VERSION}_{TRIAL_NO}')
use_cols.good_columns += [c for c in use_cols.rdkit_cols if c != 'id']
use_cols.good_columns += [c for c in use_cols.babel_cols if c != 'id']
use_cols.good_columns = np.unique(use_cols.good_columns).tolist()
# use_cols = importlib.import_module(f'use_cols')
print(use_cols.good_columns)


type_name ={ 0 : "1JHC",
             1 : "1JHN",
             2 : "2JHC",
             3 : "2JHH",
             4 : "2JHN",
             5 : "3JHC",
             6 : "3JHH",
             7 : "3JHN", }


def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    """
    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling
    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric
    """
    maes = (y_true - y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()


def hold_out_lgb_validation(X, y, params, eval_metric='mae', columns=None,
                            plot_feature_importance=False,
                            verbose=10000, early_stopping_rounds=200, n_estimators=50000, ):
    columns = X.columns if columns is None else columns

    # to set up scoring parameters
    metrics_dict = {'mae': {'lgb_metric_name': 'mae',
                            'catboost_metric_name': 'MAE',
                            'sklearn_scoring_function': metrics.mean_absolute_error},
                    'group_mae': {'lgb_metric_name': 'mae',
                                  'catboost_metric_name': 'MAE',
                                  'scoring_function': group_mean_log_mae},
                    'mse': {'lgb_metric_name': 'mse',
                            'catboost_metric_name': 'MSE',
                            'sklearn_scoring_function': metrics.mean_squared_error}
                    }

    result_dict = {}

    X_train, X_valid, y_train, y_valid = train_test_split(X[columns], y, test_size=0.1, random_state=42)
    eval_result = {}
    callbacks = [lgb.record_evaluation(eval_result)]
    model:lgb.LGBMRegressor = lgb.LGBMRegressor(**params, n_estimators=n_estimators, n_jobs=-1, importance_type='gain')
    print(model)
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_valid, y_valid)],
              eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
              verbose=verbose, early_stopping_rounds=early_stopping_rounds,
              callbacks=callbacks)

    y_pred_valid = model.predict(X_valid)

    if eval_metric != 'group_mae':
        score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)
    else:
        score = metrics_dict[eval_metric]['scoring_function'](y_valid, y_pred_valid, X_valid['type'])

    if plot_feature_importance:
        # feature importance
        feature_importance = pd.DataFrame()
        feature_importance["feature"] = columns
        feature_importance["importance"] = model.feature_importances_
    else:
        feature_importance = None

    try:
        cv_score_msg = f'{DATA_VERSION}_{TRIAL_NO}' + f' HOLD_OUT score: {score:.4f} .'
        print(cv_score_msg)
        if not DEBUG and LINE_MSG:
            send_message(cv_score_msg)
    except Exception as e:
        print(e)
        pass

    result_dict["model"] = model
    result_dict['y_pred_valid'] = pd.DataFrame(y_pred_valid, index=X_valid.index, columns=["scalar_coupling_constant"])
    result_dict['score'] = score
    result_dict["importance"] = feature_importance
    result_dict["eval_result"] = eval_result
    result_dict["best_iteration"] = model.best_iteration_
    return result_dict


def train_lgb_regression_alldata(X, X_test, y, params, eval_metric='mae', columns=None,
                             plot_feature_importance=False, model=None,
                             verbose=10000, n_estimators=50000, mol_type=-1):
    """
    A function to train a variety of regression models.
    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.

    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: y - target
    :params: model_type - type of model to use
    :params: eval_metric - metric to use
    :params: columns - columns to use. If None - use all columns
    :params: plot_feature_importance - whether to plot feature importance of LGB
    :params: model - sklearn model, works only for "sklearn" model type

    """
    columns = X.columns if columns is None else columns
    X_test = X_test[columns]
    X_train, y_train = X[columns], y

    # to set up scoring parameters
    metrics_dict = {'mae': {'lgb_metric_name': 'mae',
                            'catboost_metric_name': 'MAE',
                            'sklearn_scoring_function': metrics.mean_absolute_error},
                    'group_mae': {'lgb_metric_name': 'mae',
                                  'catboost_metric_name': 'MAE',
                                  'scoring_function': group_mean_log_mae},
                    'mse': {'lgb_metric_name': 'mse',
                            'catboost_metric_name': 'MSE',
                            'sklearn_scoring_function': metrics.mean_squared_error}
                    }

    result_dict = {}

    model = lgb.LGBMRegressor(**params, n_estimators=n_estimators, n_jobs=-1, importance_type='gain')
    print(model)
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train)],
              eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
              verbose=verbose)

    result_dict['prediction'] = model.predict(X_test)

    if plot_feature_importance:
        # feature importance
        feature_importance = pd.DataFrame()
        feature_importance["feature"] = columns
        feature_importance["importance"] = model.feature_importances_
        result_dict['feature_importance'] = feature_importance

    return result_dict

def train_model_regression(X, X_test, y, params, folds, model_type='lgb', eval_metric='mae', columns=None,
                           plot_feature_importance=False, model=None,
                           verbose=10000, early_stopping_rounds=200, n_estimators=50000, mol_type=-1,
                           fold_group=None):
    """
    A function to train a variety of regression models.
    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.

    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: y - target
    :params: folds - folds to split data
    :params: model_type - type of model to use
    :params: eval_metric - metric to use
    :params: columns - columns to use. If None - use all columns
    :params: plot_feature_importance - whether to plot feature importance of LGB
    :params: model - sklearn model, works only for "sklearn" model type

    """
    columns = X.columns if columns is None else columns
    X_test = X_test[columns]

    # to set up scoring parameters
    metrics_dict = {'mae': {'lgb_metric_name': 'mae',
                            'catboost_metric_name': 'MAE',
                            'sklearn_scoring_function': metrics.mean_absolute_error},
                    'group_mae': {'lgb_metric_name': 'mae',
                                  'catboost_metric_name': 'MAE',
                                  'scoring_function': group_mean_log_mae},
                    'mse': {'lgb_metric_name': 'mse',
                            'catboost_metric_name': 'MSE',
                            'sklearn_scoring_function': metrics.mean_squared_error}
                    }

    result_dict = {}

    # out-of-fold predictions on train data
    oof = np.zeros(len(X))

    # averaged predictions on train data
    prediction = np.zeros(len(X_test))

    # list of scores on folds
    scores = []
    feature_importance = pd.DataFrame()
    model_list = []

    # split and train on folds
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, groups=fold_group)):
        print(f'Fold {fold_n + 1} started at {time.ctime()}')
        if type(X) == np.ndarray:
            X_train, X_valid = X[columns][train_index], X[columns][valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params, n_estimators=n_estimators, n_jobs=-1, importance_type='gain')
            print(model)
            model.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_valid, y_valid)],
                      eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                      verbose=verbose, early_stopping_rounds=early_stopping_rounds)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            params["objective"] = "reg:linear"
            params["eval_metric"] = metrics_dict[eval_metric]['lgb_metric_name']
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
                              verbose_eval=verbose, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns),
                                         ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)

            y_pred_valid = model.predict(X_valid).reshape(-1, )
            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)
            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')
            print('')

            y_pred = model.predict(X_test).reshape(-1, )

        if model_type == 'cat':
            model = CatBoostRegressor(iterations=20000, eval_metric=metrics_dict[eval_metric]['catboost_metric_name'],
                                      **params,
                                      loss_function=metrics_dict[eval_metric]['catboost_metric_name'])
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True,
                      verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)

        oof[valid_index] = y_pred_valid.reshape(-1, )
        if eval_metric != 'group_mae':
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))
        else:
            scores.append(metrics_dict[eval_metric]['scoring_function'](y_valid, y_pred_valid, X_valid['type']))

        prediction += y_pred

        if model_type == 'lgb' and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
        model_list += [model]

    if model_type == 'lgb' and plot_feature_importance:
        result_dict['importance'] = feature_importance

    prediction /= folds.n_splits
    try:
        cv_score_msg = f'{DATA_VERSION}_{TRIAL_NO}' +' CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores))
        print(cv_score_msg)
        send_message(cv_score_msg)
    except Exception as e:
        print(e)
        pass

    result_dict["models"] = model_list
    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores

    # if model_type == 'lgb':
    #     if plot_feature_importance:
    #         feature_importance["importance"] /= folds.n_splits
    #         cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
    #             by="importance", ascending=False)[:50].index
    #
    #         best_features = feature_importance.loc[feature_importance.feature.isin(cols)]
    #
    #         plt.figure(figsize=(16, 12));
    #         sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
    #         plt.title('LGB Features (avg over folds)');
    #         feature_importance.to_csv(log_path/f"importance_{mol_type}.csv")
    #

    return result_dict


def get_prev_train_test_data(prev_data_version=None, prev_trial_no=None):

    file_folder = '../input'
    train = pd.read_csv(f'{file_folder}/train.csv')
    sample_loaded = False
    prev_folder = f"../processed/{prev_data_version}/{prev_data_version}_{prev_trial_no}"
    if DEBUG:
        # v003_033
        train_path = Path(f"{prev_folder}/train_concat_{prev_data_version}_{prev_trial_no}_yiemon_123J_sampled.pkl")
        test_path = Path(f"{prev_folder}/test_concat_{prev_data_version}_{prev_trial_no}_yiemon_123J_sampled.pkl")

        if train_path.exists() and test_path.exists():
            print("sample loading")
            train = unpickle(train_path)
            test = unpickle(test_path)
            sample_loaded = True
            print("sample load finish")

    if not sample_loaded:
        print(f"loading previous dataest")
        print("train loading")
        train: pd.DataFrame = unpickle(f"{prev_folder}/train_concat_{prev_data_version}_{prev_trial_no}_yiemon_123J.pkl", )
        assert "scalar_coupling_constant" in train.columns
        print("test loading")
        test: pd.DataFrame = unpickle(f"{prev_folder}/test_concat_{prev_data_version}_{prev_trial_no}_yiemon_123J.pkl", )
        print(f"loading finished")

    if DEBUG and not sample_loaded:
        n_sample = 5000
        print(f"sampling {n_sample} rows.")
        train = train.sample(n=n_sample)
        test = test.sample(n=n_sample)
        Path(f"../processed/{prev_data_version}/{prev_data_version}_{prev_trial_no}").mkdir(parents=True,
                                                                                            exist_ok=True)
        to_pickle(f"{prev_folder}/train_concat_{prev_data_version}_{prev_trial_no}_yiemon_123J_sampled.pkl", train)
        to_pickle(f"{prev_folder}/test_concat_{prev_data_version}_{prev_trial_no}_yiemon_123J_sampled.pkl", test)
        print("saved.")

    return train, test

####################################################################################################
# Settings
print(f"STARTING : {current_time()}")

DEBUG = False
MODEL_TYPE = "lgb"
LINE_MSG = True
GROUP_K_FOLD = True

TRAIN_ALL_DATA = True
CV_FOLD = False

assert int(TRAIN_ALL_DATA) + int(CV_FOLD) != 2, f"TRAIN_ALL_DATA:{TRAIN_ALL_DATA}, CV_FOLD:{CV_FOLD}"
assert MODEL_TYPE in ["lgb", "mlp", "lgb_holdout"]
if DEBUG:
    print("=" * 30 + " DEBUG " + "=" * 30)

# DATA_VERSION = "v001"
# TRIAL_NO = "006"

debug_str = "_debug" if DEBUG else ""

####################################################################################################
# path setting
save_path = Path(f"../processed/{DATA_VERSION}")
save_path.mkdir(parents=True, exist_ok=True)
model_path = Path(f"../model/{DATA_VERSION}_{TRIAL_NO}{debug_str}")
model_path.mkdir(parents=True, exist_ok=True)
submit_path = Path(f"../submit/{DATA_VERSION}_{TRIAL_NO}{debug_str}")
submit_path.mkdir(parents=True, exist_ok=True)
log_path = Path(f"../log/{DATA_VERSION}_{TRIAL_NO}{debug_str}")
log_path.mkdir(parents=True, exist_ok=True)

####################################################################################################
# Data Loading
train, test = get_prev_train_test_data(prev_data_version="v003", prev_trial_no="033")
use_cols_revised = [c for c in use_cols.good_columns if c not in use_cols.remove_cols]
use_cols_revised = [c for c in use_cols_revised if c in train.columns]
# high_importance_dict = get_high_importance_cols(data_version="v003", trial_version="045", verbose=False)

mol_name = train.molecule_name

###################################################################################################
# final data preparation for train
X: pd.DataFrame = train[use_cols_revised].copy()
y: pd.Series = train['scalar_coupling_constant']
y_fc: pd.Series = train['fc']
X_test: pd.DataFrame = test[use_cols_revised].copy()
print(f"X.shape: {X.shape}, X_test.shape: {X_test.shape}")


# export colnames
pd.DataFrame({"columns": X.columns.tolist()}).to_csv(log_path / f"use_cols.csv")

####################################################################################################
# Model Fitting
print("start fitting")
n_fold = 5

if GROUP_K_FOLD:
    folds = GroupKFold(n_splits=n_fold)
else:
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)

#########################################################################################################
# 1st layer model
#seed_base = [0, 2019, 71, 1228, 1988, 1879, 92, 3018, 1234, 185289]
seed_base = [2019]
seed_list = np.array(seed_base) + 50
#seed_list = np.array(seed_base) + 51
#seed_list = np.array(seed_base) + 52
#seed_list = np.array(seed_base) + 53
#seed_list = np.array(seed_base) + 54

current_seed = -1


def train_main(seed, type_):
    print(f"==================== seed: {seed} ====================")
    params = { #'num_leaves': 128,
              'min_child_samples': 79,
              'objective': 'regression',
              'max_depth': -1, #9,
              'learning_rate': 0.2,
              "boosting_type": "gbdt",
              "subsample_freq": 1,
              "subsample": 0.9,
              "metric": 'mae',
              "verbosity": -1,
              'reg_alpha': 0.1,
              'reg_lambda': 0.3,
              'colsample_bytree': 1.0,
              'num_threads' : -1,
             }

    params["seed"] = seed
    params["bagging_seed"] = seed + 1
    params["feature_fraction_seed"] = seed + 2

    n_estimators = 5 #10000
    params["num_leaves"] = 256
    if DEBUG:
        n_estimators  = 5

    X_short = pd.DataFrame({'ind': list(X.index), 'type': X['type'].values,
                            'oof': [0] * len(X), 'target': y.values, 'fc':y_fc.values})

    X_short_test = pd.DataFrame({'ind': list(X_test.index), 'type': X_test['type'].values,
                                 'prediction': [0] * len(X_test)})

    print(f'{current_time()} Training of type {type_} / {X["type"].unique()}')
    X_t = X.loc[X['type'] == type_]
    X_test_t = X_test.loc[X_test['type'] == type_]
    y_fc_t = X_short.loc[X_short['type'] == type_, 'fc']
    y_t = X_short.loc[X_short['type'] == type_, 'target']
    mol_name_t = mol_name.loc[X['type'] == type_][X_t.index] if GROUP_K_FOLD else None
    print(f"X_t.shape: {X_t.shape}, X_test_t.shape: {X_test_t.shape}, y_t.shape: {y_t.shape}")

    ########################################################################################################
    # fc
    print("="*30 + " fc " +  "="*30 )
    result_dict_lgb1 = train_model_regression(X=X_t,
                                              X_test=X_test_t,
                                              y=y_fc_t,
                                              params=params,
                                              folds=folds,
                                              model_type='lgb',
                                              eval_metric='group_mae',
                                              plot_feature_importance=False,
                                              verbose=1000,
                                              early_stopping_rounds=200,
                                              n_estimators=n_estimators,
                                              fold_group=mol_name.values)

    X['oof_fc'] = result_dict_lgb1['oof']
    X_test['oof_fc'] = result_dict_lgb1['prediction']

    to_pickle(submit_path/f"train_oof_fc_{DATA_VERSION}_{TRIAL_NO}_{type_}_{seed}.pkl", X['oof_fc'])
    to_pickle(submit_path/f"test_oof_fc_{DATA_VERSION}_{TRIAL_NO}_{type_}_{seed}.pkl", X_test['oof_fc'])
    to_pickle(model_path/f"first_model_list_{DATA_VERSION}_{TRIAL_NO}_{type_}_{seed}.pkl", result_dict_lgb1["models"])

    #########################################################################################################
    # 2nd layer model
    params["seed"] = seed + 3
    params["bagging_seed"] = seed + 4
    params["feature_fraction_seed"] = seed + 5
    params["num_leaves"] = 256 # num_leaves_dict[t]
    start_time = current_time()
    bairitsu = 256 / params["num_leaves"]
    n_estimators = 5#int(15000 * bairitsu)

    if DEBUG:
        n_estimators = 5

    if TRAIN_ALL_DATA:
        print("============= 2nd layer TRIAN ALL DATA ================")
        result_dict = train_lgb_regression_alldata(X=X_t,
                                                   X_test=X_test_t,
                                                   y=y_t,
                                                   params=params,
                                                   eval_metric='group_mae',
                                                   plot_feature_importance=True,
                                                   verbose=5000,
                                                   n_estimators=int(n_estimators*1.6),
                                                   mol_type=type_)

        X_short_test.loc[X_short_test['type'] == type_, 'prediction'] = result_dict['prediction']
        X_short_test.to_csv(submit_path / f"sub_{DATA_VERSION}_{TRIAL_NO}_{type_}_{seed}.csv")

    elif CV_FOLD:
        print("============= 2nd layer CV ================")
        result_dict = train_model_regression(X_t, X_test_t, y_t,
                                             params, folds,
                                             model_type='lgb',
                                             eval_metric='mae',
                                             columns=None,
                                             plot_feature_importance=True,
                                             model=None,
                                             verbose=1000,
                                             early_stopping_rounds=200,
                                             n_estimators=n_estimators,
                                             mol_type=-1,
                                             fold_group=mol_name_t)

        result_dict["start_time"] = start_time
        result_dict["n_estimator"] = n_estimators
        result_dict["X_t_len"] = X_t.shape[0]
        result_dict["type"] = type_
        result_dict["type_name"] = type_name[type_]

        X_short.loc[X_short['type'] == type_, 'oof'] = result_dict['oof']
        X_short.to_csv(submit_path / f"oof_{DATA_VERSION}_{TRIAL_NO}_{type_}_{seed}.csv")

        X_short_test.loc[X_short_test['type'] == type_, 'prediction'] = result_dict['prediction']
        X_short_test.to_csv(submit_path / f"sub_{DATA_VERSION}_{TRIAL_NO}_{type_}_{seed}.csv")

    else:
        print("============= 2nd layer hold out ================")
        result_dict = hold_out_lgb_validation(X=X_t,
                                                  y=y_t,
                                                  params=params,
                                                  eval_metric='mae',
                                                  plot_feature_importance=True,
                                                  verbose=5000,
                                                  early_stopping_rounds=200,
                                                  n_estimators=n_estimators)

        result_dict["start_time"] = start_time
        result_dict["n_estimator"] = n_estimators
        result_dict["X_t_len"] = X_t.shape[0]
        result_dict["type"] = type_
        result_dict["type_name"] = type_name[type_]

        eval_result: list = result_dict["eval_result"]["valid_1"]["l1"]
        training_log_df: pd.DataFrame = pd.DataFrame(eval_result, index=np.arange(len(eval_result)) + 1)
        training_log_df.columns = ["l1"]
        training_log_df.index.name = "iter"
        training_log_df.to_csv(log_path / f"train_log_{DATA_VERSION}_{TRIAL_NO}_{type_}.csv")

        to_pickle(model_path / f"hold_out_model_{DATA_VERSION}_{TRIAL_NO}_{type_}_{seed}.pkl", result_dict["model"])
    #
    #
    #     to_pickle(log_path / f"result_dict_{type_}_{seed}.pkl", result_dict)
    #     importance_path = log_path / f'importance_{DATA_VERSION}_{TRIAL_NO}_{type_}_{seed}.csv'
    #     result_dict["importance"].to_csv(importance_path, index=True)
    #
    # for type_, s in zip(X['type'].unique(), score_list):
    #     print(f"type {type_}, score: {s:0.5f}")

    if TRAIN_ALL_DATA or CV_FOLD:
        #########################################################################################################
        # create oof & submission file.
        sub = pd.read_csv(f'../input/sample_submission.csv')
        sub['scalar_coupling_constant'] = X_short_test['prediction']
        sub.to_csv(submit_path / f'submission_t_{DATA_VERSION}_{TRIAL_NO}_{seed}.csv', index=False)
        print(sub.head())
        send_message(f"finish all_data train_{DATA_VERSION}_{TRIAL_NO}_{seed}")

    if CV_FOLD:
        oof_log_mae = group_mean_log_mae(X_short['target'], X_short['oof'], X_short['type'], floor=1e-9)
        print(f"oof_log_mae: {oof_log_mae}")

        df_oof = pd.DataFrame(index=train.id)
        df_oof["scalar_coupling_constant"] = X_short['oof']
        df_oof.to_csv(submit_path/f'oof_{DATA_VERSION}_{TRIAL_NO}_{seed}.csv', index=True)
        send_message(f"finish train_{DATA_VERSION}_{TRIAL_NO}_{seed}, oof_log_mae: {oof_log_mae}")

print(f"finished. : {current_time()}")