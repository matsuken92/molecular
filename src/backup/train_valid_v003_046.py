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
from lib.utils import RankGaussScalar

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

import keras
from keras.models import Model
from keras.layers import Dense, BatchNormalization, Dropout, Input
from keras.callbacks import ModelCheckpoint


DATA_VERSION = __file__.split("_")[-2]
TRIAL_NO = __file__.split("_")[-1].replace(".py", "")

sys.path.append(".")
import importlib

use_cols = importlib.import_module(f'use_cols_{DATA_VERSION}_{TRIAL_NO}')
use_cols.good_columns += [c for c in use_cols.rdkit_cols if c != 'id']
use_cols.good_columns += [c for c in use_cols.babel_cols if c != 'id']
use_cols.good_columns += [c for c in use_cols.yiemon_cols if c != 'id']
use_cols.good_columns += use_cols.feat_2J_atom_info_cols
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

def build_model(ncols, dropout_rate=0.25, activation='relu', start_neurons=256, ):
    inputs = []

    input_numeric = Input(shape=(ncols,), name='main_input')
    inputs.append(input_numeric)

    x = Dense(start_neurons, activation=activation)(input_numeric)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(start_neurons // 2, activation=activation)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(start_neurons // 4, activation=activation)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(start_neurons // 8, activation=activation)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate / 2)(x)

    main_output = Dense(1, activation='linear', name='main_output')(x)

    model = Model(inputs=inputs, outputs=main_output)

    # opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # opt = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    opt = 'adam'
    model.compile(loss="mae", optimizer=opt, metrics=['mae'])
    return model


def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    """
    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling
    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric
    """
    maes = (y_true - y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()


def hold_out_nn_validation(X, y, params, eval_metric='mae', columns=None, model_path="./"):
    # params = {"epochs":100,
    #           "batch_size": 256,
    #           "dropout_rate": 0.3,
    #           "activation": "relu",
    #           "start_neurons": 512,
    #           "pred_batch_size": 1024}

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

    columns = X.columns if columns is None else columns
    X_train, X_valid, y_train, y_valid = train_test_split(X[columns], y, test_size=0.1, random_state=42)

    model: Model = build_model(ncols=X_train.shape[1],
                               dropout_rate=params["dropout_rate"],
                               activation=params["activation"],
                               start_neurons=params["start_neurons"], )

    es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=params["patience"],
                                          verbose=1, mode='auto',
                                          restore_best_weights=True)
    checkPoint = ModelCheckpoint(str(model_path / f"mlp_model_{DATA_VERSION}_{TRIAL_NO}.model"),
                                 monitor='val_loss', mode='min', save_best_only=True, verbose=0)
    print(f"X_train: {X_train.shape}, X_valid: {X_valid.shape}")
    model.fit(x=X_train, y=y_train,
              validation_data=[X_valid, y_valid],
              epochs=params["epochs"],
              verbose=2,
              shuffle=True,
              callbacks=[es_cb, checkPoint])

    model.save_weights(str(model_path / f"mlp_model_{DATA_VERSION}_{TRIAL_NO}_best.model"))
    y_pred_valid = model.predict(X_valid, batch_size=params["pred_batch_size"])

    if eval_metric != 'group_mae':
        score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)
    else:
        score = metrics_dict[eval_metric]['scoring_function'](y_valid, y_pred_valid, X_valid['type'])

    result_dict = {}
    result_dict["model"] = model
    result_dict['y_pred_valid'] = pd.DataFrame(y_pred_valid, index=X_valid.index, columns=["scalar_coupling_constant"])
    result_dict['score'] = score
    return result_dict


def cv_nn_validation(X, y, X_test, params, folds, eval_metric='mae', columns=None, model_path="./"):
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

    columns = X.columns if columns is None else columns
    oof = np.zeros(len(X))
    prediction = np.zeros((len(X_test), folds.n_splits))
    model_list = []

    # X_train, X_valid, y_train, y_valid = train_test_split(X[columns], y, test_size=0.1, random_state=42)

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print(f'Fold {fold_n + 1} started at {time.ctime()}')

        X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        model: Model = build_model(ncols=X_train.shape[1],
                                   dropout_rate=params["dropout_rate"],
                                   activation=params["activation"],
                                   start_neurons=params["start_neurons"], )

        es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=params["patience"],
                                              verbose=1, mode='auto',
                                              restore_best_weights=True)
        checkPoint = ModelCheckpoint(str(model_path / f"mlp_model_{DATA_VERSION}_{TRIAL_NO}.model"),
                                     monitor='val_loss', mode='min', save_best_only=True, verbose=0)
        print(f"X_train: {X_train.shape}, X_valid: {X_valid.shape}")
        model.fit(x=X_train, y=y_train,
                  validation_data=[X_valid, y_valid],
                  epochs=params["epochs"],
                  verbose=2 if DEBUG else 1,
                  shuffle=True,
                  callbacks=[es_cb, checkPoint])
        model_list += [model]

        model.save_weights(str(model_path / f"mlp_model_{DATA_VERSION}_{TRIAL_NO}_best.model"))
        pred_valid = model.predict(X_valid, batch_size=params["pred_batch_size"])
        oof[valid_index] = pred_valid.flatten()
        y_pred = model.predict(X_test, batch_size=params["pred_batch_size"])
        prediction[:, fold_n] = y_pred.flatten()

        if eval_metric != 'group_mae':
            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, oof[valid_index])
        else:
            score = metrics_dict[eval_metric]['scoring_function'](y_valid, oof[valid_index], X_valid['type'])
        print(f"hold {fold_n} :score {score}")


    if eval_metric != 'group_mae':
        score = metrics_dict[eval_metric]['sklearn_scoring_function'](y, oof)
    else:
        score = metrics_dict[eval_metric]['scoring_function'](y, oof, X_valid['type'])

    result_dict = {}
    result_dict["model"] = model_list
    result_dict["oof"] = pd.DataFrame(oof, index=X.index,
                                      columns=["scalar_coupling_constant"])
    result_dict["prediction"] = pd.DataFrame(prediction.mean(axis=1), index=X_test.index,
                                             columns=["scalar_coupling_constant"])
    result_dict['score'] = score
    return result_dict


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
        cv_score_msg = f'{DATA_VERSION}_{TRIAL_NO}' + f'HOLD_OUT score: {score:.4f} .'
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

    eval_result = {}
    callbacks = [lgb.record_evaluation(eval_result)]
    model = lgb.LGBMRegressor(**params, n_estimators=n_estimators, n_jobs=-1, importance_type='gain')
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train)],
              eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
              verbose=verbose, callbacks=callbacks)

    result_dict['prediction'] = model.predict(X_test)
    result_dict["eval_result"] = eval_result

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
            model = lgb.LGBMRegressor(**params, n_estimators=n_estimators, n_jobs=-1, importance_type='gain', )
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
    prediction /= folds.n_splits
    try:
        cv_score_msg = f'{DATA_VERSION}_{TRIAL_NO}' + 'CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores),
                                                                                                     np.std(scores))
        print(cv_score_msg)
        if LINE_MSG:
            send_message(cv_score_msg)
    except Exception as e:
        print(e)
        pass

    result_dict["models"] = model_list
    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores

    if model_type == 'lgb':
        if plot_feature_importance:
            feature_importance["importance"] /= folds.n_splits
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');
            feature_importance.to_csv(log_path / f"importance_{mol_type}.csv")
            result_dict['feature_importance'] = feature_importance

    return result_dict


def map_atom_info(df, structures, atom_idx):
    df = pd.merge(df, structures, how='left',
                  left_on=['molecule_name', f'atom_index_{atom_idx}'],
                  right_on=['molecule_name', 'atom_index'])

    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df


def angle_feat(df):
    df_feat = pd.DataFrame({"id": df.id.values}, index=df.index.values)
    for axis in ["x", "y", "z"]:
        df_feat[f"{axis}_diff"] = df[f"{axis}_0"] - df[f"{axis}_1"]

    df_feat["diff_norm"] = (df_feat.x_diff ** 2 + df_feat.y_diff ** 2 + df_feat.z_diff ** 2) ** 0.5
    df_feat["zero_norm"] = (df.x_0 ** 2 + df.y_0 ** 2 + df.z_0 ** 2) ** 0.5

    for axis in ["x", "y", "z"]:
        df_feat[f"{axis}_diff"] = df_feat[f"{axis}_diff"].values / df_feat["diff_norm"].values
        df_feat[f"{axis}_0"] = df[f"{axis}_0"].values / df_feat["zero_norm"].values

    df_feat["f004:angle"] = df_feat.x_diff * df_feat.x_0 + df_feat.x_diff * df_feat.y_0 + df_feat.x_diff * df_feat.z_0
    df_feat["f004:angle_abs"] = np.abs(df_feat["f004:angle"])
    return df_feat[["id", "f004:angle", "f004:angle_abs"]]


def angle_feature_conv(structures):
    train_ = pd.read_csv('../input/train.csv')
    test_ = pd.read_csv('../input/test.csv')
    train_ = map_atom_info(train_, structures, 0)
    train_ = map_atom_info(train_, structures, 1)

    test_ = map_atom_info(test_, structures, 0)
    test_ = map_atom_info(test_, structures, 1)
    angle_df_train = angle_feat(train_)
    angle_df_test = angle_feat(test_)
    return angle_df_train, angle_df_test


def create_features(df):
    df['molecule_couples'] = df.groupby('molecule_name')['id'].transform('count')
    df['molecule_dist_mean'] = df.groupby('molecule_name')['dist'].transform('mean')
    df['molecule_dist_min'] = df.groupby('molecule_name')['dist'].transform('min')
    df['molecule_dist_max'] = df.groupby('molecule_name')['dist'].transform('max')
    df['atom_0_couples_count'] = df.groupby(['molecule_name', 'atom_index_0'])['id'].transform('count')
    df['atom_1_couples_count'] = df.groupby(['molecule_name', 'atom_index_1'])['id'].transform('count')

    df[f'molecule_atom_index_0_x_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['x_1'].transform('std')
    df[f'molecule_atom_index_0_y_1_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('mean')
    df[f'molecule_atom_index_0_y_1_mean_diff'] = df[f'molecule_atom_index_0_y_1_mean'] - df['y_1']
    df[f'molecule_atom_index_0_y_1_mean_div'] = df[f'molecule_atom_index_0_y_1_mean'] / df['y_1']
    df[f'molecule_atom_index_0_y_1_max'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('max')
    df[f'molecule_atom_index_0_y_1_max_diff'] = df[f'molecule_atom_index_0_y_1_max'] - df['y_1']
    df[f'molecule_atom_index_0_y_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('std')
    df[f'molecule_atom_index_0_z_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['z_1'].transform('std')
    df[f'molecule_atom_index_0_dist_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('mean')
    df[f'molecule_atom_index_0_dist_mean_diff'] = df[f'molecule_atom_index_0_dist_mean'] - df['dist']
    df[f'molecule_atom_index_0_dist_mean_div'] = df[f'molecule_atom_index_0_dist_mean'] / df['dist']
    df[f'molecule_atom_index_0_dist_max'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('max')
    df[f'molecule_atom_index_0_dist_max_diff'] = df[f'molecule_atom_index_0_dist_max'] - df['dist']
    df[f'molecule_atom_index_0_dist_max_div'] = df[f'molecule_atom_index_0_dist_max'] / df['dist']
    df[f'molecule_atom_index_0_dist_min'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('min')
    df[f'molecule_atom_index_0_dist_min_diff'] = df[f'molecule_atom_index_0_dist_min'] - df['dist']
    df[f'molecule_atom_index_0_dist_min_div'] = df[f'molecule_atom_index_0_dist_min'] / df['dist']
    df[f'molecule_atom_index_0_dist_std'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('std')
    df[f'molecule_atom_index_0_dist_std_diff'] = df[f'molecule_atom_index_0_dist_std'] - df['dist']
    df[f'molecule_atom_index_0_dist_std_div'] = df[f'molecule_atom_index_0_dist_std'] / df['dist']
    df[f'molecule_atom_index_1_dist_mean'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('mean')
    df[f'molecule_atom_index_1_dist_mean_diff'] = df[f'molecule_atom_index_1_dist_mean'] - df['dist']
    df[f'molecule_atom_index_1_dist_mean_div'] = df[f'molecule_atom_index_1_dist_mean'] / df['dist']
    df[f'molecule_atom_index_1_dist_max'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('max')
    df[f'molecule_atom_index_1_dist_max_diff'] = df[f'molecule_atom_index_1_dist_max'] - df['dist']
    df[f'molecule_atom_index_1_dist_max_div'] = df[f'molecule_atom_index_1_dist_max'] / df['dist']
    df[f'molecule_atom_index_1_dist_min'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('min')
    df[f'molecule_atom_index_1_dist_min_diff'] = df[f'molecule_atom_index_1_dist_min'] - df['dist']
    df[f'molecule_atom_index_1_dist_min_div'] = df[f'molecule_atom_index_1_dist_min'] / df['dist']
    df[f'molecule_atom_index_1_dist_std'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('std')
    df[f'molecule_atom_index_1_dist_std_diff'] = df[f'molecule_atom_index_1_dist_std'] - df['dist']
    df[f'molecule_atom_index_1_dist_std_div'] = df[f'molecule_atom_index_1_dist_std'] / df['dist']
    df[f'molecule_atom_1_dist_mean'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('mean')
    df[f'molecule_atom_1_dist_min'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('min')
    df[f'molecule_atom_1_dist_min_diff'] = df[f'molecule_atom_1_dist_min'] - df['dist']
    df[f'molecule_atom_1_dist_min_div'] = df[f'molecule_atom_1_dist_min'] / df['dist']
    df[f'molecule_atom_1_dist_std'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('std')
    df[f'molecule_atom_1_dist_std_diff'] = df[f'molecule_atom_1_dist_std'] - df['dist']
    df[f'molecule_type_0_dist_std'] = df.groupby(['molecule_name', 'type_0'])['dist'].transform('std')
    df[f'molecule_type_0_dist_std_diff'] = df[f'molecule_type_0_dist_std'] - df['dist']
    df[f'molecule_type_dist_mean'] = df.groupby(['molecule_name', 'type'])['dist'].transform('mean')
    df[f'molecule_type_dist_mean_diff'] = df[f'molecule_type_dist_mean'] - df['dist']
    df[f'molecule_type_dist_mean_div'] = df[f'molecule_type_dist_mean'] / df['dist']
    df[f'molecule_type_dist_max'] = df.groupby(['molecule_name', 'type'])['dist'].transform('max')
    df[f'molecule_type_dist_min'] = df.groupby(['molecule_name', 'type'])['dist'].transform('min')
    df[f'molecule_type_dist_std'] = df.groupby(['molecule_name', 'type'])['dist'].transform('std')
    df[f'molecule_type_dist_std_diff'] = df[f'molecule_type_dist_std'] - df['dist']

    # df = reduce_mem_usage(df)
    return df


def dist12(name='xy', a='x', b='y'):
    train_p_0 = train[[a + '_0', b + '_0']].values
    train_p_1 = train[[a + '_1', b + '_1']].values
    test_p_0 = test[[a + '_0', b + '_0']].values
    test_p_1 = test[[a + '_1', b + '_1']].values

    train[name] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
    test[name] = np.linalg.norm(test_p_0 - test_p_1, axis=1)
    train['abs_' + name] = np.linalg.norm(train_p_0 - train_p_1, axis=1, ord=1)
    test['abs_' + name] = np.linalg.norm(test_p_0 - test_p_1, axis=1, ord=1)


def map_ob_charges(df, ob_charges, atom_idx):
    ob_charges_col = ['eem', 'mmff94', 'gasteiger', 'qeq',
                      'qtpie', 'eem2015ha', 'eem2015hm', 'eem2015hn', 'eem2015ba',
                      'eem2015bm', 'eem2015bn']
    df = pd.merge(df, ob_charges, how='left',
                  left_on=['molecule_name', f'atom_index_{atom_idx}'],
                  right_on=['molecule_name', 'atom_index'])

    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={c: f"{c}_{atom_idx}" for c in ob_charges_col})
    return df


def get_train_test_data(use_prev=False, prev_data_version=None, prev_trial_no=None):
    if use_prev:
        assert prev_data_version is not None
        assert prev_trial_no is not None

    file_folder = '../input'
    train = pd.read_csv(f'{file_folder}/train.csv')
    if not use_prev:
        test = pd.read_csv(f'{file_folder}/test.csv')
        structures = pd.read_csv(f'{file_folder}/structures.csv')
        scalar_coupling_contributions = pd.read_csv(f'{file_folder}/scalar_coupling_contributions.csv')

        # train_cos = unpickle(save_path / "train_003.df.pkl", )[["id", "f003:cos_0_1", "f003:cos_1"]]
        # test_cos = unpickle(save_path / "test_003.df.pkl", )[["id", "f003:cos_0_1", "f003:cos_1"]]

        train_add = unpickle(save_path / "train_006.df.pkl", )
        test_add = unpickle(save_path / "test_006.df.pkl", )

        babel_train = pd.read_csv(save_path / "babel_train.csv", usecols=use_cols.babel_cols)
        babel_test = pd.read_csv(save_path / "babel_test.csv", usecols=use_cols.babel_cols)

        use_cols.good_columns += [c for c in use_cols.rdkit_cols if c != 'id']
        rdkit_train = pd.read_csv(save_path / "rdkit_train.csv", usecols=use_cols.rdkit_cols)
        rdkit_test = pd.read_csv(save_path / "rdkit_test.csv", usecols=use_cols.rdkit_cols)

        coulomb_train = pd.read_csv(save_path / "coulomb_interaction_train.csv")
        coulomb_test = pd.read_csv(save_path / "coulomb_interaction_test.csv")

        bond_calc_train = unpickle(save_path / "bond_calc_feat_train.pkl")
        bond_calc_test = unpickle(save_path / "bond_calc_feat_test.pkl")

        ob_charges = pd.read_csv(save_path / "ob_charges.csv", index_col=0)

        tda_radius_df = pd.read_csv(save_path / "tda_radius_df.csv", index_col=0)

        tda_radius_df_03 = pd.read_csv(save_path / "tda_radius_df_v003.csv", index_col=0)

        pca_feat = unpickle(save_path / "pca_feat_df.pkl")

        ####################################################################################################
        # Feature Engineering

        train = pd.merge(train, scalar_coupling_contributions, how='left',
                         left_on=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'],
                         right_on=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'])

        train = map_atom_info(train, 0)
        train = map_atom_info(train, 1)
        test = map_atom_info(test, 0)
        test = map_atom_info(test, 1)

        train_p_0 = train[['x_0', 'y_0', 'z_0']].values
        train_p_1 = train[['x_1', 'y_1', 'z_1']].values
        test_p_0 = test[['x_0', 'y_0', 'z_0']].values
        test_p_1 = test[['x_1', 'y_1', 'z_1']].values

        train['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
        test['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)
        train['dist_x'] = (train['x_0'] - train['x_1']) ** 2
        test['dist_x'] = (test['x_0'] - test['x_1']) ** 2
        train['dist_y'] = (train['y_0'] - train['y_1']) ** 2
        test['dist_y'] = (test['y_0'] - test['y_1']) ** 2
        train['dist_z'] = (train['z_0'] - train['z_1']) ** 2
        test['dist_z'] = (test['z_0'] - test['z_1']) ** 2

        train['type_0'] = train['type'].apply(lambda x: x[0])
        test['type_0'] = test['type'].apply(lambda x: x[0])

        train['abs_dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1, ord=1)
        test['abs_dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1, ord=1)
        dist12('dist_xy', 'x', 'y')
        dist12('dist_xz', 'x', 'z')
        dist12('dist_yz', 'y', 'z')

        atom_count = structures.groupby(['molecule_name', 'atom']).size().unstack(fill_value=0)
        train = pd.merge(train, atom_count, how='left', left_on='molecule_name', right_on='molecule_name')
        test = pd.merge(test, atom_count, how='left', left_on='molecule_name', right_on='molecule_name')

        train = create_features(train)
        test = create_features(test)

        angle_df_train, angle_df_test = angle_feature_conv(structures)
        train = train.merge(angle_df_train, on="id", how="left")
        test = test.merge(angle_df_test, on="id", how="left")

        train = train.merge(train_add, on="id", how="left")
        test = test.merge(test_add, on="id", how="left")

        # train = train.merge(train_cos, on="id", how="left")
        # test = test.merge(test_cos, on="id", how="left")

        train = train.merge(babel_train, on="id", how="left")
        test = test.merge(babel_test, on="id", how="left")

        train = train.merge(rdkit_train, on="id", how="left")
        test = test.merge(rdkit_test, on="id", how="left")

        train = train.merge(coulomb_train, on="id", how="left")
        test = test.merge(coulomb_test, on="id", how="left")

        train = train.merge(bond_calc_train, on="id", how="left")
        test = test.merge(bond_calc_test, on="id", how="left")

        train = train.merge(tda_radius_df, on="molecule_name", how="left")
        test = test.merge(tda_radius_df, on="molecule_name", how="left")

        train = train.merge(tda_radius_df_03, on="molecule_name", how="left")
        test = test.merge(tda_radius_df_03, on="molecule_name", how="left")

        train = train.merge(pca_feat, on="molecule_name", how="left")
        test = test.merge(pca_feat, on="molecule_name", how="left")

        train = map_ob_charges(train, ob_charges, 0)
        train = map_ob_charges(train, ob_charges, 1)
        test = map_ob_charges(test, ob_charges, 0)
        test = map_ob_charges(test, ob_charges, 1)

        train = reduce_mem_usage(train)
        test = reduce_mem_usage(test)

        for f in ['atom_1', 'type_0', 'type']:
            if f in use_cols.good_columns:
                lbl = LabelEncoder()
                lbl.fit(list(train[f].values) + list(test[f].values))
                train[f] = lbl.transform(list(train[f].values))
                test[f] = lbl.transform(list(test[f].values))

        Path(save_path / f"{DATA_VERSION}_{TRIAL_NO}").mkdir(parents=True, exist_ok=True)
        to_pickle(save_path / f"{DATA_VERSION}_{TRIAL_NO}/train_concat_{DATA_VERSION}_{TRIAL_NO}.pkl", train)
        to_pickle(save_path / f"{DATA_VERSION}_{TRIAL_NO}/test_concat_{DATA_VERSION}_{TRIAL_NO}.pkl", test)
    else:
        sample_loaded = False
        prev_folder = f"../processed/{prev_data_version}/{prev_data_version}_{prev_trial_no}"
        if DEBUG:
            # v003_033
            train_path = Path(f"{prev_folder}/train_concat_{prev_data_version}_{prev_trial_no}_yiemon_sampled.pkl")
            test_path = Path(f"{prev_folder}/test_concat_{prev_data_version}_{prev_trial_no}_yiemon_sampled.pkl")

            if train_path.exists() and test_path.exists():
                print("sample loading")
                train = unpickle(train_path)
                test = unpickle(test_path)
                sample_loaded = True
                print("sample load finish")

        if not sample_loaded:
            print(f"loading previous dataest")
            print("train w/yiemon loading")
            train: pd.DataFrame = unpickle(f"{prev_folder}/train_concat_{prev_data_version}_{prev_trial_no}_yiemon.pkl", )
            assert "scalar_coupling_constant" in train.columns
            print("test w/yiemon  loading")
            test: pd.DataFrame = unpickle(f"{prev_folder}/test_concat_{prev_data_version}_{prev_trial_no}_yiemon.pkl", )
            print(f"loading finished")

        if DEBUG and not sample_loaded:
            n_sample = 5000
            print(f"sampling {n_sample} rows.")
            train = train.sample(n=n_sample)
            test = test.sample(n=n_sample)
            Path(f"../processed/{prev_data_version}/{prev_data_version}_{prev_trial_no}").mkdir(parents=True,
                                                                                                exist_ok=True)
            to_pickle(f"{prev_folder}/train_concat_{prev_data_version}_{prev_trial_no}_yiemon_sampled.pkl", train)
            to_pickle(f"{prev_folder}/test_concat_{prev_data_version}_{prev_trial_no}_yiemon_sampled.pkl", test)
            print("saved.")

        ###################################################################################################
        # add additional feature for trying

        # Path(save_path / f"{DATA_VERSION}_{TRIAL_NO}").mkdir(parents=True, exist_ok=True)
        # to_pickle(save_path / f"{DATA_VERSION}_{TRIAL_NO}/train_concat_{DATA_VERSION}_{TRIAL_NO}.pkl", train)
        # to_pickle(save_path / f"{DATA_VERSION}_{TRIAL_NO}/test_concat_{DATA_VERSION}_{TRIAL_NO}.pkl", test)
    return train, test


def train_lgb_each_type(X, y, X_test, seed, params, eval_metric, model_type="lgb"):
    print("start fitting")

    n_fold = 5
    folds = KFold(n_splits=n_fold)

    params["seed"] = seed
    params["bagging_seed"] = seed + 1
    params["feature_fraction_seed"] = seed + 2
    score_list = []
    X_short = pd.DataFrame({'ind': list(X.index), 'type': X['type'].values, 'oof': [0] * len(X), 'target': y.values})
    X_short_test = pd.DataFrame(
        {'ind': list(X_test.index), 'type': X_test['type'].values, 'prediction': [0] * len(X_test)})

    num_leaves_dict = {0: 64,
                       1: 64,
                       2:128,
                       3: 64,
                       4: 64,
                       5:128,
                       6: 64,
                       7:128,
                    }


    print(f"X['type'].unique(): {X['type'].unique()}")
    for t in X['type'].unique():
        #if t not in [5, 7]: continue
        # if seed == current_seed and t in [0, 3, 1, 4]: continue  # [0, 3, 1, 4, 2, 6]
        print(f'{current_time()} Training of type {t}[{type_name[t]}] / {X["type"].unique()}')
        X_t: pd.DataFrame = X.loc[X['type'] == t]
        y_t: pd.Series = X_short.loc[X_short['type'] == t, 'target']
        X_test_t: pd.DataFrame = X_test.loc[X_test['type'] == t]
        print(f"X_t.shape: {X_t.shape}, y_t.shape: {y_t.shape}")
        params["num_leaves"] = num_leaves_dict[t]

        high_importance_cols = [c for c in high_importance_dict[t] if c in X_t.columns.values]
        drop_cols = [c for c in use_cols.feat_2J_atom_info_cols if c in X_t.columns.values]
        if model_type == "lgb":
            start_time = current_time()
            bairitsu = 128 / params["num_leaves"]
            n_estimators = int(8000 * bairitsu)
            result_dict = hold_out_lgb_validation(X=X_t.drop(drop_cols, axis=1)[high_importance_cols]
                                                    if t in [0, 1, 5, 6, 7] else X_t, # 2J以外はdrop
                                                  y=y_t,
                                                  params=params,
                                                  eval_metric=eval_metric,  # "mae", #'group_mae',
                                                  plot_feature_importance=True,
                                                  verbose=5000,
                                                  early_stopping_rounds=200,
                                                  n_estimators=n_estimators)

            result_dict["type"] = t
            result_dict["type_name"] = type_name[t]
            result_dict["start_time"] = start_time
            result_dict["n_estimator"] = n_estimators
            result_dict["X_len"] = X_t.shape[0]
            importance_path = log_path / f'importance_{DATA_VERSION}_{TRIAL_NO}_{t}_{seed}.csv'
            result_dict["importance"].to_csv(importance_path, index=True)

            eval_result: list = result_dict["eval_result"]["valid_1"]["l1"]
            training_log_df: pd.DataFrame = pd.DataFrame(eval_result, index=np.arange(len(eval_result)) + 1)
            training_log_df.columns = ["l1"]
            training_log_df.index.name = "iter"
            training_log_df.to_csv(log_path / f"train_log_{DATA_VERSION}_{TRIAL_NO}_{t}.csv")

            to_pickle(model_path / f"hold_out_model_{DATA_VERSION}_{TRIAL_NO}_{t}_{seed}.pkl", result_dict["model"])
            result_dict['y_pred_valid'].to_csv(submit_path / f'holdout_pred_{DATA_VERSION}_{TRIAL_NO}_{t}_{seed}.csv',
                                               index=True)
            to_pickle(log_path/f"result_dict_{t}_{params['num_leaves']}_{seed}.pkl", result_dict)

        elif model_type == "mlp":
            result_dict = cv_nn_validation(X_t, y_t,
                                           X_test_t,
                                           params,
                                           folds,
                                           eval_metric='mae',
                                           columns=None,
                                           model_path=model_path)

            X_short.loc[X_short['type'] == t, 'oof'] = result_dict['oof']
            X_short_test.loc[X_short_test['type'] == t, 'prediction'] = result_dict['prediction']
            X_short_test.to_csv(submit_path / f"tmp_sub_{t}.csv")

        else:
            assert False, f"model_type: {model_type}"

        print(f"oof_log_mae: {result_dict['score']}")
        score_list += [result_dict['score']]
    return score_list

def train_lgb(X, y, seed, params, eval_metric):
    print("start fitting")

    params["seed"] = seed
    params["bagging_seed"] = seed + 1
    params["feature_fraction_seed"] = seed + 2
    n_estimator = 128000
    start_time = current_time()
    result_dict = hold_out_lgb_validation(X=X,
                                          y=y,
                                          params=params,
                                          eval_metric=eval_metric,
                                          plot_feature_importance=True,
                                          verbose=5000,
                                          early_stopping_rounds=200,
                                          n_estimators=n_estimator)

    result_dict["start_time"] = start_time
    result_dict["n_estimator"] = n_estimator
    result_dict["X_len"] = X.shape[0]
    importance_path = log_path / f'importance_{DATA_VERSION}_{TRIAL_NO}_{seed}.csv'
    result_dict["importance"].to_csv(importance_path, index=True)

    eval_result: list = result_dict["eval_result"]["valid_1"]["l1"]
    training_log_df: pd.DataFrame = pd.DataFrame(eval_result, index=np.arange(len(eval_result)) + 1)
    training_log_df.columns = ["l1"]
    training_log_df.index.name = "iter"
    training_log_df.to_csv(log_path / f"train_log_{DATA_VERSION}_{TRIAL_NO}.csv")

    to_pickle(model_path / f"hold_out_model_{DATA_VERSION}_{TRIAL_NO}_{seed}.pkl", result_dict["model"])
    result_dict_path = submit_path / f'holdout_pred_{DATA_VERSION}_{TRIAL_NO}_{seed}.csv'
    result_dict['y_pred_valid'].to_csv(result_dict_path, index=True)
    to_pickle(log_path / f"result_dict_{params['num_leaves']}_{seed}.pkl", result_dict)


def get_high_importance_cols(data_version, trial_version, thresh=0.0005, seed=71, verbose=False):
    importance_list = []
    ret_dict = {}
    for i in range(8):
        path = f"../log/{data_version}_{trial_version}/importance_{data_version}_{trial_version}_{i}_{seed}.csv"
        try:
            importance_df = pd.read_csv(path, index_col=0).set_index("feature").sort_values("importance",
                                                                                            ascending=False)
            importance_df["importance"] /= importance_df["importance"].sum()
            if verbose:
                print(f"type_{i}")
                print(importance_df)
            importance_df = importance_df[importance_df.importance > thresh]
            ret_dict[i] = importance_df.index.values
            # break
        except Exception as e:
            print(e)

    return ret_dict

####################################################################################################
# Settings
print(f"STARTING : {current_time()}")

DEBUG = False
MODEL_TYPE = "lgb"
LINE_MSG = True

assert MODEL_TYPE in ["lgb", "mlp", "lgb_holdout"]
if DEBUG:
    print("=" * 30 + " DEBUG " + "=" * 30)

# GROUP_K_FOLD = False
# TRAIN_ALL_DATA = False
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
train, test = get_train_test_data(use_prev=True, prev_data_version="v003", prev_trial_no="033")
use_cols_revised = [c for c in use_cols.good_columns if c not in use_cols.remove_cols]
use_cols_revised = [c for c in use_cols_revised if c in train.columns]
high_importance_dict = get_high_importance_cols(data_version="v003", trial_version="045", verbose=False)
###################################################################################################
# final data preparation for train
X: pd.DataFrame = train[use_cols_revised].copy()
y: pd.Series = train['scalar_coupling_constant']
y_fc: pd.Series = train['fc']
X_test: pd.DataFrame = test[use_cols_revised].copy()
print(f"X.shape: {X.shape}, X_test.shape: {X_test.shape}")

# X.to_csv("../info/X_sampled.csv")

# export colnames
pd.DataFrame({"columns": X.columns.tolist()}).to_csv(log_path / f"use_cols.csv")

####################################################################################################
# Model Fitting

seed = 71
params = { #'num_leaves': 128,
          'min_child_samples': 79,
          'objective': 'regression',
          'max_depth': 9,
          'learning_rate': 0.2,
          "boosting_type": "gbdt",
          "subsample_freq": 1,
          "subsample": 0.9,
          "metric": 'mae',
          "verbosity": -1,
          'reg_alpha': 0.1,
          'reg_lambda': 0.3,
          'colsample_bytree': 1.0,
          'num_threads': -1,
          }

result_dict_list = []
eval_metric = "mae" # "mae" if DEBUG else "group_mae"

if MODEL_TYPE == "lgb":
    train_lgb_each_type(X, y, X_test, seed, params, eval_metric, model_type=MODEL_TYPE)

elif MODEL_TYPE == "lgb_holdout":
    params["num_leaves"] = 8
    score_list = train_lgb(X, y, seed, params, eval_metric)

elif MODEL_TYPE == "mlp":
    params = {"epochs": 200,
              "patience": 30,
              "batch_size": 512,
              "dropout_rate": 0.3,
              "activation": "relu",
              "start_neurons": 512,
              "pred_batch_size": 2048}

    if DEBUG:
        params["epochs"] = 5
        params["patience"] = 1

    rgs = RankGaussScalar()
    rgs.fit(X)
    X_rgs = rgs.transform(X.fillna(0))
    del X
    gc.collect()
    X_test_rgs = rgs.transform(X_test.fillna(0))
    del X_test
    gc.collect()

    X_rgs.fillna(0, inplace=True)
    X_test_rgs.fillna(0, inplace=True)

    score_list = train_lgb_each_type(X_rgs, y, X_test_rgs, seed, params, eval_metric, model_type=MODEL_TYPE)
    print(f"mean score: {np.mean(score_list)}")
    result_dict_list += [{"score_list": score_list, "score_mean": np.mean(score_list)}]
    result_df = pd.DataFrame(result_dict_list)
    result_df.to_csv(log_path / f"result_df_{MODEL_TYPE}.csv")
else:
    assert False, f"error MODEL_TYPE: {MODEL_TYPE}"


# result_dictの解析は ../notebook/result_dict_analysis.ipynb にて！