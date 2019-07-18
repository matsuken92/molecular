#!/usr/bin/env python
# coding: utf-8

import matplotlib as mpl
mpl.use('Agg')

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

import matplotlib.pyplot as plt
import seaborn as sns
# import lightgbm as lgb
# import xgboost as xgb
# from catboost import CatBoostRegressor
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


import keras
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Dropout, Input, Reshape, Concatenate, LeakyReLU
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.utils import to_categorical
import tensorflow as tf
from keras import backend as K
from keras.layers.embeddings import Embedding

def plot_history(history, label):
    plt.plot(history.history['loss'], rasterized=True)
    plt.plot(history.history['val_loss'], rasterized=True)
    plt.title('Loss for %s' % label)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    _= plt.legend(['Train','Validation'], loc='upper left')
    plt.savefig(log_path/f"histroy_{label}.png", dpi=128)
    plt.close()


def build_model(ncols, dropout_rate=0.25, activation='relu', start_neurons=256, ):
    inputs = []

    input_numeric = Input(shape=(ncols,), name='main_input')
    inputs.append(input_numeric)

    x = Dense(256)(input_numeric)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Dropout(0.4)(x)
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Dropout(0.2)(x)
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Dropout(0.2)(x)
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Dropout(0.4)(x)
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.05)(x)
    #x = Dropout(0.4)(x)
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Dropout(0.4)(x)
    # out1 = Dense(2, activation="linear")(x)#mulliken charge 2
    # out2 = Dense(6, activation="linear")(x)#tensor 6(xx,yy,zz)
    # out3 = Dense(12, activation="linear")(x)#tensor 12(others)
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Dropout(0.2)(x)
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Dropout(0.2)(x)

    # x = Dense(start_neurons, activation=activation)(input_numeric)
    # x = BatchNormalization()(x)
    # x = Dropout(dropout_rate)(x)
    #
    # x = Dense(start_neurons // 2, activation=activation)(x)
    # x = BatchNormalization()(x)
    # x = Dropout(dropout_rate)(x)
    #
    # x = Dense(start_neurons // 4, activation=activation)(x)
    # x = BatchNormalization()(x)
    # x = Dropout(dropout_rate)(x)
    #
    # x = Dense(start_neurons // 8, activation=activation)(x)
    # x = BatchNormalization()(x)
    # x = Dropout(dropout_rate / 2)(x)

    main_output = Dense(1, activation='linear', name='main_output')(x)

    model = Model(inputs=inputs, outputs=main_output)

    # opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # opt = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    opt = 'adam'
    model.compile(loss="mse", optimizer=opt, metrics=['mae'])
    return model


def cv_nn_validation(X, y, X_test, params, folds, mol_type, eval_metric='mae', columns=None, model_path="./", fold_group=None):
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

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, groups=fold_group)):
        print(f'Fold {fold_n + 1} started at {time.ctime()}')

        X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        model: Model = build_model(ncols=X_train.shape[1],
                                   dropout_rate=params["dropout_rate"],
                                   activation=params["activation"],
                                   start_neurons=params["start_neurons"], )

        es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=params["patience"],
                                              verbose=1, mode='auto', min_delta=0.0001,
                                              restore_best_weights=True)
        rlr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7,
                                                min_lr=1e-6, mode='auto', verbose=1)

        checkPoint = ModelCheckpoint(str(model_path / f"mlp_model_{DATA_VERSION}_{TRIAL_NO}.model"),
                                     monitor='val_loss', mode='min', save_best_only=True, verbose=0)
        print(f"X_train: {X_train.shape}, X_valid: {X_valid.shape}")
        history = model.fit(x=X_train, y=y_train,
                  validation_data=[X_valid, y_valid],
                  epochs=params["epochs"],
                  verbose=2 if DEBUG else 1,
                  shuffle=True,
                  callbacks=[es_cb, rlr, checkPoint])
        model_list += [model]

        model.save_weights(str(model_path / f"mlp_model_{DATA_VERSION}_{TRIAL_NO}_{mol_type}_{fold_n+1}_best.model"))
        pred_valid = model.predict(X_valid, batch_size=params["pred_batch_size"])
        oof[valid_index] = pred_valid.flatten()
        y_pred = model.predict(X_test, batch_size=params["pred_batch_size"])
        prediction[:, fold_n] = y_pred.flatten()

        plot_history(history, f"{mol_type}_{fold_n+1}")


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

def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    """
    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling
    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric
    """
    maes = (y_true - y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()


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

        ###################################################################################################
        # add additional feature for trying

        # Path(save_path / f"{DATA_VERSION}_{TRIAL_NO}").mkdir(parents=True, exist_ok=True)
        # to_pickle(save_path / f"{DATA_VERSION}_{TRIAL_NO}/train_concat_{DATA_VERSION}_{TRIAL_NO}.pkl", train)
        # to_pickle(save_path / f"{DATA_VERSION}_{TRIAL_NO}/test_concat_{DATA_VERSION}_{TRIAL_NO}.pkl", test)
    return train, test

####################################################################################################
# Settings
print(f"STARTING : {current_time()}")

DEBUG = False
MODEL_TYPE = "mlp"
LINE_MSG = True
GROUP_K_FOLD = True

TRAIN_ALL_DATA = False
CV_FOLD = True

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
prev_data_version="v003"
prev_trial_no="033"

prev_folder = f"../processed/{prev_data_version}/{prev_data_version}_{prev_trial_no}"
train_data_path = f"{prev_folder}/X_{prev_data_version}_{prev_trial_no}_yiemon_123J_ss{debug_str}.pkl"
test_data_path = f"{prev_folder}/X_test_{prev_data_version}_{prev_trial_no}_yiemon_123J_ss{debug_str}.pkl"
y_data_path = f"{prev_folder}/y_{prev_data_version}_{prev_trial_no}_yiemon_123J_ss{debug_str}.pkl"
mol_name_data_path = f"{prev_folder}/mol_name_{prev_data_version}_{prev_trial_no}_yiemon_123J_ss{debug_str}.pkl"

if Path(train_data_path).exists() and Path(test_data_path).exists() and \
    Path(y_data_path).exists() and Path(mol_name_data_path).exists():
    print("loading exist files")
    X_rgs = unpickle(train_data_path)
    X_test_rgs = unpickle(test_data_path)
    y = unpickle(y_data_path)
    mol_name = unpickle(mol_name_data_path)
    print("loaded exist files")
else:
    print("gathering files for model train.")
    train, test = get_prev_train_test_data(prev_data_version=prev_data_version, prev_trial_no=prev_trial_no)
    use_cols_revised = [c for c in use_cols.good_columns if c not in use_cols.remove_cols]
    use_cols_revised = [c for c in use_cols_revised if c in train.columns]
    # high_importance_dict = get_high_importance_cols(data_version="v003", trial_version="045", verbose=False)

    mol_name = train.molecule_name

    ###################################################################################################
    # final data preparation for train
    X: pd.DataFrame = train[use_cols_revised].copy()
    y: pd.Series = train['scalar_coupling_constant']
    # y_fc: pd.Series = train['fc']
    X_test: pd.DataFrame = test[use_cols_revised].copy()
    print(f"X.shape: {X.shape}, X_test.shape: {X_test.shape}")

    # X.to_csv("../info/X_sampled.csv")

    # export colnames
    pd.DataFrame({"columns": X.columns.tolist()}).to_csv(log_path / f"use_cols.csv")

    type_train = X["type"].values
    type_test = X_test["type"].values

    X_fillna = X.fillna(0)
    X_test_fillna =  X_test.fillna(0)
    ss = StandardScaler()
    ss.fit(pd.concat([X_fillna, X_test_fillna], axis=0))
    X_rgs =  pd.DataFrame(ss.transform(X_fillna), index=X_fillna.index, columns=X_fillna.columns)
    X_test_rgs = pd.DataFrame(ss.transform(X_test_fillna), index=X_test_fillna.index, columns=X_test_fillna.columns)

    # rgs = RankGaussScalar()
    # rgs.fit(X_fillna)
    # X_rgs = rgs.transform(X_fillna)
    # X_test_rgs = rgs.transform(X_test.fillna(0))

    del X;
    gc.collect()
    del X_test;
    gc.collect()

    X_rgs.fillna(0, inplace=True)
    X_test_rgs.fillna(0, inplace=True)

    X_rgs["type"] = type_train
    X_test_rgs["type"] = type_test

    to_pickle(train_data_path, X_rgs)
    to_pickle(test_data_path, X_test_rgs)
    to_pickle(y_data_path, y)
    to_pickle(mol_name_data_path, mol_name)
    print("saved files for model train.")

####################################################################################################
# Model Fitting
print("start fitting")
n_fold = 5
if DEBUG:
    n_fold = 3

if GROUP_K_FOLD:
    folds = GroupKFold(n_splits=n_fold)
else:
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)

#########################################################################################################
# 1st layer model
#seed_base = [0, 2019, 71, 1228, 1988, 1879, 92, 3018, 1234, 185289]
seed_base = [0]
seed_list = np.array(seed_base) + 60
#seed_list = np.array(seed_base) + 51
#seed_list = np.array(seed_base) + 52
#seed_list = np.array(seed_base) + 53
#seed_list = np.array(seed_base) + 54

current_seed = -1

print(seed_list)
for seed in seed_list:
    print(f"==================== seed: {seed} ====================")

    params = {"epochs": 300,
              "patience": 30,
              "batch_size": 2048,
              "dropout_rate": 0.3,
              "activation": "relu",
              "start_neurons": 512,
              "pred_batch_size": 2048}
    if DEBUG:
        params["batch_size"] = 256
        params["epochs"] = 1
        params["patience"] = 1


    #########################################################################################################
    # nn model
    score_list = []
    X_short = pd.DataFrame({'ind': list(X_rgs.index), 'type': X_rgs['type'].values, 'oof': [0] * len(X_rgs), 'target': y.values})

    if seed==current_seed:
        X_short_test = pd.read_csv(submit_path/f"tmp_sub_6.csv")
    else:
        X_short_test = pd.DataFrame(
            {'ind': list(X_test_rgs.index), 'type': X_test_rgs['type'].values, 'prediction': [0] * len(X_test_rgs)})

    print(f"X['type'].unique(): {X_rgs['type'].unique()}")
    for t in X_rgs['type'].unique():
        start_time = current_time()
        print(f"start_time: {start_time}")
        print(f'{current_time()} Training of type {t} / {X_rgs["type"].unique()}')
        X_t = X_rgs.loc[X_rgs['type'] == t]
        X_test_t = X_test_rgs.loc[X_test_rgs['type'] == t]
        y_t = X_short.loc[X_short['type'] == t, 'target']
        mol_name_t = mol_name.loc[X_rgs['type'] == t][X_t.index] if GROUP_K_FOLD else None
        print(f"X_t.shape: {X_t.shape}, X_test_t.shape: {X_test_t.shape}, y_t.shape: {y_t.shape}")

        result_dict = cv_nn_validation(X_t.drop(["type"], axis=1),
                                       y_t,
                                       X_test_t.drop(["type"], axis=1),
                                       params,
                                       folds,
                                       mol_type=t,
                                       eval_metric='mae',
                                       columns=None,
                                       model_path=model_path,
                                       fold_group=mol_name_t)

        X_short.loc[X_short['type'] == t, 'oof'] = result_dict['oof']
        X_short_test.loc[X_short_test['type'] == t, 'prediction'] = result_dict['prediction']
        X_short_test.to_csv(submit_path / f"tmp_sub_{t}.csv")

        print(f"oof_log_mae: {result_dict['score']}")
        score_list += [result_dict['score']]

    for t, s in zip(X_rgs['type'].unique(), score_list):
        print(f"type {t}, score: {s:0.5f}")

    sub = pd.read_csv(f'../input/sample_submission.csv')
    sub['scalar_coupling_constant'] = X_short_test['prediction']
    sub.to_csv(submit_path / f'submission_nn_{DATA_VERSION}_{TRIAL_NO}_{MODEL_TYPE}_{seed}.csv', index=False)
    print(sub.head())

    oof_log_mae = group_mean_log_mae(X_short['target'], X_short['oof'], X_short['type'], floor=1e-9)
    print(f"oof_log_mae: {oof_log_mae}")

    train_ids = pd.read_csv(f'../input/train.csv')["id"].values
    df_oof = pd.DataFrame(index=train_ids)
    df_oof["scalar_coupling_constant"] = X_short['oof']
    df_oof.to_csv(submit_path / f'oof_nn_{DATA_VERSION}_{TRIAL_NO}_{MODEL_TYPE}_{seed}.csv', index=True)

    if not DEBUG:
        send_message(f"{MODEL_TYPE}: finish train_{DATA_VERSION}_{TRIAL_NO}_{seed}, oof_log_mae: {oof_log_mae}")

print(f"finished. : {current_time()}")