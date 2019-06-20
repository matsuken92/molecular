#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 基本ライブラリ
import pandas as pd
import pandas.io.sql as psql
import numpy as np
import numpy.random as rd
import gc
import multiprocessing as mp
import os
import sys
import pickle
from collections import defaultdict
from glob import glob
import math
from datetime import datetime as dt
from pathlib import Path
import scipy.stats as st
import re
import shutil
from tqdm import tqdm_notebook as tqdm
import datetime

# In[2]:


import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold

from sklearn import metrics
import json

import warnings

warnings.filterwarnings("ignore")

sys.path.append('..')
from lib.line_notif import send_message
from lib.utils import reduce_mem_usage, current_time, unpickle, to_pickle
from lib.utils import one_hot_encoder, apply_agg, multi_combine_categorical_feature
from lib.utils import import_data, get_split_indexer


class ModelExtractionCallback(object):
    """Callback class for retrieving trained model from lightgbm.cv()
    NOTE: This class depends on '_CVBooster' which is hidden class, so it might doesn't work if the specification is changed.
    """

    def __init__(self):
        self._model = None

    def __call__(self, env):
        # Saving _CVBooster object.
        self._model = env.model

    def _assert_called_cb(self):
        if self._model is None:
            # Throw exception if the callback class is not called.
            raise RuntimeError('callback has not called yet')

    @property
    def boosters_proxy(self):
        self._assert_called_cb()
        # return Booster object
        return self._model

    @property
    def raw_boosters(self):
        self._assert_called_cb()
        # return list of Booster
        return self._model.boosters

    @property
    def best_iteration(self):
        self._assert_called_cb()
        # return boosting round when early stopping.
        return self._model.best_iteration


# In[3]:


print(f"STARTING : {current_time()}")

DATA_VERSION = __file__.split("_")[-2]
TRIAL_NO = __file__.split("_")[-1].replace(".py","")
# DATA_VERSION = "s001"
# TRIAL_NO = "001"

save_path = Path(f"../processed/{DATA_VERSION}")
save_path.mkdir(parents=True, exist_ok=True)
model_path = Path(f"../model/{DATA_VERSION}_{TRIAL_NO}")
model_path.mkdir(parents=True, exist_ok=True)
submit_path = Path(f"../submit/{DATA_VERSION}_{TRIAL_NO}")
submit_path.mkdir(parents=True, exist_ok=True)
#
# print("start loading...")
# train = unpickle(save_path / "train_002.df.pkl", )
# print("train loaded.")
# test = unpickle(save_path / "test_002.df.pkl", )
# print("test loaded.")
# y = train["scalar_coupling_constant"]
# train.drop("scalar_coupling_constant", axis=1, inplace=True)
#
# train.set_index("id", inplace=True)
# test.set_index("id", inplace=True)

oof_train = pd.DataFrame()
pred_test = pd.DataFrame()
y = pd.Series()

lgbm_params = {
    "boosting_type": "gbdt",
    'objective': 'regression',
    "metric": 'mae',
    # 'n_estimator': 10000,
    'n_jobs': -1,
    "seed": 71,
    "verbosity": -1,

    'learning_rate': 0.1,

    'max_depth': 9,
    'num_leaves': 128,

    "subsample_freq": 1,
    "subsample": 0.8,
    'colsample_bytree': 1.0,

    'min_child_samples': 200,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0,
}

lgb_train = lgb.Dataset(oof_train, y)

# Training settings
FOLD_NUM = 5
fold_seed = 11
folds = KFold(n_splits=FOLD_NUM, shuffle=True, random_state=fold_seed)

extraction_cb = ModelExtractionCallback()
callbacks = [
    extraction_cb
]

# Fitting
print("start fitting...")
ret = lgb.cv(params=lgbm_params,
             train_set=lgb_train,
             #categorical_feature=categorical,
             folds=folds,
             num_boost_round=5000,
             verbose_eval=500,
             early_stopping_rounds=200,
             callbacks=callbacks,
             )
#df_ret = pd.DataFrame(ret)
print("finish fitting.")

# Retrieving booster and training information.
proxy = extraction_cb.boosters_proxy
boosters = extraction_cb.raw_boosters
best_iteration = extraction_cb.best_iteration
to_pickle(model_path / 'extraction_cb.pkl', extraction_cb)

# Create oof prediction result
print("create oof preds.")
fold_iter = folds.split(oof_train, y)
oof_preds = np.zeros_like(y)
for n_fold, ((trn_idx, val_idx), booster) in enumerate(zip(fold_iter, boosters)):
    print(val_idx)
    valid = oof_train.iloc[val_idx]
    oof_preds[val_idx] = booster.predict(valid, num_iteration=best_iteration)
print(f"mae on oof preds: {mean_absolute_error(y, oof_preds)}")
np.save(submit_path / 'oof.npy', oof_preds)

# Averaging prediction result for test data.
y_pred_proba_list = proxy.predict(pred_test, num_iteration=best_iteration)
y_pred_proba_avg = np.array(y_pred_proba_list).mean(axis=0)

sub = pd.read_csv('../input/sample_submission.csv')
sub['scalar_coupling_constant'] = y_pred_proba_avg
sub.to_csv(submit_path / f'sub_stacking_{DATA_VERSION}_{TRIAL_NO}.csv', index=False)
sub.head()

print("finish.")