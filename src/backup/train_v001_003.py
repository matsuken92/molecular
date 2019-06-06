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
ts_conv = np.vectorize(datetime.datetime.fromtimestamp) # 秒ut(10桁) ⇒ 日付

# グラフ描画系
import matplotlib
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc

from matplotlib import animation as ani
from IPython.display import Image

plt.rcParams["patch.force_edgecolor"] = True
#rc('text', usetex=True)
from IPython.display import display # Allows the use of display() for DataFrames
import seaborn as sns
sns.set(style="whitegrid", palette="muted", color_codes=True)
sns.set_style("whitegrid", {'grid.linestyle': '--'})
red = sns.xkcd_rgb["light red"]
green = sns.xkcd_rgb["medium green"]
blue = sns.xkcd_rgb["denim blue"]

#カラム内の文字数。デフォルトは50
pd.set_option("display.max_colwidth", 100)

#行数
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
#
pd.options.display.float_format = '{:,.5f}'.format

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:



import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold

from sklearn import metrics
import json

import warnings
warnings.filterwarnings("ignore")

sys.path.append('..')
from lib.line_notif import send_message
from lib.utils import reduce_mem_usage, current_time, unpickle, to_pickle
from lib.utils import one_hot_encoder, apply_agg, multi_combine_categorical_feature
from lib.utils import import_data, get_split_indexer 

def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    """
    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling
    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric
    """
    maes = (y_true-y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()
    
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


DATA_VERSION = "v001"
TRIAL_NO = "001"
save_path = Path(f"../processed/{DATA_VERSION}")
save_path.mkdir(parents=True, exist_ok=True)
model_path = Path(f"../model/{DATA_VERSION}_{TRIAL_NO}")
model_path.mkdir(parents=True, exist_ok=True)
submit_path = Path(f"../submit/{DATA_VERSION}_{TRIAL_NO}")
submit_path.mkdir(parents=True, exist_ok=True)


print("start loading...")
train = unpickle(save_path/"train_002.df.pkl", )
print(f"train loaded.")
test  = unpickle(save_path/"test_002.df.pkl", )
print(f"test loaded.")
y = train["scalar_coupling_constant"]
train.drop("scalar_coupling_constant", axis=1, inplace=True)

train.set_index("id", inplace=True)
test.set_index("id", inplace=True)

print(f"train: {train.shape}, test: {test.shape}")

groups = unpickle(save_path/"lbl_molecule_name.pkl", )
groups = pd.Series(groups).value_counts().sort_index().values
print(f"groups: {groups.shape}")
print(train.shape, test.shape, y.shape)

categorical = ['atom_index_0', 'atom_index_1', 'atom_1', 'atom_0', 'type_0', 'type']
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
    'colsample_bytree': 0.8,
    
    'min_child_samples': 100,
    'reg_alpha': 0.1,
    'reg_lambda': 0.3,
}

lgb_train = lgb.Dataset(train, y, group=groups)

# Training settings
FOLD_NUM = 5
fold_seed = 71
folds = KFold(n_splits=FOLD_NUM, shuffle=True, random_state=fold_seed)
# folds = GroupKFold(n_splits=5, )

extraction_cb = ModelExtractionCallback()
callbacks = [
    ModelExtractionCallback()
]

# Fitting
print("start fitting...")
ret = lgb.cv(params=lgbm_params,
               train_set=lgb_train,
               categorical_feature=categorical,
               folds=folds,
               num_boost_round=30000,
               verbose_eval = 500,
               early_stopping_rounds=200,
               callbacks=callbacks,
               )
df_ret = pd.DataFrame(ret)
display(df_ret)
print("finish fitting.")

# Retrieving booster and training information.
proxy = extraction_cb.boosters_proxy
boosters = extraction_cb.raw_boosters
best_iteration = extraction_cb.best_iteration
to_pickle(model_path/'extraction_cb.pkl', extraction_cb)

# Create oof prediction result
print("create oof preds.")
fold_iter = folds.split(train, y)
oof_preds = np.zeros_like(y)
for n_fold, ((trn_idx, val_idx), booster) in enumerate(zip(fold_iter, boosters)):
    print(val_idx)
    valid = train.iloc[val_idx]
    oof_preds[val_idx] = booster.predict(valid, num_iteration=best_iteration)
print(f"mae on oof preds: {mean_absolute_error(y, oof_preds)}")
np.save(submit_path/'oof.npy', oof_preds)

# Averaging prediction result for test data.
y_pred_proba_list = proxy.predict(test, num_iteration=best_iteration)
y_pred_proba_avg = np.array(y_pred_proba_list).mean(axis=0)

sub = pd.read_csv('../input/sample_submission.csv')
sub['scalar_coupling_constant'] = y_pred_proba_avg
sub.to_csv(submit_path/'submission.csv', index=False)
sub.head()

print("finish.")