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

TOP_IMP_ONLY = True

DATA_VERSION = __file__.split("_")[-2]
TRIAL_NO = __file__.split("_")[-1].replace(".py","")
# DATA_VERSION = "v001"
# TRIAL_NO = "006"

save_path = Path(f"../processed/{DATA_VERSION}")
save_path.mkdir(parents=True, exist_ok=True)
model_path = Path(f"../model/{DATA_VERSION}_{TRIAL_NO}")
model_path.mkdir(parents=True, exist_ok=True)
submit_path = Path(f"../submit/{DATA_VERSION}_{TRIAL_NO}")
submit_path.mkdir(parents=True, exist_ok=True)


test_id = np.load("../input/test_id.npy")
print(f"test_id.shape: {test_id.shape}")

print("start loading...")
train = unpickle(save_path/"train_002.df.pkl", )


print(f"train loaded.")
test  = unpickle(save_path/"test_002.df.pkl", )
print(f"test loaded.")
y = train["scalar_coupling_constant"]
train.drop("scalar_coupling_constant", axis=1, inplace=True)

train.set_index("id", inplace=True)
test.set_index("id", inplace=True)

groups = unpickle(save_path/"lbl_molecule_name.pkl", )


mol_type = int(sys.argv[1]) #1
print(f"mol_type: {mol_type}")

if mol_type==0:
    pass
elif mol_type==1:
    train_type_cut = np.load("../processed/v002/train_type1_cut.npy", )
    test_type_cut = np.load("../processed/v002/test_type1_cut.npy", )
    remove_cols = np.load("../model/v001_005/low_importance_1.npy", )
elif mol_type==2:
    train_type_cut = np.load("../processed/v002/train_type2_cut.npy", )
    test_type_cut = np.load("../processed/v002/test_type2_cut.npy", )
    remove_cols = np.load("../model/v001_005/low_importance_2.npy", )
elif mol_type==3:
    test_type_cut = np.load("../processed/v002/test_type3_cut.npy", )
    train_type_cut = np.load("../processed/v002/train_type3_cut.npy", )
    remove_cols = np.load("../model/v001_005/low_importance_3.npy", )
elif mol_type==4:
    test_type_cut = np.load("../processed/v002/test_type4_cut.npy", )
    train_type_cut = np.load("../processed/v002/train_type4_cut.npy", )
    remove_cols = np.load("../model/v001_005/low_importance_3.npy", )
else:
    assert False, f"mol_type should be 0, 1, 2, 3, 4. mol_type: {mol_type}"


use_cols = [c for c in train.columns if c not in remove_cols]

if mol_type in [1,2,3,4]:
    train = train[train_type_cut]
    test  = test[test_type_cut]
    y = y[train_type_cut]
    groups = groups[train_type_cut]
    test_id = test_id[test_type_cut]
    remove_cols += ['type']

if TOP_IMP_ONLY:
    train = train[use_cols]
    test  = test[use_cols]

print(f"train: {train.shape}, test: {test.shape}, y: {y.shape}, test_id: {test_id.shape}")

groups = pd.Series(groups).value_counts().sort_index().values
print(f"groups: {groups.shape}")

categorical = ['atom_index_0', 'atom_index_1', 'atom_1', 'atom_0', 'type_0', 'type']
categorical = [c for c in categorical if c not in remove_cols]

lgbm_params = {
    "boosting_type": "gbdt",
    'objective': 'regression',
    "metric": 'mae',
    # 'n_estimator': 10000,
    'n_jobs': -1,
    "seed": 71,
    "verbosity": -1,
    
    'learning_rate': 0.1,
    
    'max_depth': 45, #9,
    'num_leaves': 500, # 128,
    
    "subsample_freq": 1,
    "subsample": 0.5, #0.8,
    'colsample_bytree': 1, #0.8,
    
    'min_child_samples': 50, #80,
    'reg_alpha': 0.5, #0.1,
    'reg_lambda': 0.82, #0.3,
}
class ModelSaveCallback(object):
    def __init__(self, save_interval, save_path):
        self.save_interval = save_interval
        self.save_path = save_path

    def __call__(self, env):
        # Saving _CVBooster object.
        if env.iteration % self.save_interval == 0:
            for i, booster in enumerate(env.model.boosters):
                booster.save_model(f"{self.save_path}/booster_{mol_type}_{i:02d}_{env.iteration}.model")
                try:
                    rm_iter = env.iteration - self.save_interval
                    os.remove(f"{self.save_path}/booster_{mol_type}_{i:02d}_{rm_iter}.model")
                except:
                    pass

lgb_train = lgb.Dataset(train, y, group=groups)

# Training settings
FOLD_NUM = 5
fold_seed = 71
folds = KFold(n_splits=FOLD_NUM, shuffle=True, random_state=fold_seed)
# folds = GroupKFold(n_splits=5, )

extraction_cb = ModelExtractionCallback()
callbacks = [
    ModelSaveCallback(save_interval=1000, save_path=str(model_path)),
    extraction_cb
]

# Fitting
print("start fitting...")
ret = lgb.cv(params=lgbm_params,
               train_set=lgb_train,
               categorical_feature=categorical,
               folds=folds,
               num_boost_round=30000,
               verbose_eval = 500,
               early_stopping_rounds=500,
               callbacks=callbacks,
               )
df_ret = pd.DataFrame(ret)
display(df_ret.tail())
print("finish fitting.")

# Retrieving booster and training information.
proxy = extraction_cb.boosters_proxy
boosters = extraction_cb.raw_boosters
best_iteration = extraction_cb.best_iteration
print(f"best_iteration: {best_iteration}")
to_pickle(model_path/'extraction_cb.pkl', extraction_cb)
to_pickle(model_path/'boosters.pkl', boosters)
to_pickle(model_path/'proxy.pkl', proxy)

# Create oof prediction result
print("create oof preds.")
fold_iter = folds.split(train, y)
oof_preds = np.zeros_like(y)

for n_fold, ((trn_idx, val_idx), booster) in enumerate(zip(fold_iter, boosters)):
    print(val_idx)
    valid = train.iloc[val_idx]
    oof_preds[val_idx] = booster.predict(valid, num_iteration=best_iteration)
print(f"mae on oof preds: {mean_absolute_error(y, oof_preds)}")
df_oof = pd.DataFrame(index=train.index)
df_oof["scalar_coupling_constant"] = oof_preds
df_oof.to_csv(submit_path/f'oof_{mol_type}.csv', index=True)

# Averaging prediction result for test data.
y_pred_proba_list = proxy.predict(test, num_iteration=best_iteration)
y_pred_proba_avg = np.array(y_pred_proba_list).mean(axis=0)

sub = pd.DataFrame(index=test_id)
sub['scalar_coupling_constant'] = y_pred_proba_avg
sub.to_csv(submit_path/f'submission_{mol_type}.csv', index=False)
sub.head()

print("finish.")