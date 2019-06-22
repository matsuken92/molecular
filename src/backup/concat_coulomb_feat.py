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
import time

import datetime
ts_conv = np.vectorize(datetime.datetime.fromtimestamp) # 秒ut(10桁) ⇒ 日付

def merge_atom(df, df_distance):
    df_merge_0 = pd.merge(df, df_distance, left_on=['molecule_name', 'atom_index_0'],
                          right_on=['molecule_name', 'atom_index'])
    df_merge_0_1 = pd.merge(df_merge_0, df_distance, left_on=['molecule_name', 'atom_index_1'],
                            right_on=['molecule_name', 'atom_index'])
    del df_merge_0_1['atom_index_x'], df_merge_0_1['atom_index_y']
    return df_merge_0_1

def concat_coulomb_data(data_type="train"):
    assert data_type in ["train", "test"], f"data_type: {data_type}"
    df_list = []
    for f in np.sort(glob(f"../processed/v003/coulomb_feat_old/coulomb_{data_type}_*.pkl")):
        coulomb_test = np.load(f)
        df_list += [coulomb_test]
    coulomb_df = pd.concat(df_list, axis=0)

    df = pd.read_csv(f'../input/{data_type}.csv')
    assert df.molecule_name.nunique() == coulomb_df.molecule_name.nunique()

    start = time.time()
    df_train_dist = merge_atom(df, coulomb_df).sort_values("id")
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    return df_train_dist

df_train_dist = concat_coulomb_data(data_type="train")
drop_cols = ["molecule_name","atom_index_0","atom_index_1","type", "scalar_coupling_constant"]
df_train_dist.drop(drop_cols, axis=1).to_csv("../processed/v003/coulomb_interaction_train.csv")
del df_train_dist
gc.collect()

df_test_dist = concat_coulomb_data(data_type="test")
drop_cols = ["molecule_name","atom_index_0","atom_index_1","type"]
df_test_dist.drop(drop_cols, axis=1).to_csv("../processed/v003/coulomb_interaction_test.csv")

print("finished.")