#!/usr/bin/env python
# coding: utf-8

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


def map_atom_info(df_1, df_2, atom_idx):
    df = pd.merge(df_1, df_2, how = 'left',
                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                  right_on = ['molecule_name',  'atom_index'])
    df = df.drop('atom_index', axis=1)

    return df


def make_features(df):
    df['dx']=df['x_1']-df['x_0']
    df['dy']=df['y_1']-df['y_0']
    df['dz']=df['z_1']-df['z_0']
    df['distance']=(df['dx']**2+df['dy']**2+df['dz']**2)**(1/2)
    return df


def feat(df):
    df_temp = df.loc[:, ["molecule_name", "atom_index_0", "atom_index_1", "distance", "x_0", "y_0", "z_0", "x_1", "y_1",
                         "z_1"]].copy()
    df_temp_ = df_temp.copy()
    df_temp_ = df_temp_.rename(columns={'atom_index_0': 'atom_index_1',
                                        'atom_index_1': 'atom_index_0',
                                        'x_0': 'x_1',
                                        'y_0': 'y_1',
                                        'z_0': 'z_1',
                                        'x_1': 'x_0',
                                        'y_1': 'y_0',
                                        'z_1': 'z_0'})
    df_temp = pd.concat((df_temp, df_temp_), axis=0)

    df_temp["min_distance"] = df_temp.groupby(['molecule_name', 'atom_index_0'])['distance'].transform('min')
    df_temp = df_temp[df_temp["min_distance"] == df_temp["distance"]]

    df_temp = df_temp.drop(['x_0', 'y_0', 'z_0', 'min_distance'], axis=1)
    df_temp = df_temp.rename(columns={'atom_index_0': 'atom_index',
                                      'atom_index_1': 'atom_index_closest',
                                      'distance': 'distance_closest',
                                      'x_1': 'x_closest',
                                      'y_1': 'y_closest',
                                      'z_1': 'z_closest'})
    return df_temp


def add_cos_features(df):
    df["distance_0"] = ((df['x_0'] - df['x_closest_0']) ** 2 +
                        (df['y_0'] - df['y_closest_0']) ** 2 +
                        (df['z_0'] - df['z_closest_0']) ** 2) ** (1 / 2)
    df["distance_1"] = ((df['x_1'] - df['x_closest_1']) ** 2 + (df['y_1'] - df['y_closest_1']) ** 2 + (
                df['z_1'] - df['z_closest_1']) ** 2) ** (1 / 2)
    df["vec_0_x"] = (df['x_0'] - df['x_closest_0']) / df["distance_0"]
    df["vec_0_y"] = (df['y_0'] - df['y_closest_0']) / df["distance_0"]
    df["vec_0_z"] = (df['z_0'] - df['z_closest_0']) / df["distance_0"]
    df["vec_1_x"] = (df['x_1'] - df['x_closest_1']) / df["distance_1"]
    df["vec_1_y"] = (df['y_1'] - df['y_closest_1']) / df["distance_1"]
    df["vec_1_z"] = (df['z_1'] - df['z_closest_1']) / df["distance_1"]
    df["vec_x"] = (df['x_1'] - df['x_0']) / df["distance"]
    df["vec_y"] = (df['y_1'] - df['y_0']) / df["distance"]
    df["vec_z"] = (df['z_1'] - df['z_0']) / df["distance"]
    df["cos_0_1"] = df["vec_0_x"] * df["vec_1_x"] + df["vec_0_y"] * df["vec_1_y"] + df["vec_0_z"] * df["vec_1_z"]
    df["cos_0"] = df["vec_0_x"] * df["vec_x"] + df["vec_0_y"] * df["vec_y"] + df["vec_0_z"] * df["vec_z"]
    df["cos_1"] = df["vec_1_x"] * df["vec_x"] + df["vec_1_y"] * df["vec_y"] + df["vec_1_z"] * df["vec_z"]
    df = df.drop(['vec_0_x', 'vec_0_y', 'vec_0_z', 'vec_1_x', 'vec_1_y', 'vec_1_z', 'vec_x', 'vec_y', 'vec_z'], axis=1)
    return df


df_train = pd.read_csv('../input/train.csv')
n_train = df_train.shape[0]
print(f"n_train: {n_train}")
df_test = pd.read_csv('../input/test.csv')
n_test = df_test.shape[0]
print(f"n_test: {n_test}")

# df_struct = pd.read_csv('../input/structures.csv')
df_struct = pd.read_csv(f'{file_folder}/rotated_structures_71.csv')

for atom_idx in [0, 1]:
    df_train = map_atom_info(df_train, df_struct, atom_idx)
    df_test = map_atom_info(df_test, df_struct, atom_idx)
    df_train = df_train.rename(columns={'atom': f'atom_{atom_idx}',
                                        'x': f'x_{atom_idx}',
                                        'y': f'y_{atom_idx}',
                                        'z': f'z_{atom_idx}'})
    df_test = df_test.rename(columns={'atom': f'atom_{atom_idx}',
                                      'x': f'x_{atom_idx}',
                                      'y': f'y_{atom_idx}',
                                      'z': f'z_{atom_idx}'})

df_train = make_features(df_train)
assert n_train == df_train.shape[0], f"{n_train} {df_train.shape[0]}"
df_test = make_features(df_test)
assert n_test == df_test.shape[0], f"{n_test} {df_test.shape[0]}"

df_train_ = feat(df_train)
df_test_ = feat(df_test)

for atom_idx in [0, 1]:
    df_train = map_atom_info(df_train, df_train_, atom_idx)
    df_train = df_train.rename(columns={'atom_index_closest': f'atom_index_closest_{atom_idx}',
                                        'distance_closest': f'distance_closest_{atom_idx}',
                                        'x_closest': f'x_closest_{atom_idx}',
                                        'y_closest': f'y_closest_{atom_idx}',
                                        'z_closest': f'z_closest_{atom_idx}'})

    df_test = map_atom_info(df_test, df_test_, atom_idx)
    df_test = df_test.rename(columns={'atom_index_closest': f'atom_index_closest_{atom_idx}',
                                      'distance_closest': f'distance_closest_{atom_idx}',
                                      'x_closest': f'x_closest_{atom_idx}',
                                      'y_closest': f'y_closest_{atom_idx}',
                                      'z_closest': f'z_closest_{atom_idx}'})

# assert n_train == df_train.shape[0], f"{n_train} {df_train.shape[0]}"
# assert n_test == df_test.shape[0], f"{n_test} {df_test.shape[0]}"

cnt = df_train.id.value_counts()

duplicate_ids = cnt[cnt==2].index.values
reduce_duplicates_df = df_train[df_train.id.isin(duplicate_ids)].groupby("id").first()
df_train = pd.concat([df_train[~df_train.id.isin(duplicate_ids)], reduce_duplicates_df], axis=0)
df_train = add_cos_features(df_train)
df_test  = add_cos_features(df_test)

df_train.columns = [f"f003:{c}" for c in df_train.columns]
df_test.columns = [f"f003:{c}" for c in df_test.columns]
df_train.rename({"f003:id":"id"}, axis=1, inplace=True)
df_test.rename({"f003:id":"id"}, axis=1, inplace=True)

# idが抜けていたところを埋める
train_org = pd.read_csv('../input/train.csv')
train_org_s = train_org[train_org.molecule_name == "dsgdb9nsd_059818"]
train_org_s

for i, d in df_train[df_train.id.isna()].iterrows():
    # display(d)
    idx0 = d["f003:atom_index_0"]
    idx1 = d["f003:atom_index_1"]
    id_ = train_org_s.query(f"atom_index_0 == {idx0} and atom_index_1=={idx1}").iloc[0].id
    df_train.loc[d.name, "id"] = id_

df_train[df_train.id.isna()]