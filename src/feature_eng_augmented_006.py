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


def map_atom_info(df, atom_idx):
    df = pd.merge(df, structures, how='left',
                  left_on=['molecule_name', f'atom_index_{atom_idx}'],
                  right_on=['molecule_name', 'atom_index'])

    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sub = pd.read_csv('../input/sample_submission.csv')
# structures = pd.read_csv('../input/structures.csv')
structures = pd.read_csv(f'../input/rotated_structures_71.csv')

structures["mass"] = structures.atom.replace({"H":1,"C":12, "O":16, "N":14, "F":19})
mass_sum = structures.groupby("molecule_name")["mass"].sum()
mass_sum.name = "mass_sum"
mass_sum.index.name = "molecule_name"
structures = structures.merge(pd.DataFrame(mass_sum).reset_index(), on="molecule_name", how="left")

for axis in ["x", "y", "z"]:
    structures[f"{axis}_weighted"] = structures[axis]*structures.mass / structures.mass_sum

structures_origin = structures.groupby("molecule_name")[["x_weighted","y_weighted","z_weighted"]].sum()
structures_origin.columns = ["origin_x", "origin_y", "origin_z", ]
structures = structures.merge(structures_origin.reset_index(), on="molecule_name", how="left")
for axis in ["x", "y", "z"]:
    structures[f"{axis}_diff"] = structures[axis]-structures[f"origin_{axis}"]
structures["dist_from_origin"] = (structures.x_diff**2 + structures.y_diff**2 + structures.z_diff**2)**0.5
dist_origin_mean = pd.DataFrame(structures.groupby("molecule_name")["dist_from_origin"].mean()).reset_index()
dist_origin_mean.rename({"dist_from_origin":"f006:dist_origin_mean"},axis=1, inplace=True)
train = train.merge(dist_origin_mean, on="molecule_name", how="left")
test  = test.merge(dist_origin_mean, on="molecule_name", how="left")

structures.drop(["mass_sum", "x_weighted", "y_weighted", "z_weighted",
                 "origin_x", "origin_y", "origin_z",
                 "x_diff", "y_diff", "z_diff"], axis=1, inplace=True)
print("train 1")
train = map_atom_info(train, 0)
print("train 2")
train = map_atom_info(train, 1)
train.rename({"mass_x":"f006:mass_0", "mass_y":"f006:mass_1",
              "dist_from_origin_x":"f006:dist_from_origin_0",
              "dist_from_origin_y":"f006:dist_from_origin_1",}, axis=1, inplace=True)


print("test 1")
test = map_atom_info(test, 0)
print("test 2")
test = map_atom_info(test, 1)
print("test 3")
test.rename({"mass_x":"f006:mass_0", "mass_y":"f006:mass_1",
              "dist_from_origin_x":"f006:dist_from_origin_0",
              "dist_from_origin_y":"f006:dist_from_origin_1",}, axis=1, inplace=True)

Path("../processed/v003").mkdir(parents=True, exist_ok=True)
to_pickle("../processed/v003/train_augmented_006.df.pkl",train[["id",
                                           "f006:dist_origin_mean", "f006:mass_0", "f006:mass_1",
                                           "f006:dist_from_origin_0", "f006:dist_from_origin_1"]])

# to_pickle("../processed/v004/aug_test_006.df.pkl",test[["id",
#                                            "f006:dist_origin_mean", "f006:mass_0", "f006:mass_1",
#                                            "f006:dist_from_origin_0", "f006:dist_from_origin_1"]])

print("finished")
