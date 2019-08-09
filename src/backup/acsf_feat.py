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
from dscribe.descriptors import ACSF
from dscribe.core.system import System

sys.path.append('..')
from lib.line_notif import send_message
from lib.utils import matrics_rotate
from lib.utils import reduce_mem_usage, current_time, unpickle, to_pickle

SYMBOL=['H', 'C', 'N', 'O', 'F']
ACSF_GENERATOR = ACSF(
    species = SYMBOL,
    rcut = 6.0,
    g2_params=[[1, 1], [1, 2], [1, 3]],
    g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
)

def get_scsf(data):
    ret_list = []
    for molecule_name in data["mol_names"]:
        df = gb_structure.get_group(molecule_name)
        df = df.sort_values(['atom_index'], ascending=True)
        a   = df.atom.values.tolist()
        xyz = df[['x','y','z']].values

        atom = System(symbols=a, positions=xyz)
        acsf = ACSF_GENERATOR.create(atom)

        acsf_df = pd.DataFrame(acsf)
        acsf_df.columns = [f"acsf_{c}" for c in range(acsf_df.shape[1])]
        acsf_df = pd.concat([df[["molecule_name", "atom_index"]].reset_index(drop=True),
                             acsf_df.reset_index(drop=True)], axis=1)
        ret_list.append(acsf_df)
    return pd.concat(ret_list, axis=0)

print("loading structures")
structures = pd.read_csv("../input/structures.csv")

molecule_names = np.sort(structures.molecule_name.unique())
gb_structure = structures.groupby("molecule_name")

n_split = mp.cpu_count()
unit = np.ceil(len(molecule_names) / n_split).astype(int)
indexer = [[unit * (i), unit * (i + 1)] for i in range(n_split)]

split_mol_names = []
for idx in indexer:
    split_mol_names.append(molecule_names[idx[0]:idx[1]])

mp_data = [{"mol_names": m} for m in split_mol_names]

print("start multiprocessing")
num_workers = mp.cpu_count()
with mp.Pool(num_workers) as executor:
    features_chunk = executor.map(get_scsf, mp_data)

df = pd.concat(features_chunk)

to_pickle("../processed/v003/acsf_feat.pkl", df)
#df.to_csv("../processed/v003/acsf_feat.csv")
print("finished.")