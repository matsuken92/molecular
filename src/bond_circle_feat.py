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
import networkx as nx


def get_bonds(molecule_name, structures):
    """Generates a set of bonds from atomic cartesian coordinates"""
    atomic_radii = dict(C=0.77, F=0.71, H=0.38, N=0.75, O=0.73)
    cpk_colors = dict(C='black', F='green', H='white', N='blue', O='red')

    molecule = structures[structures.molecule_name == molecule_name]
    coordinates = molecule[['x', 'y', 'z']].values
    elements = molecule.atom.tolist()
    radii = [atomic_radii[element] for element in elements]

    ids = np.arange(coordinates.shape[0])
    n_atom = len(ids)
    bonds = dict()
    coordinates_compare, radii_compare, ids_compare = coordinates, radii, ids

    for _ in range(len(ids)):
        coordinates_compare = np.roll(coordinates_compare, -1, axis=0)
        radii_compare = np.roll(radii_compare, -1, axis=0)
        ids_compare = np.roll(ids_compare, -1, axis=0)
        distances = np.linalg.norm(coordinates - coordinates_compare, axis=1)
        bond_distances = (radii + radii_compare) * 1.3
        mask = np.logical_and(distances > 0.1, distances < bond_distances)
        distances = distances.round(2)
        new_bonds = {frozenset([i, j]): dist for i, j, dist in zip(ids[mask], ids_compare[mask], distances[mask])}
        bonds.update(new_bonds)
    return bonds, n_atom


def num_cycles(molecule_name):
    bonds, n_atom = get_bonds(molecule_name, structures)
    link_list = []
    for b in bonds:
        b = list(b)
        link_list.append([b[0], b[1]])
    G = nx.Graph(link_list)
    ret = nx.minimum_cycle_basis(G)
    return len(ret)


def cycles_sizes(molecule_name):
    bonds, n_atom = get_bonds(molecule_name, structures)
    link_list = []
    for b in bonds:
        b = list(b)
        link_list.append([b[0], b[1]])
    G = nx.Graph(link_list)
    ret = nx.minimum_cycle_basis(G)
    return [len(r) for r in ret]


def func(data):
    cyclic_check_result = []
    error_mols = []
    mol_names = data["mol_names"]
    for m in mol_names:
        try:
            cyclic_check_result += [cycles_sizes(m)]  # [num_cycles(m)]
        except Exception as e:
            error_mols += [m]
            cyclic_check_result += [0]
            # raise e

    return pd.DataFrame({"molecule_name": mol_names, "circle_size": cyclic_check_result})


def calc_statistics(circle_size):
    # print(circle_size)
    return {"n_circle":len(circle_size),
            "sum_circle_size": np.sum(circle_size) if len(circle_size)>0 else 0,
            "max_circle_size": np.max(circle_size) if len(circle_size)>0 else 0,
            "min_circle_size": np.min(circle_size) if len(circle_size)>0 else 0,
            "mean_circle_size": np.mean(circle_size) if len(circle_size)>0 else 0}


train = pd.read_csv("../input/train.csv")
structures = pd.read_csv("../input/structures.csv")
mol_names = structures.molecule_name.unique()

n_split = mp.cpu_count()
unit = np.ceil(len(mol_names) / n_split).astype(int)
indexer = [[unit * (i), unit * (i + 1)] for i in range(n_split)]

split_mol_names = []
for idx in indexer:
    split_mol_names.append(mol_names[idx[0]:idx[1]])

mp_data = [{"mol_names": m} for m in split_mol_names]
num_workers = n_split
with mp.Pool(num_workers) as executor:
    features_chunk = executor.map(func, mp_data)

n_circle_df = pd.concat(features_chunk, axis=0).reset_index(drop=True)

circle_feat = n_circle_df.circle_size.apply(calc_statistics)
circle_feat_df = pd.DataFrame(circle_feat.values.tolist())
n_circle_df_ = pd.concat([n_circle_df, circle_feat_df], axis=1)

n_circle_df_.to_csv("../processed/v003/circle_feat_df.csv")
print("finished.")

