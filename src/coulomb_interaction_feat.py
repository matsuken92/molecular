import os
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

import multiprocessing as mp
import sys
sys.path.append('..')
from lib.line_notif import send_message
from lib.utils import reduce_mem_usage, current_time, unpickle, to_pickle


def get_dist_matrix(df_structures, molecule):
    df_temp = df_structures.query('molecule_name == "{}"'.format(molecule))
    locs = df_temp[['x','y','z']].values
    num_atoms = len(locs)
    loc_tile = np.tile(locs.T, (num_atoms,1,1))
    dist_mat = ((loc_tile - loc_tile.T)**2).sum(axis=1)
    return dist_mat

def assign_atoms_index(df, molecule):
    se_0 = df.query('molecule_name == "{}"'.format(molecule))['atom_index_0']
    se_1 = df.query('molecule_name == "{}"'.format(molecule))['atom_index_1']
    assign_idx = pd.concat([se_0, se_1]).unique()
    assign_idx.sort()
    return assign_idx


def get_pickup_dist_matrix(df, df_structures, molecule, num_pickup=5, atoms=['H', 'C', 'N', 'O', 'F']):
    pickup_dist_matrix = np.zeros([0, len(atoms) * num_pickup])
    assigned_idxs = assign_atoms_index(df, molecule)  # [0, 1, 2, 3, 4, 5, 6] -> [1, 2, 3, 4, 5, 6]
    dist_mat = get_dist_matrix(df_structures, molecule)
    for idx in assigned_idxs:  # [1, 2, 3, 4, 5, 6] -> [2]

        dist_arr = dist_mat[idx]  # (7, 7) -> (7, )

        atoms_mole = df_structures.query('molecule_name == "{}"'.format(molecule))[
            'atom'].values  # ['O', 'C', 'C', 'N', 'H', 'H', 'H']
        atoms_mole_idx = df_structures.query('molecule_name == "{}"'.format(molecule))[
            'atom_index'].values  # [0, 1, 2, 3, 4, 5, 6]

        mask_atoms_mole_idx = atoms_mole_idx != idx  # [ True,  True, False,  True,  True,  True,  True]
        masked_atoms = atoms_mole[mask_atoms_mole_idx]  # ['O', 'C', 'N', 'H', 'H', 'H']
        masked_atoms_idx = atoms_mole_idx[mask_atoms_mole_idx]  # [0, 1, 3, 4, 5, 6]
        masked_dist_arr = dist_arr[
            mask_atoms_mole_idx]  # [ 5.48387003, 2.15181049, 1.33269675, 10.0578779, 4.34733927, 4.34727838]

        sorting_idx = np.argsort(masked_dist_arr)  # [2, 1, 5, 4, 0, 3]
        sorted_atoms_idx = masked_atoms_idx[sorting_idx]  # [3, 1, 6, 5, 0, 4]
        sorted_atoms = masked_atoms[sorting_idx]  # ['N', 'C', 'H', 'H', 'O', 'H']
        sorted_dist_arr = 1 / masked_dist_arr[
            sorting_idx]  # [0.75035825,0.46472494,0.23002898,0.23002576,0.18235297,0.09942455]

        target_matrix = np.zeros([len(atoms), num_pickup])
        for a, atom in enumerate(atoms):
            pickup_atom = sorted_atoms == atom  # [False, False,  True,  True, False,  True]
            pickup_dist = sorted_dist_arr[pickup_atom]  # [0.23002898, 0.23002576, 0.09942455]
            num_atom = len(pickup_dist)
            if num_atom > num_pickup:
                target_matrix[a, :] = pickup_dist[:num_pickup]
            else:
                target_matrix[a, :num_atom] = pickup_dist
        pickup_dist_matrix = np.vstack([pickup_dist_matrix, target_matrix.reshape(-1)])
    return pickup_dist_matrix

def get_dist_mat(data):
    df_data = data["df"]
    mol = data["mol"]
    #print(mol)
    assigned_idxs = assign_atoms_index(df_data, mol)
    dist_mat_mole = get_pickup_dist_matrix(df_data, df_structures, mol, num_pickup=num)
    mol_name_arr = [mol] * len(assigned_idxs)

    return (mol_name_arr, assigned_idxs, dist_mat_mole)


def feat(data, num = 5):
    try:
        number = data["number"]
        df = data["df"]
        mols = data["mols"]
        #mols = df['molecule_name'].unique()
        # dist_mat = np.zeros([0, num*5])
        # atoms_idx = np.zeros([0], dtype=np.int32)
        # molecule_names = np.empty([0])

        start = time.time()

        mp_data = [{"mol": mol, "df":df} for mol in mols]
        dist_mats = [get_dist_mat(d) for d in mp_data]
        #dist_mats = Parallel(n_jobs=NCORES)(delayed(get_dist_mat)(data) for data in mp_data)

        molecule_names = np.hstack([x[0] for x in dist_mats])
        atoms_idx = np.hstack([x[1] for x in dist_mats])
        dist_mat = np.vstack([x[2] for x in dist_mats])

        col_name_list = []
        atoms = ['H', 'C', 'N', 'O', 'F']
        for a in atoms:
            for n in range(num):
                col_name_list.append('dist_{}_{}'.format(a, n))

        se_mole = pd.Series(molecule_names, name='molecule_name')
        se_atom_idx = pd.Series(atoms_idx, name='atom_index')
        df_dist = pd.DataFrame(dist_mat, columns=col_name_list)
        df_distance = pd.concat([se_mole, se_atom_idx,df_dist], axis=1)

        elapsed_time = time.time() - start
        print ("elapsed_time:{0:.2f}".format(elapsed_time) + "[sec]")

        first_mol_name = df.molecule_name.iloc[0].replace("dsgdb9nsd_","")
        last_mol_name = df.molecule_name.iloc[-1].replace("dsgdb9nsd_","")

        np.save(f"mols_{first_mol_name}_{last_mol_name}.npy", mols)

        to_pickle(coulomb_feat/f"coulomb_train_{number}_{first_mol_name}_{last_mol_name}.pkl", df_distance)
    except Exception as e:
        print(e)
        print(mols[:5])
        print(mols[-5:])
        display(df.head())
        display(df.tail())
        raise e
    #return df_distance

FOLDER = '../input/'
OUTPUT = '../input/preprocessed/'
NCORES = mp.cpu_count()
os.listdir(FOLDER)

df_train = pd.read_csv(FOLDER + 'train.csv')
df_test = pd.read_csv(FOLDER + 'test.csv')
df_structures = pd.read_csv(FOLDER + 'structures.csv')

num = 5
from pathlib import Path
coulomb_feat = Path("../processed/v003/coulomb_feat")
coulomb_feat.mkdir(exist_ok=True, parents=True)

processed_mol_names = pd.read_csv("./processed_mol_names.csv", index_col=0)


############################################################################################
# Train
mols = df_train['molecule_name'].unique()
if True:
    mols = np.setdiff1d(mols, processed_mol_names.molecule_name)

n_each_split = len(mols) // NCORES + 1
mols_split = [mols[n_each_split*i:n_each_split*(i+1)] for i in range(NCORES)]

mp_data = [{"number": i,
            "df": df_train[df_train.molecule_name.isin(mols_split[i])],
            "mols": mols_split[i]} for i in range(NCORES)]

num_workers = NCORES
with mp.Pool(num_workers) as executor:
    features_chunk = executor.map(feat, mp_data)


############################################################################################
# Test
# mols = df_test['molecule_name'].unique()
# n_each_split = len(mols) // NCORES + 1
# mols_split = [mols[n_each_split*i:n_each_split*(i+1)] for i in range(NCORES)]
#
# mp_data = [{"number": f"test-{i:08d}",
#             "df": df_test[df_test.molecule_name.isin(mols_split[i])],
#             "mols": mols_split[i]} for i in range(NCORES)]
#
# num_workers = NCORES
# with mp.Pool(num_workers) as executor:
#     features_chunk = executor.map(feat, mp_data)
#

#
# def merge_atom(df, df_distance):
#     df_merge_0 = pd.merge(df, df_distance, left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index'])
#     df_merge_0_1 = pd.merge(df_merge_0, df_distance, left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index'])
#     del df_merge_0_1['atom_index_x'], df_merge_0_1['atom_index_y']
#     return df_merge_0_1
#
# start = time.time()
# df_train_dist = merge_atom(df_train, distance_train).sort_values("id")
# elapsed_time = time.time() - start
# print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
#
# start = time.time()
# df_test_dist = merge_atom(df_test, distance_test).sort_values("id")
# elapsed_time = time.time() - start
# print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
#
# drop_cols = ["molecule_name	atom_index_0","atom_index_1","type","scalar_coupling_constant"]
# df_train_dist.drop(drop_cols, axis=1).to_csv("../processed/v003/coulomb_interaction_train.csv")
# df_test_dist.drop(drop_cols, axis=1).to_csv("../processed/v003/coulomb_interaction_test.csv")

print("finished.")