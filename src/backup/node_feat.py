import numpy as np
import pandas as pd
import os
import sys
from glob import glob
sys.path.append('..')
from lib.line_notif import send_message
from lib.utils import reduce_mem_usage, current_time, unpickle, to_pickle
from tqdm import tqdm_notebook as tqdm
import multiprocessing as mp

def func(data):
    graph_list = data["graph_list"]
    node_list = []
    for j in range(len(graph_list)):
        graph_name = graph_list[j]
        graph_name = graph_name.split("/")[-1].replace(".pickle","")
        g = unpickle(graph_list[j])
        node_df = pd.concat([structure[structure.molecule_name==graph_name][["molecule_name", "atom_index"]].reset_index(drop=True), 
                   pd.DataFrame(np.concatenate(g.node, -1), columns=[f"node_{i}" for i in range(13)])], axis=1)
        node_list += [node_df]
    return node_list

structure = pd.read_csv("../input/structures.csv")
graph_list = glob("../input/graph/*.pickle")
print(len(graph_list))
n_split = mp.cpu_count()
unit = np.ceil(len(graph_list) / n_split).astype(int)
indexer = [[unit * (i), unit * (i + 1)] for i in range(n_split)]

split_graph_list = []
for idx in indexer:
    split_graph_list.append(graph_list[idx[0]:idx[1]])

mp_data = [{"graph_list": m} for m in split_graph_list]

num_workers = mp.cpu_count()
with mp.Pool(num_workers) as executor:
    features_chunk = executor.map(func, mp_data)
    
concat_list = []
for i in range(len(features_chunk)):
    concat_list += features_chunk[i]

node_df = pd.concat(concat_list, axis=0)

to_pickle("../processed/v003/node_df.pkl", node_df)
