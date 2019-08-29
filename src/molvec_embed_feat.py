# coding: utf-8
import pandas as pd
import sys
sys.path.append('..')
from lib.utils import current_time, unpickle, to_pickle

# sklearn
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA, TruncatedSVD, FastICA, FactorAnalysis
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

df = unpickle("../processed/v003/mol_vec_df.pkl").set_index("molecule_name")

SEED = 71
N_COMP = 10
num_clusters2 = 10

fa   = FactorAnalysis(n_components=N_COMP, )
pca  = PCA(n_components=N_COMP, random_state=SEED)
tsvd = TruncatedSVD(n_components=N_COMP, random_state=SEED)
ica  = FastICA(n_components=N_COMP, random_state=SEED)
grp  = GaussianRandomProjection(n_components=N_COMP, eps=0.1, random_state=SEED)
srp  = SparseRandomProjection(n_components=N_COMP, dense_output=True, random_state=SEED)
mbkm = MiniBatchKMeans(n_clusters=num_clusters2, random_state=SEED)
tsne = TSNE(n_components=3, random_state=SEED)

ss = StandardScaler()
df_ss = pd.DataFrame(ss.fit_transform(df.fillna(df.mean(axis=0))), columns=df.columns)

decomp_cols = []
comp_results = []
comp_names = ["fa", "pca", "tsvd", "ica", "grp", "srp", "mbkm"] #, "tsne"] # removing tsne
for name, transform in zip(comp_names, [fa, pca, tsvd, ica, grp, srp, mbkm, tsne]):
    print(current_time(), "{} converting...".format(name), flush=True)
    n_components = N_COMP
    if name == 'mbkm':
        n_components = num_clusters2
    elif name == "tsne":
        n_components = 2
    df_results = pd.DataFrame(transform.fit_transform(df_ss))
    decomp_col = ["{0}_{1:02d}".format(name, i) for i in range(n_components)]
    df_results.columns = decomp_col
    decomp_cols.extend(decomp_col)
    df_results.reset_index(inplace=True)
    del df_results['index']
    comp_results.append(df_results)

comp_results_df = pd.concat(comp_results, axis=1)

comp_results_df = pd.concat([df.reset_index(drop=False)["molecule_name"],
                             comp_results_df.reset_index(drop=True)], axis=1)

to_pickle(f"../processed/v003/comp_molvec_df_{N_COMP}.pkl", comp_results_df)
print("finished.")