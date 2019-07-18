
good_columns = [
'molecule_atom_index_0_dist_min',
'molecule_atom_index_0_dist_max',
'molecule_atom_index_1_dist_min',
'molecule_atom_index_0_dist_mean',
'molecule_atom_index_0_dist_std',
'dist', 'abs_dist',
'x_0', 'y_0', 'z_0',
'x_1', 'y_1', 'z_1',
'molecule_atom_index_1_dist_std',
'molecule_atom_index_1_dist_max',
'molecule_atom_index_1_dist_mean',
'molecule_atom_index_0_dist_max_diff',
'molecule_atom_index_0_dist_max_div',
'molecule_atom_index_0_dist_std_diff',
'molecule_atom_index_0_dist_std_div',
'atom_0_couples_count',
'molecule_atom_index_0_dist_min_div',
'molecule_atom_index_1_dist_std_diff',
'molecule_atom_index_0_dist_mean_div',
'atom_1_couples_count',
'molecule_atom_index_0_dist_mean_diff',
'molecule_couples',
'atom_index_1',
'molecule_dist_mean',
'molecule_atom_index_1_dist_max_diff',
'molecule_atom_index_0_y_1_std',
'molecule_atom_index_1_dist_mean_diff',
'molecule_atom_index_1_dist_std_div',
'molecule_atom_index_1_dist_mean_div',
'molecule_atom_index_1_dist_min_diff',
'molecule_atom_index_1_dist_min_div',
'molecule_atom_index_1_dist_max_div',
'molecule_atom_index_0_z_1_std',
'molecule_type_dist_std_diff',
'molecule_atom_1_dist_min_diff',
'molecule_atom_index_0_x_1_std',
'molecule_dist_min',
'molecule_atom_index_0_dist_min_diff',
'molecule_atom_index_0_y_1_mean_diff',
'molecule_type_dist_min',
'molecule_atom_1_dist_min_div',
'atom_index_0',
'molecule_dist_max',
'molecule_atom_1_dist_std_diff',
'molecule_type_dist_max',
'molecule_atom_index_0_y_1_max_diff',
'molecule_type_0_dist_std_diff',
'molecule_type_dist_mean_diff',
'molecule_atom_1_dist_mean',
'molecule_atom_index_0_y_1_mean_div',
'molecule_type_dist_mean_div',
'type', "f004:angle", "f004:angle_abs",
# "f003:cos_0_1", "f003:cos_1",
"f006:dist_origin_mean", # "f006:mass_0", "f006:mass_1",
"f006:dist_from_origin_0", "f006:dist_from_origin_1",
'Angle', 'Torsion', 'cos2T', 'cosT', # 'sp',
'dist_xy', 'dist_xz', 'dist_yz',
"C","H","N","O", # "F",
'eem_0', 'mmff94_0', 'gasteiger_0', 'qeq_0', 'qtpie_0', 'eem2015ha_0',
'eem2015hm_0', 'eem2015hn_0', 'eem2015ba_0', 'eem2015bm_0', 'eem2015bn_0',
'eem_1', 'mmff94_1', 'gasteiger_1', 'qeq_1', 'qtpie_1', 'eem2015ha_1',
'eem2015hm_1', 'eem2015hn_1', 'eem2015ba_1', 'eem2015bm_1', 'eem2015bn_1',
'dist_H_0_x', 'dist_H_1_x', 'dist_H_2_x', 'dist_H_3_x',
'dist_H_4_x', 'dist_C_0_x', 'dist_C_1_x', 'dist_C_2_x', 'dist_C_3_x',
'dist_C_4_x', 'dist_N_0_x', 'dist_N_1_x', 'dist_N_2_x', 'dist_N_3_x',
'dist_N_4_x', 'dist_O_0_x', 'dist_O_1_x', 'dist_O_2_x', 'dist_O_3_x',
'dist_F_0_x', 'dist_F_1_x', # 'dist_F_3_x', 'dist_O_4_x',
# 'dist_F_4_x', 'dist_F_2_x', 'dist_F_2_y',
'dist_H_0_y', 'dist_H_1_y', 'dist_H_2_y', 'dist_H_3_y',
'dist_H_4_y', 'dist_C_0_y', 'dist_C_1_y', 'dist_C_2_y', 'dist_C_3_y',
'dist_C_4_y', 'dist_N_0_y', 'dist_N_1_y', 'dist_N_2_y', 'dist_N_3_y',
'dist_N_4_y', 'dist_O_0_y', 'dist_O_1_y', 'dist_O_2_y', 'dist_O_3_y',
'dist_F_0_y', 'dist_F_1_y', # 'dist_F_3_y',
# 'dist_F_4_y', 'dist_O_4_y',
#'EN_x', 'rad_x', 'n_bonds_x', 'bond_lengths_mean_x',
#'EN_y', 'rad_y', 'n_bonds_y', 'bond_lengths_mean_y',
#'tda_radius',
"tda_max_radius", "tda_min_radius",
# "sum_radius", "tda_cocycles_shape",
# "tda_mean_radius",  "tda_num_circle"
"pca_exp_1", "pca_exp_2", "pca_exp_3",
]


rdkit_cols = ['id', 'a1_degree', 'a1_hybridization',
              'a1_inring', 'a1_inring3', 'a1_inring4', 'a1_inring5', 'a1_inring6',
              'a1_inring7', 'a1_inring8', 'a1_nb_h', 'a1_nb_o', 'a1_nb_c', 'a1_nb_n',
              'a1_nb_na', 'a0_nb_degree', 'a0_nb_hybridization', 'a0_nb_inring',
              'a0_nb_inring3', 'a0_nb_inring4', 'a0_nb_inring5', 'a0_nb_inring6',
              'a0_nb_inring7', 'a0_nb_inring8', 'a0_nb_nb_h', 'a0_nb_nb_o',
              'a0_nb_nb_c', 'a0_nb_nb_n', 'a0_nb_nb_na', 'x_a0_nb', 'y_a0_nb',
              'z_a0_nb', 'a1_nb_degree', 'a1_nb_hybridization', 'a1_nb_inring',
              'a1_nb_inring3', 'a1_nb_inring4', 'a1_nb_inring5', 'a1_nb_inring6',
              'a1_nb_inring7', 'a1_nb_inring8', 'a1_nb_nb_h', 'a1_nb_nb_o',
              'a1_nb_nb_c', 'a1_nb_nb_n', 'a1_nb_nb_na', 'x_a1_nb', 'y_a1_nb',
              'z_a1_nb', 'dist_to_type_mean']

babel_cols = ['id', 'Angle', 'Torsion', 'cos2T', 'cosT', 'sp']

remove_cols = """a0_nb_nb_na
a1_nb_nb_na
dist_N_4_y
N
a0_nb_inring8
dist_N_4_x
dist_O_3_x
a1_inring8
dist_F_1_y
dist_F_1_x
a1_nb_inring8
dist_O_3_y
a1_nb_inring7
dist_N_3_x
dist_N_3_y
a1_nb_h
a1_nb_na
a1_nb_inring6
a0_nb_inring7
O
dist_F_0_y
f004:angle_abs
dist_F_0_x
a1_nb_c
a1_nb_inring
atom_index_0
dist_N_2_y
f004:angle
molecule_atom_index_0_x_1_std
a1_inring7
a1_nb_nb_o
molecule_atom_index_0_z_1_std
a1_nb_nb_c
a1_nb_inring3
z_a0_nb
dist_N_2_x
dist_O_2_x
z_0
x_a0_nb
C
molecule_atom_index_0_y_1_mean_div
z_1
dist_O_2_y
x_0
molecule_atom_1_dist_mean
molecule_dist_max
molecule_atom_index_0_y_1_std
a1_nb_hybridization
z_a1_nb
a0_nb_inring6
a1_nb_nb_h
x_a1_nb
a1_nb_inring5
molecule_atom_index_0_y_1_mean_diff
x_1
f006:dist_origin_mean
a1_nb_inring4
dist_yz
tda_min_radius
molecule_atom_index_0_y_1_max_diff
molecule_couples
a1_inring6
eem2015ha_0
a1_nb_nb_n
y_0
y_a0_nb
H
Torsion
dist_xy
molecule_dist_mean
a1_nb_degree
eem2015ba_0
a1_inring
atom_index_1
y_a1_nb
atom_1_couples_count
molecule_atom_index_0_dist_std
eem2015hn_0
dist_xz
y_1
dist_N_1_y
a0_nb_nb_h
molecule_atom_index_1_dist_std
atom_0_couples_count
eem2015bn_0
molecule_atom_index_1_dist_mean
eem2015hm_0
dist_H_4_x
a0_nb_nb_c
eem2015hn_1
abs_dist
a1_nb_n
a0_nb_degree
molecule_type_dist_mean_div
f006:dist_from_origin_0
molecule_atom_index_1_dist_std_div
dist_H_3_x
""".split("\n")