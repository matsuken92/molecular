
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
'1J1st_IsPolarHydrogen', '1Jlast_IsPolarHydrogen',
'1Jlast_GetFormalCharge', '1Jlast_GetAtomicNum',
'1Jlast_GetSpinMultiplicity', '1Jlast_GetValence', '1Jlast_GetHyb',
'1Jlast_GetImplicitValence', '1Jlast_GetHvyValence',
'1Jlast_GetHeteroValence', '1Jlast_GetPartialCharge',
'1J1st_CountFreeOxygens', '1J1st_ExplicitHydrogenCount',
'1J1st_MemberOfRingCount', '1J1st_MemberOfRingSize',
'1J1st_SmallestBondAngle', '1J1st_AverageBondAngle',
'1J1st_AveSmallestBondAngle_diff', '1J1st_BOSum', '1J1st_IsAromatic',
'1J1st_IsInRing', '1J1st_IsCarboxylOxygen', '1J1st_IsNitroOxygen',
'1J1st_IsChiral', '1J1st_IsAxial', '2J1st_IsPolarHydrogen',
'2Jlast_IsPolarHydrogen', '2J2nd_isInRing', '2J2nd_isAroma',
'2J2nd_isChiral', '2J2nd_IsAxial', '2J2nd_AverageBondAngle',
'2J2nd_SmallestBondAngle', '2J2nd_MemberOfRingSize', '2J2nd__C',
'2J2nd__N', '2J2nd__O', '2J2nd__nan', '3J1st_IsPolarHydrogen',
'3Jlast_IsPolarHydrogen', '3J2nd_isInRing', '3J2nd_isAroma',
'3J2nd_isChiral', '3J2nd_IsAxial', '3J2nd_AverageBondAngle',
'3J2nd_SmallestBondAngle', '3J2nd_MemberOfRingSize', '3J3rd_isInRing',
'3J3rd_isAroma', '3J3rd_isChiral', '3J3rd_IsAxial',
'3J3rd_AverageBondAngle', '3J3rd_SmallestBondAngle',
'3J3rd_MemberOfRingSize', '3Jlast_isInRing', '3Jlast_isAroma',
'3Jlast_isChiral', '3Jlast_IsAxial', '3Jlast_AverageBondAngle',
'3Jlast_SmallestBondAngle', '3Jlast_MemberOfRingSize',
'interBond_BondOrder', 'interBond_Length', 'interBond_EquibLength',
'interBond_IsAromatic', 'interBond_IsInRing', 'interBond_IsRotor',
'interBond_IsAmider', 'interBond_IsEster', 'interBond_IsCarbonyl',
'interBond_IsClosure', 'interBond_IsUp', 'interBond_IsDown',
'interBond_IsWedge', 'interBond_IsHash',
'interBond_IsDoubleBondGeometry', '3J2nd__C', '3J2nd__N', '3J2nd__O',
'3J2nd__nan', '3J3rd__C', '3J3rd__N', '3J3rd__O', '3J3rd__nan',
'mean_dist_C_from2nd',
'mean_angle_C_from2nd',
'd_O_from2nd_ratio_0',
'mean_dist_ratio_C_from2nd',
'mean_dist_ratio_O_from1st',
'mean_dist_ratio_O_from2nd',
'd_O_from1st_ratio_0',
'mean_angle_O_from2nd',
'd_O_from1st_0',
'mean_dist_C_from1st',
"max_circle_size","mean_circle_size","min_circle_size","n_circle","sum_circle_size",
'2Jd_idx1_2nd',
'2Jd_idx0_2nd',
'2Jangle_from2nd_max',
'2Jangle_from2nd_min',
'3Jd_idx1_2nd',
'2JGetHeteroValence',
'2Jdist_from2nd_min',
'2JHasAlphaBetaUnsat',
'3Jangle_from2nd_max',
'2Jangle_from2nd_mean',
'2Jdist_from2nd_mean',
'3Jangle_from2nd_mean',
'3JGetPartialCharge',
'3Jdist_from2nd_var',
'2Jdist_from2nd_max_mean_diff',
'2JGetPartialCharge',
'3JExplicitHydrogenCount',
'3Jdist_from2nd_mean',
'2JExplicitHydrogenCount',
'2Jdist_from2nd_var',
'3Jdist_from2nd_max_mean_diff',
#### seg_H1J_bond_extension1
'1J_ex1_angle_fromEx1_mean',
'1J_ex1_angle_fromEx1_min',
'1J_ex1_angle_fromEx1_std',
'1J_ex1_Angle_0_1_max',
'1J_ex1_Angle_0_1_min',
'1J_ex1_dist_from_first_std',
'1J_ex1_dist_from_first_max_min_diff',
'1J_ex1_Angle_0_1_mean',
'1J_ex1_AverageBondAngle_max',
'1J_ex1_dist_1_std',
'1J_ex1_SmallestBondAngle_max',
'1J_ex1_dist_1_min',
'1J_ex1_dist_0_min',
'1J_ex1_dist_0_std',
'1J_ex1_dist_from_first_min',
'1J_ex1_cos2T_F_L_EX1_mean',
'1J_ex1_dist_from_first_max',
'1J_ex1_dist_0_mean',
'1J_ex1_dist_0_max_min_diff',
'1J_ex1_cos2T_F_L_EX1_std',
'1J_ex1_dist_1_max_min_diff',
'1J_ex1_dist_from_first_mean',
'1J_ex1_angle_fromEx1_max',
'1J_ex1_dist_1_mean',
'1J_ex1_Angle_0_1_std',
'1J_ex1_angle_fromEx1_max_min_diff',
'acsf_0_0', 'acsf_1_0', 'acsf_2_0', 'acsf_3_0', 'acsf_4_0', 'acsf_5_0',
'acsf_6_0', 'acsf_7_0', 'acsf_8_0', 'acsf_9_0', 'acsf_10_0',
'acsf_11_0', 'acsf_12_0', 'acsf_13_0', 'acsf_14_0', 'acsf_15_0',
'acsf_16_0', 'acsf_17_0', 'acsf_18_0', 'acsf_19_0', 'acsf_20_0',
'acsf_21_0', 'acsf_22_0', 'acsf_23_0', 'acsf_24_0', 'acsf_25_0',
'acsf_26_0', 'acsf_27_0', 'acsf_28_0', 'acsf_29_0', 'acsf_30_0',
'acsf_31_0', 'acsf_32_0', 'acsf_33_0', 'acsf_34_0', 'acsf_35_0',
'acsf_36_0', 'acsf_37_0', 'acsf_38_0', 'acsf_39_0', 'acsf_40_0',
'acsf_41_0', 'acsf_42_0', 'acsf_43_0', 'acsf_44_0', 'acsf_45_0',
'acsf_46_0', 'acsf_47_0', 'acsf_48_0', 'acsf_49_0', 'acsf_50_0',
'acsf_51_0', 'acsf_52_0', 'acsf_53_0', 'acsf_54_0', 'acsf_55_0',
'acsf_56_0', 'acsf_57_0', 'acsf_58_0', 'acsf_59_0', 'acsf_60_0',
'acsf_61_0', 'acsf_62_0', 'acsf_63_0', 'acsf_64_0', 'acsf_65_0',
'acsf_66_0', 'acsf_67_0', 'acsf_68_0', 'acsf_69_0', 'acsf_70_0',
'acsf_71_0', 'acsf_72_0', 'acsf_73_0', 'acsf_74_0', 'acsf_75_0',
'acsf_76_0', 'acsf_77_0', 'acsf_78_0', 'acsf_79_0',
'acsf_0_1', 'acsf_1_1', 'acsf_2_1', 'acsf_3_1', 'acsf_4_1', 'acsf_5_1',
'acsf_6_1', 'acsf_7_1', 'acsf_8_1', 'acsf_9_1', 'acsf_10_1',
'acsf_11_1', 'acsf_12_1', 'acsf_13_1', 'acsf_14_1', 'acsf_15_1',
'acsf_16_1', 'acsf_17_1', 'acsf_18_1', 'acsf_19_1', 'acsf_20_1',
'acsf_21_1', 'acsf_22_1', 'acsf_23_1', 'acsf_24_1', 'acsf_25_1',
'acsf_26_1', 'acsf_27_1', 'acsf_28_1', 'acsf_29_1', 'acsf_30_1',
'acsf_31_1', 'acsf_32_1', 'acsf_33_1', 'acsf_34_1', 'acsf_35_1',
'acsf_36_1', 'acsf_37_1', 'acsf_38_1', 'acsf_39_1', 'acsf_40_1',
'acsf_41_1', 'acsf_42_1', 'acsf_43_1', 'acsf_44_1', 'acsf_45_1',
'acsf_46_1', 'acsf_47_1', 'acsf_48_1', 'acsf_49_1', 'acsf_50_1',
'acsf_51_1', 'acsf_52_1', 'acsf_53_1', 'acsf_54_1', 'acsf_55_1',
'acsf_56_1', 'acsf_57_1', 'acsf_58_1', 'acsf_59_1', 'acsf_60_1',
'acsf_61_1', 'acsf_62_1', 'acsf_63_1', 'acsf_64_1', 'acsf_65_1',
'acsf_66_1', 'acsf_67_1', 'acsf_68_1', 'acsf_69_1', 'acsf_70_1',
'acsf_71_1', 'acsf_72_1', 'acsf_73_1', 'acsf_74_1', 'acsf_75_1',
'acsf_76_1', 'acsf_77_1', 'acsf_78_1', 'acsf_79_1',
 'node_0_0',
 'node_1_0',
 'node_2_0',
 'node_3_0',
 'node_4_0',
 'node_5_0',
 'node_6_0',
 'node_7_0',
 'node_8_0',
 'node_9_0',
 'node_10_0',
 'node_11_0',
 'node_12_0',
 'node_0_1',
 'node_1_1',
 'node_2_1',
 'node_3_1',
 'node_4_1',
 'node_5_1',
 'node_6_1',
 'node_7_1',
 'node_8_1',
 'node_9_1',
 'node_10_1',
 'node_11_1',
 'node_12_1',
 'atom_2', 'atom_3', 'atom_4', 'atom_5', 'atom_6', 'atom_7', 'atom_8', 'atom_9',
 'd_1_0', 'd_2_0', 'd_2_1', 'd_3_0', 'd_3_1', 'd_3_2', 'd_4_0', 'd_4_1', 'd_4_2', 'd_4_3', 'd_5_0',
 'd_5_1', 'd_5_2', 'd_5_3', 'd_6_0', 'd_6_1', 'd_6_2', 'd_6_3', 'd_7_0', 'd_7_1', 'd_7_2', 'd_7_3',
 'd_8_0', 'd_8_1', 'd_8_2', 'd_8_3', 'd_9_0', 'd_9_1', 'd_9_2', 'd_9_3',
'at2Jsub1_0th_fromidx0_bond_GetBO',
'at2Jsub1_0th_fromidx0_bond_IsRotor',
'at2Jsub1_0th_fromsecond_AverageBondAngle',
'at2Jsub1_0th_fromsecond_CountRingBonds',
'at2Jsub1_0th_fromsecond_GetAtomicNum',
'at2Jsub1_0th_fromsecond_HasAromaticBond',
'at2Jsub1_0th_fromsecond_HasDoubleBond',
'at2Jsub1_0th_fromsecond_HasNonSingleBond',
'at2Jsub1_0th_fromsecond_HasSingleBond',
'at2Jsub1_0th_fromsecond_IsAromatic',
'at2Jsub1_0th_fromsecond_IsInRing',
'at2Jsub1_0th_fromsecond_IsInRingSize3',
'at2Jsub1_0th_fromsecond_IsInRingSize4',
'at2Jsub1_0th_fromsecond_IsInRingSize5',
'at2Jsub1_0th_fromsecond_IsInRingSize6',
'at2Jsub1_0th_fromsecond_IsInRingSize7',
'at2Jsub1_0th_fromsecond_IsInRingSize8',
'at2Jsub1_0th_fromsecond_IsNegativeStereo',
'at2Jsub1_0th_fromsecond_MemberOfRingCount',
'at2Jsub1_0th_fromsecond_MemberOfRingSize',
'at2Jsub1_0th_fromsecond_cos_second_asOrigin',
'at2Jsub1_1th_fromidx0_bond_GetBO',
'at2Jsub1_1th_fromidx0_bond_IsRotor',
'at2Jsub1_1th_fromsecond_AverageBondAngle',
'at2Jsub1_1th_fromsecond_CountRingBonds',
'at2Jsub1_1th_fromsecond_GetAtomicNum',
'at2Jsub1_1th_fromsecond_HasAromaticBond',
'at2Jsub1_1th_fromsecond_HasDoubleBond',
'at2Jsub1_1th_fromsecond_HasNonSingleBond',
'at2Jsub1_1th_fromsecond_HasSingleBond',
'at2Jsub1_1th_fromsecond_IsAromatic',
'at2Jsub1_1th_fromsecond_IsInRing',
'at2Jsub1_1th_fromsecond_IsInRingSize3',
'at2Jsub1_1th_fromsecond_IsInRingSize4',
'at2Jsub1_1th_fromsecond_IsInRingSize5',
'at2Jsub1_1th_fromsecond_IsInRingSize6',
'at2Jsub1_1th_fromsecond_IsInRingSize7',
'at2Jsub1_1th_fromsecond_IsInRingSize8',
'at2Jsub1_1th_fromsecond_IsNegativeStereo',
'at2Jsub1_1th_fromsecond_MemberOfRingCount',
'at2Jsub1_1th_fromsecond_MemberOfRingSize',
'at2Jsub1_1th_fromsecond_cos_second_asOrigin',
'at2Jsub1_Last_AverageBondAngle', 'at2Jsub1_Last_CountRingBonds',
'at2Jsub1_Last_GetAtomicNum', 'at2Jsub1_Last_GetSpinMultiplicity',
'at2Jsub1_Last_Hyb', 'at2Jsub1_Last_IsAromatic',
'at2Jsub1_Last_IsInRing', 'at2Jsub1_Last_IsInRingSize3',
'at2Jsub1_Last_IsInRingSize4', 'at2Jsub1_Last_IsInRingSize5',
'at2Jsub1_Last_IsInRingSize6', 'at2Jsub1_Last_IsInRingSize7',
'at2Jsub1_Last_IsInRingSize8', 'at2Jsub1_Last_MemberOfRingCount',
'at2Jsub1_Last_MemberOfRingSize', 'at2Jsub1_Last_SmallestBondAngle',
'at2Jsub1_firstsecond_dist', 'at2Jsub1_secLast_b_BO',
'at2Jsub1_secLast_b_IsAmide', 'at2Jsub1_secLast_b_IsAromatic',
'at2Jsub1_secLast_b_IsCarbonyl', 'at2Jsub1_secLast_b_IsCisOrTrans',
'at2Jsub1_secLast_b_IsPrimaryAmide', 'at2Jsub1_secLast_b_IsRotor',
'at2Jsub1_secLast_b_IsWedge', 'at2Jsub1_secondLast_Angle',
'at2Jsub1_secondLast_dist', 'at2Jsub1_second_AverageBondAngle',
'at2Jsub1_second_CountRingBonds', 'at2Jsub1_second_GetAtomicNum',
'at2Jsub1_second_GetSpinMultiplicity', 'at2Jsub1_second_Hyb',
'at2Jsub1_second_IsAromatic', 'at2Jsub1_second_IsInRing',
'at2Jsub1_second_IsInRingSize3', 'at2Jsub1_second_IsInRingSize4',
'at2Jsub1_second_IsInRingSize5', 'at2Jsub1_second_IsInRingSize6',
'at2Jsub1_second_IsInRingSize7', 'at2Jsub1_second_IsInRingSize8',
'at2Jsub1_second_MemberOfRingCount', 'at2Jsub1_second_MemberOfRingSize',
'at2Jsub1_second_SmallestBondAngle',
] + [f"submol_FPMC{i}" for i in range(167)]


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
molecule_atom_index_0_dist_mean_diff
d_O_from2nd_ratio_0
molecule_atom_index_0_dist_mean
interBond_IsInRing
dist_N_1_x
qeq_0
molecule_atom_index_0_dist_std_div
3J2nd_AverageBondAngle
qtpie_0
3J3rd__C
molecule_dist_min
2Jdist_from2nd_max_mean_diff
a1_inring5
molecule_type_dist_mean_diff
interBond_IsRotor
mean_circle_size
n_circle
2Jdist_from2nd_mean
mean_dist_ratio_O_from2nd
1J1st_CountFreeOxygens
molecule_atom_index_1_dist_max_diff
interBond_BondOrder
max_circle_size
3J3rd_isChiral
a0_nb_inring4
1J1st_BOSum
a0_nb_inring3
1J1st_IsAromatic
molecule_atom_index_1_dist_mean_diff
3Jdist_from2nd_max_mean_diff
3J3rd__N
molecule_atom_index_1_dist_mean_div
pca_exp_1
pca_exp_2
2J2nd_isChiral
3Jlast_MemberOfRingSize
1Jlast_GetValence
3J3rd__O
a0_nb_nb_n
1J1st_IsInRing
min_circle_size
1Jlast_GetImplicitValence
3J2nd_isChiral
1Jlast_GetAtomicNum
interBond_IsAmider
3Jlast_isChiral
interBond_IsDoubleBondGeometry
interBond_IsClosure
1J1st_ExplicitHydrogenCount
3J3rd_isAroma
1J1st_IsChiral
1Jlast_GetHvyValence
a0_nb_inring
1Jlast_GetSpinMultiplicity
a0_nb_hybridization
3J3rd_isInRing
3Jlast_isAroma
2J2nd_isAroma
3J1st_IsPolarHydrogen
3Jlast_IsAxial
a0_nb_nb_o
3J2nd_IsAxial
2J1st_IsPolarHydrogen
3J2nd__N
1Jlast_GetHyb
2J2nd_IsAxial
2J2nd_isInRing
3J2nd_isAroma
3J2nd__C
2Jlast_IsPolarHydrogen
interBond_IsAromatic
3Jlast_IsPolarHydrogen
3J2nd_isInRing
3Jlast_isInRing
3J3rd_IsAxial
3J2nd__O
2J2nd__N
2J2nd__O
interBond_IsEster
sp
interBond_IsWedge
2J2nd__C
1Jlast_GetFormalCharge
1Jlast_IsPolarHydrogen
interBond_IsUp
2J2nd__nan
3J2nd__nan
1J1st_IsNitroOxygen
1J1st_IsCarboxylOxygen
interBond_IsCarbonyl
interBond_IsDown
interBond_IsHash
3J3rd__nan
acsf_64_0
acsf_76_1
acsf_65_0
acsf_19_0
acsf_19_1
submol_FPMC146
acsf_18_1
submol_FPMC38
submol_FPMC77
acsf_65_1
acsf_60_0
acsf_63_0
submol_FPMC37
acsf_78_1
submol_FPMC21
acsf_77_1
acsf_75_0
acsf_68_1
acsf_68_0
submol_FPMC124
acsf_72_1
submol_FPMC106
acsf_74_0
acsf_69_1
acsf_18_0
acsf_77_0
acsf_79_1
acsf_70_1
acsf_71_0
acsf_74_1
acsf_71_1
acsf_73_1
acsf_76_0
acsf_73_0
acsf_61_0
submol_FPMC0
acsf_69_0
acsf_75_1
submol_FPMC52
submol_FPMC28
submol_FPMC25
acsf_70_0
acsf_72_0
acsf_79_0
submol_FPMC102
submol_FPMC69
acsf_78_0
submol_FPMC24
submol_FPMC140
submol_FPMC107
submol_FPMC23
submol_FPMC130
submol_FPMC70
submol_FPMC135
submol_FPMC101
submol_FPMC68
submol_FPMC65
submol_FPMC26
submol_FPMC58
submol_FPMC56
submol_FPMC59
submol_FPMC6
node_0_0
submol_FPMC60
node_12_0
submol_FPMC61
submol_FPMC62
submol_FPMC63
submol_FPMC64
node_0_1
submol_FPMC67
submol_FPMC88
submol_FPMC87
submol_FPMC7
submol_FPMC94
submol_FPMC71
node_11_1
submol_FPMC73
node_11_0
node_10_1
node_10_0
submol_FPMC9
submol_FPMC81
node_3_0
submol_FPMC55
node_9_0
submol_FPMC19
submol_FPMC18
node_4_0
submol_FPMC166
node_4_1
node_5_0
node_6_0
submol_FPMC162
node_7_0
node_8_0
submol_FPMC1
submol_FPMC20
submol_FPMC15
submol_FPMC148
submol_FPMC144
submol_FPMC143
submol_FPMC103
submol_FPMC14
submol_FPMC113
submol_FPMC12
submol_FPMC134
submol_FPMC125
submol_FPMC2
node_3_1
node_1_0
submol_FPMC13
submol_FPMC51
submol_FPMC5
submol_FPMC49
submol_FPMC48
submol_FPMC47
submol_FPMC46
node_2_0
submol_FPMC44
node_2_1
submol_FPMC42
submol_FPMC40
submol_FPMC27
submol_FPMC4
submol_FPMC39
submol_FPMC36
submol_FPMC35
submol_FPMC33
submol_FPMC32
submol_FPMC31
submol_FPMC30
submol_FPMC3
submol_FPMC29
submol_FPMC10
at2Jsub1_1th_fromsecond_IsInRingSize3
at2Jsub1_second_MemberOfRingCount
submol_FPMC142
at2Jsub1_1th_fromsecond_IsInRingSize6
submol_FPMC54
submol_FPMC85
submol_FPMC78
submol_FPMC43
submol_FPMC111
acsf_63_1
at2Jsub1_second_IsInRingSize7
acsf_64_1
submol_FPMC22
acsf_62_0
submol_FPMC97
at2Jsub1_Last_IsAromatic
submol_FPMC123
submol_FPMC80
submol_FPMC120
acsf_17_0
node_8_1
submol_FPMC79
submol_FPMC136
submol_FPMC161
acsf_62_1
acsf_61_1
at2Jsub1_secLast_b_IsAromatic
at2Jsub1_Last_IsInRingSize7
at2Jsub1_Last_IsInRing
at2Jsub1_0th_fromsecond_IsInRingSize8
at2Jsub1_1th_fromsecond_IsInRingSize8
at2Jsub1_0th_fromsecond_IsInRing
at2Jsub1_Last_IsInRingSize8
at2Jsub1_second_IsInRingSize8
at2Jsub1_second_IsInRingSize3
at2Jsub1_1th_fromsecond_IsInRing
at2Jsub1_secLast_b_IsPrimaryAmide
at2Jsub1_1th_fromidx0_bond_GetBO
at2Jsub1_secLast_b_IsCarbonyl
at2Jsub1_secLast_b_IsCisOrTrans
at2Jsub1_1th_fromsecond_IsNegativeStereo
at2Jsub1_second_IsInRing
at2Jsub1_0th_fromsecond_HasAromaticBond
at2Jsub1_secLast_b_IsWedge
at2Jsub1_0th_fromsecond_IsNegativeStereo
at2Jsub1_second_IsAromatic
at2Jsub1_1th_fromsecond_HasAromaticBond
at2Jsub1_1th_fromsecond_HasSingleBond
""".split("\n")
