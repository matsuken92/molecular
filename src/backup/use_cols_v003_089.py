
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
'vec_0',
 'vec_1',
 'vec_2',
 'vec_3',
 'vec_4',
 'vec_5',
 'vec_6',
 'vec_7',
 'vec_8',
 'vec_9',
 'vec_10',
 'vec_11',
 'vec_12',
 'vec_13',
 'vec_14',
 'vec_15',
 'vec_16',
 'vec_17',
 'vec_18',
 'vec_19',
 'vec_20',
 'vec_21',
 'vec_22',
 'vec_23',
 'vec_24',
 'vec_25',
 'vec_26',
 'vec_27',
 'vec_28',
 'vec_29',
 'vec_30',
 'vec_31',
 'vec_32',
 'vec_33',
 'vec_34',
 'vec_35',
 'vec_36',
 'vec_37',
 'vec_38',
 'vec_39',
 'vec_40',
 'vec_41',
 'vec_42',
 'vec_43',
 'vec_44',
 'vec_45',
 'vec_46',
 'vec_47',
 'vec_48',
 'vec_49',
 'vec_50',
 'vec_51',
 'vec_52',
 'vec_53',
 'vec_54',
 'vec_55',
 'vec_56',
 'vec_57',
 'vec_58',
 'vec_59',
 'vec_60',
 'vec_61',
 'vec_62',
 'vec_63',
 'vec_64',
 'vec_65',
 'vec_66',
 'vec_67',
 'vec_68',
 'vec_69',
 'vec_70',
 'vec_71',
 'vec_72',
 'vec_73',
 'vec_74',
 'vec_75',
 'vec_76',
 'vec_77',
 'vec_78',
 'vec_79',
 'vec_80',
 'vec_81',
 'vec_82',
 'vec_83',
 'vec_84',
 'vec_85',
 'vec_86',
 'vec_87',
 'vec_88',
 'vec_89',
 'vec_90',
 'vec_91',
 'vec_92',
 'vec_93',
 'vec_94',
 'vec_95',
 'vec_96',
 'vec_97',
 'vec_98',
 'vec_99',
 'vec_100',
 'vec_101',
 'vec_102',
 'vec_103',
 'vec_104',
 'vec_105',
 'vec_106',
 'vec_107',
 'vec_108',
 'vec_109',
 'vec_110',
 'vec_111',
 'vec_112',
 'vec_113',
 'vec_114',
 'vec_115',
 'vec_116',
 'vec_117',
 'vec_118',
 'vec_119',
 'vec_120',
 'vec_121',
 'vec_122',
 'vec_123',
 'vec_124',
 'vec_125',
 'vec_126',
 'vec_127',
 'vec_128',
 'vec_129',
 'vec_130',
 'vec_131',
 'vec_132',
 'vec_133',
 'vec_134',
 'vec_135',
 'vec_136',
 'vec_137',
 'vec_138',
 'vec_139',
 'vec_140',
 'vec_141',
 'vec_142',
 'vec_143',
 'vec_144',
 'vec_145',
 'vec_146',
 'vec_147',
 'vec_148',
 'vec_149',
 'vec_150',
 'vec_151',
 'vec_152',
 'vec_153',
 'vec_154',
 'vec_155',
 'vec_156',
 'vec_157',
 'vec_158',
 'vec_159',
 'vec_160',
 'vec_161',
 'vec_162',
 'vec_163',
 'vec_164',
 'vec_165',
 'vec_166',
 'vec_167',
 'vec_168',
 'vec_169',
 'vec_170',
 'vec_171',
 'vec_172',
 'vec_173',
 'vec_174',
 'vec_175',
 'vec_176',
 'vec_177',
 'vec_178',
 'vec_179',
 'vec_180',
 'vec_181',
 'vec_182',
 'vec_183',
 'vec_184',
 'vec_185',
 'vec_186',
 'vec_187',
 'vec_188',
 'vec_189',
 'vec_190',
 'vec_191',
 'vec_192',
 'vec_193',
 'vec_194',
 'vec_195',
 'vec_196',
 'vec_197',
 'vec_198',
 'vec_199',
 'vec_200',
 'vec_201',
 'vec_202',
 'vec_203',
 'vec_204',
 'vec_205',
 'vec_206',
 'vec_207',
 'vec_208',
 'vec_209',
 'vec_210',
 'vec_211',
 'vec_212',
 'vec_213',
 'vec_214',
 'vec_215',
 'vec_216',
 'vec_217',
 'vec_218',
 'vec_219',
 'vec_220',
 'vec_221',
 'vec_222',
 'vec_223',
 'vec_224',
 'vec_225',
 'vec_226',
 'vec_227',
 'vec_228',
 'vec_229',
 'vec_230',
 'vec_231',
 'vec_232',
 'vec_233',
 'vec_234',
 'vec_235',
 'vec_236',
 'vec_237',
 'vec_238',
 'vec_239',
 'vec_240',
 'vec_241',
 'vec_242',
 'vec_243',
 'vec_244',
 'vec_245',
 'vec_246',
 'vec_247',
 'vec_248',
 'vec_249',
 'vec_250',
 'vec_251',
 'vec_252',
 'vec_253',
 'vec_254',
 'vec_255',
 'vec_256',
 'vec_257',
 'vec_258',
 'vec_259',
 'vec_260',
 'vec_261',
 'vec_262',
 'vec_263',
 'vec_264',
 'vec_265',
 'vec_266',
 'vec_267',
 'vec_268',
 'vec_269',
 'vec_270',
 'vec_271',
 'vec_272',
 'vec_273',
 'vec_274',
 'vec_275',
 'vec_276',
 'vec_277',
 'vec_278',
 'vec_279',
 'vec_280',
 'vec_281',
 'vec_282',
 'vec_283',
 'vec_284',
 'vec_285',
 'vec_286',
 'vec_287',
 'vec_288',
 'vec_289',
 'vec_290',
 'vec_291',
 'vec_292',
 'vec_293',
 'vec_294',
 'vec_295',
 'vec_296',
 'vec_297',
 'vec_298',
 'vec_299',
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
""".split("\n")