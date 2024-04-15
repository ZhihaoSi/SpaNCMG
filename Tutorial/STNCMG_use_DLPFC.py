# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:40:14 2024

@author: szh
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys
from sklearn.metrics.cluster import adjusted_rand_score
from STNCMG import Mix_STNCMG
from Train import train_STNCMG
from Mix_adj import Transfer_pytorch_Data, Mix_adj, mclust_R

section_id = '151675'
print('Current slice %s'%(section_id))
input_dir = os.path.join('/home/szh/upload/DATA/DLPFC', section_id) 
adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5')
adata.var_names_make_unique()
#data preprocessing
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

#Original drawing
Ann_df = pd.read_csv(os.path.join('/home/szh/upload/DATA/DLPFC', section_id, section_id+'_truth.txt'), sep='\t', header=None, index_col=0)
Ann_df.columns = ['Ground Truth']
adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.spatial(adata, img_key="hires", color=["Ground Truth"],show=False)

#Build the SNN diagram
Mix_adj(adata, k_cutoff=6,rad_cutoff=150) 

#training model
adata = train_STNCMG(adata,n_epochs=1000) 
print("Completion of training")
sc.pp.neighbors(adata, use_rep='STNCMG')
sc.tl.umap(adata)
print("Start clustering")


adata = mclust_R(adata, num_cluster=7,used_obsm='STNCMG' )
obs_df = adata.obs.dropna()

ARI = adjusted_rand_score(obs_df['mclust'], obs_df['Ground Truth'])
print('ARI = %.2f' %ARI)


plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.umap(adata, color=["mclust", "Ground Truth"], title=['STNCMG (ARI=%.2f)'%ARI, "Ground Truth"],save='_%.2f_U.png'%ARI)
plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.spatial(adata, color=["mclust", "Ground Truth"], title=['STNCMG (ARI=%.2f)'%ARI, "Ground Truth"],save='_%.2f_C.png'%ARI)





