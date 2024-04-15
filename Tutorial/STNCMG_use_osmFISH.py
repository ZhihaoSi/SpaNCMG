"""
Created on Thu Jan 11 15:37:43 2024

@author: szh
"""

import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys

import torch
from STNCMG import Mix_STNCMG
from Train import train_STNCMG
from Mix_adj import Transfer_pytorch_Data, Mix_adj, mclust_R

from sklearn.metrics.cluster import adjusted_rand_score

adata = sc.read_h5ad('/home/szh/upload/STAGATE01/osmFISH/osmFISH_cortex.h5ad')#5328 cell 33 genes

layer_num_dict = {
    'Pia Layer 1':1,
    'Layer 2-3 lateral':2.5,
    'Layer 2-3 medial':2.5,
    'Layer 3-4':3.5,
    'Layer 4':4,
    'Layer 5':5,
    'Layer 6':6,
}
layer_list = layer_num_dict.keys()

adata_layer = adata[adata.obs['Region'].isin(layer_num_dict.keys())] #3405 cell 33 genes

sc.pl.embedding(adata_layer,basis='spatial',color=['Region'],show=False, frameon=False)#,save='_osmFISH0.png'
adata = adata_layer
#location information
#a0 = adata.obsm["spatial"]

#Normalization
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
Mix_adj(adata, k_cutoff=6,rad_cutoff=500)
adata = train_STNCMG(adata,n_epochs=450)


sc.pp.neighbors(adata, use_rep='STNCMG')
sc.tl.umap(adata)
print("Start clustering")

adata = mclust_R(adata, num_cluster=7,used_obsm='STNCMG' )

obs_df = adata.obs.dropna()

ARI = adjusted_rand_score(obs_df['mclust'], obs_df['Region'])
print('ARI = %.2f' %ARI)

plt.rcParams["figure.figsize"] = (5, 5)
sc.pl.umap(adata, color="mclust", title='STNCMG (ARI=%.2f)'%ARI)
plt.rcParams["figure.figsize"] = (5, 5)
sc.pl.embedding(adata,basis='spatial',color="mclust", title='STNCMG (ARI=%.2f)'%ARI,show=False)
