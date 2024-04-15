
"""
Created on Fri Sep  1 18:43:02 2023

@author: szh
"""
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys
import torch
import sklearn.neighbors
from STNCMG import Mix_STNCMG
from Train import train_STNCMG
from Mix_adj import Transfer_pytorch_Data, Mix_adj, mclust_R
import ot


# the number of clusters
n_clusters = 22 # E9.5 22 
file_path = ''/home/szh/upload/DATA/Mouse_embryo/'
adata = sc.read_h5ad(file_path +'E9.5_E1S1.MOSTA.h5ad' )
adata.var_names_make_unique()

#Normalization
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
#Build the SNN diagram
Mix_adj(adata, k_cutoff=3,rad_cutoff=1)
adata = train_STNCMG(adata,n_epochs=550)
sc.pp.neighbors(adata, use_rep='STNCMG')
sc.tl.umap(adata)

print('Start clustering')
adata = mclust_R(adata, used_obsm='STNCMG', num_cluster=n_clusters)
obs_df = adata.obs.dropna()

import matplotlib.pyplot as plt
adata.obsm['spatial'][:, 1] = -1*adata.obsm['spatial'][:, 1] # picture inversion
plt.rcParams["figure.figsize"] = (2.5, 3)
plot_color=["#F56867","#556B2F","#C798EE","#59BE86","#006400","#8470FF",
            "#CD69C9","#EE7621","#B22222","#FFD700","#CD5555","#DB4C6C",
            "#8B658B","#1E90FF","#AF5F3C","#CAFF70", "#F9BD3F","#DAB370",
          "#877F6C","#268785", '#82EF2D', '#B4EEB4']
sc.pl.embedding(adata, basis="spatial", color="mclust",s=30, show=False,palette=plot_color, title='STNCMG')#,save='_stereo_embryo01.png'
plt.axis('off')
sc.pl.umap(adata, color='mclust', title='STNCMG',save='_stereo_embryo02.png')


