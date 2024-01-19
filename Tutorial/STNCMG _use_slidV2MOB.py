
"""
Created on Thu Jan 11 15:36:40 2024

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

input_dir = 'F://DATA//小鼠嗅球//Slide-seqV2'
counts_file = os.path.join(input_dir, 'Puck_200127_15.digital_expression.txt')
coor_file = os.path.join(input_dir, 'Puck_200127_15_bead_locations.csv')
#reading data
counts = pd.read_csv(counts_file, sep='\t', index_col=0)
coor_df = pd.read_csv(coor_file, index_col=0)

coor_df=coor_df.set_index('barcode')

print(counts.shape, coor_df.shape)

adata = sc.AnnData(counts.T)
adata.var_names_make_unique()
#Add spatial location information
coor_df = coor_df.loc[adata.obs_names, ['xcoord', 'ycoord']]
adata.obsm["spatial"] = coor_df.to_numpy()

sc.pp.calculate_qc_metrics(adata, inplace=True)
adata

plt.rcParams["figure.figsize"] = (6,5)
#Original tissue area, some scattered spots
sc.pl.embedding(adata, basis="spatial", color="log1p_total_counts",s=6, show=False,save='_MOB01_slide.png')
plt.title('')
plt.axis('off')


used_barcode = pd.read_csv(os.path.join(input_dir, 'used_barcodes.txt'), sep='\t', header=None)
used_barcode = used_barcode[0]

adata = adata[used_barcode,]

plt.rcParams["figure.figsize"] = (5,5)
#Get rid of the scattered spots and get the main organizational area
sc.pl.embedding(adata, basis="spatial", color="log1p_total_counts",s=10, show=False, title='the main tissue area',save='_MOB02_standard.png')
plt.axis('off')

sc.pp.filter_genes(adata, min_cells=50)
print('After flitering: ', adata.shape)

#Normalization
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

#Build the SNN diagram
Mix_adj(adata,k_cutoff=3, rad_cutoff=50) 

adata = train_STNCMG(adata,n_epochs=300)#300

sc.pp.neighbors(adata, use_rep='STNCMG')
sc.tl.umap(adata)

sc.tl.louvain(adata, resolution=0.5)
adata.obsm["spatial"] = adata.obsm["spatial"] * (-1)

plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.embedding(adata, basis="spatial", color="louvain",s=6, show=False, title='STNCMG',save='_MOB03_STNCMG.png')
plt.axis('off')

sc.pl.umap(adata, color='louvain', title='STNCMG',save='_MOB04_UMAP.png')






