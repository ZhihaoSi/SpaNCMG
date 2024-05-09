# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:31:58 2024

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
import torch
from SpaNCMG import Mix_SpaNCMG
from Train import train_SpaNCMG
from Mix_adj import Transfer_pytorch_Data, Mix_adj, mclust_R

inpath='/home/szh/upload/DATA/Stereoseq_MOB'
counts_file = os.path.join(inpath,'RNA_counts.tsv')
coor_file = os.path.join(inpath,'position.tsv')

counts = pd.read_csv(counts_file, sep='\t', index_col=0)
coor_df = pd.read_csv(coor_file, sep='\t')
print(counts.shape, coor_df.shape)

counts.columns = ['Spot_'+str(x) for x in counts.columns] 
coor_df.index = coor_df['label'].map(lambda x: 'Spot_'+str(x))
coor_df = coor_df.loc[:, ['x','y']]
coor_df.head()

adata = sc.AnnData(counts.T)
adata.var_names_make_unique()
adata

coor_df = coor_df.loc[adata.obs_names, ['y', 'x']]
adata.obsm["spatial"] = coor_df.to_numpy()
sc.pp.calculate_qc_metrics(adata, inplace=True) #inplace=True, the quality control indicator is added to adata.obs, adata.var

plt.rcParams["figure.figsize"] = (5,4)
sc.pl.embedding(adata, basis="spatial", color="n_genes_by_counts", show=False,save='_stereo_MOB01_UMAP.png')
plt.title("")
plt.axis('off')

used_barcode = pd.read_csv(os.path.join(inpath,'used_barcodes.txt'), sep='\t', header=None)
used_barcode = used_barcode[0]
adata = adata[used_barcode,]
adata

plt.rcParams["figure.figsize"] = (5,4)
sc.pl.embedding(adata, basis="spatial", color="n_genes_by_counts", show=False,save='_stereo_MOB02_UMAP.png')
plt.title("")
plt.axis('off')

sc.pp.filter_genes(adata, min_cells=50)
print('After flitering: ', adata.shape)

#Normalization
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
#Build the SNN diagram
Mix_adj(adata, k_cutoff=6,rad_cutoff=50)

adata = train_SpaNCMG(adata,n_epochs=100)#100
sc.pp.neighbors(adata, use_rep='SpaNCMG')
sc.tl.umap(adata)
sc.tl.louvain(adata, resolution=0.8)
plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.embedding(adata, basis="spatial", color="louvain",s=6, show=False, title='SpaNCMG')#,save='_stereo_MOB03_UMAP.png')
plt.axis('off')

sc.pl.umap(adata, color='louvain', title='SpaNCMG') #,save='_stereo_MOB04_UMAP.png')


