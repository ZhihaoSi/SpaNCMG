# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 14:30:11 2023

@author: szh
"""
import torch
import torch.nn as nn

import pandas as pd
import numpy as np
import sklearn.neighbors
import scipy.sparse as sp
import seaborn as sns
import matplotlib.pyplot as plt
import ot
from torch_geometric.data import Data

def Transfer_pytorch_Data(adata):
    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])

    edgeList = np.nonzero(G)
    if type(adata.X) == np.ndarray:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X))  # .todense()
    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))  # .todense()
    return data

 
def Mix_adj(adata, rad_cutoff=None, k_cutoff=None):
    
    coor = pd.DataFrame(adata.obsm['spatial']) 
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']
    n_spot = coor.shape[0]
    
    #Find the nearest neighbor based on the radius
    nbrs1 = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
    #Return two array distances An array stores the distances of each point to other points
    #indices The second array contains its index
    distances1, indices1 = nbrs1.radius_neighbors(coor, return_distance=True)
    interaction1 = np.zeros([n_spot, n_spot]) 
    for i in range(n_spot):
        interaction1[i,indices1[i]] = 1
    adj1 = interaction1

    KNN_list1 = []
    for it in range(indices1.shape[0]):
        KNN_list1.append(pd.DataFrame(zip([it]*indices1[it].shape[0], indices1[it], distances1[it])))
    
    #KNN
    nbrs2 = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1,metric='chebyshev').fit(coor)#,metric='cosine'
    distances2, indices2 = nbrs2.kneighbors(coor)

    x = indices2[:, 0].repeat(k_cutoff)
    y = indices2[:, 1:].flatten()
    interaction2 = np.zeros([n_spot, n_spot])
    interaction2[x, y] = 1
    interaction2[y, x] = 1
    
    adj2 = interaction2
    adj2 = adj2 + adj2.T
    adj2 = np.where(adj2>1, 1, adj2)
   
    KNN_list2 = []
    for it in range(indices2.shape[0]):
        KNN_list2.append(pd.DataFrame(zip([it]*indices2.shape[1],indices2[it,:], distances2[it,:])))

    adata.obsm['graph_neigh1'] = interaction1
    adata.obsm['graph_neigh2'] = interaction2
    adata.obsm['adj1'] = adj1
    adata.obsm['adj2'] = adj2    
    
    KNN_df1 = pd.concat(KNN_list1)
    KNN_df2 = pd.concat(KNN_list2)
    df = pd.concat([KNN_df1,KNN_df2],ignore_index=True)
    df.columns = ['Cell1', 'Cell2', 'Distance']
    KNN_df = df.drop_duplicates(subset=["Cell1", "Cell2"],ignore_index=True)
    

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,] #It removes its own distance
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index) ))#Establish correspondence between indexes and cells
    #Map the index to the cell
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)

    adata.uns['Spatial_Net'] = Spatial_Net


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='SpaNCMG', random_seed=2023):

    from sklearn.decomposition import KernelPCA
    kpca = KernelPCA(n_components=20)
    embedding = kpca.fit_transform(adata.obsm[used_obsm].copy())
    adata.obsm['emb_pca'] = embedding

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm['emb_pca']), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    
    return adata
    
