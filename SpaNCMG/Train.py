# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 16:39:35 2023

@author: szh
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp

from SpaNCMG import Mix_SpaNCMG
from Mix_adj import Transfer_pytorch_Data

import torch
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from torch import nn

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()
    
def preprocess_adj(adj):
    #Preprocessing of adjacency matrix of simple GCN model and conversion of tuple representation
    adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    return adj_normalized 

def permutation(feature):
    #Feature random arrangement
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids) 
    feature_permutated = feature[ids]
    return feature_permutated 
    
def add_contrastive_label(adata):
    # contrastive label
    n_spot = adata.n_obs
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
    adata.obsm['label_CSL'] = label_CSL 
    
def get_feature(adata, deconvolution=False):
    if deconvolution:
       adata_Vars = adata
    else:   
       adata_Vars =  adata[:, adata.var['highly_variable']] #Only 3000 highly variable genes are selected
       
    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
       feat = adata_Vars.X.toarray()[:, ]
    else:
       feat = adata_Vars.X[:, ] 
    # data augmentation
    feat_a = permutation(feat) 
    
    adata.obsm['feat'] = feat
    adata.obsm['feat_a'] = feat_a    

def train_SpaNCMG(adata, hidden_dims=[512, 30], n_epochs=1000, lr=0.001,key_added='SpaNCMG',
                gradient_clipping=5.,  weight_decay=0.0001, verbose=True, 
                random_seed=0, save_loss=False, save_reconstrction=False, 
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                dim_input=3000,dim_output=64,alpha = 10,beta = 1,deconvolution = False):  
    """\
    Training graph attention auto-encoder.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    hidden_dims
        The dimension of the encoder.
    n_epochs
        Number of total epochs in training.
    lr
        Learning rate for AdamOptimizer.
    key_added
        The latent embeddings are saved in adata.obsm[key_added].
    gradient_clipping
        Gradient Clipping.
    weight_decay
        Weight decay for AdamOptimizer.
    save_loss
        If True, the training loss is saved in adata.uns['SpaNCMG_loss'].
    save_reconstrction
        If True, the reconstructed expression profiles are saved in adata.layers['SpaNCMG_ReX'].
    device
        See torch.device.
    dim_input : int, optional
        Dimension of input feature. The default is 3000.
    dim_output : int, optional
        Dimension of output representation. The default is 64.
    alpha : float, optional
        Weight factor to control the influence of reconstruction loss in representation learning. 
        The default is 10.
    beta : float, optional
        Weight factor to control the influence of contrastive loss in representation learning. 
        The default is 1.
    deconvolution : bool, optional
        Deconvolution task? The default is False.  
        
    Returns
    -------
    AnnData
    """
    seed=random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
   
    adata.X = sp.csr_matrix(adata.X)
    
    if 'highly_variable' in adata.var.columns:
        adata_Vars =  adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata

    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Construct two kinds of spatial neighborhood network first!")
      
    print('Size of Input: ', adata_Vars.shape)
    add_contrastive_label(adata)
    get_feature(adata) 
    
    features = torch.FloatTensor(adata.obsm['feat'].copy()).to(device)
    features_a = torch.FloatTensor(adata.obsm['feat_a'].copy()).to(device)
    label_CSL = torch.FloatTensor(adata.obsm['label_CSL']).to(device)
    adj1 = adata.obsm['adj1']
    adj2 = adata.obsm['adj2']
    graph_neigh1 = torch.FloatTensor(adata.obsm['graph_neigh1'].copy() + np.eye(adj1.shape[0])).to(device)
    graph_neigh2 = torch.FloatTensor(adata.obsm['graph_neigh2'].copy() + np.eye(adj2.shape[0])).to(device)
    
    dim_input = features.shape[1]
    # dim_output = dim_output
    
    adj1 = preprocess_adj(adj1) 
    adj1 = torch.FloatTensor(adj1).to(device)
    adj2 = preprocess_adj(adj2) 
    adj2 = torch.FloatTensor(adj2).to(device)
 
    model = Mix_SpaNCMG(dim_input, dim_output).to(device)
    
    loss_CSL = nn.BCEWithLogitsLoss()
    #data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr,weight_decay=weight_decay) 
    print('Begin to train ST data...')
    model.train()
    for epoch in tqdm(range(n_epochs)): 
        model.train()
          
        features_a = permutation(features) 
        hiden_feat, emb, ret, ret_a = model(features, features_a, adj1, adj2, graph_neigh1, graph_neigh2)
        
        loss_sl_1 = loss_CSL(ret, label_CSL)
        loss_sl_2 = loss_CSL(ret_a, label_CSL)
        loss_feat = F.mse_loss(features, emb)  
        
        loss =  alpha*loss_feat + beta*(loss_sl_1 + loss_sl_2) 
        
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
    with torch.no_grad():
        model.eval()
        if deconvolution:
           emb_rec = model(features, features_a, adj1, adj2)[1]
        else: 
            emb_rec = model(features, features_a, adj1, adj2, graph_neigh1, graph_neigh2)[1].detach().cpu().numpy()
            
        adata.obsm[key_added] = emb_rec  
        if save_loss:
            adata.uns['SpaNCMG_loss'] = loss
        if save_reconstrction:
            ReX = emb.to('cpu').detach().numpy()
            ReX[ReX<0] = 0
            adata.layers['SpaNCMG_ReX'] = ReX   
        return adata 

