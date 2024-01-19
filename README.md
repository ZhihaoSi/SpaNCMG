# STNCMG: precisely describing the spatial domains of spatial transcriptomics using a neighborhood-complementary mixed-view graph convolutional network
The advancement of spatial transcriptomics (ST) technology contributes to a more profound comprehension of the spatial properties of gene expression within tissues.  However, due to challenges including high dimensionality, pronounced noise, and dynamic limitations in ST data, the extraction and integration of gene expression and spatial information to accurately identify spatial domains remain challenging. Based on a neighborhood-complementary mixed-view graph convolutional network, we propose algorithm, STNCMG, to accurately describe and position spatial domains. For the first time, the algorithm integrates the local information captured by KNN and the global structure captured by the r-radius into a complementary neighborhood graph, enabling better adaptation to ST data at different resolutions. The STNCMG also introduces an attention mechanism to achieve adaptive fusion of different reconstructed expressions and utilizes the KPCA method for dimensionality reduction. Application of STNCMG on 5 datasets from 4 different sequencing platforms demonstrates superior performance in spatial domain identification compared to 7 existing advanced methods. Specifically, the algorithm achieved ARI accuracies of 0.62 and 0.52 on the datasets of the human dorsolateral prefrontal cortex and the mouse somatosensory cortex, respectively. Moreover, it accurately identified the spatial locations of marker genes in the mouse olfactory bulb tissue and inferred the biological functions of different regions. Furthermore, when handling larger datasets such as mouse embryos, STNCMG not only identified the main tissue structures but also explored unlabeled domains. Overall, STNCMG demonstrates good generalization ability and scalability, making it an outstanding spatial domain identification tool with significant implications for understanding tissue structure, disease mechanisms, and drug action.
## Overview
![image]([./figures/Overview.png](https://github.com/ZhihaoSi/STNCMG/blob/main/figures/Overview.png))

Fig. 1. The overall workflow of STNCMG. (a) Data preprocessing and the construction of different spatial adjacency matrices with normalization. (b) Neighborhood-complementary multi-view graph convolutional network for feature training of different complementary neighborhood graphs, outputting the reconstructed feature graphs. (c) The reconstruction matrices mapped from the reconstructed feature graphs of B is fused through attention mechanism to obtain the final reconstructed gene expression. (d) The final reconstructed gene expression matrix's biological applications.
## Dependencies
- Python=3.8.16
- torch=2.0.0 
- torch-cluster=1.6.0+pt112cpu
- torch-scatter=2.1.0+pt112cpu
- torch-sparse=0.6.15+pt112cpu
- torch-spline-conv=1.2.1+pt112cpu
- torch-geometric=2.3.0
- scanpy=1.9.1
- numpy=1.22.0
- pandas=1.5.0
- sklearn=1.1.1
- scipy=1.9.1
## Tutorial
Check the Tutorial folder for detailed instructions.














