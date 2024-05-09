# SpaNCMG: precisely describing the spatial domains of spatial transcriptomics using a neighborhood-complementary mixed-view graph convolutional network
The advancement of spatial transcriptomics (ST) technology contributes to a more profound comprehension of the spatial properties of gene expression within tissues. However, due to challenges of high dimensionality, pronounced noise, and dynamic limitations in ST data, the integration of gene expression and spatial information to accurately identify spatial domains remains challenging. This paper proposes a SpaNCMG algorithm for the purpose of achieving precise spatial domain description and localization based on a neighborhood-complementary mixed-view graph convolutional network. The algorithm enables better adaptation to ST data at different resolutions by integrating the local information from KNN and the global structure from r-radius into a complementary neighborhood graph. It also introduces an attention mechanism to achieve adaptive fusion of different reconstructed expressions, and utilizes KPCA method for dimensionality reduction. The application of SpaNCMG on 5 datasets from 4 sequencing platforms demonstrates superior performance to 8 existing advanced methods. Specifically, the algorithm achieved highest ARI accuracies of 0.63 and 0.52 on the datasets of the human dorsolateral prefrontal cortex and mouse somatosensory cortex, respectively. It accurately identified the spatial locations of marker genes in the mouse olfactory bulb tissue and inferred the biological functions of different regions. When handling larger datasets such as mouse embryos, the SpaNCMG not only identified the main tissue structures but also explored unlabeled domains. Overall, the good generalization ability and scalability of SpaNCMG make it an outstanding tool for understanding tissue structure and disease mechanisms.
## Overview
![image](https://github.com/ZhihaoSi/SpaNCMG/blob/main/figure/Fig%201.png)

Fig. 1. The overall workflow of SpaNCMG. (a) Data preprocessing and the construction of different spatial adjacency matrices with normalization. (b) Neighborhood-complementary multi-view graph convolutional network for feature training of different complementary neighborhood graphs, outputting the reconstructed feature graphs. (c) The reconstruction matrices mapped from the reconstructed feature graphs of B is fused through attention mechanism to obtain the final reconstructed gene expression. (d) The final reconstructed gene expression matrix's biological applications.
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
Detailed instructions are provided in the Tutorial folder. Please refer to the [SpaNCMG_DLPFC.ipynb](https://github.com/ZhihaoSi/SpaNCMG/blob/main/Tutorial/SpaNCMG_DLPFC.ipynb) file for details.














