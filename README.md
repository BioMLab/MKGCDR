
#Interpretable Cancer Drug Response Prediction Based on Meta-path Guided Knowledge Graphs.

##Datasets

*In this study, 103,336 cancer drug response records involving 221 drugs on 568 cell lines are collected from the publicly free access database GDSC (https://www.cancerrxgene.org/)
. In each record, the IC50 value is provided to illustrate how sensitive a cell line is to the specific drug. Meanwhile,
the IC50 values of the involved drugs are downloaded from
GDSC. For each drug, a specific threshold for IC50 value
is determined. If the IC50 value of a drug for a particular
cell line exceeds the threshold, the record of the drug and
the cell line is defined as a positive sample; otherwise, it
is defined as a negative sample.

##Description
* **dataloader4KGNN.py**: Constructs knowledge graph indexes and generates adjacency lists for GNN processing.
* **load_gat.py**: Loads knowledge graph data and normalizes adjacency matrices for models.
* **main.py**: Run model
* **model.py**: Defines MKGCDR model architecture and contrastive loss functions.
* **utils.py**: Provides utility functions for data handling and evaluation metrics.


##Run step
* Run main.py to train the model and obtain the predicted scores for cancer drug response prediction.
 
##Requirements
* Python == 3.7.0
* Numpy == 1.21.6
* Pandas == 1.2.3
* Scikit-learn == 1.0.2
* Scipy == 1.7.3
* Seaborn == 0.12.2
* Pytorch == 1.10.0
* Torch-geometric == 2.2.0
* Rdkit == 2018.09.2
* Torch-cluster == 1.5.9
* Torch-scatter == 2.0.9
* Torch-sparse == 0.6.12
* Torch-spline-conv == 1.2.1


## Citation
If there is a requirement for you to reference the paper, code or dataset, please ensure to cite the source accurately.
