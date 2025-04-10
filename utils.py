import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import pickle
import math
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import json
import torch.utils.data as data
from sklearn.metrics import roc_auc_score,precision_recall_curve,accuracy_score

class MyDataset(Dataset):
    def __init__(self, cell_dict, edge_index):
        super(MyDataset, self).__init__()
        self.cell =  cell_dict
        # self.edge_index = torch.tensor(edge_index, dtype=torch.long)
        self.edge_index = edge_index

    def __len__(self):
        return len(self.value)

    def __getitem__(self, index):
        self.cell[self.Cell_line_name[index]].edge_index = self.edge_index
        return (self.cell[self.Cell_line_name[index]])


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# device = torch.device('cpu')

class PairsData(data.Dataset):
    def __init__(self, data1, data2):
        self.d1 = data1
        self.d2 = data2

    def __len__(self):
        return len(self.d1)

    def __getitem__(self, index):  # 返回的是tensor
        s1 = torch.FloatTensor(self.d1[index])
        s2 = torch.FloatTensor(self.d2[index])
        data = torch.cat((s1, s2), 0)
        return data.to(device)






def readTriple(path,sep=None):
    with open(path,'r',encoding='utf-8') as f:
        for line in f.readlines():
            if sep:
                lines = line.strip().split(sep)
            else:
                lines=line.strip().split()
            #if len(lines)!=4 :continue
            yield lines

def readFile(path,sep=None):
    with open(path,'r',encoding='utf-8') as f:
        for line in f.readlines():
            if sep:
                lines = line.strip().split(sep)
            else:
                lines = line.strip().split()
            if len(lines)==0:continue
            yield lines

def getJson(path):
    with open(path,'r',encoding='utf-8') as f:
        d=json.load(f)
    return d

def dumpJson(obj,path):
    with open(path,'w+',encoding='utf-8') as f:
        json.dump(obj,f)



def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


def metrics_graph(yt, yp):

    precision, recall, _, = precision_recall_curve(yt, yp)
    aupr = -np.trapz(precision, recall)
    auc = roc_auc_score(yt, yp)
    #---f1,acc,recall, specificity, precision
    real_score=np.mat(yt)
    predict_score=np.mat(yp)
    sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]
    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN
    tpr = TP / (TP + FN)
    recall_list = tpr
    precision_list = TP / (TP + FP)
    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)
    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return auc, aupr, f1_score[0, 0], accuracy[0, 0],recall[0, 0], specificity[0, 0], precision[0, 0]