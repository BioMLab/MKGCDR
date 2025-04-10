import scipy.sparse as sp

from utils import *


def normalize_adj(mx):
    """Row Normalized Sparse Matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)



def readKGData_ymx(path='data/Knowledge Graph/relationship.txt'):
    print('Read knowledge graph data...')
    entity_set = set()
    relation_set = set()
    triples = []
    for h, r, t in readTriple(path, sep=','):
        entity_set.add(int(h))
        entity_set.add(int(t))
        relation_set.add(int(r))
        triples.append([int(h), int(r), int(t)])
    return list(entity_set), list(relation_set), triples

def readRecData_ymx(path='data/Knowledge Graph/news_data_标签为0.txt', test_ratio=0.2):
    print('Read Drug Combination Synergy Data...')
    cell, drug = set(), set()
    triples = []
    for c, d ,r in readTriple(path, sep=','):
        drug.add(int(d))
        cell.add(int(c))
        triples.append((int(c), int(d), int(r)))

    return list(drug), list(cell),  triples


