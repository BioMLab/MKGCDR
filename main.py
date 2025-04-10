from sklearn.model_selection import KFold, train_test_split
import numpy as np
import pandas as pd
import dataloader4KGNN
import load_gat
from model import KGANS, InfoNCELoss
from utils import metrics_graph
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import torch
import torch.nn as nn

# Set the device: use GPU if available, otherwise use CPU.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def arg_parse():
    """
    Parse command-line arguments for the model.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0, help='device')
    # Knowledge Graph parameters
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--n_heads', type=int, default=6, help='Number of multi-heads')
    parser.add_argument('--n_neighbors', type=int, default=7, help='Number of neighbors to sample')  # originally 6
    parser.add_argument('--e_dim', type=int, default=64, help='Dimension of user and entity embeddings')
    parser.add_argument('--r_dim', type=int, default=64, help='Dimension of user and relation embeddings')
    parser.add_argument('--n_iter', type=int, default=3,
                        help='Number of iterations for entity representation computation')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout_ratio', type=float, default=0.2, help='Dropout ratio')
    parser.add_argument('--train_test_mode', type=int, default=1, help='Determine if training or testing')
    parser.add_argument('--meta_heads', type=int, default=4, help='Number of meta path multi-heads')
    return parser.parse_args()


# Using KFold to generate training and testing indices from the response data.
for i in range(2, 5):
    # Read knowledge graph data
    args = arg_parse()
    drug, cells, triples = load_gat.readRecData_ymx()
    entitys, relations, kgTriples = load_gat.readKGData_ymx()  # entitys: list of all entity indices; relations: list of all relations; kgTriples: nested list of KG relationships.
    triples_DF = pd.DataFrame(triples)
    triples_DF.columns = ['Cell', 'Drug', 'label']

    # Read response data
    data_new = pd.read_csv('data/data_new_entity.csv', index_col=0)
    train_temp_list = []
    test_temp_list = []
    kf = KFold(n_splits=5, shuffle=True, random_state=77)
    for train_index_temp, test_index_temp in kf.split(data_new):  # Split data using KFold
        train_temp_list.append(train_index_temp)
        test_temp_list.append(test_index_temp)
    train_index = train_temp_list[i].tolist()
    test_index = test_temp_list[i].tolist()
    train = triples_DF.loc[train_index]
    test = triples_DF.loc[test_index]
    train_np = np.array(train)
    train_list = train_np.tolist()
    test_np = np.array(test)
    test_list = test_np.tolist()

# Reading knowledge graph related data.
args = arg_parse()
drug, cells, triples = load_gat.readRecData_ymx()
entitys, relations, kgTriples = load_gat.readKGData_ymx()  # entitys: list of all entity indices; relations: list of all relations; kgTriples: nested list of KG relationships.
triples_DF = pd.DataFrame(triples)
triples_DF.columns = ['Cell', 'Drug', 'label']

# Reading response data.
data_new = pd.read_csv('data/data_new_entity.csv', index_col=0)
train, test = train_test_split(data_new, test_size=0.1, random_state=77)

train_np = np.array(train)
train_list = train_np.tolist()
test_np = np.array(test)
test_list = test_np.tolist()

# ---------------------- Subgraph generation based on meta-path ---------------------- #
train_order = ['Cell', 'label', 'Drug']
train_1 = train[train_order]
train_1 = train_1[train_1["label"] == 1]
train_relation_np = np.array(train_1)
train_relation_list = train_relation_np.tolist()

# Generate bidirectional relationships.
train_order = ['Drug', 'label', 'Cell']
train_2 = train[train_order]
train_2 = train_2[train_2["label"] == 1]
train_relation_np_2 = np.array(train_2)
train_relation_list_2 = train_relation_np_2.tolist()
for item in train_relation_list:
    kgTriples.append(item)
for item2 in train_relation_list_2:
    kgTriples.append(item2)
kgTriples_df = pd.DataFrame(kgTriples)
kgTriples_df.columns = ['Entity1', 'Relation', 'Entity2']
kgTriples_df.to_csv("data/Knowledge Graph/fold/initial_metapath.csv")
print("done1111")

# ---------------------- Read meta-path files ---------------------- #
meta_path_1_kgTriples = pd.read_csv('data/Knowledge Graph/fold/Fold_meta_path_1.csv', index_col=None, header=0)
meta_path_2_kgTriples = pd.read_csv('data/Knowledge Graph/fold/Fold_meta_path_2.csv', index_col=None, header=0)
meta_path_3_kgTriples = pd.read_csv('data/Knowledge Graph/fold/Fold_meta_path_3.csv', index_col=None, header=0)
meta_path_4_kgTriples = pd.read_csv('data/Knowledge Graph/fold/Fold_meta_path_4_gene.csv', index_col=None, header=0)
meta_path_5_kgTriples = pd.read_csv('data/Knowledge Graph/fold/Fold_meta_path_5_gene.csv', index_col=None, header=0)

meta_path_5_kgTriples_np = np.array(meta_path_5_kgTriples)
meta_path_5_kgTriples_list = meta_path_5_kgTriples_np.tolist()

meta_path_4_kgTriples_np = np.array(meta_path_4_kgTriples)
meta_path_4_kgTriples_list = meta_path_4_kgTriples_np.tolist()

meta_path_3_kgTriples_np = np.array(meta_path_3_kgTriples)
meta_path_3_kgTriples_list = meta_path_3_kgTriples_np.tolist()

meta_path_2_kgTriples_np = np.array(meta_path_2_kgTriples)
meta_path_2_kgTriples_list = meta_path_2_kgTriples_np.tolist()

meta_path_1_kgTriples_np = np.array(meta_path_1_kgTriples)
meta_path_1_kgTriples_list = meta_path_1_kgTriples_np.tolist()

kgTriples_list = [
    meta_path_1_kgTriples_list,
    meta_path_2_kgTriples_list,
    meta_path_3_kgTriples_list,
    meta_path_4_kgTriples_list,
    meta_path_5_kgTriples_list
]

# ---------------------- Load entity sets for various types ---------------------- #
cell_set_pd = pd.read_csv('./data/Knowledge Graph/Cell_entity_set.csv', index_col=0, header=None)  # Cell set
drug_set_pd = pd.read_csv('./data/Knowledge Graph/Drug_entity_set.csv', index_col=0, header=None)
gene_set_pd = pd.read_csv('./data/Knowledge Graph/Gene_entity_set.csv', index_col=0, header=None)
tissue_set_pd = pd.read_csv('./data/Knowledge Graph/Tissue_entity_set.csv', index_col=0, header=None)

cell_set = list(cell_set_pd.index)
drug_set = list(drug_set_pd.index)
gene_set = list(gene_set_pd.index)
tissue_set = list(tissue_set_pd.index)

# ---------------------- Construct KG indexes and adjacency lists ---------------------- #
kg_indexes = dataloader4KGNN.getKgIndexsFromKgTriples(kgTriples_list)
# kg_indexes is a dictionary with keys as entity indices and values as lists like [entity number, relationship]

(adj_entity_1, adj_relation_1,
 adj_entity_2, adj_relation_2,
 adj_entity_3, adj_relation_3,
 adj_entity_4, adj_relation_4,
 adj_entity_5, adj_relation_5) = dataloader4KGNN.construct_adj(args.n_neighbors, kg_indexes, len(entitys))

adj_entity_list = [adj_entity_1, adj_entity_2, adj_entity_3, adj_entity_4, adj_entity_5]
adj_relation_list = [adj_relation_1, adj_relation_2, adj_relation_3, adj_relation_4, adj_relation_5]

# ---------------------- Define the model and optimizers ---------------------- #
KGANSnet = KGANS(args, max(entitys) + 1, max(relations) + 2, args.e_dim, args.r_dim, adj_entity_list,
                 adj_relation_list).to(device)
criterion = nn.BCELoss().to(device)
opt_KGANS = torch.optim.Adam(KGANSnet.parameters(), lr=args.lr)
myloss = nn.BCELoss().to(device)
contrastive_loss_fn = InfoNCELoss().to(device)


def train_2(KGANSnet, train_loader, loss_fn, opt_KGANS):
    """
    Train function for one epoch.
    Args:
        KGANSnet (nn.Module): The KGANS model.
        train_loader (DataLoader): DataLoader for training data.
        loss_fn (nn.Module): Loss function (here BCELoss).
        opt_KGANS (Optimizer): Optimizer.
    """
    KGANSnet.train()

    for data in tqdm(train_loader, desc='Train_Iteration'):
        cell, drug, label = data
        drug, cell, label = drug.to(device), cell.to(device), label.to(device)
        output2, weighted_sum, attention_test_train = KGANSnet(cell, drug)
        loss = loss_fn(output2, label.float())
        # Calculate contrastive loss
        contrastive_loss = contrastive_loss_fn(weighted_sum)
        total_loss = loss + 0.5 * contrastive_loss
        # Zero gradients before backpropagation
        opt_KGANS.zero_grad()
        # Backward propagation to compute gradients
        total_loss.backward()
        # Update network parameters based on computed gradients
        opt_KGANS.step()


def test_2(KGANSnet, test_loader, loss_fn):
    """
    Test function to evaluate the model.
    Args:
        KGANSnet (nn.Module): The KGANS model.
        test_loader (DataLoader): DataLoader for testing data.
        loss_fn (nn.Module): Loss function.
    Returns:
        Tuple: Test loss, evaluation metrics (AUC, AUPR, F1, ACC, recall, specificity, precision), results DataFrame,
               true labels, predicted values, and attention weights.
    """
    KGANSnet.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in tqdm(test_loader, desc='Test_Iteration'):
            cell_test, drug_test, label_test = data
            cell_test, drug_test, label_test = cell_test.to(device), drug_test.to(device), label_test.to(device)
            output2, weighted_sum, attention_test = KGANSnet(cell_test, drug_test)
            y_true.append(label_test.view(-1, 1))
            y_pred.append(output2)
            loss_test = loss_fn(output2, label_test.float())
            output2 = output2.detach().cpu().numpy()
            label_test = label_test.detach().cpu().numpy()

    y_true = torch.cat(y_true, dim=0)
    y_true = torch.squeeze(y_true)
    y_pred = torch.cat(y_pred, dim=0)
    y_pred = torch.squeeze(y_pred)
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    df = np.array([y_pred.squeeze().cpu().numpy(), y_true.squeeze().cpu().numpy()])
    df = pd.DataFrame(df.T, columns=['y_pred', 'y_true'])
    AUC, AUPR, F1, ACC, recall, specificity, precision = metrics_graph(y_true_np, y_pred_np)
    return loss_test.item(), AUC, AUPR, F1, ACC, recall, specificity, precision, df, y_true_np, y_pred_np, attention_test


# ---------------------- Main training loop ---------------------- #
final_AUC = 0
final_AUPR = 0
final_F1 = 0
final_ACC = 0
Final_recall = 0
Final_specificity = 0
Final_precision = 0
num = 0

for epoch in range(1, args.epochs + 1):
    print(f"===== Epoch {epoch} =====")
    num += 1

    train_loader = DataLoader(train_list, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_list, batch_size=args.batch_size, shuffle=True)
    train_2(KGANSnet, train_loader, myloss, opt_KGANS)
    loss_test, AUC, AUPR, F1, ACC, recall, specificity, precision, df, y_true_np, y_pred_np, attention_test = test_2(
        KGANSnet, test_loader, myloss)

    if AUC > final_AUC:
        best_epoch = epoch
        final_AUC = AUC
        final_AUPR = AUPR
        final_F1 = F1
        final_ACC = ACC
        final_recall = recall
        final_specificity = specificity
        final_precision = precision
        final_df = df
        final_ypred = y_pred_np.copy()
        final_ytrue = y_true_np.copy()

    print('test loss: ', str(round(loss_test, 4)))
    print('test auc: ' + str(round(AUC, 4)) + '  test aupr: ' + str(round(AUPR, 4)) +
          '  test f1: ' + str(round(F1, 4)) + '  test acc: ' + str(round(ACC, 4)) + '  test recall: ' + str(
        round(recall, 4))
          + '  test  specificity: ' + str(round(specificity, 4)) + '  test final_precision: ' + str(
        round(precision, 4))
          )
