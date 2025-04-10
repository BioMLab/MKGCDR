import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set the device: use GPU if available, otherwise use CPU.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class InfoNCELoss(nn.Module):
    """
    Enhanced InfoNCE Loss:
      1. Normalizes input features using the L2 norm.
      2. Computes a pairwise similarity matrix and applies temperature scaling.
      3. Selects positive and negative pairs using the specified thresholds.
      4. Dynamically adjusts the number of positive and negative samples.
      5. Uses binary cross entropy loss with logits and adds an L2 regularization term.
    """
    def __init__(self, temperature=2, pos_threshold=0.8, neg_threshold=0.2, reg_lambda=1e-4):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.reg_lambda = reg_lambda

    def forward(self, features):
        """
        Args:
            features (Tensor): Input tensor of shape [batch_size, feature_dim].
        Returns:
            Tensor: Computed loss.
        """
        # Normalize the features along the feature dimension.
        features = F.normalize(features, p=2, dim=1)

        # Compute the similarity matrix and apply temperature scaling.
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Create masks for positive and negative samples using the defined thresholds.
        pos_mask = similarity_matrix > self.pos_threshold
        neg_mask = similarity_matrix < self.neg_threshold

        # Remove self-similarity by masking the diagonal elements.
        mask = torch.eye(similarity_matrix.size(0), dtype=torch.bool, device=features.device)
        pos_mask = pos_mask & ~mask
        neg_mask = neg_mask & ~mask

        # If there are no positive or no negative samples, return zero loss.
        num_pos_samples = pos_mask.sum().item()
        num_neg_samples = neg_mask.sum().item()
        if num_pos_samples == 0 or num_neg_samples == 0:
            return torch.tensor(0.0, device=features.device)

        # Extract similarity values for positive and negative pairs.
        pos_similarities = similarity_matrix[pos_mask]
        neg_similarities = similarity_matrix[neg_mask]

        # Adjust the number of samples to the smaller one between positives and negatives.
        if pos_similarities.numel() > num_neg_samples:
            pos_similarities = pos_similarities[:num_neg_samples]
        elif neg_similarities.numel() > num_pos_samples:
            neg_similarities = neg_similarities[:num_pos_samples]

        # Create labels and combine the logits for positive and negative samples.
        labels = torch.cat([
            torch.ones(pos_similarities.size(0), device=features.device),
            torch.zeros(neg_similarities.size(0), device=features.device)
        ], dim=0)
        logits = torch.cat([pos_similarities, neg_similarities], dim=0)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        # Add L2 regularization on the normalized features.
        reg_loss = self.reg_lambda * (features ** 2).sum() / features.size(0)
        return loss + reg_loss


class LayerAttention(nn.Module):
    """
    Layer Attention Module:
      Computes attention weights across layers using a two-layer MLP (with Tanh activation)
      and returns a weighted sum of the features along with the attention weights.
    """
    def __init__(self, in_size):
        super(LayerAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, in_size),
            nn.Tanh(),
            nn.Linear(in_size, 1, bias=False)
        )

    def forward(self, z):
        """
        Args:
            z (Tensor): Input tensor of shape [batch_size, num_layers, feature_dim].
        Returns:
            Tuple[Tensor, Tensor]: Aggregated features and the attention weights.
        """
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


def calc_kg_loss(entity_embeddings, relation_embeddings, triples):
    """
    Calculate the Knowledge Graph Loss:
      1. Computes the scores for positive samples using the input triples.
      2. Randomly generates negative samples and computes their scores.
      3. Uses a margin-based ranking loss formulation.
    Args:
        entity_embeddings (Tensor): Entity embedding matrix.
        relation_embeddings (Tensor): Relation embedding matrix.
        triples (Tensor): Tensor containing triples [head, relation, tail].
    Returns:
        Tensor: Computed knowledge graph loss.
    """
    positive_score = torch.sum(
        entity_embeddings[triples[:, 0]] *
        relation_embeddings[triples[:, 1]] *
        entity_embeddings[triples[:, 2]],
        dim=1
    )
    negative_samples = torch.randint(0, entity_embeddings.size(0), triples.size(), dtype=torch.long, device=triples.device)
    negative_score = torch.sum(
        entity_embeddings[negative_samples[:, 0]] *
        relation_embeddings[negative_samples[:, 1]] *
        entity_embeddings[negative_samples[:, 2]],
        dim=1
    )
    margin = 1.0
    kg_loss = torch.mean(F.relu(positive_score - negative_score + margin))
    return kg_loss


def calc_ap_loss(predicted_attributes, true_attributes):
    """
    Calculate the Attribute Prediction Loss using Mean Squared Error.
    Args:
        predicted_attributes (Tensor): Predicted attribute values.
        true_attributes (Tensor): Ground truth attribute values.
    Returns:
        Tensor: Computed attribute prediction loss.
    """
    return F.mse_loss(predicted_attributes, true_attributes)


class KGANS(nn.Module):
    """
    KGANS Model:
      Learns embeddings for drug and cell nodes through multi-layer GAT message passing.
      For each meta path, embeddings are computed separately using a combination of neighbor
      sampling and aggregation. The resulting representations are then fused using multi-head
      attention and fed into a regression network for final prediction.
    """
    def __init__(self, args, n_entities, n_relations, e_dim, r_dim,
                 adj_entity_list, adj_relation_list, agg_method='Bi-Interaction'):
        """
        Args:
            args: Hyperparameters (e.g., dropout_ratio, n_iter, n_heads, meta_heads).
            n_entities (int): Number of entities.
            n_relations (int): Number of relations.
            e_dim (int): Dimensionality of entity embeddings.
            r_dim (int): Dimensionality of relation embeddings.
            adj_entity_list (list): List of neighbor entity lists for each meta path (nested lists).
            adj_relation_list (list): List of neighbor relation lists for each meta path (nested lists).
            agg_method (str): Aggregation method ('Bi-Interaction', 'concat', or 'sum').
        """
        super(KGANS, self).__init__()

        # Initialize entity and relation embeddings with a maximum norm constraint.
        self.entity_embs = nn.Embedding(n_entities, e_dim, max_norm=1)
        self.relation_embs = nn.Embedding(n_relations, r_dim, max_norm=1)

        self.dropout = args.dropout_ratio
        self.n_iter = args.n_iter
        self.dim = e_dim
        self.n_heads = args.n_heads

        # Store adjacent entities and relations for the 5 meta paths.
        self.adj_entity_meta_paths = [adj_entity_list[i] for i in range(5)]
        self.adj_relation_meta_paths = [adj_relation_list[i] for i in range(5)]

        # Define the attention network for GAT message passing.
        self.attention = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim, bias=False),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim, bias=False),
            nn.ReLU(),
            nn.Linear(self.dim, 1, bias=False),
            nn.Sigmoid(),
        )
        self._init_weight()

        self.dropout_layer = nn.Dropout(self.dropout)
        self.agg_method = agg_method
        self.agg = 'attention'  # Layer aggregation method: 'concat', 'sum', or 'attention'.
        self.leakyRelu = nn.LeakyReLU(negative_slope=0.2)

        # Define the regression network for final predictions.
        self.regression = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 1)
        )
        self.act = nn.Sigmoid()

        # Set up the linear transformation layers depending on the aggregation method.
        if agg_method == 'concat':
            self.W_concat = nn.Linear(e_dim * 2, e_dim)
        else:
            self.W1 = nn.Linear(e_dim * self.n_heads, e_dim * 2)
            if agg_method == 'Bi-Interaction':
                self.W2 = nn.Linear(e_dim * self.n_heads, e_dim * 2)

        # Define parameters for single-head attention across meta paths.
        self.attention_weights = nn.Parameter(torch.ones(5, requires_grad=True))
        self.softmax = nn.Softmax(dim=0)

        # Define parameters for multi-head attention fusion.
        self.meta_heads = args.meta_heads
        self.num_tensors = 5
        self.attention_weights_multi = nn.Parameter(torch.randn(self.meta_heads, self.num_tensors))
        self.softmax_multi = nn.Softmax(dim=-1)

        # Initialize the layer attention module for aggregating multi-layer features.
        self.layer_att = LayerAttention(self.dim * 2)
        # Define a linear transformation for aggregation in the 'sum' method.
        self.WW = nn.Linear(256, 128, bias=False)

    def _init_weight(self):
        """
        Initialize weights for entity/relation embeddings and the attention network using Xavier uniform initialization.
        """
        nn.init.xavier_uniform_(self.entity_embs.weight)
        nn.init.xavier_uniform_(self.relation_embs.weight)
        for layer in self.attention:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def meta_weight(self, tensors):
        """
        Single-head attention fusion for meta path features.
        Args:
            tensors (list): List of feature tensors for each meta path.
        Returns:
            Tuple[Tensor, Tensor]: Fused feature tensor and normalized attention weights.
        """
        weighted_tensors = [tensor * self.attention_weights[i] for i, tensor in enumerate(tensors)]
        weighted_sum = sum(weighted_tensors)
        normalized_attention = self.softmax(self.attention_weights)
        return weighted_sum, normalized_attention

    def meta_weight_multi(self, *tensors):
        """
        Multi-head attention fusion for meta path features.
        Args:
            *tensors: Variable number of feature tensors (expected 5 for the meta paths).
        Returns:
            Tuple[Tensor, Tensor]: Fused feature tensor and updated attention weights.
        """
        normalized_attention_weights = self.softmax_multi(self.attention_weights_multi)
        weighted_sums = []
        for i in range(self.meta_heads):
            weighted_sum = sum(tensor * normalized_attention_weights[i, idx]
                               for idx, tensor in enumerate(tensors))
            weighted_sums.append(weighted_sum)
        # Average the outputs from all heads.
        combined_weighted_sum = torch.stack(weighted_sums).mean(dim=0)
        new_attention = self.softmax(normalized_attention_weights.mean(dim=0))
        return combined_weighted_sum, new_attention

    def get_neighbors(self, nodes, meta_path_idx, node_type='cell'):
        """
        Retrieve neighbor node embeddings and the corresponding relation embeddings for a given meta path.
        Args:
            nodes (Tensor): Tensor of node indices of shape [batch_size].
            meta_path_idx (int): The index of the meta path (0 to 4).
            node_type (str): A string indicating whether the node is a 'cell' or 'drug' (for clarity).
        Returns:
            Tuple[List[Tensor], List[Tensor]]: Lists of neighbor entity embeddings and neighbor relation embeddings for each layer.
        """
        nodes = nodes.view(-1, 1).to(device)
        entities = [nodes]
        relations = []
        for _ in range(self.n_iter):
            neighbor_entities = torch.LongTensor(
                self.adj_entity_meta_paths[meta_path_idx][entities[-1].to('cpu')]
            ).view(entities[-1].shape[0], -1).to(device)
            neighbor_relations = torch.LongTensor(
                self.adj_relation_meta_paths[meta_path_idx][entities[-1].to('cpu')]
            ).view(entities[-1].shape[0], -1).to(device)
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        neighbor_entities_embs = [self.entity_embs(entity) for entity in entities]
        neighbor_relations_embs = [self.relation_embs(relation) for relation in relations]
        return neighbor_entities_embs, neighbor_relations_embs

    def sum_aggregator(self, embs):
        """
        Layer Aggregator:
          Aggregates multi-layer embeddings based on the selected method.
        Args:
            embs (list): List of embeddings for different layers.
        Returns:
            Tensor: Aggregated embedding.
        """
        e_u = embs[0]
        if self.agg == 'concat':
            for i in range(1, len(embs)):
                e_u = torch.cat((embs[i], e_u), dim=-1)
        elif self.agg == 'sum':
            for i in range(1, len(embs)):
                e_u += self.WW(embs[i])
        elif self.agg == 'attention':
            layer_tensor = torch.stack(embs[1:], dim=1)
            aggregated, _ = self.layer_att(layer_tensor)
            e_u = torch.cat((aggregated, e_u), dim=-1)
        return e_u

    def GATMessagePass(self, h_embs, r_embs, t_embs):
        """
        Multi-head GAT Message Passing:
          For each attention head:
            1. Concatenates head entity and relation vectors.
            2. Computes attention weights and normalizes them with softmax.
            3. Computes a weighted sum of neighbor tail entity vectors.
            4. Applies a linear transformation to the aggregated neighbor messages.
        Args:
            h_embs (Tensor): Head entity embeddings of shape [batch_size, n_neighbors, e_dim].
            r_embs (Tensor): Relation embeddings of shape [batch_size, n_neighbors, r_dim].
            t_embs (Tensor): Neighbor tail entity embeddings of shape [batch_size, n_neighbors, e_dim].
        Returns:
            Tensor: Concatenated output from all attention heads with shape [batch_size, n_heads * e_dim].
        """
        head_outputs = []
        h_embs, r_embs, t_embs = h_embs.to(device), r_embs.to(device), t_embs.to(device)
        for _ in range(self.n_heads):
            att_input = torch.cat((h_embs, r_embs), dim=-1)
            att_weights = self.attention(att_input).squeeze(-1)
            att_weights_norm = F.softmax(att_weights, dim=-1)
            weighted_t = att_weights_norm.unsqueeze(-1) * t_embs
            aggregated = weighted_t.sum(dim=1)
            # Dynamically create a linear layer (this can be defined outside the loop for efficiency)
            Wx = nn.Linear(self.dim, self.dim).to(device)
            head_output = Wx(aggregated)
            head_outputs.append(head_output)
        return torch.cat(head_outputs, dim=-1)

    def aggregate(self, h_embs, Nh_embs, method='Bi-Interaction'):
        """
        Fuse the original head entity embedding with the aggregated neighbor message.
        Args:
            h_embs (Tensor): Original head entity embedding of shape [batch_size, e_dim].
            Nh_embs (Tensor): Aggregated neighbor message of shape [batch_size, e_dim].
            method (str): Fusion method: 'Bi-Interaction', 'concat', or 'sum'.
        Returns:
            Tensor: Fused embedding.
        """
        if method == 'Bi-Interaction':
            out = self.leakyRelu(self.W1(h_embs + Nh_embs)) + self.leakyRelu(self.W2(h_embs * Nh_embs))
        elif method == 'concat':
            out = self.leakyRelu(self.W_concat(torch.cat([h_embs, Nh_embs], dim=-1)))
        else:  # sum
            out = self.leakyRelu(self.W1(h_embs + Nh_embs))
        return out

    def forward(self, d, c):
        """
        Forward Pass:
          1. Compute cell and drug embeddings using multi-layer message passing.
          2. For each meta path, process the neighbors separately.
          3. Fuse the outputs from the 5 meta paths using multi-head attention.
          4. Pass the fused features through a regression network to obtain final predictions.
        Args:
            d (Tensor): Tensor of drug node indices.
            c (Tensor): Tensor of cell node indices.
        Returns:
            Tuple[Tensor, Tensor, Tensor]:
              - Final prediction after applying the sigmoid activation.
              - Fused feature representation.
              - Attention weights computed during meta path fusion.
        """
        d, c = d.to(device), c.to(device)

        # ----- Cell Embedding Learning -----
        t_embs_cell, r_embs_cell = [], []
        # Retrieve neighbor embeddings for cells for each meta path.
        for meta_idx in range(5):
            t_embs, r_embs = self.get_neighbors(c, meta_idx, node_type='cell')
            t_embs_cell.append(t_embs)
            r_embs_cell.append(r_embs)
        h_embs = self.entity_embs(c).to(device)

        # Perform multi-layer message passing for cell nodes (n_iter iterations).
        t_vectors_cell = [[h_embs] for _ in range(5)]
        for i in range(self.n_iter):
            for m in range(5):
                # Broadcast head entity embeddings to match the neighbor dimensions.
                h_broadcast = torch.cat([h_embs.unsqueeze(1)] * t_embs_cell[m][i+1].shape[1], dim=1).to(device)
                vector = self.GATMessagePass(h_broadcast, r_embs_cell[m][i], t_embs_cell[m][i+1])
                vector = self.leakyRelu(vector)
                if i == 0:
                    h_expanded = torch.cat([h_embs for _ in range(self.n_heads)], dim=1)
                    cell_emb = self.aggregate(h_expanded, vector, self.agg_method)
                else:
                    h_sum = torch.sum(t_embs_cell[m][i], dim=1)
                    h_broadcast = torch.cat([h_sum.unsqueeze(1)] * t_embs_cell[m][i+1].shape[1], dim=1)
                    vector = self.GATMessagePass(h_broadcast, r_embs_cell[m][i], t_embs_cell[m][i+1])
                    vector = self.leakyRelu(vector)
                    h_sum_expanded = torch.cat([torch.sum(t_embs_cell[m][i], dim=1) for _ in range(self.n_heads)], dim=1)
                    cell_emb = self.aggregate(h_sum_expanded, vector, self.agg_method)
                t_vectors_cell[m].append(cell_emb)
        # Aggregate embeddings from all layers for each meta path.
        cell_embs = [self.sum_aggregator(t_vectors_cell[m]).to(device) for m in range(5)]

        # ----- Drug Embedding Learning -----
        t_embs_drug, r_embs_drug = [], []
        for meta_idx in range(5):
            t_embs, r_embs = self.get_neighbors(d, meta_idx, node_type='drug')
            t_embs_drug.append(t_embs)
            r_embs_drug.append(r_embs)
        h_embs_drug = self.entity_embs(d).to(device)
        t_vectors_drug = [[h_embs_drug] for _ in range(5)]
        for i in range(self.n_iter):
            for m in range(5):
                h_broadcast = torch.cat([h_embs_drug.unsqueeze(1)] * t_embs_drug[m][i+1].shape[1], dim=1).to(device)
                vector = self.GATMessagePass(h_broadcast, r_embs_drug[m][i], t_embs_drug[m][i+1])
                vector = self.leakyRelu(vector)
                if i == 0:
                    h_expanded = torch.cat([h_embs_drug for _ in range(self.n_heads)], dim=1)
                    drug_emb = self.aggregate(h_expanded, vector, self.agg_method)
                else:
                    h_sum = torch.sum(t_embs_drug[m][i], dim=1)
                    h_broadcast = torch.cat([h_sum.unsqueeze(1)] * t_embs_drug[m][i+1].shape[1], dim=1)
                    vector = self.GATMessagePass(h_broadcast, r_embs_drug[m][i], t_embs_drug[m][i+1])
                    vector = self.leakyRelu(vector)
                    h_sum_expanded = torch.cat([torch.sum(t_embs_drug[m][i], dim=1) for _ in range(self.n_heads)], dim=1)
                    drug_emb = self.aggregate(h_sum_expanded, vector, self.agg_method)
                t_vectors_drug[m].append(drug_emb)
        drug_embs = [self.sum_aggregator(t_vectors_drug[m]).to(device) for m in range(5)]

        # ----- Fuse Meta Path Features -----
        meta_path_embeddings = [
            torch.cat((drug_embs[m], cell_embs[m]), dim=1) for m in range(5)
        ]
        weighted_sum, new_attention = self.meta_weight_multi(*meta_path_embeddings)
        logits_FNN = self.regression(weighted_sum).squeeze()
        logits_FNN = torch.sigmoid(logits_FNN)

        return logits_FNN, weighted_sum, new_attention
