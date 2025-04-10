import collections
import numpy as np

from load_gat import readKGData_ymx


def construct_kg(kgTriples):
    """
    Generate the knowledge graph index.

    For each triple in kgTriples, the function adds bidirectional edges:
    both (tail, relation) are added to head's list and (head, relation) are
    added to tail's list.

    Args:
        kgTriples (list): List of triples, each triple is [head, relation, tail].

    Returns:
        dict: A dictionary mapping each entity to a list of (neighbor, relation) tuples.
    """
    print('生成知识图谱索引图')
    kg = dict()
    for triple in kgTriples:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))
    return kg


def getKgIndexsFromKgTriples(kgTriples_list):
    """
    Generate KG indexes from a list of KG triples lists.

    For each meta-path in kgTriples_list, this function creates a dictionary
    mapping the head entity (as a string) to a list of [tail, relation] pairs.

    Args:
        kgTriples_list (list): List of lists of triples.

    Returns:
        list: A list of dictionaries (one per meta-path) representing KG indexes.
    """
    kg_indexes_list = []
    for triple_list in kgTriples_list:
        kg_indexs = collections.defaultdict(list)
        for h, r, t in triple_list:
            kg_indexs[str(h)].append([int(t), int(r)])  # Head entity's neighbors and relation
        kg_indexes_list.append(kg_indexs)
    return kg_indexes_list


def filetDateSet(dataSet, user_pos):
    """
    Filter the dataset to include only records whose first element is in user_pos.

    Args:
        dataSet (list): List of records.
        user_pos (iterable): Collection of keys to filter by.

    Returns:
        list: Filtered dataset.
    """
    return [i for i in dataSet if str(i[0]) in user_pos]


def construct_adj(neighbor_sample_size, kg_indexes_list, entity_num):
    """
    Generate entity and relation adjacency lists based on the KG indexes.

    For each meta-path (assumed to be 5), this function creates two numpy arrays
    of shape [entity_num, neighbor_sample_size] for neighbor entities and their corresponding
    relations. For each entity, neighbor indices are sampled without replacement if the
    number of neighbors is sufficient; otherwise, sampling is done with replacement.

    Args:
        neighbor_sample_size (int): Number of neighbors to be sampled.
        kg_indexes_list (list): List of KG index dictionaries (one per meta-path).
        entity_num (int): Total number of entities.

    Returns:
        tuple: (adj_entity_1, adj_relation_1, adj_entity_2, adj_relation_2,
                adj_entity_3, adj_relation_3, adj_entity_4, adj_relation_4,
                adj_entity_5, adj_relation_5)
    """
    num_meta = 5  # The original code expects 5 meta-paths.
    adj_entities_list = []
    adj_relations_list = []

    for idx in range(num_meta):
        # Initialize adjacency arrays for current meta-path.
        adj_entity = np.zeros([entity_num, neighbor_sample_size], dtype=np.int64)
        adj_relation = np.zeros([entity_num, neighbor_sample_size], dtype=np.int64)
        kg_indexes = kg_indexes_list[idx]

        for entity in range(entity_num):
            neighbors = kg_indexes.get(str(entity), [])
            n_neighbors = len(neighbors)
            if n_neighbors == 0:
                continue
            if n_neighbors >= neighbor_sample_size:
                sampled_indices = np.random.choice(range(n_neighbors), size=neighbor_sample_size, replace=False)
            else:
                sampled_indices = np.random.choice(range(n_neighbors), size=neighbor_sample_size, replace=True)
            adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
            adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])

        adj_entities_list.append(adj_entity)
        adj_relations_list.append(adj_relation)

    # Return the arrays in the specific order required
    return (adj_entities_list[0], adj_relations_list[0],
            adj_entities_list[1], adj_relations_list[1],
            adj_entities_list[2], adj_relations_list[2],
            adj_entities_list[3], adj_relations_list[3],
            adj_entities_list[4], adj_relations_list[4])

