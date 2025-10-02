import numpy as np
import torch
import random
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


dir = './dataset'

def get_cora_casestudy(SEED=0):
    data_Y, data_edges = parse_cora()

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Load data
    data_name = 'cora'
    dataset = Planetoid(dir, data_name, transform=T.NormalizeFeatures())
    data = dataset[0]

    data.edge_index = torch.tensor(data_edges).long()
    data.y = torch.tensor(data_Y).long()
    data.num_nodes = len(data_Y)
    # print(len(data_Y))

    return data

def parse_cora():
    path = dir + '/cora_orig/cora'
    idx_features_labels = np.genfromtxt(
        "{}.content".format(path), dtype=np.dtype(str))
    labels = idx_features_labels[:, -1]
    class_map = {x: i for i, x in enumerate(['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
                                            'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory'])}
    
    data_Y = np.array([class_map[l] for l in labels])
    data_citeid = idx_features_labels[:, 0]
    idx = np.array(data_citeid, dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(
        "{}.cites".format(path), dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(
        edges_unordered.shape)
    data_edges = np.array(edges[~(edges == None).max(1)], dtype='int')
    data_edges = np.vstack((data_edges, np.fliplr(data_edges)))
    return data_Y, np.unique(data_edges, axis=0).transpose()

def random_drop_nodes(data, drop_ratio=0.1, seed=42):
    """
    Randomly drop nodes and their related edges.
    Returns:
        - Modified graph data
        - Indices of kept nodes
    """
    torch.manual_seed(seed)
    assert 0 <= drop_ratio < 1, "drop_ratio should be in [0, 1) range"

    num_nodes = data.num_nodes
    num_drop_nodes = int(num_nodes * drop_ratio)

    if num_drop_nodes == 0:
        return data, torch.arange(num_nodes)

    drop_nodes = torch.randperm(num_nodes, device=data.edge_index.device)[:num_drop_nodes]
    keep_nodes = torch.tensor(
        [i for i in range(num_nodes) if i not in drop_nodes],
        dtype=torch.long,
        device=data.edge_index.device
    )

    row, col = data.edge_index
    edge_mask = torch.isin(row, keep_nodes) & torch.isin(col, keep_nodes)
    new_edge_index = data.edge_index[:, edge_mask]

    node_map = torch.zeros(num_nodes, dtype=torch.long, device=data.edge_index.device)
    node_map[keep_nodes] = torch.arange(len(keep_nodes), device=data.edge_index.device)
    new_edge_index = node_map[new_edge_index]

    new_data = data.clone()
    new_data.x = data.x[keep_nodes] if data.x is not None else None
    new_data.y = data.y[keep_nodes]
    new_data.edge_index = new_edge_index
    new_data.num_nodes = len(keep_nodes)

    return new_data, keep_nodes


def reduce_same_type_edges(data, reduction_ratio=0.5, seed=42):
    """
    Reduce edges between nodes of the same type by a given ratio.
    Args:
        data (torch_geometric.data.Data): Graph data.
        reduction_ratio (float): Ratio of same-type edges to remove, range [0, 1).
        seed (int): Random seed.
    Returns:
        torch_geometric.data.Data: Modified graph data.
    """
    torch.manual_seed(seed)

    row, col = data.edge_index
    same_type_edges = torch.where(data.y[row] == data.y[col])[0]
    num_same_type_edges = len(same_type_edges)
    num_to_remove = int(num_same_type_edges * reduction_ratio)
    if num_to_remove == 0:
        return data
    remove_indices = torch.randperm(num_same_type_edges)[:num_to_remove]
    edges_to_remove = same_type_edges[remove_indices]
    edge_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool, device=data.edge_index.device)
    edge_mask[edges_to_remove] = False

    data.edge_index = data.edge_index[:, edge_mask]
    return data

def reduce_global_feature_density_on_embed(embed, drop_ratio=0.5, seed=42):
    np.random.seed(seed)
    mask = embed != 0
    all_nonzero = np.argwhere(mask)
    num_to_drop = int(len(all_nonzero) * drop_ratio)

    if num_to_drop > 0:
        drop_indices = np.random.choice(len(all_nonzero), size=num_to_drop, replace=False)
        drop_positions = all_nonzero[drop_indices]
        embed[drop_positions[:, 0], drop_positions[:, 1]] = 0

    return embed

def split_data_by_ratio(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    """
    Split dataset into training, validation, and test sets by ratio.
    Args:
        data (torch_geometric.data.Data): Graph data.
        train_ratio (float): Training set ratio.
        val_ratio (float): Validation set ratio.
        test_ratio (float): Test set ratio.
        seed (int): Random seed.
    Returns:
        train_mask, val_mask, test_mask: Boolean masks for each set.
    """
    assert train_ratio + val_ratio + test_ratio <= 1.0, "Sum of train, val, test ratios cannot exceed 1.0"

    np.random.seed(seed)
    num_nodes = data.num_nodes
    node_indices = np.arange(num_nodes)
    np.random.shuffle(node_indices)

    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)
    test_size = int(num_nodes * test_ratio)

    train_idx = node_indices[:train_size]
    val_idx = node_indices[train_size:train_size + val_size]
    test_idx = node_indices[train_size + val_size:train_size + val_size + test_size]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask

def load_cora(args, per_classnum=10, seed=0, thred=1.0, lam=-1, model_type=''):
    data = get_cora_casestudy(seed)
    np.random.seed(seed)

    # Randomly drop nodes (optional, uncomment if needed)
    data, keep_nodes = random_drop_nodes(data, drop_ratio=args.node_drop_ratio, seed=seed)

    # Reduce same-type edges
    data = reduce_same_type_edges(data, reduction_ratio=args.same_type_edge_reduction_ratio, seed=seed)

    # Split data into training, validation, and testing sets
    train_mask, val_mask, test_mask = split_data_by_ratio(data, train_ratio=args.train_ratio, val_ratio=0.2, test_ratio=0.2, seed=seed)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # Load embeddings
    new_embed = np.load(dir + '/cora_process/new_nodes_optimized_supcon_rag_simple.npy')
    text_embed = np.load(dir + '/cora_process/original_nodes_optimized_supcon_rag_simple.npy')
    text_embed = text_embed[keep_nodes]

    # Apply feature density reduction to text_embed
    text_embed = reduce_global_feature_density_on_embed(text_embed, drop_ratio=args.feature_drop_ratio, seed=seed)

    # Add new nodes and update labels
    N = data.num_nodes
    M = new_embed.shape[0]
    C = data.y.max() + 1
    
    new_y = []
    samples_per_category = M // C
    remaining_samples = M % C
    
    category_counts = [samples_per_category] * C
    for i in range(remaining_samples):
        category_counts[i] += 1
    
    cycle_index = 0
    for _ in range(M):
        category = cycle_index % C
        new_y.append(category)
        cycle_index += 1
    
    print(f"New node label distribution: {np.bincount(new_y)}")
    data.y = torch.cat((data.y, torch.tensor(new_y)))
    data.num_nodes = N + M

    new_train_mask = torch.cat([data.train_mask, torch.ones(M, dtype=torch.bool)])
    new_val_mask = torch.cat([data.val_mask, torch.zeros(M, dtype=torch.bool)])
    new_test_mask = torch.cat([data.test_mask, torch.zeros(M, dtype=torch.bool)])

    data.train_mask = new_train_mask
    data.val_mask = new_val_mask
    data.test_mask = new_test_mask

    assert data.train_mask.shape[0] == data.num_nodes, "train_mask shape doesn't match number of nodes"
    assert data.val_mask.shape[0] == data.num_nodes, "val_mask shape doesn't match number of nodes"
    assert data.test_mask.shape[0] == data.num_nodes, "test_mask shape doesn't match number of nodes"

    # Connect new nodes to the graph by generating edges based on similarity
    sim = torch.mm(torch.from_numpy(new_embed), torch.from_numpy(text_embed).t())
    new_edge = torch.nonzero(sim > 0.7)  # Filter based on similarity threshold
    # print(new_edge)
    new_edge[:, 0] += N  # Adjust for new nodes' indices

    # Update edge index to include new edges
    data.edge_index = torch.cat([data.edge_index, new_edge.transpose(0, 1)], dim=1)

    # For edge prediction tasks, prepare edges and labels
    if model_type == 'Edge':
        E = data.edge_index.size(1)
        sparse_matrix = torch.sparse.FloatTensor(data.edge_index,
                                                 torch.ones(E),
                                                 torch.Size([N + M, N + M]))
        adj = sparse_matrix.to_dense()
        missing_edge = torch.nonzero(adj != 1)
        missing_edge_idx = np.arange(0, missing_edge.size(0))
        np.random.shuffle(missing_edge_idx)
        missing_edge_index = missing_edge[missing_edge_idx[0:E]].transpose(0, 1)
        data.edge_y = torch.cat([torch.ones(E), torch.zeros(E)])
        data.train_edge = torch.cat([data.edge_index, missing_edge_index], dim=1)

    return data, np.concatenate((text_embed, new_embed))