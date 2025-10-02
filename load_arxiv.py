import os
import random
import numpy as np
import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_undirected, negative_sampling

from utils import set_params

DATA_DIR = './dataset'


def random_drop_nodes(data, drop_ratio=0.1, seed=42):
    """Randomly drop nodes and their incident edges."""
    torch.manual_seed(seed)
    assert 0 <= drop_ratio < 1, "drop_ratio should be in [0, 1) range"

    num_nodes = data.num_nodes
    num_drop_nodes = int(num_nodes * drop_ratio)

    if num_drop_nodes == 0:
        keep_nodes = torch.arange(num_nodes, device=data.edge_index.device)
        return data, keep_nodes

    drop_nodes = torch.randperm(num_nodes, device=data.edge_index.device)[:num_drop_nodes]
    keep_nodes = torch.tensor(
        [i for i in range(num_nodes) if i not in drop_nodes],
        dtype=torch.long,
        device=data.edge_index.device,
    )

    row, col = data.edge_index
    edge_mask = torch.isin(row, keep_nodes) & torch.isin(col, keep_nodes)
    new_edge_index = data.edge_index[:, edge_mask]

    node_map = torch.zeros(num_nodes, dtype=torch.long, device=data.edge_index.device)
    node_map[keep_nodes] = torch.arange(len(keep_nodes), device=data.edge_index.device)
    new_edge_index = node_map[new_edge_index]

    new_data = data.clone()
    new_data.edge_index = new_edge_index
    new_data.num_nodes = len(keep_nodes)

    if data.x is not None:
        new_data.x = data.x[keep_nodes]
    new_data.y = data.y[keep_nodes]
    if hasattr(data, 'node_year') and data.node_year is not None:
        new_data.node_year = data.node_year[keep_nodes]

    return new_data, keep_nodes


def reduce_same_type_edges(data, reduction_ratio=0.5, seed=42):
    """Randomly remove a proportion of edges connecting nodes with identical labels."""
    torch.manual_seed(seed)

    row, col = data.edge_index
    same_type_edges = torch.where(data.y[row] == data.y[col])[0]
    num_same_type_edges = same_type_edges.numel()
    num_to_remove = int(num_same_type_edges * reduction_ratio)

    if num_to_remove == 0:
        return data

    remove_indices = torch.randperm(num_same_type_edges, device=data.edge_index.device)[:num_to_remove]
    edges_to_remove = same_type_edges[remove_indices]
    edge_mask = torch.ones(data.edge_index.size(1), dtype=torch.bool, device=data.edge_index.device)
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
    """Generate boolean masks via random splits according to provided ratios."""
    assert train_ratio + val_ratio + test_ratio <= 1.0, "Sum of ratios cannot exceed 1.0"

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


def load_arxiv(args=None, per_classnum=10, seed=0, thred=1.0, lam=-1, model_type=''):
    """Load the ogbn-arxiv dataset with preprocessing aligned to other loaders."""
    if args is None:
        args = set_params()

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=DATA_DIR)
    data = dataset[0]

    data.y = data.y.squeeze()
    if data.y.dim() == 0:
        data.y = data.y.unsqueeze(0)
    data.y = data.y.long()
    data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)

    data, keep_nodes = random_drop_nodes(data, drop_ratio=args.node_drop_ratio, seed=seed)
    data = reduce_same_type_edges(data, reduction_ratio=args.same_type_edge_reduction_ratio, seed=seed)
    data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)

    train_mask, val_mask, test_mask = split_data_by_ratio(
        data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=seed,
    )
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    arxiv_dir = os.path.join(DATA_DIR, 'ogbn_arxiv_process')
    new_embed = np.load(os.path.join(arxiv_dir, 'new_sample.npy')).astype(np.float32)
    text_embed = np.load(os.path.join(arxiv_dir, 'raw_text.npy')).astype(np.float32)

    keep_indices = keep_nodes.cpu().numpy()
    text_embed = text_embed[keep_indices]
    text_embed = reduce_global_feature_density_on_embed(
        text_embed,
        drop_ratio=args.feature_drop_ratio,
        seed=seed,
    )

    N = data.num_nodes
    M = new_embed.shape[0]
    C = int(data.y.max().item() + 1)

    new_y = []
    samples_per_category = M // C
    remaining_samples = M % C

    for class_id in range(C):
        count = samples_per_category + (1 if class_id < remaining_samples else 0)
        new_y.extend([class_id] * count)
    if len(new_y) != M:
        new_y.extend([new_y[-1]] * (M - len(new_y)))

    data.y = torch.cat((data.y, torch.tensor(new_y, dtype=data.y.dtype)))
    data.num_nodes = N + M

    data.train_mask = torch.cat([data.train_mask, torch.ones(M, dtype=torch.bool)])
    data.val_mask = torch.cat([data.val_mask, torch.zeros(M, dtype=torch.bool)])
    data.test_mask = torch.cat([data.test_mask, torch.zeros(M, dtype=torch.bool)])

    assert data.train_mask.shape[0] == data.num_nodes
    assert data.val_mask.shape[0] == data.num_nodes
    assert data.test_mask.shape[0] == data.num_nodes

    sim = torch.mm(torch.from_numpy(new_embed), torch.from_numpy(text_embed).t())
    new_edge = torch.nonzero(sim > 0.7, as_tuple=False)
    if new_edge.numel() > 0:
        new_edge[:, 0] += N
        new_edge = new_edge.t()
        data.edge_index = torch.cat([data.edge_index, new_edge], dim=1)
        data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)

    combined_embed = np.concatenate((text_embed, new_embed), axis=0)

    if model_type == 'Edge':
        E = data.edge_index.size(1)
        neg_edge_index = negative_sampling(
            data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=E,
            method='sparse',
        )
        data.edge_y = torch.cat([torch.ones(E), torch.zeros(neg_edge_index.size(1))])
        data.train_edge = torch.cat([data.edge_index, neg_edge_index], dim=1)

        predict_sim = torch.mm(
            torch.from_numpy(new_embed),
            torch.from_numpy(combined_embed).t(),
        )
        predict_new_edge = torch.nonzero(predict_sim > thred, as_tuple=False)
        if predict_new_edge.numel() > 0:
            predict_new_edge[:, 0] += N
            data.predict_edge_index = predict_new_edge.t()
        else:
            data.predict_edge_index = torch.empty((2, 0), dtype=torch.long)
    elif model_type == 'Node':
        if lam != -1:
            adj_path = os.path.join(arxiv_dir, 'adj', f'lam_{lam}_thred_{thred}.pt')
            if os.path.exists(adj_path):
                data.edge_index = torch.load(adj_path)
            else:
                raise FileNotFoundError(f"Adjacency matrix not found at {adj_path}")
    else:
        if model_type:
            raise ValueError(f"Unsupported model_type: {model_type}")

    return data, combined_embed
