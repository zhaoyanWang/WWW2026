import argparse

def set_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', type=int, default=20)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--l2_coef', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--lam', type=int, default=-1)
    parser.add_argument('--thred', type=float, default=1.0)
    parser.add_argument('--model_type', type=str, default='')
    parser.add_argument("--node_drop_ratio", type=float, default=0.5, help="Proportion of nodes to randomly drop.")
    parser.add_argument("--same_type_edge_reduction_ratio", type=float, default=0.5, help="Proportion of same-type edges to reduce.")
    parser.add_argument("--feature_drop_ratio", type=float, default=0.5, help="Proportion of features to randomly drop.")
    parser.add_argument("--train_ratio", type=int, default=0.2, help="Number of nodes in training set.")
    parser.add_argument("--val_ratio", type=int, default=0.2, help="Number of nodes in validation set.")
    parser.add_argument("--test_ratio", type=int, default=0.2, help="Number of nodes in test set.")
    parser.add_argument("--k_neighbors", type=int, default=5, help="Number of neighbors for class-constrained k-NN edge generation.")
    parser.add_argument("--nodes_per_class", type=int, default=10, help="Number of new nodes to select per class.")

    args, _ = parser.parse_known_args()
    return args