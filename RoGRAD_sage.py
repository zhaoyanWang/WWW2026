import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from load_cora import load_cora
from load_pubmed import load_pubmed
from load_arxiv import load_arxiv
import random
import numpy as np
from utils import set_params
import os
import pandas as pd

class GraphSAGE_Encoder(torch.nn.Module):                           
    def __init__(self, in_channels, hidden_dim, out_channels):
        super(GraphSAGE_Encoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, out_channels)

    def forward(self, x, edge_index, drop):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=drop, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=drop, training=self.training)
        x = self.out(x)
        return F.log_softmax(x, dim=1)
        
def node_classification():
    node_drop_ratios = [0, 0.5, 0.9]
    same_type_edge_reduction_ratios = [0, 0.5, 0.9]
    feature_drop_ratios = [0, 0.5, 0.9]
    train_ratios = [0.6,0.4,0.2]
    val_ratio = 0.2
    test_ratio = 0.2

    os.makedirs("results", exist_ok=True)
    results_file = "results/all_results_label_pubmedGraphSAGE.csv"

    if not os.path.exists(results_file):
        pd.DataFrame(columns=[
            "node_drop_ratio", 
            "same_type_edge_reduction_ratio", 
            "feature_drop_ratio", 
            "train_ratio", 
            "val_ratio", 
            "test_ratio", 
            "test_acc"
        ]).to_csv(results_file, index=False)

    for node_drop_ratio in node_drop_ratios:
        for same_type_edge_reduction_ratio in same_type_edge_reduction_ratios:
            for feature_drop_ratio in feature_drop_ratios:
                for train_ratio in train_ratios:
                    args = set_params()
                    args.node_drop_ratio = node_drop_ratio
                    args.same_type_edge_reduction_ratio = same_type_edge_reduction_ratio
                    args.feature_drop_ratio = feature_drop_ratio
                    args.train_ratio = train_ratio
                    args.val_ratio = val_ratio
                    args.test_ratio = test_ratio
                    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
                    result = []
                    # print(args)
                    # print(device)
                    if args.model_type != 'Node':
                        print("error")
                        exit()
                    for s in [42, 48, 13, 25, 68]:
                        torch.manual_seed(s)
                        torch.cuda.manual_seed(s)
                        np.random.seed(s) 
                        random.seed(s)

                        if args.dataset == 'cora':
                            data, text_embed = load_cora(args=args, seed=s, thred=args.thred, lam=args.lam, model_type=args.model_type)
                            # print(data.y)
                        elif args.dataset == 'pubmed':
                            data, text_embed = load_pubmed(args=args,seed=s, thred=args.thred, lam=args.lam, model_type=args.model_type)
                        elif args.dataset == 'arxiv':
                            data, text_embed = load_arxiv(args=args, seed=s, thred=args.thred, lam=args.lam, model_type=args.model_type)
                        else:
                            print("error")
                            exit()
                        
                        # parameters
                        hidden_dim = args.hidden
                        out_channels = int(data.y.max() + 1)
                        dropout = args.dropout
                        epochs = 500

                        data = data.to(device)
                        text_embed = torch.from_numpy(text_embed).to(device)
                        input_feature = text_embed

                        # model
                        num_features = input_feature.size(1)
                        Node = GraphSAGE_Encoder(num_features, hidden_dim, out_channels).to(device)
                        optimizer_n = torch.optim.Adam(Node.parameters(), lr=args.lr, weight_decay=args.l2_coef)


                        max_val = 0
                        best_test = 0
                        
                        for epoch in range(1, epochs + 1):
                            Node.train()
                            optimizer_n.zero_grad()
                            out_node = Node(input_feature, data.edge_index, dropout)
                            loss_node = F.nll_loss(out_node[data.train_mask], data.y[data.train_mask])
                            loss_node.backward()
                            optimizer_n.step()

                            Node.eval()
                            logits = out_node.argmax(dim=1)
                            train_acc = logits[data.train_mask].eq(data.y[data.train_mask]).sum() / data.train_mask.sum()
                            val_acc = logits[data.val_mask].eq(data.y[data.val_mask]).sum() / data.val_mask.sum()
                            test_acc = logits[data.test_mask].eq(data.y[data.test_mask]).sum() / data.test_mask.sum()
                            if val_acc > max_val:
                                max_val = val_acc
                                best_test = test_acc
                            if epoch % 5 == 0 :
                                pass
                                # print(loss_node, args.lam * loss_edge, 0.001 * predict_edge_norm)
                                # print('loss:', loss)
                                # print('Train_acc: {:.4f},  Val_acc: {:.4f}, Test_acc: {:.4f}'.format(train_acc, val_acc, test_acc))
                        # print('Best epoch: {:d}, Best test acc: {:.4f}'.format(max_epoch, best_test))
                        result.append(best_test.cpu().detach())

                    result = np.array(result)
                    # print(result)
                    # print('Ratio:{:d} Results:{:.2f}({:.2f})'.format(args.ratio, result.mean()*100, result.std()*100))
                    experiment_result = {
                        "node_drop_ratio": node_drop_ratio,
                        "same_type_edge_reduction_ratio": same_type_edge_reduction_ratio,
                        "feature_drop_ratio": feature_drop_ratio,
                        "train_ratio": train_ratio,
                        "val_ratio": val_ratio,
                        "test_ratio": test_ratio,
                        "test_acc": f"{result.mean()*100:.2f} ± {result.std():.2f}"
                    }
                    pd.DataFrame([experiment_result]).to_csv("results/all_results_label_pubmedGraphSAGE.csv", mode='a', header=False, index=False)
                    print(experiment_result)
    print("Results saved to results/all_results_label_pubmedGraphSAGE.csv")
  
if __name__ == '__main__':
    node_classification()
