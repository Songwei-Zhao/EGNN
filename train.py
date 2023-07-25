import numpy as np
import torch
import argparse

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from model import DNN
from Operator import *
import os
from os import path
import gdown
from datalo import load_planetoid_dataset
import torch_geometric

def propagate(features, k):
    feature_list = []
    feature_list.append(features)
    for i in range(1, k):
        feature_list.append(torch.spmm(adj_norm, feature_list[-1]))
    return feature_list

def train(epoch, model, feature, record, idx_train, idx_val, idx_test, labels):
    model.train()
    optimizer.zero_grad()
    output = model(feature)
    
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    model.eval()
    output = model(feature)

    acc_val = accuracy(output[idx_val], labels[idx_val])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    record[acc_val.item()] = acc_test.item()
    return  acc_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-1, help='Initial learning rate.')
    parser.add_argument('--k1', type=int, default=21, help='Value of K in stage (1).')
    parser.add_argument('--k2', type=int, default=6, help='Value of K in stage (3).')
    parser.add_argument('--epsilon1', type=float, default=0.03, help='Value of epsilon in stage (1).')
    parser.add_argument('--epsilon2', type=float, default=0.05, help='Value of epsilon in stage (2).')
    parser.add_argument('--hidden', type=int, default=16, help='Dim of hidden layer.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate of input and hidden layers.')
    # parser.add_argument('--dataset', type=str, default='pubmed', help='type of dataset.')
    parser.add_argument('--runs', type=int, default=1, help='Number of run times.')
    args = parser.parse_args()

    fin_ACC = []

    datasetname = 'cora'
    dataset = load_planetoid_dataset(datasetname)
    print("▓▓▓▓▓▓", datasetname, "▓▓▓▓▓▓")

    split_idx = dataset.get_idx_split()
    idx_train, idx_val, idx_test = split_idx["train"], split_idx["valid"], split_idx["test"]
    adj = dataset.graph['edge_index']
    adj = torch_geometric.utils.to_scipy_sparse_matrix(adj)
    labels = dataset.label
    features = dataset.graph['node_feat']

    print("| # of nodes : {}".format(adj.shape[0]))
    print("| # of edges : {}".format(adj.sum().sum()))
    print("| # of features : {}".format(features.shape[1]))
    print("| # of clases   : {}".format(labels.max() + 1))
    print("| # of train set : {}".format(len(idx_train)))
    print("| # of val set   : {}".format(len(idx_val)))
    print("| # of test set  : {}".format(len(idx_test)))

    node_sum = adj.shape[0]
    edge_sum = adj.sum()/2
    row_sum = (adj.sum(1) + 1)
    norm_a_inf = row_sum/ (2*edge_sum+node_sum)

    adj_norm = sparse_mx_to_torch_sparse_tensor(aug_random_walk(adj))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = F.normalize(features, p=1)
    feature_list = []
    feature_list.append(features)
    for i in range(1, args.k1):
        feature_list.append(torch.spmm(adj_norm, feature_list[-1]))

    norm_a_inf = torch.Tensor(norm_a_inf).view(-1, node_sum)
    norm_fea_inf = torch.mm(norm_a_inf, features)
    n_class = (labels.max()+1).item()

    print("Local Smoothing is done.")

    hops, input_feature, ACC = GWO(F2, 0, 20, node_sum, 30, 100, feature_list, norm_fea_inf, args, n_class, labels, idx_train, idx_val, idx_test)

    input_feature = input_feature.to(device)

    print("Start training...")
    for i in range(args.runs):
        best_acc = 0
        record = {}
        model = DNN(features.shape[1], args.hidden, n_class, args.dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        for epoch in range(args.epochs):
            acc_val = train(epoch, model, input_feature, record, idx_train, idx_val, idx_test, labels.to(device))
        if acc_val > best_acc:
            best_acc = acc_val
            torch.save(model, './best.pt')

        bit_list = sorted(record.keys())
        bit_list.reverse()

        model = torch.load('./best.pt')
        model.eval()
        output = model(input_feature).cpu()

        final_acc = accuracy(output[idx_test], labels[idx_test])

        output_list = propagate(output, args.k2)

        norm_softmax_inf = torch.mm(norm_a_inf, output)


        hops, output_final, ACC = GWO(F3, 0, 5, node_sum, 30, 50,  output_list , norm_softmax_inf, args, n_class, labels, idx_train, idx_val, idx_test)

        print(f'Run {i + 1}: Test Accuracy {ACC}')
        fin_ACC.append(ACC)

    print(fin_ACC)

    print(f'Test Mean Accuracy {np.mean(fin_ACC)}, std {np.std(fin_ACC)}')
    print('\n')
