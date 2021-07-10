import pathlib
import pickle
import sys
import importlib
import torch
import os
import torch.nn.functional as F
from algorithm import convert_to_DGLGraph
import numpy as np
import networkx as nx
import argparse
from dgl.data.utils import load_graphs
from sage.all import *

def read_graph(filename):
    graph_path = "/".join(filename.split(".")[0:-1]) + ".bin"
    feature_path = "/".join(filename.split(".")[0:-1]) + ".pkl"
    graph = load_graphs(graph_path)
    data = pickle.load(open(feature_path, 'rb'))
    # with open(filename, 'rb') as f:
    #     data = pickle.load(f)
    return (graph, data)

def num_one(source_array):
    count = 0
    for x in source_array:
        if x == 1:
            count += 1
    return count

def cal_center(g, distance=False):
    G = g.to_networkx(node_attrs=['x'], edge_attrs=['weight'])
    try:
        adj = nx.adjacency_matrix(G, weight='weight')
    except:
        print('The graph has no nodes.')
    else:
        dense_adj = adj.todense()
        sums = np.max(dense_adj, axis=1)
        min_index, min_dist = min(enumerate(sums), key=operator.itemgetter(1))
        center = g.parent_nid[min_index-1].numpy().tolist()
    if distance:
        return center, min_dist[0,0]
    else:
        return min_dist

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-nn', '--nnodes',
        help='nodes numbers',
        choices=['50', '100', '200', '500', '1000'],
        default=50,
    )
    parser.add_argument(
        '-nc', '--ncenter',
        help='center numbers',
        choices=['3', '6', '10'],
        default=3,
    )
    parser.add_argument(
        '-m', '--model',
        help='GNN model to be trained.',
        type=str,
        default='GAT',
        choices=['GCN', 'GAT', 'GraphSAGE']
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    args = parser.parse_args()

    k=args.ncenter
##################################################################
####LOAD_DATA
##################################################################
    sample_num = 1
    sample_file = f'data/samples/{args.ncenter}_center/{args.nnodes}/sample_{sample_num}.pkl'
    samples = read_graph(sample_file)

###################################################################
####IMPORT MODEL
####################################################################
    sys.path.insert(0, os.path.abspath(f'models/{args.model}'))
    import model
    importlib.reload(model)
    if args.model == 'GAT':
        net = model.GAT(in_dim=30,
                        hidden_dim=k*5,
                        out_dim=k,
                        num_heads=5)
    elif args.model == 'GCN':
        net = model.GCN(in_dim=30,
                        hidden_dim=k*5,
                        out_dim=k)
    elif args.model == 'GraphSAGE':
        net = model.GraphSAGE(in_feats=30,
                              n_hidden=k*5,
                              n_classes=k,
                              n_layers=1,
                              activation=F.relu,
                              dropout=True,
                              aggregator_type='mean')
    else:
        raise NotImplementedError
    del sys.path[0]

####################################################################
    ####READ GRAPH AND FEATURE
    running_dir = f"trained_models/p_center/{args.nnodes}_{args.ncenter}/{sample_num}/{args.model}"
    graph_dataset, dataset = samples
    g, data_ = graph_dataset[0][0], dataset
    feature, c, l, m, clsts = data_
    from torch.autograd import Variable
    feature = Variable(feature)
    clsts = Variable(torch.LongTensor(clsts))
    #########LOAD MODEL PARAMETERS
    net.load_state_dict(torch.load(os.path.join(running_dir, 'best_params.pkl')))
    logits = net(g, feature)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp, clsts)

    _, indices = torch.max(logp, dim=1)
    correct = torch.sum(indices == clsts)
    acc = correct.item() * 1.0 / len(clsts)

    if args.ncenter == 3:
        subg = [[],[],[]]
        s = 0
        for i in indices:
            if i == 0:
                subg[0].append(s)
                s += 1
            elif i == 1:
                subg[1].append(s)
                s += 1
            elif i == 2:
                subg[2].append(s)
                s += 1
    elif args.ncenter == 6:
        subg = []
        s = 0
        for i in indices:
            if i == 0:
                subg[0].append(s)
                s += 1
            elif i == 1:
                subg[1].append(s)
                s += 1
            elif i == 2:
                subg[2].append(s)
                s += 1
            elif i == 3:
                subg[3].append(s)
                s += 1
            elif i == 4:
                subg[4].append(s)
                s += 1
            elif i == 5:
                subg[5].append(s)
                s += 1

    G = [[],[],[]]
    centers = [[],[],[]]
    dists = [[],[],[]]
    for i in range(k):
        G[i] = g.subgraph(subg[i])
        G[i].ndata['x'] = g.ndata['z'][G[i].parent_nid]
        G[i].edata['weight'] = g.edata['weight'][G[i].parent_eid]
        G[i].edata['e'] = g.edata['e'][G[i].parent_eid]
        centers[i], dists[i] = cal_center(G[i], distance=True)
    dist = max(dists)
    print(dists)
    print(centers)
    print(dist)