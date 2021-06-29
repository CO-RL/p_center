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

def load_data(dataloader):
    graph_datasets = []
    datasets = []
    for data in dataloader:
        dataset = read_graph(data)
        graph, data = dataset
        graph_datasets.append(graph)
        datasets.append(data)

    return (graph_datasets, datasets)

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
    sample_files = list(pathlib.Path(f'data/samples/{args.ncenter}_center/{args.nnodes}').glob('sample_*.pkl'))
    sample_files = [str(x) for x in sample_files]
    samples = load_data(sample_files)

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
    for step in range(len(samples)):
        ####READ GRAPH AND FEATURE
        running_dir = f"trained_models/p_center/{args.nnodes}_{args.ncenter}/{step+1}/{args.model}"
        graph_datasets, datasets = samples
        graph_dataset = graph_datasets[step]
        dataset = datasets[step]
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

        g_0 = []
        g_1 = []
        g_2 = []
        s = 0
        for i in indices:
            if i == 0:
                g_0.append(s)
                s += 1
            elif i == 1:
                g_1.append(s)
                s += 1
            elif i == 2:
                g_2.append(s)
                s += 1
        g0 = g.subgraph(g_0)
        g1 = g.subgraph(g_1)
        g2 = g.subgraph(g_2)
        g0.ndata['x'] = g.ndata['z'][g0.parent_nid]
        g0.edata['weight'] = g.edata['weight'][g0.parent_eid]
        g0.edata['e'] = g.edata['e'][g0.parent_eid]
        g1.ndata['x'] = g.ndata['z'][g1.parent_nid]
        g1.edata['weight'] = g.edata['weight'][g1.parent_eid]
        g1.edata['e'] = g.edata['e'][g1.parent_eid]
        g2.ndata['x'] = g.ndata['z'][g2.parent_nid]
        g2.edata['weight'] = g.edata['weight'][g2.parent_eid]
        g2.edata['e'] = g.edata['e'][g2.parent_eid]

        center0, dist0 = cal_center(g0, distance=True)
        center1, dist1 = cal_center(g1, distance=True)
        center2, dist2 = cal_center(g2, distance=True)
        centers = [center0, center1, center2]
        all_dist = [dist0, dist1, dist2]
        dist = max(dist0, dist1, dist2)
        print(all_dist)
        print(centers)
        print(dist)