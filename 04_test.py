import pathlib
import pickle
import importlib
import torch
import torch.nn.functional as F
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-nn', '--nnodes',
        help='nodes numbers',
        choices=['100', '200', '500', '1000'],
        default=100,
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
        default='GCN',
        choices=['GCN', 'GAT', 'GraphSAGE']
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    args = parser.parse_args()
    n = args.nnodes
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
    sys.path.insert(0, os.path.abspath(f'model/{args.model}'))
    import model
    importlib.reload(model)
    if args.model == 'GAT':
        net = model.GAT(in_dim=2,
                        hidden_dim=k*5,
                        out_dim=k,
                        num_heads=5)
    elif args.model == 'GCN':
        net = model.GCN(in_dim=2,
                        hidden_dim=k*5,
                        out_dim=k)
    elif args.model == 'GraphSAGE':
        net = model.GraphSAGE(in_feats=2,
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

    _, indices = logp.max(dim=1, keepdim=True)  # 输出分类的结果
    indices = indices.reshape(n)
    indices = indices.detach().numpy()
    indices = indices.tolist()

    def get_same_element_index(ob_list, word):
        return [i for (i, v) in enumerate(ob_list) if v == word]

    v = []
    for i in range(k):
        a = get_same_element_index(indices, i)
        v.append(a)
    G = []
    for i in range(k):
        G.append(g.subgraph(v[i]))
    center = []
    for j in range(k):
        c = G[j].center(by_weight=True)
        center.append(c[0])
    dist = []
    for i in range(k):
        dists = []
        for j in v[i]:
            dists.append(g.distance(center[i], j, by_weight=True))
        dist.append(np.max(dists))
    minmax_dist = np.max(dist)
    print("GCN solution")
    print('Centers:', center)
    print('Maximum distance to any point:' + str(minmax_dist));
    print('\n')