import os
import pickle
from algorithm import k_center, k_center_approximation
from dgl.data.utils import load_graphs
import argparse
import time
import sys
import os
import importlib
import torch
import torch.nn.functional as F
import numpy as np

def read_graph(filename):
    with open(filename, 'rb') as f:
        G = pickle.load(f)
    return G

def read_dgl_graph(filename):
    graph_path = "/".join(filename.split(".")[0:-1]) + ".bin"
    feature_path = "/".join(filename.split(".")[0:-1]) + ".pkl"
    graph = load_graphs(graph_path)
    data = pickle.load(open(feature_path, 'rb'))
    # with open(filename, 'rb') as f:
    #     data = pickle.load(f)
    return (graph, data)

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
    args = parser.parse_args()

    n = args.nnodes
    k=args.ncenter
###########################
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
##########################################################3
    instance_path = os.listdir(f'data/graph/{args.ncenter}_center/{args.nnodes}')
    instance_path = sorted(instance_path, key=lambda x: int(x.split('_')[1].split('.')[0]))
    file = instance_path[0]

    sample_num = 1
    sample_files = f'data/samples/{args.ncenter}_center/{args.nnodes}/sample_{sample_num}.pkl'
    samples = read_dgl_graph(sample_files)
    graph_dataset, dataset = samples
    file_path = f'data/graph/{args.ncenter}_center/{args.nnodes}/{file}'

    G = read_graph(file_path)
    dgl_g = graph_dataset[0][0]

################################## GCN

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

    Gs = []
    for i in range(k):
        Gs.append(G.subgraph(v[i]))
    center = []
    for j in range(k):
        c = Gs[j].center(by_weight=True)
        center.append(c[0])

    dist = []
    for i in range(k):
        dists = []
        for j in v[i]:
            dists.append(G.distance(center[i], j, by_weight=True))
        dist.append(np.max(dists))
    minmax_dist = np.max(dist)
    print("GCN solution")
    print('Centers:', center)
    print('Maximum distance to any point:' + str(minmax_dist));
    print('\n')
#########################################################################################

################### Greedy algorithm
    t1 = time.time()
    centers, dist = k_center_approximation(G, k=k, distance=True, weighted=True)
    ta = time.time() - t1
    p_approximate = G.plot(layout='circular', vertex_colors={'red': centers}, vertex_labels=False, edge_labels=True)

    print("K-Center approximation")
    print('Centers:', centers)
    print('Maximum distance to any point:' + str(dist));
    print('\n')
    print('runtime: %d', ta)
    p_approximate.show(figsize=15)
#################### Exact algorithm
    t0 = time.time()
    centers, dist = k_center(G, k=k, distance=True)
    te = time.time() - t0
    p_exact = G.plot(layout='circular', vertex_colors={'red': centers}, vertex_labels=False, edge_labels=True)

    print("Exact solution")
    print('Centers:', centers)
    print('Maximum distance to any point:' + str(dist));
    print('\n')
    print('runtime: %d', te)
    p_exact.show(figsize=15)
