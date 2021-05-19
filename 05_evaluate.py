import sys
import importlib
import torch
import pathlib
import pickle
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from algorithm import k_center, k_center_approximation
def read_graph(filename):
    with open(filename, 'rb') as f:
        G, features = pickle.load(f)
    return G, features

def one_center(G, weighted=False, distance=False):
    '''
    Function computes distances between all vertices of graph and iterates through edge weights to compute vertices that minimize weights.
    Resulting vertices that fit are returned as a list.
    Weighted as false reduces computational time marginally.
    '''
    distances = G.distance_matrix(by_weight=weighted)
    sums = sum(distances.columns())
    vertices = G.vertices()
    centers = []
    min_dist = sums[0]
    # iterates through each
    for vertex in range(len(sums)):
        dists = sums[vertex]
        if dists < min_dist:
            min_dist = dists
            centers = [vertex]
        elif dists == min_dist:
            centers.append(vertex)
    if distance:
        return [vertices[c] for c in centers], min_dist
    return centers

if __name__ == '__main__':
    # Hyper parameter
    k = 2

    # load samples
    transfer_files = list(pathlib.Path(f"data/samples/p_center/transfer").glob('sample_*.pkl'))
    transfer_files = [str(x) for x in transfer_files]

    # load models
    sys.path.insert(0, os.path.abspath(f"models/GAT"))
    import model
    importlib.reload(model)
    del sys.path[0]
    net = model.GAT(in_dim=30,
                    hidden_dim=k * 2,
                    out_dim=2,
                    num_heads=5)
    running_dir = f"trained_models/p_center/GAT/0"
    net.load_state_dict(torch.load(f"trained_models/p_center/GAT/0/best_params.pkl"))

    for sample in transfer_files:
        g, feature = read_graph(sample)

        centers, dist = k_center(g, k=2, distance=True)
        p_exact = g.plot(layout='circular', vertex_colors={'red': centers}, vertex_labels=False, edge_labels=True)

        print("Exact solution")
        print('Maximum distance to any point:' + str(dist));
        print('\n')
        p_exact.show(figsize=15)

        centers, dist = k_center_approximation(g, k=3, distance=True, weighted=True)
        p_approximate = g.plot(layout='circular', vertex_colors={'red': centers}, vertex_labels=False, edge_labels=True)

        print("K-Center approximation")
        print('Maximum distance to any point:' + str(dist));
        print('\n')
        p_approximate.show(figsize=15)

        logits = net(g, feature)
        logp = F.log_softmax(logits, 1)
        _, indices = logp.max(dim=1, keepdim=True)
        g_1 = []
        g_2 = []
        for i in indices:
            if i == 0:
                g_1.append(g[i])
            else:
                g_2.append(g[i])
        center_1, dist_1 = one_center(g_1, weighted=True, distance=True)
        center_2, dist_2 = one_center(g_1, weighted=True, distance=True)
        centers = [center_1, center_2]
        dist = max(dist_1, dist_2)
        print("K-Center approximation with GAT")
        print('Maximum distance to any point:' + str(dist));
        print('\n')
        p_approximate.show(figsize=15)