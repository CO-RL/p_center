from sage.all import *
import numpy as np
import torch
import networkx as nx
import matplotlib
from algorithm import k_center
import matplotlib.pyplot as plt
import pickle
import os
def random_connected_graph(n, p, seed=None, weighted=True):
    '''
        n - number of vertices
        p - probability there is an edge between two vertices
        uses uniform distribution for edge labeling
    '''
    G = graphs.RandomGNP(n, p, seed=seed)  # ensures that G is completely connected

    sd = seed
    while len(G.connected_components()) > 1:
        if sd != None:
            sd += 1
        G = graphs.RandomGNP(n, p, seed=sd)
    np.random.seed(seed)
    if weighted:
        for edge in G.edges():
            G.set_edge_label(edge[0], edge[1], RR(np.random.random_sample()))
    return G

def extract_features(G, dim=10, weighted=True):
    features = np.zeros((G.order(), dim))
    resolution = float(G.diameter(by_weight=weighted)) / dim
    vertices = G.vertices()
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            distance = G.distance(vertices[i], vertices[j], by_weight=weighted)
            section = min(int(math.floor(distance/resolution)), dim - 1)
            features[i, section] += 1
            features[j, section] += 1
    return torch.FloatTensor(features)

def generate_samples(datasize, n ,p, out_dir, dim):
    print('generating samples...')
    for i in range (datasize):
        G = random_connected_graph(n, p, weighted=True)
        features = extract_features(G, dim = dim, weighted=True)
        with open(f'{out_dir}/sample_{i+1}.pkl', 'wb') as f:
            pickle.dump(G, f)
            pickle.dump(features, f)
        f.close()

if __name__ == '__main__':
    train_size = 30
    valid_size = 10
    test_size = 10
    transfer_size = 5
    #
    out_dir = 'data/samples/p_center'
    train_dir = out_dir + '/train'
    valid_dir = out_dir + '/valid'
    test_dir = out_dir + '/test'
    transfer_dir = out_dir + '/transfer'
    print(f"{train_size} samples in {train_dir}")
    print(f"{valid_size} samples in {valid_dir}")
    print(f"{test_size} samples in {test_dir}")
    print(f"{transfer_size} samples in {transfer_dir}")
    os.makedirs(train_dir)
    os.makedirs(valid_dir)
    os.makedirs(test_dir)
    os.makedirs(transfer_dir)

    generate_samples(datasize=train_size, n=50, p=0.4, out_dir=train_dir, dim=10)

    generate_samples(datasize=valid_size, n=50, p=0.4, out_dir=valid_dir, dim=10)

    generate_samples(datasize=test_size, n=50, p=0.4, out_dir=test_dir, dim=10)

    generate_samples(datasize=transfer_size, n=50, p=0.4, out_dir=transfer_dir, dim=10)

    # G = [random_connected_graph(50, 0.4, weighted=True) for _ in range(1)]
    #
    #
    # with open("./data/sample.pkl", 'wb') as f:
    #     pickle.dump(G, f)
    # #
    # with open('./data/sample.pkl', 'rb') as f:
    #     K = pickle.load(f)
    # print(K)
    #画图 计算距离
    # g = G[0]
    # g.weighted(True)
    # centers, dist = k_center(g, k=3, distance=True)
    # p_exact = g.plot(layout='circular', vertex_colors={'red': centers}, vertex_labels=False, edge_labels=True)
    # print("Exact solution")
    # print('Maximum distance to any point:' + str(dist));
    # print('\n')
    # p_exact.show(figsize=15)

    # features = [extract_features(graph, dim=30, weighted=True) for graph in G]
    # print(G[0])
