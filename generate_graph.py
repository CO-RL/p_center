from sage.all import *
import numpy as np
import torch
import networkx as nx
import matplotlib
from algorithm import k_center
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    n = 3
    # G = []
    #生成图片集合
    G = [random_connected_graph(50, 0.4, weighted=True) for _ in range(1)]

    #画图 计算距离
    # g = G[0]
    # g.weighted(True)
    # centers, dist = k_center(g, k=3, distance=True)
    # p_exact = g.plot(layout='circular', vertex_colors={'red': centers}, vertex_labels=False, edge_labels=True)
    # print("Exact solution")
    # print('Maximum distance to any point:' + str(dist));
    # print('\n')
    # p_exact.show(figsize=15)

    features = [extract_features(graph, dim=30, weighted=True) for graph in G]
    print(G[0])
