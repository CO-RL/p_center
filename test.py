import pathlib
import model
import pickle
import os
import argparse
import time
from sage.all import *
import numpy as np
import matplotlib.pyplot as plt
import torch

def convert_to_complete(G, weighted=False):
    '''
    Function that converts graph to complete graph. Edge weights are distances between the vertices.
    Weighted as false reduced computational time.
    '''
    all_edges = []
    vertices = G.vertices()
    # distance is symmetric hence iterating one direction is enough
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            all_edges.append((vertices[i], vertices[j], G.distance(vertices[i], vertices[j], by_weight=weighted)))
    complete = Graph(all_edges)
    return complete

def read_graph(filename):
    with open(filename, 'rb') as f:
        G = pickle.load(f)
    return G
def k_center(G, k=3, distance=False):
    '''
    Function takes argument graph and modifies into complete graph before implementing efficient k-center algorithm
    Detailed description of algorithm can be found above

    '''
    # sorted edge list by edge weight
    complete = convert_to_complete(G, weighted=True)
    weights = sorted(set([edge[2] for edge in complete.edges()]))
    high = len(weights) - 1
    low = 0
    while high - low > 1:
        mid = int(math.ceil((high + low) / 2))
        r_max = weights[mid]
        bottleneck_graph = complete.copy()
        # removes all edges from G that have a weight larger than the maximum permitted radius
        edges_to_remove = [edge for edge in complete.edges() if edge[2] > r_max]
        bottleneck_graph.delete_edges(edges_to_remove)
        centers = bottleneck_graph.dominating_set()
        # binary search within weights
        if len(centers) <= k:
            high = mid
        else:
            low = mid
    if len(centers) > k:
        mid += 1
        r_max = weights[mid]
        bottleneck_graph = complete.copy()
        edges_to_remove = [edge for edge in complete.edges() if edge[2] > r_max]
        bottleneck_graph.delete_edges(edges_to_remove)
        centers = bottleneck_graph.dominating_set()
    if distance:
        return centers, r_max
    else:
        return centers

def k_center_approximation(G, k=3, weighted=False, seed=None, distance=False):
    ''' This function uses greedy approximation to find centers.
        Returns the resulting k-centers as specified as well as the maximum distance if required.
    '''
    G.weighted(True)
    vertices = G.vertices()
    if seed != None:
        np.random.seed(seed)
    starting_index = np.random.randint(len(vertices))
    C = [starting_index]
    deltas = []
    while len(C) < k:
        maximized_distance = 0
        for v in range(len(vertices)):
            if v in C:
                continue
            dists = [float(G.distance(vertices[v], vertices[c], by_weight=weighted)) for c in C]
            min_dist = min(dists)
            if min_dist > maximized_distance:
                maximized_distance = min_dist
                best_candidate = v
        deltas.append(maximized_distance)
        C.append(best_candidate)
    if distance:
        return [vertices[c] for c in C], min(deltas)
    else:
        return [vertices[c] for c in C]

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
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '-problem',
    #     help='the type of p_center.',
    #     choices = ['2center'],
    #     default='2center',
    # )
    # args = parser.parse_args()
    #
    # print(f"problem: {args.problem}")
    # print(f"gpu: {args.gpu}")
    #
    # os.makedirs("results", exist_ok=True)
    #
    # seeds = [0]
    # models = ['GAT']
    #
    # problem_folders = {
    #     '2center': 'p_center'
    # }
    # problem_folder = problem_folders[args.problem]
    #
    # result_file = f"results/{args.problem}_test_{time.strftime('%Y%m%d-%H%M%S')}"
    #
    # result_file = result_file + '.csv'
    # os.makedirs('results', exist_ok=True)
    #
    # test_files = list(pathlib.Path(f'data/samples/p_center/test').glob('sample_*.pkl'))
    # test_files = [str(x) for x in test_files]
    #
    # for data in test_files:

    net = model.GAT(in_dim=30,
                    hidden_dim=k * 2,
                    out_dim=2,
                    num_heads=5)
    running_dir = f"trained_models/p_center/GAT/0"
    net.load_state_dict(torch.load(os.path.join(running_dir, 'best_params.pkl')))


    g = random_connected_graph(50, 0.4, weighted=True)
    g_features = extract_features(g, dim=30, weighted=True)
    result_g = net(g_features).detach().numpy()

    g.weighted(True)
    centers, dist = k_center(g, k=2, distance=True)
    p_exact = g.plot(layout='circular', vertex_colors={'red': centers}, vertex_labels=False, edge_labels=True)

    print("Exact solution")
    print('Maximum distance to any point:' + str(dist));
    print('\n')
    p_exact.show(figsize=15)
    plt.show()
    centers, dist = k_center_approximation(g, k=3, distance=True, weighted=True)
    p_approximate = g.plot(layout='circular', vertex_colors={'red': centers}, vertex_labels=False, edge_labels=True)

    print("K-Center approximation")
    print('Maximum distance to any point:' + str(dist));
    print('\n')
    p_approximate.show(figsize=15)
    # print(f"{len(test_files)} test samples")
    #
    # ### MODEL LOADING ###
    # # sys.path.insert(0, os.path.abspath(f'models/{args.model}'))

    # running_dir = f"trained_models/{args.problem}/{args.model}/{args.seed}"
    # net.load_state_dict(torch.load(os.path.join(running_dir, 'best_params.pkl')))
    #