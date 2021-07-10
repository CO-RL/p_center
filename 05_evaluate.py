import os
import pickle
from algorithm import k_center, k_center_approximation
from dgl.data.utils import load_graphs
import argparse

import matplotlib.pyplot as plt
import networkx as nx
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
        choices=['50', '100', '200', '500', '1000'],
        default=50,
    )
    parser.add_argument(
        '-nc', '--ncenter',
        help='center numbers',
        choices=['3', '6', '10'],
        default=3,
    )
    args = parser.parse_args()

    instance_path = os.listdir(f'data/graph/{args.ncenter}_center/{args.nnodes}')
    instance_path = sorted(instance_path, key=lambda x: int(x.split('_')[1].split('.')[0]))
    file = instance_path[0]

    sample_num = 1
    sample_files = f'data/samples/{args.ncenter}_center/{args.nnodes}/sample_{sample_num}.pkl'
    samples = read_dgl_graph(sample_files)
    graph_dataset, dataset = samples
    file_path = f'data/graph/{args.ncenter}_center/{args.nnodes}/{file}'

    g = read_graph(file_path)
    dgl_g = graph_dataset[0][0]
    centers, dist = k_center(g, k=3, distance=True)
    p_exact = g.plot(layout='circular', vertex_colors={'red': centers}, vertex_labels=False, edge_labels=True)
    G = dgl_g.to_networkx(node_attrs=['x'], edge_attrs=['weight'])
    a = nx.draw_networkx(G)
    plt.show()
    a.savefig('a.png')
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
