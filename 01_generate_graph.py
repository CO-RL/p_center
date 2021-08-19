from sage.all import *
import numpy as np
import torch
import networkx as nx
import matplotlib
from algorithm import k_center
import matplotlib.pyplot as plt
import pickle
import os
import argparse

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

def generate_graph(datasize, n ,p, out_dir):
    print(f"generating {datasize} graph in {out_dir}")
    for i in range (datasize):
        G = random_connected_graph(n, p, weighted=True)
        with open(f'{out_dir}/sample_{i+1}.pkl', 'wb') as f:
            pickle.dump(G, f)
        f.close()

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
        choices=['3', '5', '10'],
        default=5,
    )
    args = parser.parse_args()

    ngraph = 100
    out_dir = f"data/graph/{args.ncenter}_center/{args.nnodes}"
    os.makedirs(out_dir)

    generate_graph(datasize=ngraph, n=args.nnodes, p=0.4, out_dir=out_dir)

