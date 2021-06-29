import numpy as np
import torch
from algorithm import convert_to_DGLGraph
import pickle
import math
import os
import glob
from dgl.data.utils import save_graphs
import utilities
import argparse

def remove_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            remove_file(path_file)

def check_path_exist(path):
    if os.path.exists(path):
        remove_file(path)
    else:
        os.makedirs(path)


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

def read_graph(filename):
    with open(filename, 'rb') as f:
        G = pickle.load(f)
    return G

def generate_samples(filepath, k, out_dir, dim):
    print(f"generating samples from {out_dir}")
    i = 0
    for file in filepath:
        G = read_graph(file)
        feature = extract_features(G, dim = dim, weighted=True)
        g, c, l, m, clsts = convert_to_DGLGraph(G, k=k, weighted=True)

        save_graphs("{}/sample_{}.bin".format(out_dir, i+1), g)
        data = (feature, c, l, m, clsts)
        with open(f'{out_dir}/sample_{i+1}.pkl', 'wb') as f:
            pickle.dump(data, f)
        # data = [g, feature, c, l, m, clsts]
        # with open(f'{out_dir}/sample_{i+1}.pkl', 'wb') as f:
        #     pickle.dump(data, f)
        # f.close()
        i +=1
        print(f"{i} / {len(filepath)} samples written.")

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

    dim = 30  #feature dimision

    instances_graph = glob.glob(f"data/graph/{args.ncenter}_center/{args.nnodes}/*.pkl")
    out_dir = f"data/samples/{args.ncenter}_center/{args.nnodes}/"
    check_path_exist(out_dir)
    generate_samples(filepath=instances_graph, k=args.ncenter, out_dir=out_dir, dim=dim)
