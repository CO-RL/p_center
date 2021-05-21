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

def generate_samples(filepath, k, out_dir, dim, n_samples):
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
        print(f"{i} / {n_samples} samples written.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='center numbers',
        choices=['2_center', '6_center', '10_center'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=utilities.valid_seed,
        default=0,
    )
    parser.add_argument(
        '-j', '--njobs',
        help='Number of parallel jobs.',
        type=int,
        default=1,
    )
    args = parser.parse_args()
    print(f"seed {args.seed}")

    train_size = 1000
    valid_size = 300
    test_size = 300

    if args.problem == '2_center':
        k = 2
        instances_train = glob.glob('data/graph/p_center/train/*.pkl')
        instances_valid = glob.glob('data/graph/p_center/valid/*.pkl')
        instances_test = glob.glob('data/graph/p_center/test/*.pkl')
        out_dir = 'data/samples/2_center'
    elif args.problem == '6_center':
        k = 6
        instances_train = glob.glob('data/graph/p_center/train/*.pkl')
        instances_valid = glob.glob('data/graph/p_center/valid/*.pkl')
        instances_test = glob.glob('data/graph/p_center/test/*.pkl')
        out_dir = 'data/samples/6_center'
    if args.problem == '10_center':
        k = 10
        instances_train = glob.glob('data/graph/p_center/train/*.pkl')
        instances_valid = glob.glob('data/graph/p_center/valid/*.pkl')
        instances_test = glob.glob('data/graph/p_center/test/*.pkl')
        out_dir = 'data/samples/10_center'

    train_dir = out_dir + '/train'
    valid_dir = out_dir + '/valid'
    test_dir = out_dir + '/test'
    # transfer_dir = out_dir + '/transfer'
    print(f"{train_size} samples in {train_dir}")
    print(f"{valid_size} samples in {valid_dir}")
    print(f"{test_size} samples in {test_dir}")
    # print(f"{transfer_size} samples in {transfer_dir}")
    os.makedirs(train_dir)
    os.makedirs(valid_dir)
    os.makedirs(test_dir)
    # os.makedir(transfer_dir)

    generate_samples(filepath=instances_train, k=k, out_dir=train_dir, dim=30, n_samples=train_size)
    generate_samples(filepath=instances_valid, k=k, out_dir=valid_dir, dim=30, n_samples=valid_size)
    generate_samples(filepath=instances_valid, k=k, out_dir=test_dir, dim=30, n_samples=test_size)
