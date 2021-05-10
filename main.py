import torch
import dgl
from generate_graph import random_connected_graph, extract_features
from algorithm import k_center_approximation, clusters
import time
import torch.nn.functional as F
import numpy as np
import argparse
import model
import pickle

import matplotlib.pyplot as plt

def convert_to_DGLGraph(G, k=3, weighted=True):
    if weighted:
        nxgraph = G.networkx_graph(weight_function = lambda edge: float(edge[2]))
    else:
        nxgraph = G.networkx_graph(weight_function = lambda edge: float(1))
    dgl_G = dgl.DGLGraph()
    dgl_G.from_networkx(nxgraph, edge_attrs=['weight'])
    centers, distance = k_center_approximation(G, k=k, distance=True)
    centers = [int(c) for c in centers]
    labels = torch.LongTensor([int(1) if node in centers else int(0) for node in G.vertices()])
    mask = torch.BoolTensor([True for _ in range(len(labels))])
    clusts = clusters(G, centers, weighted=weighted)
    return dgl_G, centers, labels, mask, clusts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     'problem',
    #     help='the type of p_center.',
    #     default ='2center',
    #     choices = ['2ceter']
    # )
    # parser.add_argument(
    #     '-m', '--model',
    #     help = 'GNN model to be trained.',
    #     type=str,
    #     default='GAT',
    #     choices=['GAT', 'GCN']
    # )
    # parser.add_argument(
    #     '-s', '--seed',
    #     help='Random generator seed.',
    #     default=0
    # )

    args = parser.parse_args()
    args.problem = '2center'
    args.model = 'GAT'
    # args.model = 'GraphConv'

    ### HYPER PARAMETERS ###
    max_epochs = 1000
    lr = 0.001
    patience = 10
    early_stopping = 20

    # problem_folder = problem_folder[args.problem]
    # running_dir = f"trained_models/{args.problem}/{args.model}"
    # os.makedirs(running_dir)
    #
    # logfile = os.path.join(running_dir, 'log.txt')
## Generate graph data and features
    if args.problem == '2center':
        weighted = True
        G = [random_connected_graph(30, 0.4, weighted=True) for _ in range(3)]
        features = [extract_features(graph, dim=30, weighted=True) for graph in G]

        g = []
        centers = []
        labels = []
        masks = []
        clusts = []

        k = 2
        for graph in G:
            dgl_graph, c, l, m, clsts = convert_to_DGLGraph(graph, k=k, weighted=weighted)
            g.append(dgl_graph)
            centers.append(c)
            labels.append(l)
            masks.append(m)
clusts.append(torch.LongTensor(clsts))

    if args.model == 'GAT':
        net = model.GAT(in_dim=features[0].size()[1],
                        hidden_dim=k * 2,
                        out_dim=2,
                        num_heads=5)
    elif args.model == 'GCN':
        net = model.GCN(in_dim=features[0].size()[1],
                        hidden_dim=k * 2,
                        out_dim=2)
    elif args.model == 'GraphConv':
        net= model.GraphSAGE(in_feats=features[0].size()[1],
                             n_hidden=k*2,
                             n_classes=2,
                             n_layers=1,
                             activation=F.relu,
                             dropout=True,
                             aggregator_type='mean')
    else:
        raise NotImplementedError
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # best_loss = np.inf
    # for epoch in range(max_epochs + 1):

    duration = []
    losses = []
    count = 0
    for k in range(1):
        for i in range(len(g)):
            for epoch in range(25):
                net.train()
                count += 1
                if epoch >= 3:
                    t0 = time.time()
                logits = net(g[i], features[i])
                logp = F.log_softmax(logits, 1)
                loss = F.nll_loss(logp, clusts[i])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if epoch >= 3:
                    duration.append(time.time() - t0)
                    losses.append(loss.item())

                print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
                    epoch*i, loss.item(), np.mean(duration)))

    plt.plot([*range(len(losses))], losses)
    plt.title("Clustering-Based GNN Loss for k=2")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    # out_png1 = './img/loss.png'
    # plt.savefig(out_png1, dpi=150)


    # test_graph = random_connected_graph(50, 0.4, weighted=True)
    # test_features = extract_features(test_graph, dim=30, weighted=True)
    # test_centers = k_center_approximation(test_graph, k=2, distance=False)
    # test_clusters = clusters(test_graph, test_centers, weighted=True, as_dict=True)
    # results = net(test_features).detach().numpy()
    # plt.scatter(results[:, 0], results[:, 1])
    # plt.scatter(results[test_centers, 0], results[test_centers, 1], c='red')
    # plt.title("Visualization of Clustering-GNN output for k=2")
    # out_png2 = './img/graph.png'
    # plt.savefig(out_png2, dpi=150)
