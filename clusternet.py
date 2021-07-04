import argparse
from models import GCNClusterNet
import numpy as np
import torch
import os
import networkx as nx
from dgl.data.utils import load_graphs
import pickle
import pathlib

class CenterObjective():
    def __init__(self, dist, dmax, temp, hardmax=False):
        '''
        dist: (num customers) * (num locations) matrix

        dmax: maximum distance that can be suffered by any customer (e.g., if 
              no facilities are chosen)

        temp: how hard to make the softmax over customers
        '''
        self.dmax = dmax
        dist, order = torch.sort(dist, dim=1)
        self.order = order
        dmax_vec = dmax * torch.ones(dist.shape[0], 1)
        off_one = torch.cat((dist[:, 1:], dmax_vec), dim=1)
        self.m = dist - off_one
        self.temp = temp
        self.hardmax = hardmax

    def __call__(self, x):
        '''
        Evaluates E_S[softmax_{customers} min_{i \in S} dist(customer, i)] where 
        the expectation is over the set of facility locations S. Every 
        location is included in S independently with probability x_i. 
        '''
        x_sort = x[self.order]
        probs = 1 - torch.cumprod(1 - x_sort, dim=1)
        vals = self.dmax + (self.m * probs).sum(dim=1)
        if self.hardmax:
            return vals.max()
        weights = torch.softmax(self.temp * vals, dim=0)
        return torch.dot(vals, weights)

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

def read_graph(filename):
    graph_path = "/".join(filename.split(".")[0:-1]) + ".bin"
    feature_path = "/".join(filename.split(".")[0:-1]) + ".pkl"
    graph = load_graphs(graph_path)
    data = pickle.load(open(feature_path, 'rb'))
    # with open(filename, 'rb') as f:
    #     data = pickle.load(f)
    return (graph, data)

def load_data(dataloader):
    graph_datasets = []
    datasets = []
    for data in dataloader:
        dataset = read_graph(data)
        graph, data = dataset
        graph_datasets.append(graph)
        datasets.append(data)

    return (graph_datasets, datasets)

def loss_kcenter(mu, r, embeds, dist, bin_adj, obj, args):
    if obj == None:
        return torch.tensor(0).float()
    x = torch.softmax(dist*args.kcentertemp, 0).sum(dim=1)
    x = 2*(torch.sigmoid(4*x) - 0.5)
    if x.sum() > args.K:
        x = args.K*x/x.sum()
    loss = obj(x)
    return loss

def make_all_dists(bin_adj, dmax, use_weights=False):
    g = nx.from_numpy_array(bin_adj.detach().numpy())
    if not use_weights:
        lengths = nx.shortest_path_length(g)
    else:
        lengths = nx.shortest_path_length(g, weight='weight')
    dist = torch.zeros_like(bin_adj)
    for u, lens_u in lengths:
        for v in range(bin_adj.shape[0]):
            if v in lens_u:
                dist[u,v] = lens_u[v]
            else:
                dist[u,v] = dmax
    return dist

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

    max_epochs = 10000
    batch_size = 1
    lr = 0.01
    patience = 100
    early_stopping = 1000
    best_loss = np.inf
    k = args.ncenter
    num_cluster_iter = 1
    ### SET-UP DATASET ###
    sample_files = f'data/samples/{args.ncenter}_center/{arg.nnodes}/sample_1.pkl'
    sample_num = 1
    samples = read_graph(sample_files)

    ### MODEL LOADING ###
    model_cluster = GCNClusterNet(nfeat=30,
                nhid=50,
                nout=50,
                dropout=args.dropout,
                K = k,
                cluster_temp = 30)

    optimizer = torch.optim.Adam(model_cluster.parameters(), lr=lr)

    losses = []

    running_dir = f"trained_models/p_center/{args.nnodes}_{args.ncenter}/{sample_num}/clusternet"
    check_path_exist(running_dir)
    logfile = os.path.join(running_dir, 'log.txt')

    graph_dataset, dataset = samples
    g, data_ = graph_dataset[0][0], dataset
    feature, c, l, m, clsts = data_
    G = g.to_networkx(node_attrs=['x'], edge_attrs=['weight'])
    adj = nx.adjacency_matrix(G, weight='weight')
    adj = adj.coalesce()
    bin_adj = (adj.to_dense() > 0).float()

    dist_all = make_all_dists(bin_adj, 100, use_weights=True)
    diameter = dist_all[dist_all < 100].max()
    dist_all[dist_all == 100] = diameter

    object = CenterObjective(dist_all, diameter, 0)

    from torch.autograd import Variable
    feature = Variable(feature)


    for epoch in range(max_epochs + 1):

        mu, r, embeds, dist = model_cluster(feature, adj, num_cluster_iter)
        loss = loss_kcenter(mu, r, embeds, dist, bin_adj, object, args)
        loss = -loss
        optimizer.zero_grad()
        loss.backward()

        if step == 500:
            num_cluster_iter = 5
        if step % 100 == 0:
            # round solution to discrete partitioning
            # evalaute test loss -- note that the best solution is
            # chosen with respect training loss. Here, we store the test loss
            # of the currently best training solution
            loss_test = loss_kcenter(mu, r, embeds, dist, bin_adj, object, args)
            # for k-center problem, keep track of the fractional x with best
            # training loss, to do rounding after
            if loss.item() < best_train_val:
                best_train_val = loss.item()
                curr_test_loss = loss_test.item()
                # convert distances into a feasible (fractional x)
                x_best = torch.softmax(dist * args.kcentertemp, 0).sum(dim=1)
                x_best = 2 * (torch.sigmoid(4 * x_best) - 0.5)
                if x_best.sum() > k:
                    x_best = k * x_best / x_best.sum()
        losses.append(loss.item())
        optimizer.step()

    print('ClusterNet value', x_best)