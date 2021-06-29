import torch
import torch.nn.functional as F
import utilities
from utilities import log
import os
import numpy as np
import pathlib
import sys
import importlib
import pickle
import matplotlib.pyplot as plt
from dgl.data.utils import load_graphs
import argparse
import time

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

# 返回一个0,1数组中1的数量
def num_one(source_array):
    count = 0
    for x in source_array:
        if x == 1:
            count += 1
    return count

def load_data(dataloader):
    graph_datasets = []
    datasets = []
    for data in dataloader:
        dataset = read_graph(data)
        graph, data = dataset
        graph_datasets.append(graph)
        datasets.append(data)

    return (graph_datasets, datasets)

def plot_loss(loss):
    plt.plot([*range(len(loss))], loss)
    plt.title("Clustering-Based GNN Loss for n=50, k=3 ")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

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
    parser.add_argument(
        '-m', '--model',
        help='GNN model to be trained.',
        type=str,
        default='GAT',
        choices=['GCN', 'GAT', 'GraphSAGE']
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=utilities.valid_seed,
        default=0,
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    args = parser.parse_args()

    ### HYPER PARAMETERS ###
    max_epochs = 10000
    batch_size = 1
    lr = 0.01
    patience = 100
    early_stopping = 1000
    best_loss = np.inf
    k = args.ncenter

    # ### LOG ###
    # logfile = os.path.join(running_dir, 'log.txt')
    # log(f"max_epochs: {max_epochs}", logfile)
    # log(f"batch_size: {batch_size}", logfile)
    # log(f"lr: {lr}", logfile)
    # log(f"patience : {patience }", logfile)
    # log(f"early_stopping : {early_stopping }", logfile)

    ### SET-UP DATASET ###
    sample_files = list(pathlib.Path(f'data/samples/{args.ncenter}_center/{args.nnodes}').glob('sample_*.pkl'))

    sample_files = [str(x) for x in sample_files]

    samples = load_data(sample_files)

    ### MODEL LOADING ###
    sys.path.insert(0, os.path.abspath(f'models/{args.model}'))
    import model

    importlib.reload(model)
    if args.model == 'GAT':
        net = model.GAT(in_dim=30,
                        hidden_dim=k*5,
                        out_dim=k,
                        num_heads=5)
    elif args.model == 'GCN':
        net = model.GCN(in_dim=30,
                        hidden_dim=k*5,
                        out_dim=k)
    elif args.model == 'GraphSAGE':
        net = model.GraphSAGE(in_feats=30,
                              n_hidden=k*5,
                              n_classes=k,
                              n_layers=1,
                              activation=F.relu,
                              dropout=True,
                              aggregator_type='mean')
    else:
        raise NotImplementedError

    del sys.path[0]

    ### TRAINING LOOP ###
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)


    for step in range(len(samples[0])):
        best_loss = np.inf
        train_losses = []
        train_accs = []

        running_dir = f"trained_models/p_center/{args.nnodes}_{args.ncenter}/{step+1}/{args.model}"
        check_path_exist(running_dir)
        logfile = os.path.join(running_dir, 'log.txt')

        graph_datasets, datasets = samples
        graph_dataset = graph_datasets[step]
        dataset = datasets[step]
        duration = []
        losses = []
        accs = []

        g, data_ = graph_dataset[0][0], dataset
        feature, c, l, m, clsts = data_
        from torch.autograd import Variable
        feature = Variable(feature)
        clsts = Variable(torch.LongTensor(clsts))

        for epoch in range(max_epochs + 1):


            if epoch >= 3:
                t0 = time.time()

            logits = net(g, feature)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp, clsts)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, indices = torch.max(logp, dim=1)
            correct = torch.sum(indices == clsts)
            acc = correct.item() * 1.0 / len(clsts)

            if epoch >= 3:
                duration.append(time.time() - t0)
                losses.append(loss.item())
                accs.append(acc)

            print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | Acc(%) {:.4f}".format(
                epoch, loss.item(), np.mean(duration), np.mean(acc)))

            if loss.item() < best_loss:
                plateau_count = 0
                best_loss = loss.item()
                torch.save(net.state_dict(), os.path.join(running_dir, 'best_params.pkl'))
                log(f"  best model so far", logfile)
            else:
                plateau_count += 1
                if plateau_count % early_stopping == 0:
                    log(f"  {plateau_count} epochs without improvement, early stopping", logfile)
                    break
                if plateau_count % patience == 0:
                    lr *= 0.2
                    log(f"  {plateau_count} epochs without improvement, decreasing learning rate to {lr}", logfile)
        plot_loss(losses)
        net.load_state_dict(torch.load(os.path.join(running_dir, 'best_params.pkl')))
        logits = net(g, feature)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp, clsts)

        _, indices = torch.max(logp, dim=1)
        correct = torch.sum(indices == clsts)
        acc = correct.item() * 1.0 / len(clsts)
        log(f"Best loss: {loss:0.3f}" + "".join(f" Acc: {acc:0.3f}"), logfile)