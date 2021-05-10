import torch
import dgl
from generate_graph import random_connected_graph, extract_features
from algorithm import k_center_approximation, clusters
import time
import torch.nn.functional as F
import numpy as np
import argparse
import model
import matplotlib.pyplot as plt
import pickle
import os
import pathlib
import datetime
import sys
def log(str, logfile=None):
    str = f'[{datetime.datetime.now()}] {str}'
    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(str, file=f)

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

def read_graph(filename):
    with open(filename, 'rb') as f:
        G = pickle.load(f)
    return G
def process(model, dataloader, optimizer=None):
    mean_loss = 0
    mean_acc = np.zeros(1)
    n_samples_processed = len(dataloader)
    for batch in dataloader:
        clusts = []
        G = read_graph(batch)
        feature = extract_features(G, dim=30, weighted=True)
        g, c, l, m, clsts = convert_to_DGLGraph(G, k=k, weighted=True)
        clusts.append(torch.LongTensor(clsts))

        if optimizer:
            logits = model(g, feature)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp, clusts[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            logits = model(g, feature)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp, clusts[0])

        mean_loss += loss

        # mean_acc += acc
        # g.append(dgl_graph)
        # centers.append(c)
        # labels.append(l)
        # masks.append(m)
        # clusts.append(torch.LongTensor(clsts))
    mean_loss /= n_samples_processed
    return mean_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-problem',
        help='the type of p_center.',
        choices = ['2center'],
        default='2center',
    )
    parser.add_argument(
        '-m', '--model',
        help = 'GNN model to be trained.',
        type=str,
        default='GAT',
        choices=['GAT', 'GCN']
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        default=0
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    args = parser.parse_args()

    args.problem = '2center'
    args.model = 'GAT'
    # args.model = 'GraphConv'
    # if args.problem == '2center':
    #     weighted = True
    #     with open('./data/sample.pkl', 'rb') as f:
    #         G = pickle.load(f)
    #     # features = [extract_features(graph, dim=30, weighted=True) for graph in G]
    #     #
    #     # g = []
    #     # centers = []
    #     # labels = []
    #     # masks = []
    #     # clusts = []
    #     #
    #     # k = 2
    #     # for graph in G:
    #     #     dgl_graph, c, l, m, clsts = convert_to_DGLGraph(graph, k=k, weighted=weighted)
    #     #     g.append(dgl_graph)
    #     #     centers.append(c)
    #     #     labels.append(l)
    #     #     masks.append(m)
    #     #     clusts.append(torch.LongTensor(clsts))
    # if args.model == 'GAT':
    #     net = model.GAT(in_dim=features[0].size()[1],
    #                     hidden_dim=k * 2,
    #                     out_dim=2,
    #                     num_heads=5)
    # elif args.model == 'GCN':
    #     net = model.GCN(in_dim=features[0].size()[1],
    #                     hidden_dim=k * 2,
    #                     out_dim=2)
    # elif args.model == 'GraphConv':
    #     net= model.GraphSAGE(in_feats=features[0].size()[1],
    #                          n_hidden=k*2,
    #                          n_classes=2,
    #                          n_layers=1,
    #                          activation=F.relu,
    #                          dropout=True,
    #                          aggregator_type='mean')
    # else:
    #     raise NotImplementedError

    ### HYPER PARAMETERS ###
    max_epochs = 1000
    lr = 0.01
    patience = 10
    early_stopping = 20
    best_loss = np.inf
    k=2
    # logfile = os.path.join(running_dir, 'log.txt')
    # problem_folder = problem_folder[args.problem]
    # running_dir = f"trained_models/{args.problem}/{args.model}"
    # os.makedirs(running_dir)
    #
    # logfile = os.path.join(running_dir, 'log.txt')
    problem_folders = {
        '2center': 'p_center'
    }
    problem_folder = problem_folders[args.problem]
    running_dir = f"trained_models/{args.problem}/{args.model}/{args.seed}"
    os.makedirs(running_dir)

    ### LOG ###
    logfile = os.path.join(running_dir, 'log.txt')

    log(f"max_epochs: {max_epochs}", logfile)
    # log(f"epoch_size: {epoch_size}", logfile)
    # log(f"batch_size: {batch_size}", logfile)
    # log(f"pretrain_batch_size: {pretrain_batch_size}", logfile)
    # log(f"valid_batch_size : {valid_batch_size }", logfile)
    log(f"lr: {lr}", logfile)
    log(f"patience : {patience }", logfile)
    log(f"early_stopping : {early_stopping }", logfile)
    log(f"problem: {args.problem}", logfile)
    log(f"gpu: {args.gpu}", logfile)
    log(f"seed {args.seed}", logfile)

    ### SET-UP DATASET ###
    train_files = list(pathlib.Path(f'data/samples/{problem_folder}/train').glob('sample_*.pkl'))
    valid_files = list(pathlib.Path(f'data/samples/{problem_folder}/valid').glob('sample_*.pkl'))

    log(f"{len(train_files)} training samples", logfile)
    log(f"{len(valid_files)} validation samples", logfile)

    train_files = [str(x) for x in train_files]
    valid_files = [str(x) for x in valid_files]


    ### MODEL LOADING ###
    # sys.path.insert(0, os.path.abspath(f'models/{args.model}'))
    net = model.GAT(in_dim=30,
                    hidden_dim=k * 2,
                    out_dim=2,
                    num_heads=5)

    ### TRAINING LOOP ###
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    best_loss = np.inf

    # duration = []
    # losses = []
    # count = 0
    for epoch in range(max_epochs + 1):
        log(f"EPOCH {epoch}...", logfile)
        if epoch == 0:
            continue
        else:
            train_loss = process(net, train_files, optimizer=optimizer)
            log(f"TRAIN LOSS: {train_loss:0.3f}")
        valid_loss = process(net, valid_files, None)
        log(f"VALID LOSS: {valid_loss:0.3f}")

        if valid_loss < best_loss:
            plateau_count = 0
            best_loss = valid_loss
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

    net.load_state_dict(torch.load(os.path.join(running_dir, 'best_params.pkl')))
    valid_loss = process(net, valid_files, None)
    log(f"VALID LOSS: {valid_loss:0.3f}")
    #     for g in G:
    #         net.train()
    #         count += 1
    #
    #         t0 = time.time()
    #         logits = net(g[i], features[i])
    #         logp = F.log_softmax(logits, 1)
    #         loss = loss + F.nll_loss(logp, clusts[i])
    #
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     if loss < best_loss:
    #         plateau_count = 0
    #         best_loss = loss
    #         # model.save()
    #     else:
    #         plateau_count += 1
    #         if plateau_count % early_stopping == 0:
    #             print('20 epochs without improvement, early stopping')
    #             # log(f"  {plateau_count} epochs without improvement, early stopping", logfile)
    #             break
    #         if plateau_count % patience == 0:
    #             lr *= 0.2
    #             print('10 epochs without improvement, decreasing learning rate to {:.8f}'.format(lr))
    #             # log(f"  {plateau_count} epochs without improvement, decreasing learning rate to {lr}", logfile)
    #         # if epoch >= 3:
    #     duration.append(time.time() - t0)
    #     losses.append(loss.item())
    #
    #     print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
    #         epoch, loss.item(), np.mean(duration)))

    # plt.plot([*range(len(losses))], losses)
    # plt.title("Clustering-Based GNN Loss for k=2")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.show()
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
