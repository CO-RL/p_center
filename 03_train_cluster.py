import torch
import torch.nn.functional as F
from utilities import log
import os
import numpy as np
import pathlib
import sys
import importlib
import pickle
import torch.utils.data as Data
import matplotlib.pyplot as plt
import dgl
from dgl.data.utils import load_labels
from dgl.data.utils import load_graphs

def read_graph(filename):
    graph_path = "/".join(filename.split(".")[0:-1])+".bin"
    feature_path = "/".join(filename.split(".")[0:-1])+".pkl"
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

def load_data(dataloader, batch_size=6):
    graph_datasets = []
    datasets = []
    for data in dataloader:
        dataset = read_graph(data)
        graph, data = dataset
        graph_datasets.append(graph)
        datasets.append(data)
    # graph_loader = Data.DataLoader(
    #     dataset=datasets,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=2,
    # )
    return (graph_datasets, datasets)

def process(model, dataloader, optimizer=None, weighted=True):
    graph_datasets, datasets = dataloader
    mean_loss = 0
    mean_acc = 0
    for step in range(len(dataloader)):
        g, data_ = graph_datasets[step][0][0], datasets[step]
        feature, c, l, m, clsts = data_
        # if weighted:
        #     nxgraph = G.networkx_graph(weight_function=lambda edge: float(edge[2]))
        # else:
        #     nxgraph = G.networkx_graph(weight_function=lambda edge: float(1))
        # g = dgl.DGLGraph()
        # g.from_networkx(nxgraph, edge_attrs=['weight'])

        if optimizer:
            logits = model(g, feature)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp, torch.LongTensor(clsts))

            _, indices = logp.max(dim=1, keepdim=True)
            num = [int(a==b) for a,b in zip(indices, clsts)]
            acc_num = num_one(num)
            batch_size = 6
            node_num = 50
            acc = acc_num/(batch_size*node_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            logits = model(g, feature)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp, torch.LongTensor(clsts))
            _, indices = logp.max(dim=1, keepdim=True)
            num = [int(a==b) for a,b in zip(indices, clsts)]
            acc_num = num_one(num)
            acc = acc_num/50

    mean_loss += loss
    mean_acc += acc

    mean_loss /= step
    mean_acc /= step
    return mean_loss, mean_acc

if __name__ == '__main__':
    ### HYPER PARAMETERS ###
    max_epochs = 1000
    lr = 0.01
    patience = 10
    early_stopping = 20
    best_loss = np.inf
    k = 2
    running_dir = f"trained_models/p_center/GAT"
    os.makedirs(running_dir)

    ### LOG ###
    logfile = os.path.join(running_dir, 'log.txt')
    log(f"max_epochs: {max_epochs}", logfile)
    log(f"lr: {lr}", logfile)
    log(f"patience : {patience }", logfile)
    log(f"early_stopping : {early_stopping }", logfile)

    ### SET-UP DATASET ###
    train_files = list(pathlib.Path(f'data/samples/p_center/train').glob('sample_*.pkl'))
    valid_files = list(pathlib.Path(f'data/samples/p_center/valid').glob('sample_*.pkl'))

    log(f"{len(train_files)} training samples", logfile)
    log(f"{len(valid_files)} validation samples", logfile)

    train_files = [str(x) for x in train_files]
    valid_files = [str(x) for x in valid_files]

    train_loader = load_data(train_files)
    valid_loader = load_data(valid_files)

    ### MODEL LOADING ###
    sys.path.insert(0, os.path.abspath(f'models/GAT'))
    import model
    importlib.reload(model)
    net = model.GAT(in_dim=30,
                    hidden_dim=k * 2,
                    out_dim=2,
                    num_heads=5)
    del sys.path[0]

    ### TRAINING LOOP ###
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    best_loss = np.inf
    train_losses = []
    train_accs = []

    valid_losses = []
    valid_accs = []
    for epoch in range(max_epochs + 1):
        log(f"EPOCH {epoch}...", logfile)
        if epoch == 0:
            print("pretrain...")
        else:
            train_loss, train_acc= process(net, train_loader, optimizer=optimizer, weighted=True)
            log(f"TRAIN LOSS: {train_loss:0.3f}" + "".join(f" acc: {train_acc:0.3f}"), logfile)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
        valid_loss, valid_acc = process(net, valid_loader, None)
        log(f"VALID LOSS: {valid_loss:0.3f}" + "".join(f" acc: {valid_acc:0.3f}"), logfile)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)

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
    valid_loss, valid_acc = process(net, valid_files, None)
    log(f"VALID LOSS: {valid_loss:0.3f}" + "".join(f" acc: {valid_acc:0.3f}"), logfile)

    plt.plot([*range(len(train_losses))], train_losses)
    plt.title("Clustering-Based GNN Loss for k=2")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()