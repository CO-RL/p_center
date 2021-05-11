import torch
import torch.nn.functional as F
from utilities import log
import os
import numpy as np
import pathlib
import sys
import importlib
import pickle
from algorithm import convert_to_DGLGraph

def read_graph(filename):
    with open(filename, 'rb') as f:
        G = pickle.load(f)
        feature = pickle.load(f)
    return G, feature
# 返回一个0,1数组中1的数量
def num_one(source_array):
    count = 0
    for x in source_array:
        if x == 1:
            count += 1
    return count

def process(model, dataloader, optimizer=None):
    mean_loss = 0
    mean_acc = 0
    count = 0
    n_samples_processed = len(dataloader)
    for batch in dataloader:
        count += 1
        clusts = []
        G, feature= read_graph(batch)
        g, c, l, m, clsts = convert_to_DGLGraph(G, k=k, weighted=True)
        clusts.append(torch.LongTensor(clsts))

        if optimizer:
            logits = model(g, feature)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp, clusts[0])
            _, indices = logp.max(dim=1, keepdim=True)
            num = [int(a==b) for a,b in zip(indices, clusts[0])]
            acc_num = num_one(num)
            acc = acc_num/len(g)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            logits = model(g, feature)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp, clusts[0])
            _, indices = logp.max(dim=1, keepdim=True)
            num = [int(a==b) for a,b in zip(indices, clusts[0])]
            acc_num = num_one(num)
            acc = acc_num

        mean_loss += loss
        mean_acc += acc_num
        # mean_acc += acc
        # g.append(dgl_graph)
        # centers.append(c)
        # labels.append(l)
        # masks.append(m)
        # clusts.append(torch.LongTensor(clsts))
    mean_loss /= n_samples_processed
    mean_acc = mean_acc/(len(g)*count)
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

    ### MODEL LOADING ###
    sys.path.insert(0, os.path.abspath(f'models/GAT'))
    import model
    importlib.reload(model)
    net = model.GAT(in_dim=10,
                    hidden_dim=k * 2,
                    out_dim=2,
                    num_heads=5)
    del sys.path[0]

    ### TRAINING LOOP ###
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    best_loss = np.inf

    for epoch in range(max_epochs + 1):
        log(f"EPOCH {epoch}...", logfile)
        if epoch == 0:
            continue
        else:
            train_loss, train_acc= process(net, train_files, optimizer=optimizer)
            log(f"TRAIN LOSS: {train_loss:0.3f}" + "".join(f" acc: {train_acc:0.3f}"), logfile)
        valid_loss, valid_acc = process(net, valid_files, None)
        log(f"VALID LOSS: {valid_loss:0.3f}" + "".join(f" acc: {valid_acc:0.3f}"), logfile)

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