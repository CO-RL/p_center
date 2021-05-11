import pathlib
import pickle
import sys
import importlib
import torch
import os
import torch.nn.functional as F
from algorithm import convert_to_DGLGraph
def read_graph(filename):
    with open(filename, 'rb') as f:
        G, features = pickle.load(f)
    return G, features
def num_one(source_array):
    count = 0
    for x in source_array:
        if x == 1:
            count += 1
    return count

def process(model, dataloader):
    mean_acc = 0
    count = 0
    for batch in dataloader:
        count += 1
        clusts = []
        G, feature= read_graph(batch)
        g, c, l, m, clsts = convert_to_DGLGraph(G, k=k, weighted=True)
        clusts.append(torch.LongTensor(clsts))

        logits = model(g, feature)
        logp = F.log_softmax(logits, 1)
        _, indices = logp.max(dim=1, keepdim=True)
        num = [int(a == b) for a,b in zip(indices, clusts[0])]
        acc_num = num_one(num)

        mean_acc += acc_num
    mean_acc = mean_acc/(len(g)*count)
    return mean_acc
if __name__ == '__main__':
    test_files = list(pathlib.Path(f"data/samples/p_center/test").glob('sample_*.pkl'))
    test_files = [str(x) for x in test_files]
    k = 2
    sys.path.insert(0, os.path.abspath(f"models/GAT"))
    import model
    importlib.reload(model)
    del sys.path[0]
    net = model.GAT(in_dim=30,
                    hidden_dim=k * 2,
                    out_dim=2,
                    num_heads=5)
    running_dir = f"trained_models/p_center/GAT/0"
    net.load_state_dict(torch.load(f"trained_models/p_center/GAT/0/best_params.pkl"))
    test_acc = process(net, test_files)
    print(f" 0 " + " ".join(f"acc: {100*test_acc:4.1f}"))
