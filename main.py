import torch
import dgl
from generate_graph import random_connected_graph, extract_features
from algorithm import k_center_approximation, clusters
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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    weighted = True
    G = [random_connected_graph(50, 0.4, weighted=True) for _ in range(3)]
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

    import model
    net= model.GAT(g[0],
           in_dim=features[0].size()[1],
           hidden_dim=k * 2,
           out_dim=2,
           num_heads=5)

    optimizer = torch.optim.Adam((net.parameters(), lr=1e-3)

    duration = []
    lossed = []
    count = 0
    for k in range(1):
        for i in range(len(g)):
            for epoch in range(25):
                net_2_clustering.train()
                count += 1
                if epoch >= 3:
                    t0 = time.time()
                logits =net_2_clustering(features[i])
                logp = F.log_softmax(logits, 1)
                loss = F.nll_loss(logp, clusts[i])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if epoch >= 3:
                    duration.append(time.time() - t0)
                    losses.append(loss.item())