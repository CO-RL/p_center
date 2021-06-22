import numpy as np
import math
import networkx as nx
import time
from sage.all import *
import matplotlib as plt


def convert_to_complete(G, weighted=False):
    '''
    Function that converts graph to complete graph. Edge weights are distances between the vertices.
    Weighted as false reduced computational time.
    '''
    all_edges = []
    vertices = G.vertices()
    # distance is symmetric hence iterating one direction is enough
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            all_edges.append((vertices[i], vertices[j], G.distance(vertices[i], vertices[j], by_weight=weighted)))
    complete = Graph(all_edges)
    return complete

def k_center(G, k=3, distance=False):
    '''
    Function takes argument graph and modifies into complete graph before implementing efficient k-center algorithm
    Detailed description of algorithm can be found above

    '''
    # sorted edge list by edge weight
    complete = convert_to_complete(G, weighted=True)
    weights = sorted(set([edge[2] for edge in complete.edges()]))
    high = len(weights) - 1
    low = 0
    while high - low > 1:
        mid = int(math.ceil((high + low) / 2))
        r_max = weights[mid]
        bottleneck_graph = complete.copy()
        # removes all edges from G that have a weight larger than the maximum permitted radius
        edges_to_remove = [edge for edge in complete.edges() if edge[2] > r_max]
        bottleneck_graph.delete_edges(edges_to_remove)
        centers = bottleneck_graph.dominating_set()
        # binary search within weights
        if len(centers) <= k:
            high = mid
        else:
            low = mid
    if len(centers) > k:
        mid += 1
        r_max = weights[mid]
        bottleneck_graph = complete.copy()
        edges_to_remove = [edge for edge in complete.edges() if edge[2] > r_max]
        bottleneck_graph.delete_edges(edges_to_remove)
        centers = bottleneck_graph.dominating_set()
    if distance:
        return centers, r_max
    else:
        return centers

def k_center_approximation(G, k=3, weighted=False, seed=None, distance=False):
    ''' This function uses greedy approximation to find centers.
        Returns the resulting k-centers as specified as well as the maximum distance if required.
    '''
    G.weighted(True)
    vertices = G.vertices()
    if seed != None:
        np.random.seed(seed)
    starting_index = np.random.randint(len(vertices))
    C = [starting_index]
    deltas = []
    while len(C) < k:
        maximized_distance = 0
        for v in range(len(vertices)):
            if v in C:
                continue
            dists = [float(G.distance(vertices[v], vertices[c], by_weight=weighted)) for c in C]
            min_dist = min(dists)
            if min_dist > maximized_distance:
                maximized_distance = min_dist
                best_candidate = v
        deltas.append(maximized_distance)
        C.append(best_candidate)
    if distance:
        return [vertices[c] for c in C], min(deltas)
    else:
        return [vertices[c] for c in C]

def random_connected_graph(n, p, seed=None, weighted=True):
    '''
        n - number of vertices
        p - probability there is an edge between two vertices
        uses uniform distribution for edge labeling
    '''
    G = graphs.RandomGNP(n, p, seed=seed)  # ensures that G is completely connected

    sd = seed
    while len(G.connected_components()) > 1:
        if sd != None:
            sd += 1
        G = graphs.RandomGNP(n, p, seed=sd)
    np.random.seed(seed)
    if weighted:
        for edge in G.edges():
            G.set_edge_label(edge[0], edge[1], RR(np.random.random_sample()))
    return G

import random
def clusters(G, centers, weighted=True, as_dict=False):
    if not as_dict:
        clusts = [0] * G.order()
        for vertex in G.vertices():
            if vertex in centers:
                continue
            closest = None
            min_dist = 100000000
            # we randomly shuffle to not skew our clustering towards the first elements of centers
            # otherwise, we get a top-heavy list of clusters which, although is correct, will make classification harder
            cs = [*range(len(centers))]
            random.shuffle(cs)
            for c in cs:
                dist = G.distance(centers[c], vertex, by_weight=weighted)
                if dist < min_dist:
                    closest = c
                    min_dist = dist
            clusts[vertex] = closest
        return clusts
    else:
        clusts = {center: [] for center in range(len(centers))}
        for vertex in G.vertices():
            if vertex in centers:
                continue
            closest = None
            min_dist = 100000000
            # we randomly shuffle to not skew our clustering towards the first elements of centers
            # otherwise, we get a top-heavy list of clusters which, although is correct, will make classification harder
            cs = [*range(len(centers))]
            random.shuffle(cs)
            for c in cs:
                dist = G.distance(centers[c], vertex, by_weight=weighted)
                if dist < min_dist:
                    closest = c
                    min_dist = dist
            clusts[closest].append(vertex)
        return clusts

# # GNN imports
# !pip3 install --user dgl
# !pip3 install --user torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from dgl import DGLGraph
import dgl
import networkx as nx
import time
RealNumber = float; Integer = int

def convert_to_DGLGraph(G, k=3, weighted=True):
    if weighted:
        nxgraph = G.networkx_graph(weight_function = lambda edge: float(edge[2]))
    else:
        nxgraph = G.networkx_graph(weight_function = lambda edge: float(1))
    dgl_G = dgl.DGLGraph()
    dgl_G.from_networkx(nxgraph, edge_attrs=['weight'])
    centers, distance = k_center(G, k=k, distance=True)
    centers = [int(c) for c in centers]
    labels = torch.LongTensor([int(1) if node in centers else int(0) for node in G.vertices()])
    mask = torch.BoolTensor([True for _ in range(len(labels))])
    clusts = clusters(G, centers, weighted=weighted)
    return dgl_G, centers, labels, mask, clusts

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

import torch
import torch.nn as nn
import torch.nn.functional as F
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        # self.g = g
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h):
        # equation (1)
        self.g = g
        z = self.fc(h)
        self.g.ndata['z'] = z
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')

class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim))
        self.merge = merge

    def forward(self, g, h):
        head_outs = [attn_head(g, h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(in_dim, hidden_dim, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1)

    def forward(self, g, h):
        h = self.layer1(g, h)
        h = F.elu(h)
        h = self.layer2(g, h)
        return h

weighted = True
g = random_connected_graph(50, 0.4, seed=2020)
features = extract_features(g, dim=30, weighted=True)
k=3

centers, dist = k_center(g, k=3, distance=True)
p_exact = g.plot(layout='circular', vertex_colors={'red': centers}, vertex_labels=False, edge_labels=True)
print(centers)
print("Exact solution")
print('Maximum distance to any point:' + str(dist)) ; print('\n')
p_exact.show(figsize=15)

centers, dist = k_center_approximation(g, k=3, distance=True, weighted=True)
p_approximate = g.plot(layout='circular', vertex_colors={'red': centers}, vertex_labels=False, edge_labels=True)
print(centers)
print("K-Center approximation")
print('Maximum distance to any point:' + str(dist)) ; print('\n')
p_approximate.show(figsize=15)

dgl_graph, c, l, m, clsts = convert_to_DGLGraph(g, k=k, weighted=weighted)

from torch.autograd import Variable
features = Variable(features)
clsts = Variable(torch.LongTensor(clsts))

net_2_clustering = GAT(in_dim=30,
      hidden_dim=k*5,
      out_dim=k,
      num_heads=5)
optimizer = torch.optim.Adam(net_2_clustering.parameters(), lr=1e-3)

duration = []
losses = []
count = 0

for epoch in range(3200):
    count += 1
    if epoch >= 3:
        t0 = time.time()

    logits = net_2_clustering(dgl_graph, features)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp, clsts)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >= 3:
        duration.append(time.time() - t0)
        losses.append(loss.item())
    print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
       count, loss.item(), np.mean(duration)))

_, indices = logp.max(dim=1, keepdim=True)
g_0 = []
g_1 = []
g_2 = []
s=0
for i in indices:
    if i == 0:
        g_0.append(s)
        s+=1
    elif i == 1:
        g_1.append(s)
        s+=1
    elif i == 2:
        g_2.append(s)
        s+=1
g0 = dgl_graph.subgraph(g_0)
g1 = dgl_graph.subgraph(g_1)
g2 = dgl_graph.subgraph(g_2)
g0.ndata['x'] = dgl_graph.ndata['z'][g0.parent_nid]
g0.edata['weight'] = dgl_graph.edata['weight'][g0.parent_eid]
g0.edata['e'] = dgl_graph.edata['e'][g0.parent_eid]
g1.ndata['x'] = dgl_graph.ndata['z'][g1.parent_nid]
g1.edata['weight'] = dgl_graph.edata['weight'][g1.parent_eid]
g1.edata['e'] = dgl_graph.edata['e'][g1.parent_eid]
g2.ndata['x'] = dgl_graph.ndata['z'][g2.parent_nid]
g2.edata['weight'] = dgl_graph.edata['weight'][g2.parent_eid]
g2.edata['e'] = dgl_graph.edata['e'][g2.parent_eid]

def cal_center(g, distance=False):
    G = g.to_networkx(node_attrs=['x'], edge_attrs=['weight'])
    adj = nx.adjacency_matrix(G, weight='weight')
    dense_adj = adj.todense()
    sums = np.max(dense_adj, axis=1)
    min_index, min_dist = min(enumerate(sums), key=operator.itemgetter(1))
    center = g.parent_nid[min_index-1].numpy().tolist()
    if distance:
        return center, min_dist[0,0]
    else:
        return min_dist

center0, dist0 = cal_center(g0, distance=True)
center1, dist1 = cal_center(g1, distance=True)
center2, dist2 = cal_center(g2, distance=True)
centers = [center0, center1, center2]
all_dist = [dist0, dist1, dist2]
dist = max(dist0, dist1, dist2)
print(all_dist)
print(centers)
print(dist)

