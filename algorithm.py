from sage.all import *
import numpy as np
# from generate_graph import random_connected_graph
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

# g = random_connected_graph(100, 0.4, seed = 2020)
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

if __name__ == '__main__':
    g = random_connected_graph(50, 0.4, seed=2020)
#     c = clusters(g, k_center_approximation(g, k=9, distance=False), weighted=True, as_dict=True)
#     g.weighted(True)
#
#     centers, dist = k_center(g, k=3, distance=True)
#     p_exact = g.plot(layout='circular', vertex_colors={'red': centers}, vertex_labels=False, edge_labels=True)
#     print("Exact solution")
#     print('Maximum distance to any point:' + str(dist))
#     print('\n')
#     p_exact.show(figsize=15)
#
#     centers, dist = k_center_approximation(g, k=3, distance=True, weighted=True)
#     p_approximate = g.plot(layout='circular', vertex_colors={'red': centers}, vertex_labels=False, edge_labels=True)
#
#     print("K-Center approximation")
#     print('Maximum distance to any point:' + str(dist));
#     print('\n')
#     p_approximate.show(figsize=15)