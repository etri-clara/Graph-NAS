import torch
from grakel import Graph
from grakel.kernels import ShortestPath
from torch_cluster import knn_graph

# construct kNN graph
kernel_map = {"shortest": ShortestPath}


def kNNConstruction(data, node, method="euclidean", k=10):
    tot = len(node)
    batch = torch.tensor([0] * tot)
    if method == "cosine":
        edge_index = knn_graph(node, k=k, batch=batch, loop=False, cosine=True)
    elif method == "shortest":
        edge_index = kernel_kNN(tot, data, method, k)
    else: # if euclidean
        edge_index = knn_graph(node, k=k, batch=batch, loop=False)
    return edge_index


# kernel
def find_idx(f):
    id = []
    for i in f:
        for idx, j in enumerate(i):
            if j == 1:
                id.append(idx)
    return id


def kernel_kNN(sample, data, kernel="shortest path", k=10):
    G = []
    for i in range(sample):
        node_labels = {}
        id = find_idx(data[i]['x'].numpy())
        for key, value in enumerate(id):
            node_labels[key] = value

        edges = torch.transpose(data[i]['edge_index'], 0, 1).numpy()
        edges = [tuple(e) for e in edges]
        G.append(Graph(edges, node_labels=node_labels))

    gk = kernel_map[kernel]()
    kernel_simi = torch.tensor(gk.fit_transform(G))
    kernel_idx = torch.topk(kernel_simi, k=k, dim=1, largest=True)[1][:, 1:].numpy()

    edge_list = []
    for i in range(sample):
        for j in kernel_idx[i]:
            edge_list.append([i, j])
    edge_list = torch.transpose(torch.tensor(edge_list), 0, 1)

    return edge_list
