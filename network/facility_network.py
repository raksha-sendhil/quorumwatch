"""Facility network as a weighted adjacency matrix, using NumPy matrix operations."""
import numpy as np
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree


def build_weights(coords, knn, length_scale):
    """Symmetric kNN graph plus a spanning tree (guarantees connectivity). Weights are
    exp(-d / length_scale), scaled so that the largest weighted degree equals 1."""
    delta = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((delta ** 2).sum(axis=2))
    N = len(coords)
    order = np.argsort(dist, axis=1)[:, 1:knn + 1]
    adj = np.zeros((N, N), bool)
    adj[np.repeat(np.arange(N), knn), order.ravel()] = True
    tree = minimum_spanning_tree(dist).toarray() > 0
    adj = adj | adj.T | tree | tree.T
    W = np.where(adj, np.exp(-dist / length_scale), 0.0)
    return W / W.sum(axis=1).max(), dist


def laplacian(W):
    return np.diag(W.sum(axis=1)) - W


def n_components(W):
    return connected_components(W > 0, directed=False)[0]


def closed_neighbourhood(W):
    """Boolean matrix whose row i marks facility i and its direct neighbours."""
    return (W > 0) | np.eye(len(W), dtype=bool)


def n_directed_edges(W):
    return int((W > 0).sum())
