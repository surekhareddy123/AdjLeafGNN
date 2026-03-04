from __future__ import annotations
import numpy as np
import torch

def knn_graph(embeddings: torch.Tensor, k: int = 5, self_loops: bool = True):
    """Build KNN adjacency from embeddings (N,D) using Euclidean distance.

    Returns:
      adj: (N,N) float32 tensor with 0/1 entries
      dist: (N,N) float32 tensor of pairwise distances (for optional spread labels)
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be (N,D)")
    with torch.no_grad():
        # cdist can be memory-heavy but OK for moderate N; for very large N, batch it.
        dist = torch.cdist(embeddings, embeddings, p=2)  # (N,N)
        N = dist.size(0)
        # exclude self in knn selection
        dist_for_knn = dist.clone()
        dist_for_knn.fill_diagonal_(float("inf"))
        knn_idx = torch.topk(dist_for_knn, k=k, largest=False).indices  # (N,k)
        adj = torch.zeros((N, N), dtype=torch.float32, device=embeddings.device)
        rows = torch.arange(N, device=embeddings.device).unsqueeze(1).expand_as(knn_idx)
        adj[rows, knn_idx] = 1.0
        # symmetrize to undirected
        adj = torch.maximum(adj, adj.t())
        if self_loops:
            adj.fill_diagonal_(1.0)
        return adj, dist
