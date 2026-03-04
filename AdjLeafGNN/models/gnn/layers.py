from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # adj: (N,N) with self-loops
        # normalize: D^{-1/2} A D^{-1/2}
        deg = adj.sum(dim=1)  # (N,)
        deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
        D = torch.diag(deg_inv_sqrt)
        A_norm = D @ adj @ D
        return self.lin(A_norm @ x)

class GATLayer(nn.Module):
    """Lightweight dense GAT layer (no PyG), O(N^2) attention (OK for moderate N)."""
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.W = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, out_dim))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        h = self.W(x).view(N, self.heads, self.out_dim)  # (N,H,D)
        # attention logits e_ij per head
        # e_ij = a^T [Wh_i || Wh_j] implemented as dot with a_src/a_dst
        e_src = (h * self.a_src.unsqueeze(0)).sum(dim=-1)  # (N,H)
        e_dst = (h * self.a_dst.unsqueeze(0)).sum(dim=-1)  # (N,H)

        # broadcast to (N,N,H): e_ij = leaky(e_src_i + e_dst_j)
        e = self.leaky(e_src.unsqueeze(1) + e_dst.unsqueeze(0))  # (N,N,H)

        # mask non-edges
        mask = (adj > 0).unsqueeze(-1)  # (N,N,1)
        e = e.masked_fill(~mask, float("-inf"))

        alpha = torch.softmax(e, dim=1)  # sum over j neighbors
        alpha = self.dropout(alpha)
        # aggregate: sum_j alpha_ij * h_j
        out = torch.einsum("ijh,jhd->ihd", alpha, h)  # (N,H,D)
        out = out.reshape(N, self.heads * self.out_dim)
        return out
