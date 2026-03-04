from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lddnet.encoder import LDDEncoder
from .gnn.graph_builder import knn_graph
from .gnn.layers import GCNLayer, GATLayer

class AdjLeafGNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        base_ch: int = 64,
        aspp_dilations=(1,6,12,18),
        dropout: float = 0.3,
        gnn_type: str = "gcn",
        gnn_hidden: int = 256,
        gat_heads: int = 4,
        k_nn: int = 5,
        self_loops: bool = True,
    ):
        super().__init__()
        self.encoder = LDDEncoder(base_ch=base_ch, aspp_dilations=aspp_dilations, dropout=dropout, emb_dim=gnn_hidden)

        self.gnn_type = gnn_type.lower()
        self.k_nn = int(k_nn)
        self.self_loops = bool(self_loops)

        in_dim = self.encoder.embedding_dim

        if self.gnn_type == "gcn":
            self.gnn1 = GCNLayer(in_dim, gnn_hidden)
            self.gnn2 = GCNLayer(gnn_hidden, gnn_hidden)
            gnn_out_dim = gnn_hidden
        elif self.gnn_type == "gat":
            self.gnn1 = GATLayer(in_dim, gnn_hidden // gat_heads, heads=gat_heads, dropout=dropout)
            self.gnn2 = GATLayer(gnn_hidden, gnn_hidden // gat_heads, heads=gat_heads, dropout=dropout)
            gnn_out_dim = gnn_hidden
        else:
            raise ValueError(f"Unknown gnn_type: {gnn_type}")

        self.cls_head = nn.Linear(gnn_out_dim, num_classes)
        self.spread_head = nn.Linear(gnn_out_dim, 1)

    def forward(self, images: torch.Tensor):
        """Forward for a mini-batch; internally builds a graph over the batch nodes."""
        _, emb = self.encoder(images)  # (B,D)
        adj, dist = knn_graph(emb, k=self.k_nn, self_loops=self.self_loops)  # (B,B)

        x = emb
        if self.gnn_type == "gcn":
            x = F.relu(self.gnn1(x, adj))
            x = self.gnn2(x, adj)
        else:
            x = F.elu(self.gnn1(x, adj))
            x = F.elu(self.gnn2(x, adj))

        logits_cls = self.cls_head(x)             # (B,C)
        logits_spread = self.spread_head(x).squeeze(-1)  # (B,)
        return {
            "emb": emb,
            "adj": adj,
            "dist": dist,
            "logits_cls": logits_cls,
            "logits_spread": logits_spread,
        }
