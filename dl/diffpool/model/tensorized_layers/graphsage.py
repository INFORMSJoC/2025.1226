# Adapted from the DGL DiffPool example and modified for Tran2SP.
# Original source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/diffpool
# Licensed under the Apache License 2.0. See dl/diffpool/LICENSE.

import torch
from torch import nn as nn
from torch.nn import functional as F


class BatchedGraphSAGE(nn.Module):
    def __init__(
        self, infeat, outfeat, bn_dim, use_bn=True, mean=False, add_self=False, edge_dim=1
    ):
        super().__init__()
        self.add_self = add_self
        self.use_bn = use_bn
        self.mean = mean
        self.edge_dim = edge_dim

        self.W_list = nn.ModuleList([
            nn.Linear(infeat, outfeat, bias=True) for _ in range(edge_dim)
        ])
        for W in self.W_list:
            nn.init.xavier_uniform_(W.weight, gain=nn.init.calculate_gain("relu"))

        self.bn = nn.BatchNorm1d(bn_dim)


    def forward(self, x, adj):
        num_node_per_graph = adj.size(1)
        if self.use_bn and not hasattr(self, "bn"):
            self.bn = self.bn.to(adj.device)

        if self.add_self:
            adj = adj + torch.eye(num_node_per_graph).to(adj.device).unsqueeze(-1)

        if self.mean:
            adj = adj / (adj.sum(-2, keepdim=True) + 1e-8)

        h_k = sum(self.W_list[c](torch.matmul(adj[..., c], x)) for c in range(self.edge_dim))
        h_k = F.normalize(h_k, dim=2, p=2)
        h_k = F.relu(h_k)
        if self.use_bn:
            h_k = self.bn(h_k)
        return h_k

    def __repr__(self):
        if self.use_bn:
            return "BN" + super(BatchedGraphSAGE, self).__repr__()
        else:
            return super(BatchedGraphSAGE, self).__repr__()
