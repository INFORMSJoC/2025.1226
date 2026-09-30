# Adapted from the DGL DiffPool example and modified for Tran2SP.
# Original source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/diffpool
# Licensed under the Apache License 2.0. See dl/diffpool/LICENSE.

import time

import dgl

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import block_diag
from torch.nn import init

from .dgl_layers import DiffPoolBatchedGraphLayer, GraphSageLayer
from .model_utils import batch2tensor
from .tensorized_layers import *


class DiffPool(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        embedding_dim,
        activation,
        n_layers,
        dropout,
        n_pooling,
        linkpred,
        aggregator_type,
        assign_dim,
        pool_ratio,
        out_type,
        cat=False,
        edge_dim=1,
    ):
        super(DiffPool, self).__init__()
        self.link_pred = linkpred
        self.concat = cat
        self.n_pooling = n_pooling
        self.link_pred_loss = []
        self.entropy_loss = []
        self.out_type = out_type

        self.gc_before_pool = nn.ModuleList()
        self.diffpool_layers = nn.ModuleList()

        self.gc_after_pool = nn.ModuleList()
        self.assign_dim = assign_dim
        self.bn = True
        self.num_aggs = 1
        if not cat:
            self.skip_proj = nn.Linear(embedding_dim * 2, embedding_dim)

        self.gc_before_pool.append(
            GraphSageLayer(
                input_dim,
                hidden_dim,
                activation,
                dropout,
                aggregator_type,
                self.bn,
                edge_dim=edge_dim,
            )
        )
        for _ in range(n_layers - 2):
            self.gc_before_pool.append(
                GraphSageLayer(
                    hidden_dim,
                    hidden_dim,
                    activation,
                    dropout,
                    aggregator_type,
                    self.bn,
                    edge_dim=edge_dim,
                )
            )
        self.gc_before_pool.append(
            GraphSageLayer(hidden_dim, embedding_dim, None, dropout, aggregator_type, edge_dim=edge_dim)
        )

        assign_dims = []
        assign_dims.append(self.assign_dim)
        if self.concat:
            pool_embedding_dim = hidden_dim * (n_layers - 1) + embedding_dim
        else:
            pool_embedding_dim = embedding_dim

        self.first_diffpool_layer = DiffPoolBatchedGraphLayer(
            pool_embedding_dim,
            self.assign_dim,
            hidden_dim,
            activation,
            dropout,
            aggregator_type,
            self.link_pred,
            edge_dim=edge_dim,
        )
        gc_after_per_pool = nn.ModuleList()

        for _ in range(n_layers - 1):
            gc_after_per_pool.append(BatchedGraphSAGE(hidden_dim, hidden_dim, self.assign_dim, edge_dim=edge_dim))
        gc_after_per_pool.append(BatchedGraphSAGE(hidden_dim, embedding_dim, self.assign_dim, edge_dim=edge_dim))
        self.gc_after_pool.append(gc_after_per_pool)

        if self.concat:
            self.pred_input_dim = pool_embedding_dim * self.num_aggs * (n_pooling + 1)
        else:
            self.pred_input_dim = embedding_dim * self.num_aggs

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def gcn_forward(self, g, h, gc_layers, cat=False):
        block_readout = []
        for gc_layer in gc_layers[:-1]:
            h = gc_layer(g, h)
            block_readout.append(h)
        h = gc_layers[-1](g, h)
        block_readout.append(h)
        if cat:
            block = torch.cat(block_readout, dim=1)
        else:
            block = h
        return block

    def gcn_forward_tensorized(self, h, adj, gc_layers, cat=False):
        block_readout = []
        for gc_layer in gc_layers:
            h = gc_layer(h, adj)
            block_readout.append(h)
        if cat:
            block = torch.cat(block_readout, dim=2)
        else:
            block = h
        return block

    def forward(self, g):
        self.link_pred_loss = []
        self.entropy_loss = []
        h = g.ndata["feat"]
        h_a = h

        out_all = []

        g_embedding = self.gcn_forward(g, h, self.gc_before_pool, self.concat)

        g.ndata["h"] = g_embedding

        readout = dgl.sum_nodes(g, "h")
        out_all.append(readout)
        if self.num_aggs == 2:
            readout = dgl.max_nodes(g, "h")
            out_all.append(readout)

        adj, h = self.first_diffpool_layer(g, g_embedding)
        node_per_pool_graph = int(adj.size()[0] / len(g.batch_num_nodes()))

        h, adj = batch2tensor(adj, h, node_per_pool_graph)
        h = self.gcn_forward_tensorized(h, adj, self.gc_after_pool[0], self.concat)
        readout = torch.sum(h, dim=1)
        out_all.append(readout)
        if self.num_aggs == 2:
            readout, _ = torch.max(h, dim=1)
            out_all.append(readout)

        for i, diffpool_layer in enumerate(self.diffpool_layers):
            h, adj = diffpool_layer(h, adj)
            h = self.gcn_forward_tensorized(h, adj, self.gc_after_pool[i + 1], self.concat)
            readout = torch.sum(h, dim=1)
            out_all.append(readout)
            if self.num_aggs == 2:
                readout, _ = torch.max(h, dim=1)
                out_all.append(readout)
        if self.concat or self.num_aggs > 1:
            final_readout = torch.cat(out_all, dim=1)
        else:
            final_readout = self.skip_proj(torch.cat([out_all[0], readout], dim=1))
        if self.out_type == "raw":
            return h
        elif self.out_type == "mean":
            return final_readout.unsqueeze(1)
