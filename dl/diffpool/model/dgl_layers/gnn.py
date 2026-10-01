# Adapted from the DGL DiffPool example and modified for Tran2SP.
# Original source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/diffpool
# Licensed under the Apache License 2.0. See dl/diffpool/LICENSE.

import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.linalg import block_diag

from dl.diffpool.model.loss import EntropyLoss

from dl.diffpool.model.dgl_layers.aggregator import MeanAggregator
from dl.diffpool.model.dgl_layers.bundler import Bundler


# GraphSage layer in Inductive learning paper by hamilton
# Here, graphsage layer is a reduced function in DGL framework
class GraphSageLayer(nn.Module):

    def __init__(
        self,
        in_feats,
        out_feats,
        activation,
        dropout,
        aggregator_type,
        bn=False,
        bias=True,
        edge_dim=1,
    ):
        super(GraphSageLayer, self).__init__()
        self.use_bn = bn
        self.edge_dim = edge_dim
        self.bundler = Bundler(in_feats, out_feats, activation, dropout, bias=bias, msg_dim=edge_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.aggregator = MeanAggregator()

        if self.use_bn:
            self.bn = nn.BatchNorm1d(out_feats)

    def forward(self, g, h):
        h = self.dropout(h)
        g.ndata["h"] = h
        if "feat" in g.edata:
            g.update_all(fn.copy_e("feat", "m"), self.aggregator, self.bundler)
        else:
            g.update_all(fn.copy_u(u="h", out="m"), self.aggregator, self.bundler)
        h = g.ndata.pop("h")
        if self.use_bn:
            h = self.bn(h)
        return h


class DiffPoolBatchedGraphLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        assign_dim,
        output_feat_dim,
        activation,
        dropout,
        aggregator_type,
        link_pred,
        edge_dim=1,
    ):
        super(DiffPoolBatchedGraphLayer, self).__init__()
        self.embedding_dim = input_dim
        self.assign_dim = assign_dim
        self.hidden_dim = output_feat_dim
        self.link_pred = link_pred
        self.feat_gc = GraphSageLayer(
            input_dim, output_feat_dim, activation, dropout, aggregator_type, edge_dim=edge_dim
        )
        self.pool_gc = GraphSageLayer(input_dim, assign_dim, activation, dropout, aggregator_type, edge_dim=edge_dim)
        self.reg_loss = nn.ModuleList([])
        self.loss_log = {}
        self.reg_loss.append(EntropyLoss())

    def forward(self, g, h):
        feat = self.feat_gc(g, h)
        device = feat.device
        assign_tensor = self.pool_gc(g, h)
        assign_tensor = F.softmax(assign_tensor, dim=1)
        assign_tensor = torch.split(
            assign_tensor, g.batch_num_nodes().tolist()
        )
        assign_tensor = torch.block_diag(*assign_tensor)

        h = torch.matmul(torch.t(assign_tensor), feat)
        adj = g.adj_external(transpose=True, ctx=device)
        e_f = torch.sparse_coo_tensor(
            indices=adj.coalesce().indices(),
            values=g.edata["feat"],
            size=(adj.size()[0], adj.size()[1], g.edata["feat"].shape[-1]),
        )
        ef = []
        for i in range(g.edata["feat"].shape[1]):
            ef.append(torch.matmul(e_f.to_dense().permute(2, 0, 1)[i], assign_tensor).unsqueeze(0))
        adj_new = torch.concat(ef, dim=0)

        adj_new = torch.matmul(torch.t(assign_tensor), adj_new).permute(1, 2, 0)

        if self.link_pred:
            current_lp_loss = torch.norm(adj.to_dense() - torch.mm(assign_tensor, torch.t(assign_tensor))) / np.power(
                g.num_nodes(), 2
            )
            self.loss_log["LinkPredLoss"] = current_lp_loss

        for loss_layer in self.reg_loss:
            loss_name = str(type(loss_layer).__name__)
            self.loss_log[loss_name] = loss_layer(adj, adj_new, assign_tensor)

        return adj_new, h
