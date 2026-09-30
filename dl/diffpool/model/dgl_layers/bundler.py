# Adapted from the DGL DiffPool example and modified for Tran2SP.
# Original source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/diffpool
# Licensed under the Apache License 2.0. See dl/diffpool/LICENSE.

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bundler(nn.Module):

    def __init__(self, in_feats, out_feats, activation, dropout, bias=True, msg_dim=None):
        super(Bundler, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        actual_msg_dim = msg_dim if msg_dim is not None else in_feats
        self.linear = nn.Linear(in_feats + actual_msg_dim, out_feats, bias)
        self.activation = activation

        nn.init.xavier_uniform_(
            self.linear.weight, gain=nn.init.calculate_gain("relu")
        )

    def concat(self, h, aggre_result):
        bundle = torch.cat((h, aggre_result), 1)
        bundle = self.linear(bundle)
        return bundle

    def forward(self, node):
        h = node.data["h"]
        c = node.data["c"]
        bundle = self.concat(h, c)
        if self.activation:
            bundle = self.activation(bundle)
        return {"h": bundle}
