# Adapted from the DGL DiffPool example and modified for Tran2SP.
# Original source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/diffpool
# Licensed under the Apache License 2.0. See dl/diffpool/LICENSE.

import torch
import torch.nn as nn


class EntropyLoss(nn.Module):
    def forward(self, adj, anext, s_l):
        entropy = (torch.distributions.Categorical(probs=s_l).entropy()).sum(-1).mean(-1)
        assert not torch.isnan(entropy)
        return entropy
