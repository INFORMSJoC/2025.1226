# Adapted from the DGL DiffPool example and modified for Tran2SP.
# Original source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/diffpool
# Licensed under the Apache License 2.0. See dl/diffpool/LICENSE.

import torch
import torch.nn as nn
import torch.nn.functional as F


# Base Aggregator class. Adapting
# from PR# 403
class Aggregator(nn.Module):

    def __init__(self):
        super(Aggregator, self).__init__()

    def forward(self, node):
        neighbour = node.mailbox["m"]
        c = self.aggre(neighbour)
        return {"c": c}

    def aggre(self, neighbour):
        raise NotImplementedError


class MeanAggregator(Aggregator):

    def __init__(self):
        super(MeanAggregator, self).__init__()

    def aggre(self, neighbour):
        mean_neighbour = torch.mean(neighbour, dim=1)
        return mean_neighbour
