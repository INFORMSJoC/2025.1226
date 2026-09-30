# DiffPool

This code is adapted from the [DGL DiffPool example](https://github.com/dmlc/dgl/tree/master/examples/pytorch/diffpool), which implements the method described in Ying et al., *Hierarchical Graph Representation Learning with Differentiable Pooling*, NeurIPS 2018.

The implementation was modified for Tran2SP to support edge features in message passing and pooling and to produce embeddings for the Tran2SP model. Unused components were removed.

The code in this directory is distributed under the Apache License 2.0. A copy is included as [LICENSE](LICENSE).
