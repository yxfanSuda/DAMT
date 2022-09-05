import torch
from torch import nn
import numpy as np

def normalize_adj(adj):
    import scipy.sparse as sp
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).toarray()

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, \
                 output_dim, \
                 act_func=None, \
                 dropout_rate=0., \
                 bias=False):
        super(GraphConvolution, self).__init__()

        self.W = nn.Parameter(torch.randn(input_dim, output_dim))

        if bias:
            self.b = nn.Parameter(torch.zeros(1, output_dim))

        self.act_func = act_func
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, adj):

        x = self.dropout(x)
        pre_sup = x.matmul(self.W)
        out = adj.matmul(pre_sup)

        if self.act_func is not None:
            out = self.act_func(out)
        return out

