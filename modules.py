import math

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import GINConv, global_add_pool, GATConv, GCNConv
from torch.nn import Linear, LayerNorm, Dropout, Sequential, BatchNorm1d, ReLU

class GraphConvolution(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_, adj):
        support = torch.matmul(input_, self.weight)
        support = support.view(1, adj.shape[1], -1)
        output = torch.bmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.size = args.dim
        self.gc1 = GraphConvolution(args.in_channels, self.size)
        self.gc2 = GraphConvolution(self.size, self.size)
        self.gc3 = GraphConvolution(self.size, self.size)
        self.gc4 = GraphConvolution(self.size, self.size)
        self.bn1 = torch.nn.BatchNorm1d(self.size)
        self.bn2 = torch.nn.BatchNorm1d(self.size)
        self.bn3 = torch.nn.BatchNorm1d(self.size)
        self.bn4 = torch.nn.BatchNorm1d(self.size)
        self.sigmoid = torch.nn.Sigmoid()
        self.fc = torch.nn.Linear(self.size, args.out_channels)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.uniform_(self.gc1.weight, a=-0.05, b=0.05)
        torch.nn.init.uniform_(self.gc2.weight, a=-0.05, b=0.05)
        torch.nn.init.uniform_(self.gc3.weight, a=-0.05, b=0.05)
        torch.nn.init.uniform_(self.gc4.weight, a=-0.05, b=0.05)

    def forward(self, x, edge_index, batch=None, extract_embedding=False):
        adj = to_dense_adj(edge_index)
        x = x.float()
        x = F.relu(self.bn1(self.gc1(x, adj).transpose(2, 1)))
        x = x.transpose(1, 2)
        x = F.relu(self.bn2(self.gc2(x, adj).transpose(2, 1)))
        x = x.transpose(1, 2)
        x = F.relu(self.bn3(self.gc3(x, adj).transpose(2, 1)))
        x = x.transpose(1, 2)
        x = F.relu(self.bn4(self.gc4(x, adj).transpose(2, 1)))
        x = x.transpose(1, 2)
        x = x.reshape(-1, self.size)
        x = global_add_pool(x, batch)
        out = self.fc(x)
        if extract_embedding:
            return x
        else:
            return out


class GAT(torch.nn.Module):
    def __init__(self, args, heads=8):
        super(GAT, self).__init__()
        self.conv1 = GATConv(args.in_channels, args.dim, heads, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(args.dim * heads, args.dim, heads=1,
                             concat=False, dropout=0.6)

        self.lin1 = Linear(args.dim, args.dim)
        self.lin2 = Linear(args.dim, args.out_channels)

    def forward(self, x, edge_index, batch=None):
        x = x.float()
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x

class GIN(torch.nn.Module):
    def __init__(self, args):
        super(GIN, self).__init__()
        in_channels, dim, out_channels = args.in_channels, args.dim, args.out_channels
        self.conv1 = GINConv(
            nn=Sequential(
                Linear(in_channels, dim),
                BatchNorm1d(dim),
                ReLU(),
                Linear(dim, dim),
                ReLU())
        )

        self.conv2 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, out_channels)

    def forward(self, x, edge_index, batch=None):
        x = x.float()
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x