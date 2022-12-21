import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import to_dense_adj
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, global_max_pool, GATConv, GCNConv, GraphConv, \
    TopKPooling, dense_diff_pool
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_sparse import spmm
from utils import *


class GCNRegressor(torch.nn.Module):
    def __init__(self, n_feat, n_hidden):
        super().__init__()
        self.conv1 = GCNConv(n_feat, n_hidden)
        self.conv2 = GCNConv(n_hidden, n_hidden)
        self.lin1 = Linear(n_hidden, n_hidden)
        self.lin2 = Linear(n_hidden, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)

        x = self.lin1(x).relu()
        x = self.lin2(x)

        return x.sigmoid()


class GINRegressor(torch.nn.Module):
    def __init__(self, n_feat, n_hidden):
        super().__init__()

        self.conv1 = GINConv(
            Sequential(Linear(n_feat, n_hidden), BatchNorm1d(n_hidden), ReLU(),
                       Linear(n_hidden, n_hidden), ReLU()))

        self.conv2 = GINConv(
            Sequential(Linear(n_hidden, n_hidden), BatchNorm1d(n_hidden), ReLU(),
                       Linear(n_hidden, n_hidden), ReLU()))

        self.conv3 = GINConv(
            Sequential(Linear(n_hidden, n_hidden), BatchNorm1d(n_hidden), ReLU(),
                       Linear(n_hidden, n_hidden), ReLU()))

        self.conv4 = GINConv(
            Sequential(Linear(n_hidden, n_hidden), BatchNorm1d(n_hidden), ReLU(),
                       Linear(n_hidden, n_hidden), ReLU()))

        self.conv5 = GINConv(
            Sequential(Linear(n_hidden, n_hidden), BatchNorm1d(n_hidden), ReLU(),
                       Linear(n_hidden, n_hidden), ReLU()))

        self.lin1 = Linear(n_hidden, n_hidden)
        self.lin2 = Linear(n_hidden, 1)

        self.lin = Linear(n_hidden, n_hidden)
        self.n_hidden = n_hidden

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.lin1(x).relu()
        x = self.lin2(x)

        return x.sigmoid()


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
        self.bn1 = torch.nn.BatchNorm1d(self.size)
        self.bn2 = torch.nn.BatchNorm1d(self.size)
        self.sigmoid = torch.nn.Sigmoid()
        self.fc = torch.nn.Linear(self.size, args.out_channels)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.uniform_(self.gc1.weight, a=-0.05, b=0.05)
        torch.nn.init.uniform_(self.gc2.weight, a=-0.05, b=0.05)
        torch.nn.init.uniform_(self.gc3.weight, a=-0.05, b=0.05)
        torch.nn.init.uniform_(self.gc4.weight, a=-0.05, b=0.05)

    def forward(self, x, edge_index, batch=None, embedding=False):
        adj = to_dense_adj(edge_index)
        x = x.float()
        x = F.relu(self.bn1(self.gc1(x, adj).transpose(2, 1)))
        x = x.transpose(1, 2)
        x = F.relu(self.bn2(self.gc2(x, adj).transpose(2, 1)))
        x = x.transpose(1, 2)
        x = x.reshape(-1, self.size)
        x = global_add_pool(x, batch)

        if embedding:
            return x
        return self.fc(x)


class GAT(torch.nn.Module):
    def __init__(self, args, heads=8):
        super(GAT, self).__init__()
        self.conv1 = GATConv(args.in_channels, args.dim, heads, dropout=0.6)
        self.conv2 = GATConv(args.dim * heads, args.dim, heads=1,
                             concat=False, dropout=0.6)

        self.lin1 = Linear(args.dim, args.dim)
        self.lin2 = Linear(args.dim, args.out_channels)

    def forward(self, x, edge_index, batch=None, embedding=False):
        x = x.float()
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_add_pool(x, batch)

        if embedding:
            return x

        out = self.lin1(x).relu()
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.lin2(out)

        return out


class GIN(torch.nn.Module):
    def __init__(self, args):
        super(GIN, self).__init__()
        in_channels, dim, out_channels = args.in_channels, args.dim, args.out_channels
        self.multi_layer = args.multi_layer
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

        if self.multi_layer:
            self.lin1 = Linear(dim, dim)
            self.lin2 = Linear(dim, out_channels)
        else:
            self.lin1 = Linear(dim, out_channels)

    def forward(self, x, edge_index, batch=None, embedding=False):
        x = x.float()
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = global_add_pool(x, batch)

        if not self.multi_layer:
            return self.lin1(x)

        out = self.lin1(x).relu()
        if embedding:
            return out
        out = self.lin2(out)
        return out


class GNNMetric:
    def __init__(self, args):
        raise NotImplementedError("TODO")

    def forward(self, x, edge_index, batch=None, embedding=False):
        raise NotImplementedError("TODO")


class SparseDropout(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, input):
        input_coal = input.coalesce()
        drop_val = F.dropout(input_coal._values(), self.p, self.training)
        return torch.sparse.FloatTensor(input_coal._indices(), drop_val, input.shape)


class MixedDropout(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dense_dropout = torch.nn.Dropout(p)
        self.sparse_dropout = SparseDropout(p)

    def forward(self, input):
        if input.is_sparse:
            return self.sparse_dropout(input)
        else:
            return self.dense_dropout(input)


class MixedLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Our fan_in is interpreted by PyTorch as fan_out (swapped dimensions)
        torch.nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            _, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.bias is None:
            if input.is_sparse:
                res = torch.sparse.mm(input, self.weight)
            else:
                res = input.matmul(self.weight)
        else:
            if input.is_sparse:
                res = torch.sparse.addmm(self.bias.expand(input.shape[0], -1), input, self.weight)
            else:
                res = torch.addmm(self.bias, input, self.weight)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)


class PPNP(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        in_channels, dim, out_channels = args.in_channels, args.dim, args.out_channels

        layers = 3
        bias = False
        drop_prob = 0.5
        self.alpha = 0.1
        self.iter = 10

        fcs = [MixedLinear(in_channels, dim, bias=False)]

        for i in range(layers):
            fcs.append(torch.nn.Linear(dim, dim, bias=bias))
        fcs.append(torch.nn.Linear(dim, dim, bias=bias))
        self.fcs = torch.nn.ModuleList(fcs)
        self.reg_params = list(self.fcs[0].parameters())

        if drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, out_channels)

    def _transform_features(self, attr_matrix: torch.sparse.FloatTensor):

        layer_inner = self.fcs[0](self.dropout(attr_matrix)).relu()
        for fc in self.fcs[1:-1]:
            layer_inner = fc(layer_inner).relu()
        res = self.fcs[-1](self.dropout(layer_inner))
        return res

    def forward(self, x, edge_index, batch=None, embedding=False):

        adj = to_dense_adj(edge_index.cpu())[0]
        adj = sp.coo_matrix(adj)

        propagation = PPRPowerIteration(adj, self.alpha, self.iter)

        x = x.float()
        local_logits = self._transform_features(x)

        length = x.shape[0]
        idx = torch.tensor([i for i in range(length)])
        final_logits = propagation(local_logits, idx)

        x = global_add_pool(final_logits, batch)
        x = self.lin1(x).relu()
        x = self.lin2(x)

        return x


class PPRPowerIteration(torch.nn.Module):
    def __init__(self, adj_matrix: sp.spmatrix, alpha: float, niter: int, drop_prob: float = None):
        super().__init__()
        self.alpha = alpha
        self.niter = niter

        M = calc_A_hat(adj_matrix)

        self.register_buffer('A_hat', sparse_matrix_to_torch((1 - alpha) * M))

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

    def forward(self, local_preds: torch.FloatTensor, idx: torch.LongTensor):
        preds = local_preds
        for _ in range(self.niter):
            A_drop = self.dropout(self.A_hat)
            preds = A_drop @ preds + self.alpha * local_preds
        return preds[idx]


class PPRExact(torch.nn.Module):
    def __init__(self, adj_matrix: sp.spmatrix, alpha: float, drop_prob: float = None):
        super().__init__()

        ppr_mat = calc_ppr_exact(adj_matrix, alpha)
        self.register_buffer('mat', torch.FloatTensor(ppr_mat))

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

    def forward(self, predictions: torch.FloatTensor, idx: torch.LongTensor):
        return self.dropout(self.mat[idx]) @ predictions


########pooling
class SAGPool(torch.nn.Module):
    def __init__(self, in_channels, ratio=0.8, non_linearity=torch.tanh):
        super(SAGPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.conv = GINConv(Sequential(Linear(in_channels, 1)))
        self.non_linearity = non_linearity

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        # x = x.unsqueeze(-1) if x.dim() == 1 else x
        score = self.conv(x, edge_index).squeeze()

        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm
