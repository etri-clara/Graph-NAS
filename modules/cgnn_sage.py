import sys
sys.path.append("..")
import warnings

import dgl
import tqdm
import torch
import numpy as np
import torch.nn as nn
from torch import matmul
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F

from sklearn.metrics import r2_score
from gpytorch import logdet, solve
from utils import lp_refine, R2


warnings.filterwarnings("ignore")


def prepare_mp(g):
    """
    Explicitly materialize the CSR, CSC and COO representation of the given graph
    so that they could be shared via copy-on-write to sampler workers and GPU
    trainers.
    This is a workaround before full shared memory support on heterogeneous graphs.
    """

    g.in_degrees(0)  # HJ
    g.out_degrees(0)  # HJ
    g.find_edges([0])


def compute_r2(pred, labels):
    """
    Compute the R2 of prediction given the labels.
    """
    # return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)
    return r2_score(labels.cpu().detach().numpy(), pred.cpu().detach().numpy())


def evaluate(model, g, inputs, labels, val_mask, batch_size, device):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the R2 for.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with torch.no_grad():
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    return compute_r2(pred[val_mask], labels[val_mask])


def evaluate_test(model, g, inputs, labels, test_mask, batch_size, device, lp_dict, coeffs, meta):
    model.eval()
    with torch.no_grad():
        pred = model.inference(g, inputs, batch_size, device).view(-1)

    output = pred.cuda()
    labels = labels.cuda()
    idx_test = lp_dict['idx_test']
    idx_train = lp_dict['idx_train']
    adj = lp_dict['adj']

    labels, output, adj = labels.cpu(), output.cpu(), adj.cpu()
    loss = F.mse_loss(output[idx_test].squeeze(), labels[idx_test].squeeze())
    r2_test = compute_r2(output[test_mask], labels[test_mask])
    lp_output = lp_refine(idx_test, idx_train, labels, output, adj, torch.tanh(coeffs[0]).item(),
                          torch.exp(coeffs[1]).item())
    lp_r2_test = compute_r2(lp_output, labels[idx_test])
    lp_output_raw_conv = lp_refine(idx_test, idx_train, labels, output, adj)
    lp_r2_test_raw_conv = R2(lp_output_raw_conv, labels[idx_test])

    print("------------")
    print("election year {}".format(meta))
    print("loss:", loss.item())
    print("raw_r2:", r2_test)
    print("refined_r2:", lp_r2_test)
    print("refined_r2_raw_conv:", lp_r2_test_raw_conv)
    print("------------")

    model.train()

    return lp_r2_test


def load_subtensor(g, labels, seeds, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = g.ndata['features'][input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels


def setdiff(n, idx):
    idx = idx.cpu().detach().numpy()
    cp_idx = np.setdiff1d(np.arange(n), idx)
    return cp_idx


def loss_fcn(output, labels, idx, S, coeffs, device, add_logdet):
    rL = labels - output
    S = S.to_dense()

    Gamma = (torch.eye(S.size(0)).to(device) - torch.tanh(coeffs[0]) * S.to(device)) * torch.exp(coeffs[1])
    cp_idx = setdiff(len(S), idx)

    loss1 = rL.dot(matmul(Gamma[idx, :][:, idx], rL) - matmul(Gamma[idx, :][:, cp_idx],
                                                              solve(Gamma[cp_idx, :][:, cp_idx],
                                                                    matmul(Gamma[cp_idx, :][:, idx], rL))))  # HJ
    loss2 = 0.
    if add_logdet: loss2 = logdet(Gamma) - logdet(Gamma[cp_idx, :][:, cp_idx])
    l = loss1 - loss2

    return l / len(idx)


class NeighborSampler(object):
    def __init__(self, g, fanouts):
        self.g = g
        self.fanouts = fanouts

    def sample_blocks(self, seeds):
        seeds = torch.cuda.LongTensor(np.asarray(seeds))
        blocks = []
        for fanout in self.fanouts:
            # For each seed node, sample ``fanout`` neighbors.
            frontier = dgl.sampling.sample_neighbors(self.g, seeds, fanout, replace=True)
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, seeds)
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]

            blocks.insert(0, block)
        return blocks


class SAGERegressor(nn.Module):
    def __init__(self, n_feat, n_hidden):

        super().__init__()
        self.n_layers = 2
        self.n_hidden = n_hidden
        self.n_classes = 1
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(n_feat, n_hidden, 'mean'))
        # for i in range(1, self.n_layers - 1):
        #    self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, 1, 'mean'))
        self.dropout = nn.Dropout(0.5)
        self.activation = F.sigmoid  # ReLU

    def forward(self, blocks, x):
        # print(blocks) #HJ
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[:block.number_of_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)

        return h

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        nodes = torch.arange(g.number_of_nodes())
        for l, layer in enumerate(self.layers):
            y = torch.zeros(g.number_of_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            for start in tqdm.trange(0, len(nodes), batch_size):
                end = start + batch_size
                batch_nodes = nodes[start:end].to(device)
                block = dgl.to_block(dgl.in_subgraph(g, batch_nodes), batch_nodes)
                input_nodes = block.srcdata[dgl.NID]

                h = x[input_nodes].to(device)
                h_dst = h[:block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[start:end] = h.cpu()

            x = y
        return y




