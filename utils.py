import os
import json
import math
import random

import torch
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import r2_score
from torch.autograd import Variable
from torch.utils.data import DataLoader
from gpytorch import logdet, solve
from scipy.stats import kendalltau, spearmanr, rankdata
from torch_geometric.loader import DataLoader as GDataLoader
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm, trange
            
class Option(dict):
    def __init__(self, *args, **kwargs):
        args = [arg if isinstance(arg, dict) else json.loads(open(arg).read())
                for arg in args]
        super(Option, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    if isinstance(v, dict):
                        self[k] = Option(v)
                    else:
                        self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                if isinstance(v, dict):
                    self[k] = Option(v)
                else:
                    self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Option, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Option, self).__delitem__(key)
        del self.__dict__[key]


def prepare_mp(g):
    """
    Explicitly materialize the CSR, CSC and COO representation of the given graph
    so that they could be shared via copy-on-write to sampler workers and GPU
    trainers.
    This is a workaround before full shared memory support on heterogeneous graphs.
    """

    g.in_degrees(0)
    g.out_degrees(0)
    g.find_edges([0])


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
    rL=rL.float()
    S = S.to_dense()
    Gamma = (torch.eye(S.size(0)).to(device) - torch.tanh(coeffs[0]) * S.to(device)) * torch.exp(coeffs[1])
    cp_idx = setdiff(len(S), idx)
    #torch.matmul(Gamma[idx, :][:, idx], rL)
    #solve(Gamma[cp_idx, :][:, cp_idx], torch.matmul(Gamma[cp_idx, :][:, idx], rL))
    #torch.matmul(Gamma[idx, :][:, cp_idx], solve(Gamma[cp_idx, :][:, cp_idx], torch.matmul(Gamma[cp_idx, :][:, idx], rL)))
    
    loss1 = rL.dot(torch.matmul(Gamma[idx, :][:, idx], rL) - torch.matmul(Gamma[idx, :][:, cp_idx],
                                                              solve(Gamma[cp_idx, :][:, cp_idx],
                                                                    torch.matmul(Gamma[cp_idx, :][:, idx], rL))))  # HJ
    loss2 = 0.
    if add_logdet: loss2 = logdet(Gamma) - logdet(Gamma[cp_idx, :][:, cp_idx])
    l = loss1 - loss2
    
    return l / len(idx)

"""
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
"""

def dataset(data, num_data, num_train, num_val):
    selected_data = data[:num_data]

    train_data = selected_data[:num_train]
    val_data = selected_data[num_train:num_train + num_val]
    test_data = selected_data[num_train + num_val:]
    
    train_dataloader = GDataLoader(train_data, shuffle=True, batch_size=16)
    val_dataloader = GDataLoader(val_data, shuffle=False, batch_size=16)
    test_dataloader = GDataLoader(test_data, shuffle=False, batch_size=16)

    return train_dataloader, val_dataloader, test_dataloader

def dataset_top_k(data, train_mask, val_mask, test_mask):

    train_data = [data[i] for i in train_mask]
    val_data = [data[i] for i in val_mask]
    test_data = [data[i] for i in test_mask]
    
    train_dataloader = GDataLoader(train_data, shuffle=True, batch_size=16)
    val_dataloader = GDataLoader(val_data, shuffle=False, batch_size=16)
    test_dataloader = GDataLoader(test_data, shuffle=False, batch_size=16)

    return train_dataloader, val_dataloader, test_dataloader

def dataset2(data, num_data, num_train, num_val):


    dataloader = GDataLoader(data, shuffle=False, batch_size=num_data+1)
    
    for datas in dataloader:
        one_loader = datas
        temp=datas.batch.tolist()
    
    train_mask, val_mask, test_mask = [], [], []
    
    for idx, item in enumerate(temp):
        if temp[idx] < num_train: train_mask.append(idx)
        elif temp[idx]<num_val: val_mask.append(idx)
        else: test_mask.append(idx)
    

    return one_loader, torch.tensor(train_mask), torch.tensor(val_mask), torch.tensor(test_mask)


def l2_sim_mat(x, y):
    s1 = np.sum(x ** 2, axis=1, keepdims=True)
    s2 = np.sum(y ** 2, axis=1, keepdims=True)
    return np.sqrt(np.abs(-2 * x.dot(y.T) + s1 + s2.T))


def get_triplet(data, batch_size=20, num_nn=5):
    num_data = len(data)
    ys = []
    for g in data:
        ys.append(np.array(g.y.view(-1)))
    ys = np.array(ys)
    sim_mat = l2_sim_mat(ys, ys)
    list_nn = []
    dist_nn = []
    for idx in range(num_data):
        list_nn.append(np.argsort(sim_mat[idx])[1:])
        dist_nn.append(np.sort(sim_mat[idx])[1:])

    triplets, dists = [], []

    for idx in range(num_data):
        rand_smp = np.random.choice(num_data - num_nn - 1,
                                    batch_size - num_nn - 1, replace=False) + num_nn
        rand_smp.sort()
        rand_smp = np.concatenate((np.arange(0, num_nn), rand_smp))
        list_smp = list_nn[idx][rand_smp]
        dist_smp = dist_nn[idx][rand_smp]

        for i in range(len(list_smp)):
            for j in range(i + 1, len(list_smp)):
                triplets.append([data[idx], data[i], data[j]])
                dists.append([dist_smp[i], dist_smp[j]])
    assert len(triplets) == len(dists)

    return list(zip(triplets, dists))


def dataset_metric(data, num_data, num_train, num_val):
    selected_data = data[:num_data]

    train_data = selected_data[:num_train]
    val_data = selected_data[num_train:num_train + num_val]
    test_data = selected_data[num_train + num_val:]

    train_triplet = get_triplet(train_data)
    val_triplet = get_triplet(val_data)

    train_dataloader = GDataLoader(train_triplet, shuffle=True, batch_size=1)
    val_dataloader = GDataLoader(val_triplet, shuffle=False, batch_size=1)
    test_dataloader = GDataLoader(test_data, shuffle=False, batch_size=16)

    return train_dataloader, val_dataloader, test_dataloader


def seed_everything(seed: int = 29):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def gnn_train(model, loader, optimizer, device, baseline=False):
    model.train()
    total_loss = 0
    if baseline==False:
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            output = model(data.x, data.edge_index, data.batch, embedding=False)
            loss = F.mse_loss(output, data.y.float().reshape(output.shape[0], -1))
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * len(data)
    else:
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data.x, data.edge_index, data.batch, embedding=False).sigmoid()
            loss = F.mse_loss(output, data.y.float().reshape(output.shape[0], -1))
       
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * len(data)

    return total_loss / len(loader)

def gnn_train_log(model, loader, optimizer, device, baseline=False):
    model.train()
    total_loss = 0
    if baseline==False:
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            output = model(data.x, data.edge_index, data.batch, embedding=False)
            loss = F.mse_loss(torch.exp(output-1), torch.exp(data.y.float().reshape(output.shape[0], -1)-1))
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * len(data)
    else:
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data.x, data.edge_index, data.batch, embedding=False).sigmoid()
            loss = F.mse_loss(output, data.y.float().reshape(output.shape[0], -1))
       
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * len(data)

    return total_loss / len(loader)



@torch.no_grad()
def gnn_eval(model, loader, device):
    model.eval()
    total_mse = 0
    total_mae = 0
    
    
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch, embedding=False)
        mse = F.mse_loss(output, data.y.float().reshape(output.shape[0], -1))
        mae = F.l1_loss(output, data.y.float().reshape(output.shape[0], -1))

        
        total_mse += float(mse) * len(data)
        total_mae += float(mae) * len(data)

    return total_mse / len(loader), total_mae / len(loader)

@torch.no_grad()
def gnn_eval_baseline(model, loader, device, default=1):
    model.eval()
    pred, true=[], []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch, embedding=False).sigmoid()    
        pred.extend(torch.transpose(out, 0, 1)[-1].tolist())
        temp = torch.reshape(data.y, [out.shape[0], out.shape[1]])
        true.extend(torch.transpose(temp, 0, 1)[-1].tolist())

    tot = len(true)
    pred_acc=np.array(pred)
    true_acc=np.array(true)
    
    
    #print(pred_acc, true_acc)
    if default==0:
        return np.square(np.subtract(pred_acc, true_acc)).mean(), np.abs(np.subtract(pred_acc, true_acc)).mean()
    
    loss = np.square(np.subtract(pred_acc, true_acc)).mean()
    mae_loss = np.abs(np.subtract(pred_acc, true_acc)).mean()
    r2 = r2_score(pred_acc, true_acc)

    pred_rank = rankdata(pred_acc)
    true_rank = rankdata(true_acc)
    tau, p1 = kendalltau(pred_rank, true_rank)
    coeff, p2 = spearmanr(pred_rank, true_rank)
    
    top_arc_pred = np.argsort(pred_acc)[::-1]
    top_arc_true = np.argsort(true_acc)[::-1]
           
    #print(top_arc_pred[:10], top_arc_true[:10])
    #print(true_acc[top_arc_pred[:10]], true_acc[top_arc_true[:10]])
    precision_at_1 = precision(top_arc_true[:1], top_arc_pred[:1], 1)
    precision_at_10 = precision(top_arc_true[:10], top_arc_pred[:10], 10)
    precision_at_50 = precision(top_arc_true[:50], top_arc_pred[:50], 50)
    precision_at_100 = precision(top_arc_true[:100], top_arc_pred[:100], 100)
    metric = {'knn test mse': loss, 'knn test mae': mae_loss, 'knn test r2': r2,
              'kendall tau': tau, 'spearmanr coeff': coeff, 'top_1_correct': precision_at_1,
              'p@10': precision_at_10, 'p@50': precision_at_50,
              'p@100': precision_at_100, 'top acc': true_acc[top_arc_pred[0]]}
    return metric

@torch.no_grad()
def gnn_eval_baseline_log(model, loader, device, default=1):
    model.eval()
    pred, true=[], []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch, embedding=False).sigmoid()    
        pred.extend(torch.transpose(out, 0, 1)[-1].tolist())
        temp = torch.reshape(data.y, [out.shape[0], out.shape[1]])
        true.extend(torch.transpose(temp, 0, 1)[-1].tolist())

    tot = len(true)
    pred_acc=np.array(pred)
    true_acc=np.array(true)
    
    
    #print(pred_acc, true_acc)
    if default==0:
        return np.square(np.subtract(pred_acc, true_acc)).mean(), np.abs(np.subtract(pred_acc, true_acc)).mean()
    
    loss = np.square(np.subtract(np.exp(pred_acc-1), np.exp(true_acc-1))).mean()
    mae_loss = np.abs(np.subtract(np.exp(pred_acc-1), np.exp(true_acc-1))).mean()
    r2 = r2_score(pred_acc, true_acc)

    pred_rank = rankdata(pred_acc)
    true_rank = rankdata(true_acc)
    tau, p1 = kendalltau(pred_rank, true_rank)
    coeff, p2 = spearmanr(pred_rank, true_rank)
    
    top_arc_pred = np.argsort(pred_acc)[::-1]
    top_arc_true = np.argsort(true_acc)[::-1]
           
    #print(top_arc_pred[:10], top_arc_true[:10])
    #print(true_acc[top_arc_pred[:10]], true_acc[top_arc_true[:10]])
    precision_at_1 = precision(top_arc_true[:1], top_arc_pred[:1], 1)
    precision_at_10 = precision(top_arc_true[:10], top_arc_pred[:10], 10)
    precision_at_50 = precision(top_arc_true[:50], top_arc_pred[:50], 50)
    precision_at_100 = precision(top_arc_true[:100], top_arc_pred[:100], 100)
    metric = {'knn test mse': loss, 'knn test mae': mae_loss, 'knn test r2': r2,
              'kendall tau': tau, 'spearmanr coeff': coeff, 'top_1_correct': precision_at_1,
              'p@10': precision_at_10, 'p@50': precision_at_50,
              'p@100': precision_at_100, 'top acc': true_acc[top_arc_pred[0]]}
    return metric

def train_gog_regressor(model, optimizer, num_epoch, G):
    best_val_mae = math.inf
    best_val_model = model.state_dict()
    for epoch in trange(num_epoch):
        optimizer.zero_grad()
        out = model(G)
        loss = F.mse_loss(out[G.train_mask], G.y[G.train_mask].reshape(-1, 1))

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            val_mae = F.l1_loss(out[G.val_mask], G.y[G.val_mask].reshape(-1, 1))
            if val_mae < best_val_mae:
                best_val_mae = float(val_mae)
                best_val_model = model.state_dict()
    return best_val_model


def train_gog_regressor_log(model, optimizer, num_epoch, G):
    best_val_mae = math.inf
    best_val_model = model.state_dict()
    for epoch in trange(num_epoch):
        optimizer.zero_grad()
        out = model(G)
        loss = F.mse_loss(torch.exp(out[G.train_mask]-1), torch.exp(G.y[G.train_mask].reshape(-1, 1)-1))

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            val_mae = F.l1_loss(torch.exp(out[G.val_mask]-1), torch.exp(G.y[G.val_mask]-1).reshape(-1, 1))
            if val_mae < best_val_mae:
                best_val_mae = float(val_mae)
                best_val_model = model.state_dict()
    return best_val_model

def train_gog_regressor_top_k(model, optimizer, num_epoch, G):
    best_val_mae = math.inf
    best_val_model = model.state_dict()
    num = 5
    masks = G.train_mask.tolist() + G.val_mask.tolist()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for i in range(num):
        for epoch in trange(num_epoch):
            optimizer.zero_grad()
            out = model(G)
            loss = F.mse_loss(out[G.train_mask], G.y[G.train_mask].reshape(-1, 1))
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                val_mae = F.l1_loss(out[G.val_mask], G.y[G.val_mask].reshape(-1, 1))
                if val_mae < best_val_mae:
                    best_val_mae = float(val_mae)
                    best_val_model = model.state_dict()
            
        if i!=num-1:
            tot = len(list(out))
            pred_acc = np.array([out[i].item() for i in range(tot)])
            top_acc_pred = np.argsort(pred_acc)[::-1]
            cand = []
            idx = 0
            
            while True:
                if len(cand) == 10: break
                ele = top_acc_pred[idx]
                if ele not in masks:
                    cand.append(ele)
                    masks = cand + [ele]
                    G.train_mask = torch.cat((G.train_mask, torch.tensor([ele]).to(device)), dim = 0)
                    G.test_mask = G.test_mask[G.test_mask != ele]          
                idx +=1
            
    return best_val_model, G


def cgnn_total(model, G, optimizer, device, num_data, params, epochs):
    #g = dgl.graph(
    #    (G.edge_index[0].clone().detach(), G.edge_index[1].clone().detach()))
    #g.ndata['features'] = G.x.clone().detach().requires_grad_(True)
    #prepare_mp(g)
    l = [int(i / 2) for i in range(2 * num_data)]
    #g.add_edges(l, l)

    #sampler = NeighborSampler(g, [25, 25])
    #last_loader = DataLoader(dataset=G.train_mask.cpu().numpy(), batch_size=16, collate_fn=sampler.sample_blocks,
    #                          drop_last=False, num_workers=0)

    #G.addedge
    G.edge_index=torch.cat([torch.tensor([l,l]).to(device), G.edge_index], dim=1)
    
    coeffs = Variable(torch.tensor([1., 3.]).to(device), requires_grad=True)
    coeffs_optimizer = torch.optim.SGD([coeffs], lr=1e-1, momentum=0.0)

    # initialize
    adj=to_dense_adj(G.edge_index.cpu())[0]
    
    sp_adj = sp.coo_matrix(adj)
    S = sparse_mx_to_torch_sparse_tensor(normalize(sp_adj.astype(float)))

    name = params.gog.regressor.spec.base
      
    for epoch in trange(epochs):

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.

        out = model(G)

        loss = loss_fcn(out[G.train_mask].squeeze(), G.y[G.train_mask].squeeze(), G.train_mask,
                        S, coeffs, device, False)

        optimizer.zero_grad()
        loss.backward()  # problem
        optimizer.step()

        if epoch % 10 == 0:
            model.train()
            out = model(G)
            loss = loss_fcn(out[G.train_mask].squeeze(), G.y[G.train_mask].squeeze(),
                            G.train_mask, S, coeffs, device, True)
            coeffs_optimizer.zero_grad()
            loss.backward()
            coeffs_optimizer.step()

            # print("Epoch: ", epoch, " loss: ", loss)

        metric = gog_eval(model, G)
        
    return metric

def triplet_train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    eps = 1e-10
    for triplet, dists in loader:
        optimizer.zero_grad()
        a, p, n = triplet[0].to(device), triplet[1].to(device), triplet[2].to(device)
        gt_dist_p = dists[0].to(device) + eps
        gt_dist_n = dists[1].to(device) + eps
        emb_a = model(a.x, a.edge_index, a.batch, embedding=True)
        emb_p = model(p.x, p.edge_index, p.batch, embedding=True)
        emb_n = model(n.x, n.edge_index, n.batch, embedding=True)

        emb_dist_p = torch.norm(torch.abs(emb_a - emb_p), p=2) + eps
        emb_dist_n = torch.norm(torch.abs(emb_a - emb_n), p=2) + eps

        loss = (torch.log(emb_dist_p / emb_dist_n) - torch.log(gt_dist_p / gt_dist_n)).pow(2)
        loss.backward()
        optimizer.step()

        total_loss += float(loss)

    return total_loss / len(loader)


@torch.no_grad()
def triplet_eval(model, loader, device):
    model.eval()
    total_loss = 0
    eps = 1e-10
    for triplet, dists in loader:
        a, p, n = triplet[0].to(device), triplet[1].to(device), triplet[2].to(device)
        gt_dist_p = dists[0].to(device) + eps
        gt_dist_n = dists[1].to(device) + eps
        emb_a = model(a.x, a.edge_index, a.batch, embedding=True)
        emb_p = model(p.x, p.edge_index, p.batch, embedding=True)
        emb_n = model(n.x, n.edge_index, n.batch, embedding=True)

        emb_dist_p = torch.norm(torch.abs(emb_a - emb_p), p=2) + eps
        emb_dist_n = torch.norm(torch.abs(emb_a - emb_n), p=2) + eps

        loss = (torch.log(emb_dist_p / emb_dist_n) - torch.log(gt_dist_p / gt_dist_n)).pow(2)
        total_loss += float(loss)

    return total_loss / len(loader)

@torch.no_grad()
def gog_eval(model, graph, plot=0):
    out = model(graph)
    loss = F.mse_loss(out[graph.test_mask], graph.y[graph.test_mask].reshape(-1, 1)).item()
    mae_loss = F.l1_loss(out[graph.test_mask], graph.y[graph.test_mask].reshape(-1, 1)).item()
    r2 = r2_score(graph.y[graph.test_mask].reshape(-1, 1).tolist(), out[graph.test_mask].tolist())

    tot = len(list(out))
    pred_acc = np.array([out[i].item() for i in range(tot)])
    true_acc = np.array([graph.y[i].item() for i in range(tot)])

    pred_rank = rankdata(pred_acc)
    true_rank = rankdata(true_acc)
    tau, p1 = kendalltau(pred_rank, true_rank)
    coeff, p2 = spearmanr(pred_rank, true_rank)
    
    top_arc_pred = np.argsort(pred_acc)[::-1]
    top_arc_true = np.argsort(true_acc)[::-1]
    
    
    #print(top_arc_pred[:10], top_arc_true[:10])
    #print(true_acc[top_arc_pred[:10]], true_acc[top_arc_true[:10]])
    precision_at_1 = precision(top_arc_true[:1], top_arc_pred[:1], 1)
    precision_at_10 = precision(top_arc_true[:10], top_arc_pred[:10], 10)
    precision_at_50 = precision(top_arc_true[:50], top_arc_pred[:50], 50)
    precision_at_100 = precision(top_arc_true[:100], top_arc_pred[:100], 100)
    metric = {'knn test mse': loss, 'knn test mae': mae_loss, 'knn test r2': r2,
              'kendall tau': tau, 'spearmanr coeff': coeff, 'top_1_correct': precision_at_1,
              'p@10': precision_at_10, 'p@50': precision_at_50,
              'p@100': precision_at_100, 'top acc': true_acc[top_arc_pred[0]]}

    if plot:
        plt.plot(true_rank, pred_rank, 'o', markersize=0.1)

    return metric


@torch.no_grad()
def gog_eval_log(model, graph, plot=0):
    out = model(graph)
    loss = F.mse_loss(torch.exp(out[graph.test_mask]-1), torch.exp(graph.y[graph.test_mask].reshape(-1, 1)-1)).item()
    mae_loss = F.l1_loss(torch.exp(out[graph.test_mask]-1), torch.exp(graph.y[graph.test_mask].reshape(-1, 1)-1)).item()
    r2 = r2_score(graph.y[graph.test_mask].reshape(-1, 1).tolist(), out[graph.test_mask].tolist())

    tot = len(list(out))
    pred_acc = np.array([out[i].item() for i in range(tot)])
    true_acc = np.array([graph.y[i].item() for i in range(tot)])

    pred_rank = rankdata(pred_acc)
    true_rank = rankdata(true_acc)
    tau, p1 = kendalltau(pred_rank, true_rank)
    coeff, p2 = spearmanr(pred_rank, true_rank)
    
    top_arc_pred = np.argsort(pred_acc)[::-1]
    top_arc_true = np.argsort(true_acc)[::-1]
    
    
    #print(top_arc_pred[:10], top_arc_true[:10])
    #print(true_acc[top_arc_pred[:10]], true_acc[top_arc_true[:10]])
    precision_at_1 = precision(top_arc_true[:1], top_arc_pred[:1], 1)
    precision_at_10 = precision(top_arc_true[:10], top_arc_pred[:10], 10)
    precision_at_50 = precision(top_arc_true[:50], top_arc_pred[:50], 50)
    precision_at_100 = precision(top_arc_true[:100], top_arc_pred[:100], 100)
    metric = {'knn test mse': loss, 'knn test mae': mae_loss, 'knn test r2': r2,
              'kendall tau': tau, 'spearmanr coeff': coeff, 'top_1_correct': precision_at_1,
              'p@10': precision_at_10, 'p@50': precision_at_50,
              'p@100': precision_at_100, 'top acc': true_acc[top_arc_pred[0]]}

    if plot:
        plt.plot(true_rank, pred_rank, 'o', markersize=0.1)

    return metric

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_Gamma(alpha, beta, S):
    return beta * torch.eye(S.size(0)) - beta * alpha * S

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def precision(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(k)
    return result



#####

def sparse_matrix_to_torch(X):
    coo = X.tocoo()
    indices = np.array([coo.row, coo.col])
    return torch.sparse.FloatTensor(
            torch.LongTensor(indices),
            torch.FloatTensor(coo.data),
            coo.shape)


def matrix_to_torch(X):
    if sp.issparse(X):
        return sparse_matrix_to_torch(X)
    else:
        return torch.FloatTensor(X)
    
    
def calc_A_hat(adj_matrix: sp.spmatrix) -> sp.spmatrix:
    nnodes = adj_matrix.shape[0]
    
    A = adj_matrix + sp.eye(nnodes)
    D_vec = np.sum(A, axis=1).A1
    D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
    D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)
    return D_invsqrt_corr @ A @ D_invsqrt_corr


def calc_ppr_exact(adj_matrix: sp.spmatrix, alpha: float) -> np.ndarray:
    nnodes = adj_matrix.shape[0]
    M = calc_A_hat(adj_matrix)
    A_inner = sp.eye(nnodes) - (1 - alpha) * M
    return alpha * np.linalg.inv(A_inner.toarray())

