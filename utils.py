import os
import json
import random

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import r2_score
from torch_geometric.loader import DataLoader
from scipy.stats import kendalltau, spearmanr, rankdata


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


def dataset(data, num_data, num_train, num_val):
    selected_data = data[:num_data]

    train_data = selected_data[:num_train]
    val_data = selected_data[num_train:num_train + num_val]
    test_data = selected_data[num_train + num_val:]

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16)
    val_dataloader = DataLoader(val_data, shuffle=False, batch_size=16)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=16)

    return train_dataloader, val_dataloader, test_dataloader


def dataset_metric(data, num_data, num_train, num_val):
    # metric learning에 적합한 dataloader
    raise NotImplementedError("여기도 구현하세요~")


def seed_everything(seed: int = 29):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def gnn_train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch, embedding=False)
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
def gog_eval(model, graph, plot=0):
    out = model(graph)
    loss = F.mse_loss(out[graph.test_mask], graph.y[graph.test_mask].reshape(-1, 1)).item()
    mae_loss = F.l1_loss(out[graph.test_mask], graph.y[graph.test_mask].reshape(-1, 1)).item()
    r2 = r2_score(graph.y[graph.test_mask].reshape(-1, 1).tolist(), out[graph.test_mask].tolist())

    tot = len(list(out))
    pred_acc = [out[i].item() for i in range(tot)]
    true_acc = [graph.y[i].item() for i in range(tot)]

    pred_rank = rankdata(pred_acc)
    true_rank = rankdata(true_acc)
    tau, p1 = kendalltau(pred_rank, true_rank)
    coeff, p2 = spearmanr(pred_rank, true_rank)

    metric = {'knn test mse': loss, 'knn test mae': mae_loss, 'knn test r2': r2,
              'kendall tau': tau, 'spearmanr coeff': coeff}

    if plot:
        plt.plot(true_rank, pred_rank, 'o', markersize=0.1)

    return metric
