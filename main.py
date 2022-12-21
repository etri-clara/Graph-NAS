import sys
import math
import random
import torch
import numpy as np
from tqdm import tqdm, trange
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GDataLoader

from construction import kNNConstruction
from modules.defaults import GCN, GAT, GIN, GCNRegressor, GINRegressor, GNNMetric

from utils import *

model_map = {"GCN": GCN, "GAT": GAT, "GIN": GIN, "GNN_metric": GNNMetric}
encoder_train_map = {"triplet": triplet_train, "gnn": gnn_train_log}
encoder_eval_map = {"triplet": triplet_eval, "gnn": gnn_eval_baseline, "cgnn": gnn_eval_baseline}
regressor_map = {"GCN": GCNRegressor, "GIN": GINRegressor}


class GNNPredictor:
    def __init__(self, configs, data):
        self.params = configs
        self.data = data
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.train_loader, self.val_loader, self.test_loader = dataset(data,
                                                                       self.params.dataset.num_data,
                                                                       self.params.dataset.num_train,
                                                                       self.params.dataset.num_val)

        self.encoder = model_map[self.params.encoder.spec.name](self.params.encoder.spec).to(self.device)

        if self.params.encoder.type != 'cgnn':
            self.encoder_train = encoder_train_map[self.params.encoder.type]
        self.encoder_eval = encoder_eval_map[self.params.encoder.type]

        self.regressor = regressor_map[self.params.gog.regressor.spec.name](
            self.params.encoder.spec.dim, self.params.gog.regressor.spec.hidden_dim).to(self.device)
        self.train_mask = torch.tensor([i for i in range(self.params.dataset.num_train)])
        self.val_mask = torch.tensor([i for i in range(self.params.dataset.num_train,
                                                       self.params.dataset.num_train + self.params.dataset.num_val)])
        self.test_mask = torch.tensor([i for i in range(self.params.dataset.num_train + self.params.dataset.num_val,
                                                        self.params.dataset.num_data)])

    def reload(self):
        self.train_loader, self.val_loader, self.test_loader = dataset(self.data,
                                                                       self.params.dataset.num_data,
                                                                       self.params.dataset.num_train,
                                                                       self.params.dataset.num_val)

    def encode(self):

        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.params.encoder.training.lr)
        coeffs = Variable(torch.tensor([1., 3.]).to(self.device), requires_grad=True)
        coeffs_optimizer = torch.optim.SGD([coeffs], lr=1e-1, momentum=0.0)

        loss_hist = []
        val_best_model = self.encoder.state_dict()
        S = []

        if self.params.encoder.type == "cgnn":

            for data in self.train_loader:
                xshape = data.x.shape[0]
                l = [int(i / 2) for i in range(2 * xshape)]
                yshape = data.y.shape[0]
                mask = torch.tensor([i for i in range(yshape)]).to(self.device)

                # initialize
                adj = to_dense_adj(data.edge_index.cpu())[0]

                sp_adj = sp.coo_matrix(adj)
                S.append(sparse_mx_to_torch_sparse_tensor(normalize(sp_adj.astype(float))))

            for epoch in trange(1, self.params.encoder.training.epoch):
                for idx, data in enumerate(self.train_loader):
                    data = data.to(self.device)
                    out = self.encoder(data.x, data.edge_index, data.batch, embedding=False).view(yshape).sigmoid()
                    loss = loss_fcn(out.squeeze(), data.y.squeeze(), mask, S[idx], coeffs, self.device, False)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if epoch % 10 == 0:
                        out = self.encoder(data.x, data.edge_index, data.batch, embedding=False).view(yshape).sigmoid()
                        loss = loss_fcn(out.squeeze(), data.y.squeeze(), mask, S[idx], coeffs, self.device, False)
                        coeffs_optimizer.zero_grad()
                        loss.backward()
                        coeffs_optimizer.step()

                metric = self.encoder_eval(self.encoder, self.test_loader, self.device)

        elif self.params.encoder.type == "gnn":

            best_val_mse = math.inf
            for epoch in trange(self.params.encoder.training.epoch):

                loss = self.encoder_train(self.encoder, self.train_loader, optimizer, self.device, True)
                loss_hist.append(loss)
                val_mse, val_mae = self.encoder_eval(self.encoder, self.val_loader, self.device, 0)
                if best_val_mse > val_mse:
                    val_best_model = self.encoder.state_dict()
                    metric = self.encoder_eval(self.encoder, self.test_loader, self.device)
                    best_val_mse = val_mse

        elif self.params.encoder.type == "triplet":
            best_loss = math.inf
            metric = {}
            for epoch in trange(self.params.encoder.training.epoch):
                loss = self.encoder_train(self.encoder, self.train_loader, optimizer, self.device)
                loss_hist.append(loss)
                loss = self.encoder_eval(self.encoder, self.val_loader, self.device)
                if best_loss > loss:
                    val_best_model = self.encoder.state_dict()
                    best_loss = loss
            metric['loss'] = best_loss

        self.encoder.load_state_dict(val_best_model)

        return metric

    def gog_construction(self):
        full_loader = GDataLoader(self.data[:self.params.dataset.num_data], shuffle=False, batch_size=16)
        embs = []
        ys = []

        with torch.no_grad():
            for data in full_loader:
                data = data.to(self.device)
                emb = self.encoder(data.x, data.edge_index, data.batch, embedding=True)
                y = data.y.tolist()
                embs.extend(emb.tolist())
                ys.extend(y)

        embs = torch.tensor(embs)
        ys = torch.tensor(ys).view(self.params.dataset.num_data, -1)

        ys = ys[:, -1]

        edge_index = kNNConstruction(self.data, embs, self.params.gog.construction.type, self.params.gog.construction.k)

        self.G = Data(x=embs, edge_index=edge_index, y=ys,
                      train_mask=self.train_mask, val_mask=self.val_mask, test_mask=self.test_mask).to(self.device)

    def gog_regression(self):
        optimizer = torch.optim.Adam(self.regressor.parameters(), lr=self.params.gog.regressor.training.lr,
                                     weight_decay=self.params.gog.regressor.training.weight_decay)
        if self.params.gog.regressor.spec.base == "cgnn":
            metric = cgnn_total(self.regressor, self.G, optimizer, self.device,
                                self.params.dataset.num_data, self.params, self.params.gog.regressor.training.epoch)
        else:
            best_val_model = train_gog_regressor_log(self.regressor, optimizer,
                                                     self.params.gog.regressor.training.epoch, self.G)
            self.regressor.load_state_dict(best_val_model)
            metric = gog_eval_log(self.regressor, self.G, self.device)

            out = self.regressor(self.G)
            pred_acc = np.array([out[idx].item() for idx in range(self.params.dataset.num_data)])
            true_acc = np.array([self.G.y[idx].item() for idx in range(self.params.dataset.num_data)])

            top_arc_pred = np.argsort(pred_acc)[::-1]
            low_arc_pred = np.argsort(pred_acc)

            idx = 0
            while True:
                if idx == 5:
                    break
                ele = top_arc_pred[idx]
                if ele in self.test_mask.tolist():
                    self.test_mask = self.test_mask[self.test_mask != ele]
                    self.train_mask = torch.cat((self.train_mask, torch.tensor([ele])), dim=0)
                ele = low_arc_pred[idx]
                if ele in self.test_mask.tolist():
                    self.test_mask = self.test_mask[self.test_mask != ele]
                    self.train_mask = torch.cat((self.train_mask, torch.tensor([ele])), dim=0)

                idx += 1

        return metric

    def reset(self):
        self.train_loader, self.val_loader, self.test_loader = dataset_top_k(self.data,
                                                                             self.train_mask,
                                                                             self.val_mask,
                                                                             self.test_mask)


if __name__ == "__main__":
    config = Option(sys.argv[1])
    runs = int(sys.argv[2])
    with open('./data/permutation.npy', 'rb') as f:
        split = np.load(f)

    results = {}
    results3 = {}
    results5 = {}

    nas_data = torch.load("./data/nas201_sample.pt")

    for i in range(runs):
        data = [nas_data[j] for j in split[i]]
        seed_everything()
        print("----------------Run {}----------------".format(i + 1))
        tester = GNNPredictor(config, data)

        num = 5
        for i in range(num):
            print("    ------start encoding-------")
            # stage 1
            losses = tester.encode()

            # stage 2
            tester.gog_construction()
            print("\n    ------start regression-------")
            metrics = tester.gog_regression()

            if i != num - 1:
                tester.reset()

            if i == 0:
                for key in metrics:
                    if key not in results:
                        results[key] = [metrics[key]]
                    else:
                        results[key].append(metrics[key])

            elif i == 2:
                for key in metrics:
                    if key not in results3:
                        results3[key] = [metrics[key]]
                    else:
                        results3[key].append(metrics[key])

            elif i == 4:
                for key in metrics:
                    if key not in results5:
                        results5[key] = [metrics[key]]
                    else:
                        results5[key].append(metrics[key])

    print("1.encoder:", tester.params.encoder.spec.name,
          "  2.type:", tester.params.encoder.type,
          "  3.kNN constrution:", tester.params.gog.construction.type,
          "  4.regressor:", tester.params.gog.regressor.spec.name,
          "  5.base:", tester.params.gog.regressor.spec.base)

    print("==Without top-k evolution==")
    for key in results:
        print(key, ": {:.4f}({:.4f})".format(np.mean(results[key]), np.std(results[key])))
    print("")

    print("==With two top-10 evolution==")
    for key in results3:
        print(key, ": {:.4f}({:.4f})".format(np.mean(results3[key]), np.std(results3[key])))
    print("")

    print("==With four top-10 evolution==")
    for key in results5:
        print(key, ": {:.4f}({:.4f})".format(np.mean(results5[key]), np.std(results5[key])))
