import sys
import math
import random
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from construction import kNNConstruction
from modules import GCN, GAT, GIN, GCNRegressor, GINRegressor, GNNMetric
from utils import Option, dataset, dataset_metric, seed_everything, gnn_train, gnn_eval, gog_eval

model_map = {"GCN": GCN, "GAT": GAT, "GIN": GIN, "GNN_metric": GNNMetric}
regressor_map = {"GCN": GCNRegressor, "GIN": GINRegressor}


class GNNPredictor:
    def __init__(self, configs, data):
        self.params = configs
        self.data = data
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.params.encoder.type == "metric":
            self.train_loader, self.val_loader, self.test_loader = dataset_metric(data,
                                                                                  self.params.dataset.num_data,
                                                                                  self.params.dataset.num_train,
                                                                                  self.params.dataset.num_val)
        else:
            self.train_loader, self.val_loader, self.test_loader = dataset(data,
                                                                           self.params.dataset.num_data,
                                                                           self.params.dataset.num_train,
                                                                           self.params.dataset.num_val)

        self.encoder = model_map[self.params.encoder.spec.name](self.params.encoder.spec).to(self.device)
        self.regressor = regressor_map[self.params.gog.regressor.spec.name](
            self.params.encoder.spec.dim, self.params.gog.regressor.spec.hidden_dim).to(self.device)

    def reload(self):
        self.train_loader, self.val_loader, self.test_loader = dataset(self.data,
                                                                       self.params.dataset.num_data,
                                                                       self.params.dataset.num_train,
                                                                       self.params.dataset.num_val)

    def encode_metric_learing(self):
        # metric learning 학습시키기
        # 여기서, metric learning 모델을 학습시킨 뒤에, 그 모델에서 얻은 임베딩 뒤에 MLP를 붙여서 regression을 한 후에,
        # test mse, test mae까지 리턴 해야 포맷이 맞음
        raise NotImplementedError("구현하세요 현주씨~")

    def encode(self):
        best_val_mse = math.inf
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.params.encoder.training.lr)
        loss_hist = []
        val_best_model = self.encoder.state_dict()
        for epoch in tqdm(range(self.params.encoder.training.epoch)):
            loss = gnn_train(self.encoder, self.train_loader, optimizer, self.device)
            loss_hist.append(loss)
            val_mse, val_mae = gnn_eval(self.encoder, self.val_loader, self.device)
            if best_val_mse > val_mse:
                val_best_model = self.encoder.state_dict()
                test_mse, test_mae = gnn_eval(self.encoder, self.test_loader, self.device)
                best_val_mse = val_mse
            if (epoch + 1) % 100 == 0:
                print("Epoch {} ==> Loss: {:.4f}".format(epoch + 1, loss))

        self.encoder.load_state_dict(val_best_model)

        return float(test_mse), float(test_mae)

    def gog_construction(self):
        full_loader = DataLoader(self.data[:self.params.dataset.num_data], shuffle=False, batch_size=16)
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
        ys = torch.tensor(ys)[:, 7]
        tot, _train, _val = self.params.dataset.num_data, self.params.dataset.num_train, self.params.dataset.num_val
        train_mask = torch.tensor([idx for idx in range(_train)])
        val_mask = torch.tensor([idx for idx in range(_train, _train + _val)])
        test_mask = torch.tensor([idx for idx in range(_train + _val, tot)])

        edge_index = kNNConstruction(self.data, embs, self.params.gog.construction.type, self.params.gog.construction.k)

        self.G = Data(x=embs, edge_index=edge_index, y=ys,
                      train_mask=train_mask, val_mask=val_mask, test_mask=test_mask).to(self.device)

    def gog_regression(self):
        optimizer = torch.optim.Adam(self.regressor.parameters(), lr=self.params.gog.regressor.training.lr,
                                     weight_decay=self.params.gog.regressor.training.weight_decay)

        best_val_mae = math.inf
        for epoch in tqdm(range(self.params.gog.regressor.training.epoch)):
            optimizer.zero_grad()
            out = self.regressor(self.G)
            loss = F.mse_loss(out[self.G.train_mask], self.G.y[self.G.train_mask].reshape(-1, 1))

            loss.backward()
            optimizer.step()
            with torch.no_grad():
                val_mae = F.l1_loss(out[self.G.val_mask], self.G.y[self.G.val_mask].reshape(-1, 1))
                if val_mae < best_val_mae:
                    best_val_mae = float(val_mae)
                    val_best_regressor = self.regressor.state_dict()
        self.regressor.load_state_dict(val_best_regressor)
        metric = gog_eval(self.regressor, self.G)

        return metric


if __name__ == "__main__":
    config = Option(sys.argv[1])
    runs = int(sys.argv[2])

    results = {'test mse': [], 'test mae': [], 'knn test mse': [], 'knn test mae': [],
               'knn test r2': [], 'kendall tau': [], 'spearmanr coeff': []}

    print("----------------Start load data----------------")
    nas_data = torch.load(config.dataset.data_path)
    print("----------------Done!----------------")

    for i in range(runs):
        seed_everything(i)
        # print("----------------Run {}----------------".format(i + 1))
        random.shuffle(nas_data)
        tester = GNNPredictor(config, nas_data)

        # stage 1
        if config.encoder.type == "metric":
            test_mse, test_mae = tester.encode_metric_learing()
        else:
            test_mse, test_mae = tester.encode()
        results['test mse'].append(test_mse)
        results['test mae'].append(test_mae)

        # stage 2
        tester.gog_construction()
        metrics = tester.gog_regression()

        for key in metrics:
            results[key].append(metrics[key])

    print("1.encoder:", tester.params.encoder.spec.name,
          "  2.kNN constrution:", tester.params.gog.construction.type,
          "  3.regressor:", tester.params.gog.regressor.spec.name)
    for key in results:
        print(key, ": {:.4f}({:.4f})".format(np.mean(results[key]), np.std(results[key])))
