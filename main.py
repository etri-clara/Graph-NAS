import sys
import math
import random

import torch
import numpy as np
from tqdm import tqdm

from modules import GCN, GAT, GIN
from utils import Option, dataset, seed_everything, train, eval

model_map = {"GCN": GCN, "GAT": GAT, "GIN": GIN}
class GNN_Predictor:
    def __init__(self, config_path, data):
        self.params = Option(config_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_loader, self.val_loader, self.test_loader = dataset(data,
                                                                       self.params.dataset.num_data,
                                                                       self.params.dataset.num_train,
                                                                       self.params.dataset.num_val)

        self.model = model_map[self.params.model.name](self.params.model).to(self.device)
    def reload(self):
        self.train_loader, self.val_loader, self.test_loader = dataset(self.params.dataset.data_path,
                                                                       self.params.dataset.num_data,
                                                                       self.params.dataset.num_train,
                                                                       self.params.dataset.num_val)

    def run(self):
        best_val_mse = math.inf
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.lr)
        for epoch in range(self.params.epoch):
            loss = train(self.model, self.train_loader, optimizer, self.device)
            val_mse = eval(self.model, self.val_loader, self.device)
            if best_val_mse > val_mse:
                test_mse = eval(self.model, self.test_loader, self.device)
                best_val_mse = val_mse
        return test_mse


if __name__ == "__main__":
    seed_everything(29)
    config = sys.argv[1]
    runs = int(sys.argv[2])
    test_mses = []
    data = torch.load("./nasbench_dataset.pt")
    for i in tqdm(range(runs)):
        print("----------------Run {}----------------".format(i + 1))
        random.shuffle(data)
        tester = GNN_Predictor(config, data)
        test_mse = tester.run()
        test_mses.append(test_mse)
    print("{} test mse: {}".format(tester.params.model.name, np.mean(test_mses), np.std(test_mses)))