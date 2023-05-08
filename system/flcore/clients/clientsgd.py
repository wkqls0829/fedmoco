import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.privacy import *


class clientMGDA(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.train_loader = self.load_train_data()
        self.train_iter = iter(self.train_loader)

    def train(self):
        batch = next(self.train_iter, None)
        if not batch:
            print("new batch")
            self.train_iter = iter(self.train_loader)
            batch = next(self.train_iter)
        # self.model.to(self.device)
        self.model.train()

        
        start_time = time.time()

        x, y = batch
        if type(x) == type([]):
            x[0] = x[0].to(self.device)
        else:
            x = x.to(self.device)
        y = y.to(self.device)
        output = self.model(x)
        loss = self.loss(output, y)

        loss_collector = [loss]


        # self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")

        return loss_collector
