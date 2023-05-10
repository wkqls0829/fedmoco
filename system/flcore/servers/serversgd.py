import time
from flcore.clients.clientsgd import clientSGD

from flcore.servers.serverbase import Server
from flcore.servers.serverbase2 import Server2
from threading import Thread

import numpy as np
import quadprog
import torch
import torch.nn as nn


class FedSGD(Server2):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientSGD)

        for c in self.clients:
            c.model = self.global_model

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        self.finetune_round = args.finetune_round

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.global_model.parameters(), lr=args.local_learning_rate)



    def train(self):
        self.send_models()
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            losses = []

            for cid in self.selected_client_ids:
                local_loss = self.clients[cid].train()
                losses.append(local_loss[0])
                self.local_losses[cid] = local_loss[0].item()

            self.optimizer.zero_grad()
            for loss in losses:
                loss.backward()
            self.optimizer.step()

            self.rs_local_train_losses.append(list(self.local_losses))
            self.rs_avg_train_loss.append(np.sum(self.local_losses) / len(self.selected_clients))


            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]


            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_avg_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()
