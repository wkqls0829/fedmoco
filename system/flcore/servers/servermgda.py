import time
from flcore.clients.clientmgda import clientMGDA

from flcore.servers.serverbase import Server
from flcore.servers.serverbase2 import Server2
from threading import Thread

import numpy as np
import quadprog
import torch


class FedMGDA(Server2):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientMGDA)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        self.finetune_round = args.finetune_round


    def train(self):
        self.send_models()
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            for cid in self.selected_client_ids:
                local_loss = self.clients[cid].train()
                self.local_losses[cid] = local_loss[0]

            self.rs_local_train_losses.append(list(self.local_losses))
            self.rs_avg_train_loss.append(np.sum(self.local_losses) / len(self.selected_clients))
            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if i < self.finetune_round:
                self.find_omega(self.epsilon)
            self.aggregate_parameters()

            self.send_models()
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


    def find_omega(self, epsilon=1):
        """
            searches for optimal weight omega for mgda
        """

        n = len(self.selected_clients)
        assert (n > 0)

        Wavg = np.array([self.client_weights[i] for i in self.selected_client_ids], dtype=float)

        K = np.eye(n, dtype=float)
        for i in range(0,n):
            for j in range(0,n):
                K[i,j] = 0
                for paramI , paramJ , paramG in zip(self.client_models[self.selected_client_ids[i]].parameters(), self.client_models[self.selected_client_ids[j]].parameters(), self.global_model.parameters()):
                    K[i,j] += torch.mul(paramG.data - paramI.data, paramG.data - paramJ.data).sum()

        Knorm = 0
        for i in range(0,n):
            Knorm += K[i,i]
        Knorm = Knorm / n


        Q = (K + K.T)


        p = np.zeros(n, dtype=float)
        a = np.ones(n, dtype=float).reshape(-1, 1)
        Id = np.eye(n, dtype=float)
        R = Id * self.rho * Knorm
        Q = R + Q
        neg_Id = -1 * np.eye(n, dtype=float)
        lower_b = (Wavg - epsilon) * np.ones(n,dtype=float)
        upper_b = (-Wavg - epsilon) * np.ones(n,dtype=float)
        A = np.concatenate((a,Id,Id,neg_Id),axis=1)
        b = np.zeros(n+1)
        b[0] = 1.
        b_concat = np.concatenate((b,lower_b, upper_b))
        omega = quadprog.solve_qp(Q,p,A,b_concat,meq=1)[0]
        for i, ids in enumerate(self.selected_client_ids):
            self.client_weights[ids] = omega[i]
        print(omega)







