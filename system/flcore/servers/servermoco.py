import time
from flcore.clients.clientmoco import clientMoCo

from flcore.servers.serverbase import Server
from flcore.servers.serverbase2 import Server2
from threading import Thread

import numpy as np
import quadprog
import torch
import copy


class FedMoCo(Server2):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.beta = args.beta

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientMoCo)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        self.finetune_round = args.finetune_round
        self.tracking_variables = [copy.deepcopy(self.global_model) for i in range(self.num_clients)]
        for tv in self.tracking_variables:
            for param in tv.parameters():
                param.data.zero_()



    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for cid in self.selected_client_ids:
                local_loss = self.clients[cid].train()
                self.local_losses[cid] = local_loss[0]

            self.rs_local_train_losses.append(list(self.local_losses))

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            self.update_tracking_variables()
            if i < self.finetune_round:
                self.find_omega(self.epsilon)

            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

    def update_tracking_variables(self):
        for i in self.selected_client_ids:
            for y_param, cm_param, gm_param in zip(self.tracking_variables[i].parameters(), self.client_models[i].parameters(), self.global_model.parameters()):
                y_param.data = y_param.data - self.beta * (y_param.data - cm_param.data + gm_param.data)
                


    def aggregate_parameters(self):
        assert (len(self.client_models) > 0)

        self.rs_weights.append(list(self.client_weights))
        # self.global_model = copy.deepcopy(self.uploaded_models[0])
        # for param in self.global_model.parameters():
        #     param.data.zero_()

        for i in self.selected_client_ids:
            self.add_parameters(self.client_weights[i], self.tracking_variables[i])
            

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

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
                for paramI, paramJ in zip(self.tracking_variables[self.selected_client_ids[i]].parameters(), self.tracking_variables[self.selected_client_ids[j]].parameters()):
                    K[i,j] += torch.mul(paramI.data, paramJ.data).sum()

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






