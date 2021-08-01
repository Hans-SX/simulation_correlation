# -*- coding: utf-8 -*-
"""
Created on Fri May 28 15:28:27 2021

@author: Hans
"""

import torch
from torch import nn
from torch.nn import functional as F


class Baseline_Sim_Corr_v2(nn.Module):
    def __init__(self, in_dim=8, out_dim=2, nodes=64):
        super().__init__()
        self.nodes = nodes
        if (nodes % 2) != 0:
            self.nodes = 64
            print('Input nodes is not even, set it to be 64.')
        self.model_a = self._build_model(in_dim, out_dim)
        self.model_b = self._build_model(in_dim, out_dim)
    
    def _build_model(self, in_dim, out_dim):
        n = self.nodes
        model = nn.Sequential(
                nn.Linear(in_dim, n),
                nn.Tanh(),
                nn.Linear(n, 2*n),
                nn.Tanh(),
                nn.Linear(2*n, 6*n),
                nn.Tanh(),
                nn.Linear(6*n, 3*n),
                nn.Tanh(),
                nn.Linear(3*n, int(n/2)),
                nn.Tanh(),
                nn.Linear(int(n/2), out_dim)
                )
                # nn.Sigmoid())
        return model
        
    def forward(self, x, y):
        x = self.model_a(x)
        y = self.model_b(y)
        # x.shape(batch_size, out_dim=2)
        out = torch.einsum('...i,...j->...ij', x, y)
        # out.shape (batch_size, out_dim, out_dim)
        out = out.reshape(x.shape[0], -1)
        out = F.softmax(out)
        # out.shape (batch_size, out_dim^2)
        return out, x

class Baseline_Sim_relu(Baseline_Sim_Corr_v2):
    def __init__(self, in_dim=8, out_dim=2, nodes=64):
        super().__init__(in_dim, out_dim, nodes)

    def _build_model(self, in_dim, out_dim):
        n = self.nodes
        model = nn.Sequential(
                nn.Linear(in_dim, n),
                nn.LeakyReLU(),
                nn.Linear(n, 2*n),
                nn.LeakyReLU(),
                nn.Linear(2*n, 6*n),
                nn.LeakyReLU(),
                nn.Linear(6*n, 3*n),
                nn.LeakyReLU(),
                nn.Linear(3*n, int(n/2)),
                nn.LeakyReLU(),
                nn.Linear(int(n/2), out_dim)
                )
        return model
        
class baseline_quantum_last_part(Baseline_Sim_Corr_v2):
    # To learn quantum prediction after A, B went throught local model.
    # Difference from Baseline is: go through a few layers insead of multiply.
    # From the result, it seems that quantum behavior could not be learn in this model.
    # Reasonable, since two model separately (parameters are not considered together) 
    # could be seen as two local party.
    def __init__(self, in_dim=8, out_dim=2, nodes=64):
        super().__init__(in_dim, out_dim, nodes)
        self.q_model = self._build_q_model()
        
    def _build_q_model(self):
        q_model = nn.Sequential(
            nn.Linear(4, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 4)
            )
        return q_model
        
    def forward(self, x, y):
        x = self.model_a(x)
        y = self.model_b(y)
        # x.shape(batch_size, out_dim=2)
        z = torch.cat((x, y), 1)
        out = self.q_model(z)
        out = F.softmax(out)
        return out, x

class baseline_quantum(nn.Module):
    # To learn quantum prediction from the begining.
    def __init__(self, in_dim=16, out_dim=4, nodes=64):
        super().__init__()
        self.nodes = nodes
        if (nodes % 2) != 0:
            self.nodes = 64
            print('Input nodes is not even, set it to be 64.')
        self.qmodel = self._build_model(in_dim, out_dim)
    
    def _build_model(self, in_dim, out_dim):
        n = self.nodes
        model = nn.Sequential(
                nn.Linear(in_dim, n),
                nn.Tanh(),
                nn.Linear(n, 2*n),
                nn.Tanh(),
                nn.Linear(2*n, 6*n),
                nn.Tanh(),
                nn.Linear(6*n, 3*n),
                nn.Tanh(),
                nn.Linear(3*n, int(n/2)),
                nn.Tanh(),
                nn.Linear(int(n/2), out_dim)
                # nn.Softmax(dim=1)
                )
        return model
        
    def forward(self, x, y):
        z = torch.cat((x, y), 1)
        out = self.qmodel(z)
        return out

class one_num_communitcation(nn.Module):
    # End in B model and allow only one number passing from A to B,
    # thus, predict the probability P_ab on B model instead of P_b.
    def __init__(self, in_dim=8, out_dim=2, nodes=64):
        super().__init__()
        self.nodes = nodes
        if (nodes % 2) != 0:
            self.nodes = 64
            print('Input nodes is not even, set it to be 64.')
        self.model_a = self._build_model(in_dim, out_dim + 1)
        self.model_b = self._build_model(in_dim + 1, out_dim + 2)
        
    def _build_model(self, in_dim, out_dim):
        n = self.nodes
        model = nn.Sequential(
                nn.Linear(in_dim, n),
                nn.Tanh(),
                nn.Linear(n, 2*n),
                nn.Tanh(),
                nn.Linear(2*n, 6*n),
                nn.Tanh(),
                nn.Linear(6*n, 3*n),
                nn.Tanh(),
                nn.Linear(3*n, int(n/2)),
                nn.Tanh(),
                nn.Linear(int(n/2), out_dim)
                )
                # nn.Sigmoid())
        return model
        
    # def _build_model_b(self, in_dim, out_dim):
    #     n = self.nodes
    #     model = nn.Sequential(
    #             nn.Linear(in_dim, n),
    #             nn.Tanh(),
    #             nn.Linear(n, 2*n),
    #             nn.Tanh(),
    #             nn.Linear(2*n, 6*n),
    #             nn.Tanh(),
    #             nn.Linear(6*n, 3*n),
    #             nn.Tanh(),
    #             nn.Linear(3*n, int(n/2)),
    #             nn.Tanh(),
    #             nn.Linear(int(n/2), out_dim + 2)
    #             )
    #             # nn.Sigmoid())
    #     return model

    def forward(self, x, y):
        x = self.model_a(x)
        z = torch.cat((x[:,-1].reshape(-1, 1), y), 1)
        out = self.model_b(z)
        out = F.softmax(out, dim=1)
        return out, x