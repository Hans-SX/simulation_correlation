# -*- coding: utf-8 -*-
"""
Created on Fri May 28 15:28:27 2021

@author: Hans
"""

import torch
from torch import nn


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
        # out.shape (batch_size, out_dim^2)
        return out
    

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
        
#class FC2_Sim_Corr(nn.Module):
#    def __init__(self, first_layer=4):
#        super().__init__()
