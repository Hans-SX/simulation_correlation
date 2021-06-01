# -*- coding: utf-8 -*-
"""
Created on Fri May 28 15:28:27 2021

@author: Hans
"""

import torch
from torch import nn


class Baseline_Sim_Corr(nn.Module):
    def __init__(self, in_dim=8, out_dim=2):
        super().__init__()
        
        self.model = self._build_model(in_dim, out_dim)

    
    def _build_model(self, in_dim, out_dim):
        model = torch.Sequntial(
                nn.Linear(in_dim, 16),
                torch.tanh(),
                nn.Linear(16, 32),
                torch.tanh(),
                nn.Linear(32, 16),
                torch.tanh(),
                nn.Linear(16, 8),
                torch.tanh(),
                nn.Linear(8, 4),
                torch.tanh(),
                nn.Linear(4, out_dim),
                torch.sigmoid())
        return model
        
    def forward(self, x, y):
        x = self.model(x)
        y = self.model(y)
        
        out = torch.kron(x, y)
        return out
    
    
        
#class FC2_Sim_Corr(nn.Module):
#    def __init__(self, first_layer=4):
#        super().__init__()
