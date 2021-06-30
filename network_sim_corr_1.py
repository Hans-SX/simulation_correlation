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
        
        self.model_a = self._build_model(in_dim, out_dim)
        self.model_b = self._build_model(in_dim, out_dim)
    
    def _build_model(self, in_dim, out_dim):
        model = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 16),
                nn.Tanh(),
                nn.Linear(16, 8),
                nn.Tanh(),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, out_dim),
                nn.Sigmoid())
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
    
    
        
#class FC2_Sim_Corr(nn.Module):
#    def __init__(self, first_layer=4):
#        super().__init__()
