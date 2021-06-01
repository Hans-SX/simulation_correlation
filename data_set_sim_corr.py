# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 15:26:18 2021

@author: Hans
"""

import numpy as np
import glob
from torch.utils.data import Dataset
from utils_sim_corr import generate_set_data

"""
Note:
    __init__(): load data, first 10 thousands >> test set,
        second 10 >> validation, rest >> train-set
    
    training phase: one epoch validate (performance on validation set)
                    once to check overfitting.
    Loss function: mean square difference?
    performance: look for regression performance.

"""

class sim_corr_DS(Dataset):
    def __init__(self, set_type='train'):
#        super().__init__()
        self.set = set_type
        label_paths = glob.glob('./data/prob_dist*.txt')
        input_paths = glob.glob('./data/unitary*.txt')
        
        if set_type == 'test':
            userows = np.arange(0, 10**4)
            self.U = generate_set_data(input_paths, userows)
            self.P = generate_set_data(label_paths, userows)
        elif set_type == 'validation':
            userows = np.arange(10**4+1, 2*10**4)
            self.U = generate_set_data(input_paths, userows)
            self.P = generate_set_data(label_paths, userows)
        else:
            userows = np.arange(2*10**4 + 1, 10**5)
            self.U = generate_set_data(input_paths, userows)
            self.P = generate_set_data(label_paths, userows)
        
    def __len__(self):
        return self.U.shape[0]
    
    def __getitem__(self, idx):
        return self.U[idx, 0:8], self.U[idx, 8:16], self.P[idx, :]