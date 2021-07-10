# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 15:26:18 2021

@author: Hans
"""

import numpy as np
import glob
import torch
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
        # label_paths = glob.glob('C:/Users/Ezio/Documents/SX/simulated correlations/data/prob_dist*.txt')
        # input_paths = glob.glob('C:/Users/Ezio/Documents/SX/simulated correlations/data/unitary*.txt')
        label_paths = glob.glob('/home/sxyang/files/simulation_correlation/corr/prob_dist*.txt')
        input_paths = glob.glob('/home/sxyang/files/simulation_correlation/corr/unitary*.txt')
        # Pa_Pb_paths = glob.glob('/home/sxyang/files/simulation_correlation/corr/PA_PB*')
        
        if set_type == 'test':
            userows = np.arange(0, 10**3)
            self.U = generate_set_data(input_paths, userows)
            self.P = generate_set_data(label_paths, userows)
            # self.Pa_Pb = generate_set_data(Pa_Pb_paths, userows)
        elif set_type == 'validation':
            userows = np.arange(10**3, 2*10**3)
            self.U = generate_set_data(input_paths, userows)
            self.P = generate_set_data(label_paths, userows)
            # self.Pa_Pb = generate_set_data(Pa_Pb_paths, userows)
        elif set_type == 'go_through':
            userows = np.arange(0, 10**2)
            self.U = generate_set_data(input_paths, userows)
            self.P = generate_set_data(label_paths, userows)
            # self.Pa_Pb = generate_set_data(Pa_Pb_paths, userows)
        else:
            userows = np.arange(2*10**3, 10**4)
            self.U = generate_set_data(input_paths, userows)
            self.P = generate_set_data(label_paths, userows)
            # self.Pa_Pb = generate_set_data(Pa_Pb_paths, userows)
        
        # After 200 epochs loss stat in a range,
        # expect overfitting in a less data set
        # self.U = self.U[0:100]
        # self.P = self.P[0:100]
        # self.Pa_Pb = self.Pa_Pb[0:100]
        
    def __len__(self):
        return self.U.shape[0]
    
    def __getitem__(self, idx):
        # U, P origin are numpy array float64
        in_data = torch.as_tensor(self.U[idx,:], dtype=torch.float32)
        label = torch.as_tensor(self.P[idx,:], dtype=torch.float32)
        # comparison_label = torch.as_tensor(self.Pa_Pb[idx,:], dtype=torch.float32)
        return in_data[0:8], in_data[8:16], label # comparison_label


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    # Problem arising from Spyder?  not solve the problem
    #    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    batch_size = 1
    lr = 0.001     # betas=(0.9, 0.999), eps=1e-08, weight_decay=0
    train_set = sim_corr_DS(set_type='go_through')
    x1, x2, t = train_set.__getitem__(0)
#    print(x1.shape)
#    print(x2.shape)
    print(t.shape)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                             shuffle=True, num_workers=1)
    for (x, y, target) in train_loader:
        print(x.dtype)
#        print(x.shape)
        break