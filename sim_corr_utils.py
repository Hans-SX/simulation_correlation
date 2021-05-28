# -*- coding: utf-8 -*-
"""
Created on Thu May 27 17:26:47 2021

@author: Hans
"""
import numpy as np
import glob

prob_paths = glob.glob('./data/prob_dist*.txt')
unitary_paths = glob.glob('./data/unitary*.txt')

Pab = np.array([np.loadtxt(f) for f in prob_paths])
Pab = Pab.reshape(Pab.shape[0]*Pab.shape[1], -1)
unitary = np.array([np.loadtxt(f) for f in unitary_paths])
unitary = unitary.reshape(-1, unitary.shape[2])

def generate_specific_rows(filePath, userows=[]):
    with open(filePath) as f:
        for i, line in enumerate(f):
            if i in userows:
                yield line
                
def generate_set_data(paths, userows):
    gen = generate_specific_rows
    array = np.array([np.loadtxt(gen(path, userows))
                        for path in unitary_paths])
    array = array.reshape(-1, array.shape[2])
    
    return array