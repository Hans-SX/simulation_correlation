# -*- coding: utf-8 -*-
"""
Created on Thu May 27 17:26:47 2021

@author: Hans
"""
import numpy as np
import torch

def generate_specific_rows(filePath, userows=[]):
    with open(filePath) as f:
        for i, line in enumerate(f):
            if i in userows:
                yield line
                
def generate_set_data(paths, userows):
    gen = generate_specific_rows
    array = np.array([np.loadtxt(gen(path, userows))
                        for path in paths])
    array = array.reshape(-1, array.shape[2])
    
    return array

def softXEnt(output, target):
    logprobs = torch.nn.functional.log_softmax(output, dim = 1)
    return  -(target * logprobs).sum() / output.shape[0]