# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 15:44:38 2021

@author: Hans
"""
def test_func(model, test_loader, criterion, mode=None):

    model.eval()
    tmp = 0
    tmpa = 0
    if mode == 'qbl':
        for it, (x, y, target) in enumerate(test_loader):
            # print('data from train_loader')
            output = model(x, y)
            loss = criterion(output, target)
            # print("Loss prepared.")
            tmp += loss
        Loss = tmp.item()/(it+1)
        return Loss
    else:
        for it, (x, y, target) in enumerate(test_loader):
            # print('data from train_loader')
            output, Pa = model(x, y)
            loss = criterion(output, target)
            # print("Loss prepared.")
            tmp += loss
            tmpa += Pa[0][0]    # Pa.shape = (1,2)
        Loss = tmp.item()/(it+1)
        avgPa = tmpa/(it+1)
        return Loss, avgPa