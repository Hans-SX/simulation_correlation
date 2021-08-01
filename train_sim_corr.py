# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 15:44:38 2021

@author: Hans
"""

from data_set_sim_corr import sim_corr_DS
from network_sim_corr_1 import Baseline_Sim_Corr
import network_sim_corr_2 as md2
from test_sim_corr import test_func
from utils_sim_corr import softXEnt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import os
writer = SummaryWriter()
time_mark = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
# os.mkdir('../state_dict_' + time_mark)
import matplotlib.pyplot as plt

batch_size = 10 # 50
num_epochs = 100
save_epoch = 20
init_lr = 1e-2
display_iter = 4 # 20
# betas=(0.9, 0.999), eps=1e-08, weight_decay=0
# print("Create dataset")
train_set = sim_corr_DS(set_type='go_through')
# print(len(train_set))
# raise SystemExit(0)
train_loader = DataLoader(train_set, batch_size=batch_size,
                         shuffle=True, num_workers=2)
valid_set = sim_corr_DS(set_type='validation')
valid_loader = DataLoader(valid_set, batch_size=batch_size,
                         shuffle=False, num_workers=2)

# print("Build model")
# model = Baseline_Sim_Corr()
# model = md2.Baseline_Sim_Corr_v2()
# model = md2.Baseline_Sim_relu()
# model = md2.baseline_quantum_last_part()
model = md2.baseline_quantum()
# model = md2.one_num_communitcation()

# print("Create optimizer")
optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
# print("Create loss")
# criterion = nn.MSELoss()
criterion = softXEnt
# CrossEntropyLoss is not the case here, it need the true class label.
# criterion = nn.CrossEntropyLoss()

# scheduler =  torch.optim.lr_scheduler.MultiStepLR(
#     optimizer, milestones=[5, 10, 15], gamma=0.1)



# lmbda = lambda epoch: 0.9 ** epoch  # 0.98 decreasing quite slow
# scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
lrs=[]
"""
train loss is decreasing when 
lrs =
0.007290000000000001
0.005314410000000002
0.0034867844010000016
0.0020589113209464912
0.0010941898913151243
"""
# =============================================================================
# test for learning rate decay.
# for i in range(10):
#     optimizer.step()
#     lrs.append(optimizer.param_groups[0]["lr"])
#     print("Factor = ",0.95," , Learning Rate = ",optimizer.param_groups[0]["lr"])
#     scheduler.step()
# plt.plot(range(10),lrs)
# =============================================================================

# lrs = []
# b_loss = 100
its = 0
# print("Start training")
for epoch in range(num_epochs):
    model.train()
    tmp_loss = 0
    # comp_label = Pa, Pb. But not necessary, all = 0.5, Max Ent partial trace = identity.
    for it, (x, y, target) in enumerate(train_loader):
        # print('data from train_loader')
        its += 1
        optimizer.zero_grad()
        # output, train_Pa = model(x, y)
        output = model(x, y)
        # print('put data into model')
        # print(output.shape) # (10e4, 4) = kron((10e2, 4), (10e2, 4))
        # print("x shape:", x.shape) # (10e2, 8) >> after model (10e2,4)
        # print("y shape:", y.shape) # (10e2, 8) >> after model (10e2,4)
        # print("target shape:", target.shape)
        loss = criterion(output, target)
        # print("Loss prepared.")
        
        loss.backward()
        # print("Backwarded")
        
        optimizer.step()
        # print('update parameters')
        # break
        tmp_loss += loss.item()
        if (it+1) % display_iter == 0:
            writer.add_scalar("train loss", tmp_loss/display_iter, its)
            writer.add_scalar("lr", optimizer.param_groups[0]['lr'], its)
            print(f"Epoch: {epoch+1} | it: {it} | loss: {tmp_loss/display_iter: .6f}")
            tmp_loss = 0
        
    # scheduler.step()
    
    # Loss, avgPa = test_func(model, valid_loader, criterion)
    # Loss = test_func(model, valid_loader, criterion, 'qbl')
    # writer.add_scalar("valid loss", Loss, epoch+1)
    # writer.add_scalar("Pa0 behavior", avgPa, epoch+1)
    # print('Epoch: {} - Loss: {:.6F}'.format(epoch+1, loss.item()))
    # print('valid Loss: {:.6F}'.format(Loss))
    # if (epoch+1) % save_epoch ==0:
    #     lrs.append(optimizer.param_groups[0]["lr"])
    #     scheduler.step()
        # param_path = '../state_dict_' + time_mark + '/sim_corr_' + str(epoch+1) + '_.pt'
        # torch.save(model.state_dict(), param_path)

        # break
        
writer.flush()
writer.close()
# plt.plot(range(int(num_epochs/save_epoch)), lrs)
