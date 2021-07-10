# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 15:44:38 2021

@author: Hans
"""

from data_set_sim_corr import sim_corr_DS
from network_sim_corr_1 import Baseline_Sim_Corr
from test_sim_corr import test_func
from network_sim_corr_2 import Baseline_Sim_Corr_v2, Baseline_Sim_relu

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
writer = SummaryWriter()
time_mark = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
# import matplotlib.pyplot as plt

batch_size = 50  # 10
num_epochs = 1000 # 1000
save_epoch = 10  # 10
init_lr = 1e-4   # 1e-4
display_iter = 5
# betas=(0.9, 0.999), eps=1e-08, weight_decay=0
# print("Create dataset")
train_set = sim_corr_DS(set_type='train')
train_loader = DataLoader(train_set, batch_size=batch_size,
                         shuffle=True, num_workers=0)
valid_set = sim_corr_DS(set_type='validation')
valid_loader = DataLoader(valid_set, batch_size=batch_size,
                         shuffle=False, num_workers=0)

# print("Build model")
model = Baseline_Sim_Corr()
# model = Baseline_Sim_Corr_v2(nodes=256)
# model = Baseline_Sim_relu(nodes=256)

# print("Create optimizer")
optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
# print("Create loss")
criterion = nn.MSELoss()

# lmbda = lambda epoch: 0.98 ** epoch
# scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

## for i in range(10):
##     optimizer.step()
##    lrs.append(optimizer.param_groups[0]["lr"])
# #     print("Factor = ",0.95," , Learning Rate = ",optimizer.param_groups[0]["lr"])
##     scheduler.step()
## plt.plot(range(10),lrs)

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
        output, train_Pa, Pb = model(x, y)
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
            print(f"Epoch: {epoch+1} | it: {it} | loss: {tmp_loss/display_iter: .6f}")
            tmp_loss = 0
        # if loss.item() < b_loss:
        #     b_loss = loss
    Loss, avgPa = test_func(model, valid_loader, criterion)
    writer.add_scalar("valid loss", Loss, epoch+1)
    writer.add_scalar("Pa0 behavior", avgPa, epoch+1)
    print('Epoch: {} - Loss: {:.6F}'.format(epoch+1, loss.item()))
    print('valid Loss: {:.6F}'.format(Loss))
    if (epoch+1) % save_epoch ==0:
        # lrs.append(optimizer.param_groups[0]["lr"])
        # scheduler.step()
        param_path = '../state_dict/sim_corr_' + str(epoch+1) + '_.pt'
        torch.save(model.state_dict(), param_path)

        # break
        
writer.flush()
writer.close()
# plt.plot(range(int(num_epochs/save_epoch)), lrs)
