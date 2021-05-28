# -*- coding: utf-8 -*-
"""
Created on Tue May 25 17:29:30 2021

@author: Hans
"""

import numpy as np
from scipy.stats import unitary_group
import time

'''
Goal: generate input (unitary) and label (probability distribution).
    Would save 1 file for each, bonded by time mark from the file name.
    Row: num, number of sampling.
    Column:    'Ua_00', 'iUa_00', 'Ua_01', 'iUa_01',
               'Ua_10', 'iUa_10', 'Ua_11', 'iUa_11',
               'Ub_00', 'iUb_00', 'Ub_01', 'iUb_01',
               'Ub_10', 'iUb_10', 'Ub_11', 'iUb_11',
               'P_00', 'P_01', 'P_10', 'P_11'
'''
    

def simulation_correlation(num, seed=1):
    time_mark = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    np.random.seed(seed)
    
    ket_0 = np.array([[1],[0]])
    ket_1 = np.array([[0],[1]])
    pauli_z1 = np.kron(ket_0, ket_0.T)
    pauli_z2 = np.kron(ket_1, ket_1.T)
    M_orig = np.concatenate((pauli_z1, pauli_z2)).reshape((2,2,2))
    phi = np.reshape((np.array([1,0,0,0]).T + np.array([0,0,0,1]).T)
            /np.sqrt(2),(4,1))
    rho = np.kron(phi, phi.T)
    #========================================
    # Preallocation
    unitary = np.empty([num, 4*2*2])
    Pab = np.empty([num, 4])
#    fields = [['Ua_00', 'iUa_00', 'Ua_01', 'iUa_01',
#               'Ua_10', 'iUa_10', 'Ua_11', 'iUa_11',
#               'Ub_00', 'iUb_00', 'Ub_01', 'iUb_01',
#               'Ub_10', 'iUb_10', 'Ub_11', 'iUb_11',
#              'P_00', 'P_01', 'P_10', 'P_11']]
#    data = np.concatenate((fields, data))
    #========================================
    for i in range(num):
        ua = unitary_group.rvs(2)
        ub = unitary_group.rvs(2)
        # To stack measurement outcomes (POVM), for A: 1,1,2,2 ; B: 1,2,1,2
        # Pab: P11, P12, P21, P22
        # !!! It turns out that I don't need to stack it, by applying tensor
        # product, it will.
#        A = np.concatenate((np.matlib.repmat(pauli_z1, 2, 1), np.matlib.
#                            repmat(pauli_z2, 2, 1))).reshape((4,2,2))
#        B = np.matlib.repmat(np.concatenate((pauli_z1, pauli_z2)),
#                             2, 1).reshape((4,2,2))
        
        Ma = np.einsum('...ij,jk', np.einsum('ij,...jk',
                                             np.conj(ua.T), M_orig), ua)
        Mb = np.einsum('...ij,jk', np.einsum('ij,...jk',
                                             np.conj(ub.T), M_orig), ub)
        Mab = np.kron(Ma, Mb)
        Pab[i,:] = np.einsum('...ii', np.einsum('ij,...jk',
                                           rho, Mab)).real.reshape((1,4))
        # if concate fields, i >> i + 1
        unitary[i,:] = np.column_stack([
                            ua.reshape(1,4).view(float),
                            ub.reshape(1,4).view(float)])
        
        # To save file, get date and time. formmat?
        if (i+1) % 10**3 == 0:
#        if i == 9:
            np.savetxt("./data/unitary_seed_" + str(seed) + "_" + time_mark + ".txt", unitary)
            np.savetxt("./data/prob_dist_seed_" + str(seed) + "_" + time_mark + ".txt", Pab)
            print('saving', i+1, ' data')

seed = 1
while seed < 11:
    simulation_correlation(10**5, seed)
    seed += 1
#seed = 1
#num = 10
#simulation_correlation(num, seed)
