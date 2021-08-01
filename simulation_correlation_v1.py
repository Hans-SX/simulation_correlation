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
    # M_orig: concate 2 POVM of Z
    M_orig = np.concatenate((pauli_z1, pauli_z2)).reshape((2,2,2))
    phi = np.reshape((np.array([1,0,0,0]).T + np.array([0,0,0,1]).T)
            /np.sqrt(2),(4,1))
    rho = np.kron(phi, phi.T)
    # ========================================================================
    # To partial trace along the axis of the sub system.
    # Actually, could just use eye(2) = max Ent subsystem.
    tmp_rho = rho.reshape(2,2,2,2)  # na, nb, na, nb
    # keep_a = np.transpose(tmp_rho, (1,0,2,3))
    # keep_b = np.transpose(tmp_rho, (0,1,3,2))
    keep_a = np.einsum('ij...->ji...', tmp_rho)  # nb, na, na, nb
    keep_b = np.einsum('...ij->...ji', tmp_rho)  # na, nb, nb, na
    # rho_a = np.trace(tmp_rho, axis1=1, axis2=3)
    # rho_b = np.trace(tmp_rho, axis1=0, axis2=2)
    rho_a = np.einsum('i...i', keep_a)      
    rho_b = np.einsum('i...i', keep_b)
    # ========================================================================
    #========================================
    # Preallocation
    unitary = np.empty([num, 4*2*2])
    Pab = np.empty([num, 4])
    Pa_Pb = np.empty([num, 4])
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
# ------------------------------------
        # Ma.shape = (2,2,2), first index indicates which POVM (e.g., up/down)
        # Ma = np.einsum('...ij,jk', np.einsum('ij,...jk',
        #                                      np.conj(ua.T), M_orig), ua)
        # Mb = np.einsum('...ij,jk', np.einsum('ij,...jk',
        #                                      np.conj(ub.T), M_orig), ub)
        # Pa = np.einsum('...ii', np.einsum('ij,...jk', rho_a, Ma)).real
        # Pb = np.einsum('...ii', np.einsum('ij,...jk', rho_b, Mb)).real

# this part might have problem trying to rewrite below, unfinished.
# -------------------------------------        
        Ma1 = ua.T @ pauli_z1 @ ua
        Ma2 = ua.T @ pauli_z2 @ ua
        # Ma = np.concatenate(Ma1, Ma2)
        Mb1 = ub.T @ pauli_z1 @ ub
        Mb2 = ub.T @ pauli_z2 @ ub
        # Mb = np.concatenate(Mb1, Mb2)
        Pa1 = np.trace(rho_a)
        Pa_Pb[i] = np.concatenate((Pa, Pb))
        # Mab.shape = (4,4,4),  first index indicates the combination ab=00, 01, 10, 11 
        Mab = np.kron(Ma, Mb)
        # rho times Mab[ab=1~4] separately at once. Then trace.
        Pab[i,:] = np.einsum('...ii', np.einsum('ij,...jk',
                                           rho, Mab)).real.reshape((1,4))
        # if concate fields, i >> i + 1
        unitary[i,:] = np.column_stack([
                            ua.reshape(1,4).view(float),
                            ub.reshape(1,4).view(float)])
        
        # To save file, get date and time. formmat?
        if (i+1) % 10**3 == 0:
#        if i == 9:
            np.savetxt("../corr/unitary_seed_" + str(seed) + "_" + time_mark + ".txt", unitary)
            np.savetxt("../corr/prob_dist_seed_" + str(seed) + "_" + time_mark + ".txt", Pab)
            np.savetxt("../corr/PA_PB_seed_" + str(seed) + "_" + time_mark + ".txt", Pa_Pb)
            print('saving', i+1, ' data')

if __name__ == "__main__":
    seed = 1
    while seed < 11:
        simulation_correlation(10**4, seed)
        seed += 1
#seed = 1
#num = 10
#simulation_correlation(num, seed)
