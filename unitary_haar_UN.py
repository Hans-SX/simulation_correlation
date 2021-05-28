# -*- coding: utf-8 -*-

# scipy.stats.unitary_group() is the same with randU() from Toby qubit.
# Both are sampling from the group U(N), instead of SU(N).

import numpy as np
import scipy

def sp_unitary_group(n):
    
    X = (np.random.normal(size=(n,n)) + 1j*(np.random.normal(size=(n,n))))/np.sqrt(2)
    q, r = scipy.linalg.qr(X)
#    R1 = np.diag(np.diagonal(r)/abs(np.diagonal(r)))
    R = r.diagonal()/abs(r.diagonal())
#    U1 = np.dot(q, R1)
    U = q * R
    return U
