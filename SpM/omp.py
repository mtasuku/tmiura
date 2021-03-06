# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 13:29:36 2016

@author: 高井wing
"""

import numpy as np

def OMP(b, A, k, delta):
    m, n = A.shape
    x = np.zeros(n)
    
    T = np.array([], dtype = int)
    r = b
    
    while np.size(T) <= k and np.linalg.norm(r)/np.linalg.norm(b) > delta:
        c = np.dot(A.T, r)
        s = np.argmax(np.abs(c))
        
        T = np.append(T, s)
        x[T] = np.linalg.lstsq(A[:,T],b)[0]
        r = b - np.dot(A, x)
    
    return x
rng = np.random.RandomState(40)
m, n = 300, 400

# random design
A = rng.randn(m, n)  # random design

x0 = rng.randn(n)
x0[abs(x0) < 2] = 0
b = np.dot(A, x0)
x_est = OMP(b, A, 50, 1e-7)

import matplotlib.pyplot as plt
plt.close('all')

plt.figure()
plt.stem(x0, markerfmt='b.')
#plt.stem(x_ista, markerfmt='bo')
plt.stem(x_est, markerfmt='rx')

plt.show()
