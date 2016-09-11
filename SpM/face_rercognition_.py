# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 14:05:10 2016

@author: tmiura
"""

"""
Face recognition with Yale Face Database B
Algorithm : Orthogonal Matching Pursuit(OMP) 
"""
import numpy as np
#%%
"Orthogonal Matching Pursuit(OMP)"
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
#%%
"Criate data from YaleB_Ext_Cropped_192x168_all.mat"
import scipy.io

mat = scipy.io.loadmat('./YaleB_Ext_Cropped_192x168_all.mat')

data = 1.0*np.array(mat.get('fea')) # For converting to numpy array
label = np.array(mat.get('gnd')) # For converting to numpy array
#%%
"Random Projection for dimention reduction of feature data sets 192x168->200"
m,n = 200,32256
seed = 2016
rng = np.random.RandomState(seed)
"Generate Random projection Matrix"
R = rng.randn(m,n)
D = np.dot(R,data.T)
#%%
"split data sets for train and test"
from sklearn.cross_validation import train_test_split
features_train,features_test,labels_train, labels_test = train_test_split(D.T,label, test_size=0.50,random_state=2016)
#%%
"features_train data sets Regularization"
for i in range(1207):
    norm = np.linalg.norm(features_train[i,:])
    features_train[i,:]=features_train[i,:]/norm
#%%
b = features_test[np.random.randint(1,1207),:]
A = features_train.T
x_est = OMP(b,A,100,1e-3)
#%%
"Prediction labels"
id_max = np.max(labels_train)
residual = np.zeros(id_max)
for id in range(id_max):
        b_reconst = np.dot(A[:,(labels_train.T[0] == id+1)],x_est[(labels_train.T == id+1)[0]])
        residual[id] = np.linalg.norm(b - b_reconst)##Calculate Residual 
#%%
"Plot Estimators"        
import matplotlib.pyplot as plt
axis = np.arange(0,38,1)
plt.close('all')
plt.show()
plt.bar(axis,residual,width =0.3)
plt.show()
#%%
"The result of predict lables"
id_est = np.argmin(residual,axis = 0)+1
##print ('pred_id : %d' % id_est)
##if you run pyhton2.x, please run here!!
print 'pred_id : %d' % id_est 