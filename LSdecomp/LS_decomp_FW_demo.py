# -*- coding: utf-8 -*-
"""
edit by tmiura
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as swf
import scipy.fftpack as sft
from rwt import dwt,idwt
from rwt.wavelets import daubcqf
import spmlib.solver.relaxations as relax
import scipy.sparse.linalg as linalg


def ls_decomp_FW(filename, width = 16384, max_nnz_rate = 800 / 262144,
                 sparsify = 0.01, taps = 10, wl_weight = 0.25, verbose = False, fc = 120):
    
    
    #read
    if os.path.exists(filename + '.wav') == False:
        Fs, signal  = swf.read(filename + '.WAV')
    else:
        Fs, signal = swf.read(filename + '.wav')
        
    
    
    signal = signal.astype(np.float64)
    signal = signal // 32768.0
    length = signal.shape[0]    
    signal = np.concatenate([np.zeros([int(width/2)]), signal[0:length], np.zeros([int(width)])],axis=0)
    n_wav = signal.shape[0]
    print(n_wav)
    signal_dct = np.zeros((n_wav, 1))
    signal_wl = np.zeros((n_wav, 1))
    
    pwindow = np.zeros(width, dtype = float)
    for i in range(width // 2):
        pwindow[i] = i
    for i in range(1, width // 2 + 1):
        pwindow[-i] = i-1
    pwindow = pwindow // width * 2
    
    #if fc > 0:
        #d = fdesign.hipass()
    
    w_s = 1
    w_e = width
    c1 = []
    c2 = []
    reconst_error = []
    START = np.array([n_wav])
    END = np.array([n_wav])
    i = 0
    
    while w_e < n_wav:
        print('%1.3f - %1.3f [s]\n' % (w_s/Fs, w_e/Fs))
        START[i] = w_s / Fs
        END[i] = w_e / Fs
        sig_dct, sig_wl, c1[i,1], c2[i,2], reconst_error[i,1] = LSDecompFW(signal=signal[w_s:w_e], 
                          width = 16384, max_nnz_rate = 8000 / 262144,sparsify = 0.01, taps = 10, wl_weight = 0.25, verbose = False, fc = 120 )
        signal_dct[w_s:w_e] = signal_dct[w_s:w_e] + pwindow * sig_dct
        signal_wl[w_s:w_e] = signal_wl[w_s:w_e] + pwindow * sig_wl
        
        w_s = w_s + width / 2
        w_e = w_e + width / 2
        i+=1

    dct_length = np.shape(signal_dct)
    wl_length = np.shape(signal_wl)
    #raw_length = np.shape(signal)
    
    signal_dct = signal_dct[width/2+1:dct_length]
    signal_wl = signal_wl[width/2+1:wl_length]    
    #signal_raw = signal[width/2+1:raw_length]

    swf.write(filename +'_F.wav',Fs,signal_dct)
    swf.write(filename +'_W.wav',Fs,signal_wl)
    
    print('end of ls_decomp_FW!!')

    
def nextpow2(n):
    m_f = np.log2(n)
    m_i = np.ceil(m_f)
    return int(np.log2(2**m_i))

def LSDecompFW(signal, width= 2**14,max_nnz_rate=8000/262144, sparsify = 0.01, taps = 10, 
               level = 3, wl_weight = 1, verbose = False,fc=120):
    
    MaxiterA = 60
   
    length = len(signal)
    
    n = sft.next_fast_len(length)
    
    signal = np.zeros((n))
    
    Phi = lambda c: c
    PhiT = lambda x: x
#    
    h0,h1 = daubcqf(taps,'min')
    L = level
    
    temp = np.concatenate([sft.idct(signal[0:n]),float(wl_weight)* idwt(signal[n+1:2*n],h0,h1,L)[0]],axis=0)
    temp2 = np.concatenate([sft.dct(signal),float(wl_weight)* dwt(signal,h0,h1,L)[0]],axis=0)
    print(temp2.shape)
    print(temp.shape)
    Psi = lambda c: np.concatenate([sft.idct(signal[0:n]) , float(wl_weight) * idwt(c[n+1:2*n], h0,h1,L)[0]]) 
    PsiT = lambda x: np.concatenate([sft.dct(x), float(wl_weight) * dwt(x,h0,h1,L)[0]]) 
#    
    Theta = lambda c: Phi(Psi(c))
    ThetaT = lambda y: PsiT(PhiT(y))
    
    #measurment
    

    
    #GPSR
    ###############################
    nonzeros = float("Inf")
    y =signal
    c=Theta(y)
    maxabsThetaTy = max(abs(temp))
    
    
    while nonzeros > max_nnz_rate * n:
            #FISTA
            tau = sparsify * maxabsThetaTy
            tolA = 1.0e-7
            
            fh = linalg.LinearOperator(shape=temp2.shape,matvec=temp2,rmatvec=temp2.T,dtype=np.float64)
               
            c = relax.fista(y, fh, tolA, MaxiterA, c, tau)
            
            #GPSR
            ###################
            #
            ###################
            nonzeros = np.nonzero(c)
            print('nnz = %d / %d\tat tau = %f\n' % nonzeros, n, tau)
            sparsify = sparsify * 2
            if sparsify == 0.166:
                sparsify = 0.1
                
    ###############################
    
    
if __name__ == '__main__':
    filepath = './080180500_5k'
    ls_decomp_FW(filename = filepath)