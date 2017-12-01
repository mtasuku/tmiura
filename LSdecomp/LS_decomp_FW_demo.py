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


def read_file(filename):
    #read
    if os.path.exists(filename + '.wav') == False:
        Fs, data  = swf.read(filename + '.WAV')
    else:
        Fs, data = swf.read(filename + '.wav')
    
    #normalize between -1 and 1, change dtype double
    data = data.astype(np.float64)
    data = data // 32768.0
    
    return Fs,data
    
def run_LSdecompFW(filename, width = 16384, max_nnz_rate = 8000 / 262144,
                 sparsify = 0.01, taps = 10, wl_weight = 0.25, verbose = False, fc = 120):        

    Fs,signal = read_file(filename)
    
    length = signal.shape[0]    
    signal = np.concatenate([np.zeros([int(width/2)]), signal[0:length], np.zeros([int(width)])],axis=0)
    n_wav = signal.shape[0]
    print(signal.shape)
    signal_dct = np.zeros((n_wav, 0),dtype=np.float64)
    signal_wl = np.zeros((n_wav, 0),dtype=np.float64)
    
    pwindow = np.zeros(width, dtype=np.float64)
    for i in range(width // 2):
        pwindow[i] = i
    for i in range(1, width // 2 + 1):
        pwindow[-i] = i-1
    pwindow = pwindow // width * 2
    
    #if fc > 0:
        #d = fdesign.hipass()
    
    w_s = 0
    w_e = width

    START = np.array([n_wav])
    END = np.array([n_wav])
    i = 0
    
    while w_e < n_wav:
        print('%1.3f - %1.3f [s]\n' % (w_s/Fs, w_e/Fs))
        START[i] = w_s / Fs
        END[i] = w_e / Fs
        sig_dct, sig_wl = LSDecompFW(wav=signal[w_s:w_e], width = 16384, 
                                         max_nnz_rate = 8000 / 262144, sparsify = 0.01, taps = 10, 
                                         wl_weight = 0.25, verbose = False, fc = 120 )
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
    
    print('end of run_LSdecomp_FW!!')


def LSDecompFW(wav, width= 2**14,max_nnz_rate=8000/262144, sparsify = 0.01, taps = 10, 
               level = 3, wl_weight = 1, verbose = False,fc=120):
    
    MaxiterA = 60
   
    length = len(wav)
    print('len: '+str(length))
    n = sft.next_fast_len(length)
    print('n: '+str(n))
    signal = np.zeros((n))
    signal[0:length] = wav[0:length]
     
    h0,h1 = daubcqf(taps,'min')
    L = level
    
    temp = sft.idct(signal[0:n]) + (1.0)*(wl_weight)* idwt(signal[0:n],h0,h1,L)[0]
    temp2 = np.concatenate([sft.dct(signal),(1.0)*(wl_weight)* dwt(signal,h0,h1,L)[0]],axis=0)
    temp = temp.astype(np.double)
    temp2 = temp2.astype(np.double)
    print(temp.shape)
    print(temp2.shape)
    #measurment
    

    
    #GPSR
    ###############################
    nonzeros = float("Inf")
    y = signal
    c = signal
    maxabsThetaTy = max(abs(temp))
    
    
    while nonzeros > max_nnz_rate * n:
            #FISTA
            tau = sparsify * maxabsThetaTy
            tolA = 1.0e-7
            
            fh = (temp,temp2.T)
            print(fh)
            c,r = relax.fista_scad(y, fh, tolA, MaxiterA, c, tau)
            
            #GPSR
            ###################
            #
            ###################
            nonzeros = np.nonzero(c)
            print('nnz = %d / %d\tat tau = %f\n' % nonzeros, n, tau)
            sparsify = sparsify * 2
            if sparsify == 0.166:
                sparsify = 0.1
                
    return             
    ###############################
    
    
if __name__ == '__main__':
    filepath = '../../../080180500_5k'
    run_LSdecompFW(filename = filepath)