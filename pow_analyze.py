import numpy as np
import matplotlib.pyplot as plt
import scikits.audiolab as al
from scipy.fftpack import fft
from scipy import ifft
from scipy import ceil, complex64, float64, hamming, zeros
#file_path= "/home/miura/grive/separated_ls5k"
#wav_file = "/03001_03039/Respiratory_sounds/030010500_5k_breath.wav"

file_path= "/Users/tmiura/Grive/separated_ls5k"
wav_file = "/08001_08100/Continuous_sounds/080220500_5k_continuous.wav"

wav_path = file_path+wav_file

#wavread
data,fs,fmt = al.wavread(wav_path)
f = al.Sndfile(wav_path)
frame = f.read_frames(f.nframes)
data = np.array(frame,dtype=np.float64)
f.close()

#stft
def stft(data, window, step):
    length = len(data)     
    N = len(window) 
    M = int(ceil(float(length - N + step) / step)) 
    
    new_data = zeros(N + ((M - 1) * step), dtype = float64)
    new_data[: length] = data 
    
    X = zeros([M, N], dtype = complex64)
    for m in xrange(M):
        start = step * m
        X[m, :] = fft(new_data[start : start + N] * window)
    return X

#istft
def istft(X, win, step):
    M, N = X.shape
    assert (len(win) == N), "FFT length and window length are different."

    l = (M - 1) * step + N
    x = zeros(l, dtype = float64)
    wsum = zeros(l, dtype = float64)
    for m in xrange(M):
        start = step * m
        
        x[start : start + N] = x[start : start + N] + ifft(X[m, :]).real * win
        wsum[start : start + N] += win ** 2 
    pos = (wsum != 0)
    x_pre = x.copy()
   
    x[pos] /= wsum[pos]
    return x

N = 2048
overlap = 4
shift= N / overlap
window = hamming(N)

D = stft(data,window,shift)
y =+ np.real(D)
plt.plot(abs(y)**2)
plt.xlabel("time")
plt.ylabel("Power spectrum")

plt.show()