import numpy as np
import matplotlib.pyplot as plt
import scikits.audiolab as al
import scipy.fftpack as fft

 
file_path= "/home/miura/grive/separated_ls5k"
wav_file = "/03001_03039/Respiratory_sounds/030010500_5k_breath.wav"
wav_path = file_path+wav_file

#wavread
data,fs,fmt = al.wavread(wav_path)
f = al.Sndfile(wav_path)
frame = f.read_frames(f.nframes)
data = np.array(frame,dtype=np.float64)
f.close()

length = len(data)
N = 2048
overlap = 4
shift= N / overlap

window = np.hamming(N)


ys = np.zeros(length)

# FFT & IFFT
for start in range(0, length - shift, shift):
    dft_cut = data[start: start + N]
    dft_win = dft_cut * window
    dft = fft.fft(dft_win, N)

    # some signal processing
    Pdft = dft**2
    
    # write output buffer
    ys[start: start + N] += np.real(Pdft)
