import numpy as np
import matplotlib.pyplot as plt
import scikits.audiolab as al

file_path= "/home/miura/grive/separated_ls5k"
wav_file = "/03001_03039/Respiratory_sounds/030010500_5k_breath.wav"
wav_path = file_path+wav_file

#wavread
data,fs,fmt = al.wavread(wav_path)
f = al.Sndfile(wav_path)
frame = f.read_frames(f.nframes)
data = np.array(frame,dtype=np.float64)

f.close()

n = 2048

dft = np.fft.fft(data,n)

