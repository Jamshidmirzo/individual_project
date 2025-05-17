import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.signal import butter, lfilter

file_path = 'audio.wav'
y, sr = librosa.load(file_path, sr=None)

plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.title('Исходный аудиосигнал')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.tight_layout()
plt.show()

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype='low', analog=False)

def apply_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order)
    return lfilter(b, a, data)

cutoff_freq = 4000  
filtered = apply_filter(y, cutoff_freq, sr)

plt.figure(figsize=(10, 4))
librosa.display.waveshow(filtered, sr=sr)
plt.title('Отфильтрованный аудиосигнал')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.tight_layout()
plt.show()

def plot_spectrogram(signal, sr, title):
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

plot_spectrogram(y, sr, 'Спектрограмма ДО фильтрации')
plot_spectrogram(filtered, sr, 'Спектрограмма ПОСЛЕ фильтрации')
