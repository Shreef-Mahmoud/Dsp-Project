from tkinter import filedialog
from scipy.signal import butter, filtfilt
import numpy as np
import pywt

freq = 176
waveletf='db2'
levels=3

def openFile():
    filepath = filedialog.askopenfilename(title="Select File", filetypes=(("text files", ".txt"), ("all files", ".*")))
    if filepath:
        samples = readfile(filepath)

    return samples

def readfile(filepath):
    samples = []
    with open(filepath, 'r') as f:
        for line in f:
            values = [float(val) for val in line.strip().split('\t')]
            samples.append(list(values))

    return samples


def meanRemoval(samples):
    processed_samples = []
    for sample_list in samples:
        mean_value = np.mean(sample_list)
        processed_samples.append(list(np.array(sample_list) - mean_value))
    return processed_samples

def bandpass_filter(samples):
    nyquist = 0.5 * freq
    low = 0.5 / nyquist
    high = 20 / nyquist
    b, a = butter(5, [low, high], btype='bandpass')

    filtered_samples = []
    for sample_list in samples:
        filtered_signal = filtfilt(b, a, sample_list)
        filtered_samples.append(list(filtered_signal))
    return filtered_samples

def lowpass_filter(samples):
    nyquist = 0.5 * freq
    normalized_cutoff = 20 / nyquist
    b, a = butter(5, normalized_cutoff, btype='lowpass')

    filtered_signal = filtfilt(b, a, samples)

    return filtered_signal

def normalize_signal(samples):
    normalized_samples = [] 

    for sample_list in samples:
        min_val = np.min(sample_list)
        max_val = np.max(sample_list)

        normalized_sample = 2 * (sample_list - min_val) / (max_val - min_val) - 1
        normalized_samples.append(list(normalized_sample))

    return normalized_samples


def resampling(samples):
    m = 4
    resampled_samples = []
    
    for sample_list in samples:

        filtered_samples = lowpass_filter(sample_list)

        resampled_signal = filtered_samples[::m]
        resampled_samples.append(resampled_signal)

    return resampled_samples

def preProcessing():
    samples = openFile()
    samples = meanRemoval(samples)
    samples = bandpass_filter(samples)
    samples = normalize_signal(samples)
    samples = resampling(samples)
    return samples

def wavelet(samples):
    new_samples=[]
    for sample_list in samples:
        temp=pywt.wavedec(sample_list,waveletf,mode='symmetric',level=levels)
        n=temp[0]+temp[1]
        new_samples.append(n)

    return new_samples


samples = preProcessing()
samples=wavelet(samples)

for sample_list in samples:
    print(sample_list)
    print("\n\n")
