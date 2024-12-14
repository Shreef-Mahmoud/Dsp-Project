import numpy as np

freq = 176

def meanRemoval(samples):
    mean_value = np.mean(samples)
    samples = (samples - mean_value)
    return samples

# def bandPassFilter():

#     fs = freq
#     stop_attenuation = float(simpledialog.askstring("Input", "Enter the stop band attenuation (δs):"))
#     transition_band = float(simpledialog.askstring("Input", "Enter the transition band (Hz):"))

#     f1 = 0.5
#     f2 = 20

#     delta_f_normalized = transition_band / fs

    
#     cutoff_1 = (f1 - (transition_band / 2)) / fs
#     cutoff_2 = (f2 + (transition_band / 2)) / fs

#     if stop_attenuation <= 21:
#         constant = 0.9
#         window_name = "Rectangular"
#     elif stop_attenuation <= 44:
#         constant = 3.1
#         window_name = "Hanning"
#     elif stop_attenuation <= 53:
#         constant = 3.3
#         window_name = "Hamming"
#     else:
#         constant = 5.5
#         window_name = "Blackman"

#     N = int(np.ceil(constant / delta_f_normalized))
#     if N % 2 == 0:
#         N += 1
#     middle = N // 2

#     h_d = np.zeros(N)
#     for n in range(-middle, middle + 1):
#         if n == 0:
#             h_d[n + middle] = 2 * (cutoff_2 - cutoff_1)
#         else:
#             h_d[n + middle] = (np.sin(2 * np.pi * cutoff_2 * n) - np.sin(2 * np.pi * cutoff_1 * n)) / (np.pi * n)

#     n = np.arange(-middle, middle + 1)
#     if window_name == "Rectangular":
#         window = 1
#     elif window_name == "Hanning":
#         window = 0.5 + 0.5 * np.cos(2 * np.pi * n / N)
#     elif window_name == "Hamming":
#         window = 0.54 + 0.46 * np.cos(2 * np.pi * n / N)
#     elif window_name == "Blackman":
#         window = 0.42 + 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))

#     h = h_d * window



def normalize_signal(sample):

    min_val = np.min(sample)
    max_val = np.max(sample)

    normalized_sample = 2 * (sample - min_val) / (max_val - min_val) - 1

    return normalized_sample


# def lowPassFilter(index , sample ):

#     fs = freq

#     stop_attenuation = float(simpledialog.askstring("Input", "Enter the stop band attenuation (δs):"))
#     transition_band = float(simpledialog.askstring("Input", "Enter the transition band (Hz):"))


#     cutoff_freq = float(simpledialog.askstring("Input", "Enter the cutoff frequency (Hz):"))

#     delta_f_normalized = transition_band / fs

#     cutoff = (cutoff_freq + (transition_band / 2)) / fs

#     if stop_attenuation <= 21:
#         constant = 0.9
#         window_name = "Rectangular"
#     elif stop_attenuation <= 44:
#         constant = 3.1
#         window_name = "Hanning"
#     elif stop_attenuation <= 53:
#         constant = 3.3
#         window_name = "Hamming"
#     else:
#         constant = 5.5
#         window_name = "Blackman"

#     N = int(np.ceil(constant / delta_f_normalized))
#     if N % 2 == 0:
#         N += 1
#     middle = N // 2

#     h_d = np.zeros(N)
#     for n in range(-middle, middle + 1):
#         if n == 0:
#             h_d[n + middle] = 2 * cutoff
#         else:
#             h_d[n + middle] = np.sin(2 * np.pi * cutoff * n) / (np.pi * n)

#     n = np.arange(-middle, middle + 1)
#     if window_name == "Rectangular":
#         window = 1
#     elif window_name == "Hanning":
#         window = 0.5 + 0.5 * np.cos(2 * np.pi * n / N)
#     elif window_name == "Hamming":
#         window = 0.54 + 0.46 * np.cos(2 * np.pi * n / N)
#     elif window_name == "Blackman":
#         window = 0.42 + 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))

#     h = h_d * window

def resampling(indices, samples):
    m = 4

    # filtered_indices, filtered_samples = FIR(indices, samples)

    filtered_samples = filtered_samples[::m]
    filtered_indices = filtered_indices[::m]
    index_temp = []
    i = 0
    x = filtered_indices[0]
    while (True):
        index_temp.append(x)
        i += 1
        x += 1
        if i == len(filtered_indices):
            break


        filtered_indices = index_temp

    while filtered_samples and filtered_samples[-1] == 0:
        filtered_samples.pop()    
        filtered_indices.pop()

    return filtered_samples , filtered_indices