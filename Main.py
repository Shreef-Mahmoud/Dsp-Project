from tkinter import filedialog
from scipy.signal import butter, filtfilt
import numpy as np
import pywt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
def openFile(filepath):
    # filepath = filedialog.askopenfilename(title="Select File", filetypes=(("text files", ".txt"), ("all files", ".*")))
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

def bandpass_filter(samples, lowcut=0.5, highcut=20.0, fs=176, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='bandpass')

    filtered_samples = []
    for sample_list in samples:
        filtered_signal = filtfilt(b, a, sample_list)
        filtered_samples.append(list(filtered_signal))
    return filtered_samples

def lowpass_filter(samples, highcut=20.0, fs=176, order=5):
    nyquist = 0.5 * fs
    normalized_cutoff = highcut / nyquist
    b, a = butter(order, normalized_cutoff, btype='lowpass')

    filtered_signal = filtfilt(b, a, samples)

    return filtered_signal

def normalize_signal(samples):
    normalized_samples = []
    for sample_list in samples:
        min_val = np.min(sample_list)
        max_val = np.max(sample_list)
        normalized_sample = 2 * (np.array(sample_list) - min_val) / (max_val - min_val) - 1
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

def preProcessing(filepath):
    samples = openFile(filepath)
    samples = meanRemoval(samples)
    samples = bandpass_filter(samples)
    samples = normalize_signal(samples)
    samples = resampling(samples)
    return samples

def wavelet(samples, wavelet = 'db2', level = 1):
    new_samples = []
    for sample_list in samples:
        DWT = pywt.wavedec(data=sample_list, wavelet=wavelet, mode='symmetric', level=level)
        new_samples.append(np.concatenate(DWT))
    return np.array(new_samples)

def KNN(x_train, y_train, x_test, y_test):

    model = KNeighborsClassifier(n_neighbors = 1)

    model.fit(x_train, y_train)

    print("Train  : ")

    y_predict = model.predict(x_train)

    score = accuracy_score(y_train, y_predict)

    cr = classification_report(y_train, y_predict)

    cm = confusion_matrix(y_train, y_predict)

    print("Accuracy : ", score, "\n")
    print("Classification Report : \n", cr)
    print("Confusion Matrix : \n", cm)

    print("Test  : ")

    y_predict = model.predict(x_test)

    score = accuracy_score(y_test, y_predict)

    cr = classification_report(y_test, y_predict)

    cm = confusion_matrix(y_test, y_predict)

    print("Accuracy : ", score, "\n")
    print("Classification Report : \n", cr)
    print("Confusion Matrix : \n", cm)



samples_uptrain = preProcessing("up&down/train_up.txt")
samples_uptrain = wavelet(samples_uptrain)

samples_uptest = preProcessing("up&down/test_up.txt")
samples_uptest = wavelet(samples_uptest)

samples_downtrain = preProcessing("up&down/train_down.txt")
samples_downtrain = wavelet(samples_downtrain)

samples_downtest = preProcessing("up&down/test_down.txt")
samples_downtest = wavelet(samples_downtest)

train_x = np.concatenate([samples_uptrain, samples_downtrain])
test_x = np.concatenate([samples_uptest, samples_downtest])

train_y = np.concatenate([np.array([1]*len(samples_uptrain)), np.array([0]*len(samples_downtrain))])
test_y = np.concatenate([np.array([1]*len(samples_uptest)), np.array([0]*len(samples_downtest))])

KNN(train_x, train_y, test_x, test_y)