import tkinter as tk
from tkinter import filedialog, messagebox
from scipy.signal import butter, filtfilt
import numpy as np
import pywt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def openFile():
    filepath = filedialog.askopenfilename(title="Select File", filetypes=(("text files", "*.txt"), ("all files", "*.*")))
    if filepath:
        samples = readfile(filepath)
        return samples, filepath
    else:
        return None, None

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

def normalize_signal(samples, range_type='-1 to 1'):
    normalized_samples = []
    for sample_list in samples:
        min_val = np.min(sample_list)
        max_val = np.max(sample_list)
        if range_type == '-1 to 1':

            normalized_sample = 2 * (np.array(sample_list) - min_val) / (max_val - min_val) - 1
            normalized_samples.append(list(normalized_sample))

        elif range_type == '0 to 1':

            normalized_sample = (np.array(sample_list) - min_val) / (max_val - min_val)
            normalized_samples.append(list(normalized_sample))

        else:
            raise ValueError("range_type must be either '-1 to 1' or '0 to 1'.")

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
    samples = readfile(filepath)
    samples = meanRemoval(samples)
    samples = bandpass_filter(samples)
    samples = normalize_signal(samples)
    samples = resampling(samples)
    samples = wavelet(samples)
    return samples

def wavelet(samples, wavelet = 'db2', level = 1):
    new_samples = []
    for sample_list in samples:
        DWT = pywt.wavedec(data=sample_list, wavelet=wavelet, mode='symmetric', level=level)
        new_samples.append(np.concatenate(DWT))
    return np.array(new_samples)

def KNN(train_x, train_y, test_x, test_y):

    model = RandomForestClassifier(n_estimators=100, random_state=3)

    model.fit(train_x, train_y)

    train_pred = model.predict(train_x)
    test_pred = model.predict(test_x)

    train_acc = accuracy_score(train_y, train_pred)
    test_acc = accuracy_score(test_y, test_pred)

    train_report = classification_report(train_y, train_pred)
    test_report = classification_report(test_y, test_pred)

    return train_acc, test_acc, train_report, test_report
def KNNModel(train_x, train_y, x_test):
    model = RandomForestClassifier(n_estimators=100, random_state=3)
    model.fit(train_x, train_y)
    test_pred = model.predict(x_test)
    predictions = []
    for i, prediction in enumerate(test_pred):
        label = "Up" if prediction == 1 else "Down"
        predictions.append(f"Signal {i + 1}: {label}")
    return predictions
def processAndTrain():
    global train_acc, test_acc, train_report, test_report, predictions

    train_up = readfile("up&down/train_up.txt")
    train_down = readfile("up&down/train_down.txt")
    test_up = readfile("up&down/test_up.txt")
    test_down = readfile("up&down/test_down.txt")
    test_sample, test_sample_path = openFile()

    if not (train_up and train_down and test_up and test_down and test_sample):
        messagebox.showerror("Error", "Please provide all required files.")
        return

    train_x = np.concatenate([preProcessing("up&down/train_up.txt"), preProcessing("up&down/train_down.txt")])
    test_x = np.concatenate([preProcessing("up&down/test_up.txt"), preProcessing("up&down/test_down.txt")])

    train_y = np.concatenate([np.array([1]*len(train_up)), np.array([0]*len(train_down))])
    test_y = np.concatenate([np.array([1]*len(test_up)), np.array([0]*len(test_down))])

    train_acc, test_acc, train_report, test_report = KNN(train_x, train_y, test_x, test_y)

    x_test = preProcessing(test_sample_path)

    predictions = KNNModel(train_x, train_y, x_test,)

def displaylabel():
    result_text.set(f"Predicted Labels:\n" + "\n".join(predictions))

def displayAccuracy():
    result_text.set(f"Training Accuracy: {train_acc:.2f}\nTest Accuracy: {test_acc:.2f}")

def displayReport():
    result_text.set(f"Train Report:\n{train_report}\n\nTest Report:\n{test_report}")

root = tk.Tk()
root.title("Signal Processing and Classification")
root.configure(bg="#f0f0f0")


frame = tk.Frame(root, bg="#f0f0f0")
frame.pack(padx=100, pady=10)


btn_process = tk.Button(frame, text="Process and train", command=processAndTrain, bg="#d3d3d3")
btn_process.pack(pady=10)

btn_label = tk.Button(frame, text="Show up or down", command=displaylabel, bg="#d3d3d3")
btn_label.pack(pady=10)

btn_accuracy = tk.Button(frame, text="Show Accuracy", command=displayAccuracy, bg="#d3d3d3")
btn_accuracy.pack(pady=10)

btn_report = tk.Button(frame, text="Show Report", command=displayReport, bg="#d3d3d3")
btn_report.pack(pady=10)

result_text = tk.StringVar()
result_label = tk.Label(frame, textvariable=result_text, justify="left", anchor="w", bg="#f0f0f0")
result_label.pack(fill="both", padx=10, pady=10)

root.mainloop()

