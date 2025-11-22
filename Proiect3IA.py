import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import welch
import numpy as np


import kagglehub

# Download latest version
path = kagglehub.dataset_download("danizo/eeg-dataset-for-adhd")

print("Path to dataset files:", path)

# Find the first CSV file (dataset contains .csv files)
for file in os.listdir(path):
    if file.endswith(".csv"):
        data_path = os.path.join(path, file)
        break

print("Using dataset file:", data_path)

# Load the data
df = pd.read_csv(data_path)

# Display basic info
print(df.head())
print("\nColumns:", df.columns)


# --- CLASS DISTRIBUTION PLOT ---
plt.figure(figsize=(6, 4))
class_counts = df["Class"].value_counts()
plt.bar(class_counts.index, class_counts.values, color=["#007acc", "#ff6347"])
plt.title("Distribuția claselor din setul EEG (ADHD vs Control)")
plt.xlabel("Clasă")
plt.ylabel("Număr de eșantioane")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


# --- EEG SIGNAL PLOT ---
# Select only EEG channels (exclude Class, ID)
channels = [col for col in df.columns if col not in ["Class", "ID"]]

# Select a few seconds worth of data
sampling_rate = 128  # Hz (given in dataset description)
duration_seconds = 3
samples_to_plot = sampling_rate * duration_seconds

# Pick one patient (e.g., first in dataset)
subject_data = df[df["ID"] == df["ID"].iloc[0]]
subject_data = subject_data[channels].head(samples_to_plot)

# Plot a few EEG channels
plt.figure(figsize=(12, 6))
for i, ch in enumerate(channels[:5]):  # Plot first 5 channels
    plt.plot(subject_data[ch] + i * 100, label=ch)  # Offset for clarity

plt.title(f"EEG Signals - First {duration_seconds} Seconds")
plt.xlabel("Samples (time)")
plt.ylabel("Amplitude + offset")
plt.legend()
plt.tight_layout()
plt.show()



# --- POWER SPECTRAL DENSITY (Welch) ---
plt.figure(figsize=(10, 6))
nperseg = 256  # adjust if necessary (<= segment length)
for ch in channels[:5]:
    sig = subject_data[ch].astype(float).values
    # If the segment length is shorter than nperseg, welch will warn; adjust nperseg if needed.
    f, Pxx = welch(sig, fs=sampling_rate, nperseg=min(nperseg, len(sig)))
    plt.semilogy(f, Pxx, label=ch)

plt.title("Power Spectral Density (Welch) - First 5 channels")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (V^2/Hz) [log scale]")
plt.xlim(0, sampling_rate / 2)
plt.legend()
plt.tight_layout()
plt.show()
