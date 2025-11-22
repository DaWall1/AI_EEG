import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import welch
import numpy as np


import kagglehub

# Download latest version
path = kagglehub.dataset_download("danizo/eeg-dataset-for-adhd")

print("Path to dataset files:", path)

# Se caută primul fișier CSV (Setul de date conține fișiere .csv )
for file in os.listdir(path):
    if file.endswith(".csv"):
        data_path = os.path.join(path, file)
        break

print("Using dataset file:", data_path)

# Se încarcă datele
df = pd.read_csv(data_path)


print(df.head())
print("\nColumns:", df.columns)


# --- Diagrama distribuției claselor ---
plt.figure(figsize=(6, 4))
class_counts = df["Class"].value_counts()
plt.bar(class_counts.index, class_counts.values, color=["#007acc", "#ff6347"])
plt.title("Distribuția claselor din setul EEG (ADHD vs Control)")
plt.xlabel("Clasă")
plt.ylabel("Număr de eșantioane")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


# --- Reprezentarea grafică a semnalului EEG ---
# Se selectează doar canalele EEG  (se exclud Class, ID)
channels = [col for col in df.columns if col not in ["Class", "ID"]]

# Semnalul se va reprezenta pe un interval de trei secunde
sampling_rate = 128  # Hz 
duration_seconds = 3
samples_to_plot = sampling_rate * duration_seconds

# Se alege un pacient(e.g. primul din setul de date)
subject_data = df[df["ID"] == df["ID"].iloc[0]]
subject_data = subject_data[channels].head(samples_to_plot)

# Se reprezintă primele canale EEG
plt.figure(figsize=(12, 6))
for i, ch in enumerate(channels[:5]):  # Primele 5
    plt.plot(subject_data[ch] + i * 100, label=ch)  # Offset pentru claritate

plt.title(f"EEG Signals - First {duration_seconds} Seconds")
plt.xlabel("Samples (time)")
plt.ylabel("Amplitude + offset")
plt.legend()
plt.tight_layout()
plt.show()



# --- Densitatea specctrală de putere (Welch) ---
plt.figure(figsize=(10, 6))
nperseg = 256  
for ch in channels[:5]:
    sig = subject_data[ch].astype(float).values
    # Dacă lungimea segmentului este mai scurtă decât nperseg, welch va atenționa; se va modifica valoarea nperseg dacă este necesar.
    f, Pxx = welch(sig, fs=sampling_rate, nperseg=min(nperseg, len(sig)))
    plt.semilogy(f, Pxx, label=ch)

plt.title("Power Spectral Density (Welch) - First 5 channels")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (V^2/Hz) [log scale]")
plt.xlim(0, sampling_rate / 2)
plt.legend()
plt.tight_layout()
plt.show()


