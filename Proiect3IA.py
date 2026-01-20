import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import kagglehub
from scipy.signal import welch, iirnotch, filtfilt, butter, detrend, firwin
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import GroupKFold

# --- 1. DOWNLOAD AND LOAD DATA ---
path = kagglehub.dataset_download("danizo/eeg-dataset-for-adhd")

data_path = ""
for file in os.listdir(path):
    if file.endswith(".csv"):
        data_path = os.path.join(path, file)
        break

df = pd.read_csv(data_path)
sampling_rate = 128  # Hz
channels = [col for col in df.columns if col not in ["Class", "ID"]]

# --- 2. DEFINE FIR FILTERING FOR BANDS ---

def apply_fir_filter(data, fs, lowcut, highcut, numtaps=51):
    """Applies a zero-phase FIR bandpass filter."""
    nyq = 0.5 * fs
    # Handle cases where highcut might be near Nyquist
    if highcut >= nyq:
        highcut = nyq - 0.5
    
    # Create FIR coefficients
    taps = firwin(numtaps, [lowcut, highcut], pass_zero=False, fs=fs)
    # Apply filtfilt for zero-phase distortion with reduced padding
    return filtfilt(taps, 1.0, data, padlen=min(len(data) - 1, len(taps) - 1))

def clean_and_normalize(data, fs=128):
    """Basic cleaning, 50 Hz notch filtering, and Z-score normalization."""
    
    # Cast to float
    data = data.astype(np.float64)
    
    # Detrend (removes slow linear drift)
    data = detrend(data)
    
    # Remove DC offset
    data -= np.mean(data)
    
    # --- 50 Hz NOTCH FILTER ---
    nyquist = 0.5 * fs
    b_notch, a_notch = iirnotch(50.0 / nyquist, Q=30)
    data = filtfilt(b_notch, a_notch, data)
    
    # Z-score normalization
    std = np.std(data)
    return data / std if std != 0 else data



# --- 3. TRANSFORM DATA FOR CNN (Frequency Band Stacking) ---

# --- 3. TRANSFORM DATA FOR CNN (Frequency Band Stacking - FIXED) ---

bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 63)
}

segment_length = 128
overlap = 0.5
step = int(segment_length * (1 - overlap))

X_list = []
y_list = []
subject_ids = []

print("Pre-filtering subjects (this should take seconds, not forever)...")

for subj_idx, subject_id in enumerate(df["ID"].unique()):
    subject_df = df[df["ID"] == subject_id]
    label = subject_df["Class"].iloc[0]

    # --- 1. Clean raw signals ---
    cleaned = {}
    for ch in channels:
        cleaned[ch] = clean_and_normalize(subject_df[ch].values)

    # --- 2. FIR band filtering ONCE per channel ---
    filtered = {band: {} for band in bands}

    for band, (low, high) in bands.items():
        for ch in channels:
            filtered[band][ch] = apply_fir_filter(
                cleaned[ch], sampling_rate, low, high
            )

    signal_length = len(subject_df)

    # --- 3. Segment AFTER filtering ---
    for start in range(0, signal_length - segment_length + 1, step):
        band_stack = []

        for band in bands:
            # Shape: (time, channels)
            band_segment = np.array([
                filtered[band][ch][start:start + segment_length]
                for ch in channels
            ]).T

            band_stack.append(band_segment)

        # Final shape: (time, channels, bands)
        X_list.append(np.stack(band_stack, axis=-1))
        y_list.append(label)
        subject_ids.append(subject_id)


    if (subj_idx + 1) % 10 == 0:
        print(f"Processed {subj_idx + 1} subjects")

X = np.array(X_list)
y = LabelEncoder().fit_transform(y_list)
groups = np.array(subject_ids)


print(f"Data prepared. Input shape: {X.shape}")



# --- 4. MODEL ARCHITECTURE FUNCTION ---
# We wrap the model in a function so we can re-initialize it for each fold
def build_model(input_shape):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- 5. CROSS-VALIDATION LOOP ---

n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

# Optional: Shuffle labels to test for data leakage/memorization
# y = np.random.permutation(y) # Uncomment this to run a "Sanity Check"

fold_accuracies = []
print(f"Starting {n_splits}-fold Group Cross-Validation...\n")

# X shape: (samples, time_steps, channels, bands)
input_shape = (X.shape[1], X.shape[2], X.shape[3])

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Ensure no subject overlap
    train_subjects = set(groups[train_idx])
    test_subjects = set(groups[test_idx])
    overlap = train_subjects.intersection(test_subjects)
    
    print(f"--- Fold {fold + 1} ---")
    print(f"Subjects in Train: {len(train_subjects)}, Test: {len(test_subjects)} | Overlap: {len(overlap)}")
    
    # Rebuild model from scratch for each fold
    model = build_model(input_shape)
    
    # Train (reduced epochs for speed during CV)
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1) 
    
    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    fold_accuracies.append(acc)
    print(f"Fold {fold + 1} Accuracy: {acc:.4f}")

print("\n" + "="*30)
print(f"Mean CV Accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
print("="*30) 
