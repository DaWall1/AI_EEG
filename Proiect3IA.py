
import os
import numpy as np
import pandas as pd
import kagglehub

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, DepthwiseConv2D,
                                     BatchNormalization, ELU, AveragePooling2D,
                                     Dropout, Flatten, Dense, Permute,
                                     SeparableConv2D)
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.losses import BinaryFocalCrossentropy

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

from scipy.signal import iirnotch, filtfilt, detrend, firwin

# -------------------------------
# 1. REPRODUCTIBILITATE
# -------------------------------

os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

np.random.seed(42)
tf.random.set_seed(42)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

# -------------------------------
# 2. DESCĂRCAREA SETULUI DE DATE
# -------------------------------

path = kagglehub.dataset_download("danizo/eeg-dataset-for-adhd")

data_path = ""
for file in os.listdir(path):
    if file.endswith(".csv"):
        data_path = os.path.join(path, file)
        break

df = pd.read_csv(data_path)

sampling_rate = 128
channels = [c for c in df.columns if c not in ["Class", "ID"]]
n_channels = len(channels)

# -------------------------------
# 3. PROCESAREA SEMNALULUI
#
# EEGNet primește același tensor (T, C, B) descompus pe benzi
# utilizat de celelalte patru arhitecturi, asigurând un format
# de intrare și un buget de antrenare identice pentru toate.
# -------------------------------

def clean_and_normalize(data, fs=128):
    data = data.astype(np.float64)
    data = detrend(data)                        # elimină variația lentă a nivelului de referință
    data -= np.mean(data)

    nyquist = 0.5 * fs
    b_notch, a_notch = iirnotch(50.0 / nyquist, Q=30)
    data = filtfilt(b_notch, a_notch, data)     # filtrare notch la 50 Hz, fără distorsiune de fază

    std = np.std(data)
    return data / std if std != 0 else data


def apply_fir_filter(data, fs, lowcut, highcut, numtaps=51):
    nyq = 0.5 * fs
    if highcut >= nyq:
        highcut = nyq - 0.5                     # protecție la limita Nyquist (banda gamma)

    taps = firwin(numtaps, [lowcut, highcut], pass_zero=False, fs=fs)

    return filtfilt(taps, 1.0, data,
                    padlen=min(len(data) - 1, len(taps) - 1))

# -------------------------------
# 4. AUGMENTARE (generator seeded)
# -------------------------------

rng = np.random.default_rng(42)

def add_gaussian_noise(X, noise_std=0.02, prob=0.5):
    # Zgomot gaussian proporțional cu deviația standard a segmentului
    for i in range(len(X)):
        if rng.random() < prob:
            std = np.std(X[i])
            X[i] += rng.normal(0, noise_std * std, X[i].shape)
    return X


def temporal_jitter(X, max_shift=10, prob=0.5):
    # Deplasare temporală ciclică: max ±10 eșantioane (≈78 ms)
    for i in range(len(X)):
        if rng.random() < prob:
            shift = rng.integers(-max_shift, max_shift + 1)
            X[i] = np.roll(X[i], shift, axis=0)
    return X


def amplitude_scale(X, scale_min=0.8, scale_max=1.2, prob=0.5):
    # Scalare uniformă a amplitudinii: factor ∈ [0.8, 1.2]
    for i in range(len(X)):
        if rng.random() < prob:
            factor = rng.uniform(scale_min, scale_max)
            X[i] *= factor
    return X

# -------------------------------
# 5. BENZI DE FRECVENȚĂ
# -------------------------------

bands = {
    "delta": (0.5, 4),
    "theta": (4,   8),
    "alpha": (8,  13),
    "beta":  (13, 30),
    "gamma": (30, 63)
}

segment_length = 384    # 3 secunde la 128 Hz
overlap        = 0.5
step           = int(segment_length * (1 - overlap))   # pas = 192 eșantioane

# -------------------------------
# 6. PREGĂTIREA DATELOR
#
# Forma unui segment: (T, C, B)
#   T = eșantioane de timp (384)
#   C = canale EEG
#   B = benzi de frecvență (5)
# -------------------------------

X_list, y_list, subject_ids = [], [], []

print("Pregătire segmente EEG...")

for i, subject_id in enumerate(df["ID"].unique()):
    print(f"  Subiect {i+1}...", flush=True)

    subject_df = df[df["ID"] == subject_id]
    label      = subject_df["Class"].iloc[0]

    # Curățare și normalizare per canal
    cleaned = np.stack([
        clean_and_normalize(subject_df[ch].values)
        for ch in channels
    ], axis=1)

    # Normalizare la nivel de subiect
    cleaned = (cleaned - cleaned.mean(axis=0)) / (cleaned.std(axis=0) + 1e-8)

    # Filtrare FIR pe fiecare bandă și canal
    filtered = {band: [] for band in bands}
    for band, (low, high) in bands.items():
        for ch in range(cleaned.shape[1]):
            filtered[band].append(
                apply_fir_filter(cleaned[:, ch], sampling_rate, low, high)
            )

    signal_length = len(subject_df)

    for start in range(0, signal_length - segment_length + 1, step):

        # Stivuire benzi → tensor (T, C, B)
        band_stack = []
        for band in bands:
            seg = np.array([
                filtered[band][ch][start:start + segment_length]
                for ch in range(len(channels))
            ]).T
            band_stack.append(seg)

        X_list.append(np.stack(band_stack, axis=-1))
        y_list.append(label)
        subject_ids.append(subject_id)

X      = np.array(X_list)       # (N, T, C, B)
y      = LabelEncoder().fit_transform(y_list)
groups = np.array(subject_ids)

print("Forma setului de date:", X.shape)

# -------------------------------
# 7. ARHITECTURA EEGNet — intrare descompusă pe benzi
#
# F1=16 (față de valoarea originală de 8), pentru capacitate
# suplimentară dată de intrarea pe 5 benzi.
#
# Bloc 1 — convoluție temporală (1, kern_length):
#   Detectează tipare temporale simultan pe toate cele 5 benzi.
#
# Bloc 1 — convoluție spațială depthwise (C, 1):
#   Combină electrozii per bandă. Benzile sunt tratate ca axă
#   de adâncime, similar canalelor de culoare dintr-o imagine.
#
# Bloc 2 — convoluție separabilă (1, 16):
#   Rafinează tiparele temporale din reprezentarea spațială.
# -------------------------------

def build_eegnet(input_shape, F1=16, D=2, dropout_rate=0.5, kern_length=64):
    """
    F1          : filtre temporale
    D           : F2 = F1 * D = 32
    kern_length : ≈0.5 s la 128 Hz — captează cicluri theta/alpha
    input_shape : (T, C, B)
    """
    T, C, B = input_shape
    F2 = F1 * D

    inputs = Input(shape=input_shape)           # (T, C, B)

    # Reordonare axe: (T, C, B) → (C, T, B)
    x = Permute((2, 1, 3))(inputs)

    # ---- Bloc 1: Convoluție temporală ------------------------------------------------
    x = Conv2D(F1, (1, kern_length),
               padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # Convoluție spațială — acoperă toți electrozii simultan, per bandă
    x = DepthwiseConv2D((C, 1),
                         depth_multiplier=D,
                         use_bias=False,
                         depthwise_constraint=max_norm(1.))(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = AveragePooling2D((1, 4))(x)
    x = Dropout(dropout_rate)(x)

    # ---- Bloc 2: Convoluție separabilă -----------------------------------------------
    x = SeparableConv2D(F2, (1, 16),
                         padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = AveragePooling2D((1, 8))(x)
    x = Dropout(dropout_rate)(x)

    # ---- Clasificare -----------------------------------------------------------------
    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss=BinaryFocalCrossentropy(gamma=2),
        metrics=['accuracy']
    )

    return model

# -------------------------------
# 8. VALIDARE ÎNCRUCIȘATĂ + AGREGARE LA NIVEL DE SUBIECT
#
# Configurație identică celorlalte patru pipeline-uri:
# epochs=25, EarlyStopping patience=5, ReduceLROnPlateau patience=3.
# -------------------------------

gkf         = StratifiedGroupKFold(n_splits=5)
input_shape = (X.shape[1], X.shape[2], X.shape[3])

fold_accuracies = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):

    print(f"\n--- Fold {fold+1} ---")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_test     = groups[test_idx]

    print(f"  Subiecți testare : {len(np.unique(groups_test))}")
    print(f"  Distribuție antrenare: {np.bincount(y_train)}  |  Testare: {np.bincount(y_test)}")

    # Augmentare exclusiv pe datele de antrenare
    X_train = add_gaussian_noise(X_train)
    X_train = temporal_jitter(X_train)
    X_train = amplitude_scale(X_train)

    # Ponderi de clasă calculate per fold
    cw      = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw_dict = dict(enumerate(cw))

    # Resetare seed per fold — inițializare diferită dar reproductibilă
    tf.random.set_seed(42 + fold)
    np.random.seed(42 + fold)

    model = build_eegnet(input_shape)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=3, min_lr=1e-5
        )
    ]

    model.fit(
        X_train, y_train,
        epochs=25,
        batch_size=16,
        class_weight=cw_dict,
        callbacks=callbacks,
        verbose=1
    )

    # ---------------- AGREGARE LA NIVEL DE SUBIECT ----------------
    preds = model.predict(X_test, verbose=0).flatten()

    subject_dict   = {}
    subject_labels = {}

    for p, subj, label in zip(preds, groups_test, y_test):
        subject_dict.setdefault(subj, []).append(p)
        subject_labels[subj] = label

    final_preds  = []
    final_labels = []

    for subj in subject_dict:
        # Media predicțiilor per segment → predicție unică per subiect
        avg_pred = np.mean(subject_dict[subj])
        final_preds.append(1 if avg_pred > 0.5 else 0)
        final_labels.append(subject_labels[subj])

    acc = np.mean(np.array(final_preds) == np.array(final_labels))
    fold_accuracies.append(acc)

    print(f"  Acuratețe la nivel de subiect: {acc:.4f}")

# -------------------------------
# 9. REZULTATE FINALE
# -------------------------------

print("\n==========================")
print(f"Acuratețe medie CV : {np.mean(fold_accuracies):.4f}")
print(f"Deviație standard  : {np.std(fold_accuracies):.4f}")
print("==========================")