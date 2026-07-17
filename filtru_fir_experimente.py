import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, filtfilt, freqz

sampling_rate = 128
nyquist = 0.5 * sampling_rate

bands = {
    "delta": (0.5, 4),
    "theta": (4,   8),
    "alpha": (8,  13),
    "beta":  (13, 30),
    "gamma": (30, 63)
}

def apply_fir_filter(data, fs, lowcut, highcut, numtaps=51):
    nyq = 0.5 * fs
    if highcut >= nyq:
        highcut = nyq - 0.5
    taps = firwin(numtaps, [lowcut, highcut], pass_zero=False, fs=fs)
    return filtfilt(taps, 1.0, data, padlen=min(len(data) - 1, len(taps) - 1))

# -------------------------------------------------------------
# EXPERIMENT 1 — Frequency response of each band filter
#
# Illustrates exactly which frequencies each of the 5 FIR filters
# passes and which it rejects, making the abstract band
# boundaries (e.g. theta = 4-8 Hz) visually concrete.
# -------------------------------------------------------------

fig1, ax1 = plt.subplots(figsize=(9, 5))

for band_name, (low, high) in bands.items():
    nyq = 0.5 * sampling_rate
    high_adj = high if high < nyq else nyq - 0.5
    taps = firwin(51, [low, high_adj], pass_zero=False, fs=sampling_rate)
    w, h = freqz(taps, fs=sampling_rate, worN=2000)
    ax1.plot(w, 20 * np.log10(np.abs(h) + 1e-10), label=f"{band_name} ({low}-{high} Hz)")

ax1.set_xlabel("Frecvență (Hz)")
ax1.set_ylabel("Câștig (dB)")
ax1.set_title("Răspunsul în frecvență al fiecărui filtru FIR pe bandă")
ax1.set_ylim(-60, 5)
ax1.legend(fontsize=8)
plt.tight_layout()


# -------------------------------------------------------------
# EXPERIMENT 2 — Decomposing a multi-frequency synthetic signal
#
# A signal built from FIVE known sine waves, one per band, is
# constructed. Applying apply_fir_filter() for each band should
# recover (approximately) only that band's original sine wave,
# demonstrating that the filters correctly isolate band content.
# -------------------------------------------------------------

t = np.arange(0, 3, 1 / sampling_rate)  # 3-second segment, matching pipeline

# One representative frequency per band, well inside each band's range
band_test_freqs = {"delta": 2, "theta": 6, "alpha": 10, "beta": 20, "gamma": 40}

synthetic_multiband = sum(
    np.sin(2 * np.pi * f * t) for f in band_test_freqs.values()
)

fig2, axes2 = plt.subplots(len(bands) + 1, 1, figsize=(10, 1.4 * (len(bands) + 1)), sharex=True)

axes2[0].plot(t, synthetic_multiband, linewidth=0.8, color="black")
axes2[0].set_ylabel("Semnal\ncompozit", fontsize=8)
axes2[0].tick_params(labelsize=7)

for i, (band_name, (low, high)) in enumerate(bands.items(), start=1):
    recovered = apply_fir_filter(synthetic_multiband, sampling_rate, low, high)
    axes2[i].plot(t, recovered, linewidth=0.8)
    axes2[i].set_ylabel(f"{band_name}\n({band_test_freqs[band_name]} Hz)", fontsize=8)
    axes2[i].tick_params(labelsize=7)

axes2[-1].set_xlabel("Timp (s)")
fig2.suptitle("Filtrarea FIR recuperează componenta corespunzătoare fiecărei benzi")
plt.tight_layout()


# -------------------------------------------------------------
# EXPERIMENT 3 — Effect of numtaps on filter sharpness
#
# Compares the theta-band filter's frequency response for a low
# numtaps (e.g. 11) vs. the actual pipeline value (51) vs. a much
# higher value (151), illustrating the tradeoff between sharper
# transition bands and computational cost / filter length.
# -------------------------------------------------------------

fig3, ax3 = plt.subplots(figsize=(9, 5))

low, high = bands["theta"]
for n in [11, 51, 151]:
    taps = firwin(n, [low, high], pass_zero=False, fs=sampling_rate)
    w, h = freqz(taps, fs=sampling_rate, worN=2000)
    ax3.plot(w, 20 * np.log10(np.abs(h) + 1e-10), label=f"numtaps = {n}")

ax3.axvspan(low, high, color="gray", alpha=0.15, label="banda theta (4-8 Hz)")
ax3.set_xlim(0, 20)
ax3.set_xlabel("Frecvență (Hz)")
ax3.set_ylabel("Câștig (dB)")
ax3.set_title("Efectul numărului de coeficienți (numtaps) asupra selectivității filtrului")
ax3.legend(fontsize=8)
plt.tight_layout()


# -------------------------------------------------------------
# EXPERIMENT 4 — Edge effects without sufficient padding
#
# Demonstrates why padlen matters: applying filtfilt with little
# or no padding on a short signal can introduce distortion near
# the boundaries, which the padlen calculation in the pipeline
# guards against.
# -------------------------------------------------------------

short_signal = np.sin(2 * np.pi * 6 * np.arange(0, 0.5, 1 / sampling_rate))  # very short, 0.5s

low, high = bands["theta"]
taps = firwin(51, [low, high], pass_zero=False, fs=sampling_rate)

# Proper padding, as in the pipeline
proper = filtfilt(taps, 1.0, short_signal,
                   padlen=min(len(short_signal) - 1, len(taps) - 1))

# No padding at all
try:
    no_pad = filtfilt(taps, 1.0, short_signal, padlen=0)
except Exception as e:
    no_pad = np.full_like(short_signal, np.nan)
    print("No-padding case failed or distorted:", e)

t_short = np.arange(0, 0.5, 1 / sampling_rate)

fig4, ax4 = plt.subplots(figsize=(9, 4))
ax4.plot(t_short, short_signal, label="Semnal original", color="black", linewidth=1)
ax4.plot(t_short, proper, label="filtfilt cu padding corespunzător", linewidth=1.2)
ax4.plot(t_short, no_pad, label="filtfilt fără padding", linewidth=1.2, linestyle="--")
ax4.set_xlabel("Timp (s)")
ax4.set_ylabel("Amplitudine")
ax4.set_title("Importanța padding-ului pentru semnale scurte")
ax4.legend(fontsize=8)
plt.tight_layout()


plt.show()