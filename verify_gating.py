"""Verify that saved CSV data reflects VNA time-domain gating.

Usage:
    python verify_gating.py path/to/sweep_001.csv

Reads the Real/Imag columns, IFFTs to the time domain, and plots the
impulse response.  If gating is working, the energy should be truncated
near the gate stop time.  If gating was NOT captured, the impulse tail
extends out to ~20 ns (the full cable/fixture round-trip).
"""

import sys
import numpy as np
import matplotlib.pyplot as plt


def verify(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)

    freq = df["Frequency_GHz"].values * 1e9  # Hz
    s21 = df["S21_Real"].values + 1j * df["S21_Imag"].values

    n = len(freq)
    df_hz = freq[1] - freq[0] if n > 1 else 1.0

    # IFFT → time domain
    impulse = np.fft.ifft(s21)
    t = np.fft.ifftfreq(n, d=df_hz)  # seconds
    t_ns = np.fft.fftshift(t) * 1e9
    impulse_shifted = np.fft.fftshift(impulse)
    mag = 20 * np.log10(np.abs(impulse_shifted) + 1e-15)

    # Find where energy drops below -60 dB of peak
    peak_db = mag.max()
    cutoff = peak_db - 60
    above = np.where(mag > cutoff)[0]
    if len(above) > 1:
        extent_ns = t_ns[above[-1]] - t_ns[above[0]]
    else:
        extent_ns = 0

    print(f"File:            {csv_path}")
    print(f"Points:          {n}")
    print(f"Freq span:       {freq[0]/1e9:.3f} - {freq[-1]/1e9:.3f} GHz")
    print(f"Time resolution: {1/(n * df_hz) * 1e9:.3f} ns")
    print(f"Peak impulse:    {peak_db:.1f} dB")
    print(f"Energy extent:   {extent_ns:.1f} ns (at -60 dB from peak)")
    print()
    if extent_ns < 10:
        print("PASS: Impulse response is compact — gating appears active.")
    else:
        print("CHECK: Impulse response extends to {:.0f} ns — gating may "
              "not be captured.".format(extent_ns))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    ax1.plot(freq / 1e9, 20 * np.log10(np.abs(s21) + 1e-15))
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel("|S21| (dB)")
    ax1.set_title("Frequency domain")
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_ns, mag)
    ax2.set_xlabel("Time (ns)")
    ax2.set_ylabel("Impulse response (dB)")
    ax2.set_title("Time domain (IFFT of saved data)")
    ax2.set_xlim(-5, 30)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(cutoff, color='r', ls='--', alpha=0.5, label=f'-60 dB from peak')
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    verify(sys.argv[1])
