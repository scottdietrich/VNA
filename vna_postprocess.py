#!/usr/bin/env python3
"""VNA-FMR Post-Processing Tool

Standalone tool for processing 2D VNA-FMR measurement data.
Loads sweep_NNN.csv files from a measurement folder and applies:

  - Time-domain gating: removes standing-wave reflections by
    IFFT → gate → FFT on each spectrum independently.
  - SVD background removal: removes dominant standing-wave modes
    from the 2D map.
  - Reference map subtraction: point-by-point subtraction of a
    reference measurement (e.g., off-resonance B-field).

Usage:
    python vna_postprocess.py [measurement_folder]

If no folder is given, a file dialog opens.

Requirements: numpy, matplotlib (tkinter is stdlib)
Optional: scipy (for SavGol smoothing)
"""

import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# ───────────────────────────────────────────────────────────
#  Data loading
# ───────────────────────────────────────────────────────────

def load_measurement(folder):
    """Load a 2D measurement from a folder of sweep_NNN.csv files.

    Returns:
        freq: 1D array of frequencies (GHz)
        step_vals: 1D array of step parameter values
        raw_db: 2D array (n_steps, n_freq) of S21 magnitude in dB
        s21_complex: 2D array (n_steps, n_freq) of complex S21
        norm_db: 2D array or None — pre-measured normalization
        metadata: dict of metadata from first file
    """
    sweep_files = sorted(glob.glob(os.path.join(folder, "sweep_*.csv")))
    if not sweep_files:
        raise FileNotFoundError(f"No sweep_*.csv files in {folder}")

    freq = None
    step_vals = []
    raw_db_list = []
    s21_complex_list = []
    norm_db_list = []
    metadata = {}

    for sf in sweep_files:
        with open(sf) as f:
            file_lines = f.readlines()

        # Extract metadata from comments
        for line in file_lines:
            if '# Step Value:' in line:
                step_vals.append(float(line.split(':')[1].strip()))
            if not metadata:
                if '# S-Parameter:' in line:
                    metadata['s_param'] = line.split(':')[1].strip()
                if '# Date:' in line:
                    metadata['date'] = line.split(':', 1)[1].strip()

        # Parse CSV data
        data_lines = [l.strip() for l in file_lines if not l.startswith('#')]
        header = data_lines[0].split(',')
        rows = []
        for dl in data_lines[1:]:
            rows.append([float(x) for x in dl.split(',')])
        arr = np.array(rows)

        # Column indices
        cols = {h.strip(): i for i, h in enumerate(header)}
        f_col = cols.get('Frequency_GHz', 0)
        re_col = cols.get('S21_Real', 1)
        im_col = cols.get('S21_Imag', 2)
        db_col = cols.get('S21_Mag_dB', 3)
        norm_col = cols.get('S21_Mag_dB_Normalized', None)

        if freq is None:
            freq = arr[:, f_col]

        raw_db_list.append(arr[:, db_col])
        s21_complex_list.append(arr[:, re_col] + 1j * arr[:, im_col])

        if norm_col is not None and norm_col < arr.shape[1]:
            norm_db_list.append(arr[:, norm_col])

    raw_db = np.array(raw_db_list)
    s21_complex = np.array(s21_complex_list)
    norm_db = np.array(norm_db_list) if norm_db_list else None
    step_vals = np.array(step_vals) if step_vals else np.arange(len(raw_db_list))

    return freq, step_vals, raw_db, s21_complex, norm_db, metadata


# ───────────────────────────────────────────────────────────
#  Processing functions (pure numpy, no GUI dependencies)
# ───────────────────────────────────────────────────────────

def time_gate_spectrum(s21, df_hz, gate_span_ns):
    """Time-domain gate a single complex spectrum.

    Returns gated S21 (complex) and peak arrival time (ns).
    """
    n = len(s21)
    dt_ns = 1e9 / (n * df_hz)

    h = np.fft.ifft(s21)
    h_mag = np.abs(h)

    # Find main through-path peak (first quarter)
    peak_idx = np.argmax(h_mag[:max(n // 4, 10)])
    peak_ns = peak_idx * dt_ns

    # Tukey gate
    half = max(int(gate_span_ns / (2 * dt_ns)), 2)
    gate = np.zeros(n)
    for k in range(-half, half + 1):
        idx = (peak_idx + k) % n
        frac = abs(k) / half
        if frac < 0.75:
            gate[idx] = 1.0
        else:
            gate[idx] = 0.5 * (1 + np.cos(np.pi * (frac - 0.75) / 0.25))

    return np.fft.fft(h * gate), peak_ns


def apply_time_gating(s21_complex, freq_ghz, gate_span_ns, callback=None):
    """Gate all spectra. callback(i, n) is called after each trace."""
    n_traces, n_pts = s21_complex.shape
    df_hz = (freq_ghz[-1] - freq_ghz[0]) * 1e9 / (n_pts - 1)

    gated = np.zeros_like(s21_complex)
    peak_times = []

    for i in range(n_traces):
        gated[i], pt = time_gate_spectrum(s21_complex[i], df_hz, gate_span_ns)
        peak_times.append(pt)
        if callback:
            callback(i + 1, n_traces)

    return gated, np.mean(peak_times)


def apply_svd(z, n_remove):
    """Remove the first n_remove SVD modes from 2D array z.
    Returns (filtered_z, removed_pct).
    """
    U, S, Vt = np.linalg.svd(z, full_matrices=False)
    total = np.sum(S**2)
    removed = np.sum(S[:n_remove]**2)
    pct = 100 * removed / total if total > 0 else 0
    S_f = S.copy()
    S_f[:n_remove] = 0
    return U @ np.diag(S_f) @ Vt, pct


# ───────────────────────────────────────────────────────────
#  GUI
# ───────────────────────────────────────────────────────────

class PostProcessApp:
    def __init__(self, root, folder=None):
        self.root = root
        self.root.title("VNA-FMR Post-Processing")
        self.root.geometry("1200x800")

        # Data storage
        self.freq = None
        self.step_vals = None
        self.raw_db = None
        self.s21_complex = None
        self.norm_db = None
        self.ref_raw_db = None
        self.ref_s21_complex = None
        self.ref_step_vals = None
        self.metadata = {}
        self.folder = None

        self._build_gui()

        if folder:
            self._load_folder(folder)

    def _build_gui(self):
        # Top control bar
        ctrl = ttk.Frame(self.root)
        ctrl.pack(fill='x', padx=5, pady=5)

        ttk.Button(ctrl, text="Load Measurement...",
                   command=self._browse_load).pack(side='left', padx=5)
        ttk.Button(ctrl, text="Load Reference Map...",
                   command=self._browse_ref).pack(side='left', padx=5)
        ttk.Button(ctrl, text="Clear Reference",
                   command=self._clear_ref).pack(side='left', padx=2)

        ttk.Separator(ctrl, orient='vertical').pack(side='left', fill='y', padx=10)

        ttk.Button(ctrl, text="Process",
                   command=self._process).pack(side='left', padx=5)
        ttk.Button(ctrl, text="Export CSV...",
                   command=self._export).pack(side='left', padx=5)

        self.status_var = tk.StringVar(value="No data loaded")
        ttk.Label(ctrl, textvariable=self.status_var,
                  font=('Arial', 9)).pack(side='right', padx=10)

        # Options frame
        opts = ttk.LabelFrame(self.root, text="Processing Options")
        opts.pack(fill='x', padx=5, pady=2)

        # Row 1: Display mode
        r1 = ttk.Frame(opts)
        r1.pack(fill='x', padx=5, pady=2)

        ttk.Label(r1, text="Display:").pack(side='left', padx=2)
        self.display_mode = tk.StringVar(value="Normalized")
        for m in ["Magnitude", "Normalized", "Phase"]:
            ttk.Radiobutton(r1, text=m, variable=self.display_mode,
                            value=m).pack(side='left', padx=3)

        ttk.Separator(r1, orient='vertical').pack(side='left', fill='y', padx=10)

        # Smoothing
        ttk.Label(r1, text="Smooth:").pack(side='left', padx=2)
        self.smooth_pts = tk.IntVar(value=1)
        ttk.Spinbox(r1, from_=1, to=101, increment=2,
                    textvariable=self.smooth_pts, width=4).pack(side='left', padx=2)
        ttk.Label(r1, text="pts").pack(side='left')

        # Row 2: Time-domain gating
        r2 = ttk.Frame(opts)
        r2.pack(fill='x', padx=5, pady=2)

        self.gate_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(r2, text="Time-domain gating",
                        variable=self.gate_enabled).pack(side='left', padx=2)
        ttk.Label(r2, text="Span:").pack(side='left', padx=(10, 2))
        self.gate_span = tk.StringVar(value="1.0")
        ttk.Entry(r2, textvariable=self.gate_span, width=5).pack(side='left')
        ttk.Label(r2, text="ns").pack(side='left', padx=2)

        ttk.Separator(r2, orient='vertical').pack(side='left', fill='y', padx=10)

        # SVD
        self.svd_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(r2, text="SVD removal",
                        variable=self.svd_enabled).pack(side='left', padx=2)
        ttk.Label(r2, text="Modes:").pack(side='left', padx=(5, 2))
        self.svd_modes = tk.IntVar(value=3)
        ttk.Spinbox(r2, from_=1, to=15, increment=1,
                    textvariable=self.svd_modes, width=3).pack(side='left')

        ttk.Separator(r2, orient='vertical').pack(side='left', fill='y', padx=10)

        # Ref map
        self.ref_enabled = tk.BooleanVar(value=False)
        self.ref_cb = ttk.Checkbutton(r2, text="Subtract ref map",
                                       variable=self.ref_enabled, state='disabled')
        self.ref_cb.pack(side='left', padx=2)

        self.ref_label = ttk.Label(r2, text="", font=('Arial', 8), foreground='gray')
        self.ref_label.pack(side='left', padx=5)

        # Row 3: color scale
        r3 = ttk.Frame(opts)
        r3.pack(fill='x', padx=5, pady=2)

        self.auto_color = tk.BooleanVar(value=True)
        ttk.Checkbutton(r3, text="Auto color",
                        variable=self.auto_color).pack(side='left', padx=2)
        ttk.Label(r3, text="Min:").pack(side='left', padx=(10, 2))
        self.color_min = tk.StringVar(value="-0.05")
        ttk.Entry(r3, textvariable=self.color_min, width=6).pack(side='left')
        ttk.Label(r3, text="Max:").pack(side='left', padx=(5, 2))
        self.color_max = tk.StringVar(value="0.05")
        ttk.Entry(r3, textvariable=self.color_max, width=6).pack(side='left')

        self.info_label = ttk.Label(r3, text="", font=('Arial', 8), foreground='gray')
        self.info_label.pack(side='right', padx=10)

        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode='determinate')
        self.progress.pack(fill='x', padx=5, pady=2)

        # Matplotlib figure
        self.fig = Figure(figsize=(10, 6))
        self.ax = self.fig.add_subplot(111)

        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, canvas_frame)
        toolbar.update()

    # ── Load/browse ──

    def _browse_load(self):
        folder = filedialog.askdirectory(title="Select measurement folder")
        if folder:
            self._load_folder(folder)

    def _load_folder(self, folder):
        try:
            self.freq, self.step_vals, self.raw_db, self.s21_complex, \
                self.norm_db, self.metadata = load_measurement(folder)
            self.folder = folder
            n_s, n_f = self.raw_db.shape
            name = os.path.basename(folder)
            self.status_var.set(f"Loaded: {name}  ({n_s} steps × {n_f} freq pts)")
            self._process()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load:\n{e}")

    def _browse_ref(self):
        folder = filedialog.askdirectory(title="Select reference map folder")
        if folder:
            try:
                _, ref_steps, ref_db, ref_cx, _, _ = load_measurement(folder)
                self.ref_raw_db = ref_db
                self.ref_s21_complex = ref_cx
                self.ref_step_vals = ref_steps
                self.ref_enabled.set(True)
                self.ref_cb.config(state='normal')
                self.ref_label.config(
                    text=f"{ref_db.shape[0]}×{ref_db.shape[1]} from {os.path.basename(folder)}",
                    foreground='green')
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load ref:\n{e}")

    def _clear_ref(self):
        self.ref_raw_db = None
        self.ref_s21_complex = None
        self.ref_step_vals = None
        self.ref_enabled.set(False)
        self.ref_cb.config(state='disabled')
        self.ref_label.config(text="", foreground='gray')

    # ── Processing pipeline ──

    def _process(self):
        if self.raw_db is None:
            return

        self.progress['value'] = 0
        self.root.update_idletasks()

        mode = self.display_mode.get()
        info_parts = []

        # Stage 1: Select data source (complex or magnitude)
        if self.gate_enabled.get():
            # Gate needs complex data
            self.status_var.set("Stage 1/4: Time-domain gating...")
            self.root.update_idletasks()

            try:
                span = float(self.gate_span.get())
            except ValueError:
                span = 1.0

            def gate_cb(i, n):
                self.progress['value'] = 25 * i / n
                if i % 3 == 0 or i == n:
                    self.status_var.set(f"Gating: {i}/{n}...")
                    self.root.update_idletasks()

            gated_s21, peak_ns = apply_time_gating(
                self.s21_complex, self.freq, span, callback=gate_cb)
            z = 20 * np.log10(np.abs(gated_s21) + 1e-12)
            info_parts.append(f"Peak: {peak_ns:.1f} ns")

            if mode == "Normalized" and self.norm_db is not None:
                # Apply gating correction to the original normalization
                orig_db = 20 * np.log10(np.abs(self.s21_complex) + 1e-12)
                z = self.norm_db + (z - orig_db)
            elif mode == "Phase":
                z = np.angle(gated_s21, deg=True)
        else:
            if mode == "Normalized" and self.norm_db is not None:
                z = self.norm_db.copy()
            elif mode == "Phase":
                z = np.angle(self.s21_complex, deg=True)
            else:
                z = self.raw_db.copy()

        self.progress['value'] = 25

        # Stage 2: Reference map subtraction
        if self.ref_enabled.get() and self.ref_raw_db is not None:
            self.status_var.set("Stage 2/4: Reference subtraction...")
            self.root.update_idletasks()

            if self.gate_enabled.get() and self.ref_s21_complex is not None:
                # Gate the reference too
                try:
                    span = float(self.gate_span.get())
                except ValueError:
                    span = 1.0
                ref_gated, _ = apply_time_gating(
                    self.ref_s21_complex, self.freq, span)
                ref_z = 20 * np.log10(np.abs(ref_gated) + 1e-12)
            else:
                ref_z = self.ref_raw_db

            if ref_z.shape == z.shape:
                z = z - ref_z
                info_parts.append("ref subtracted")
            else:
                n = min(ref_z.shape[0], z.shape[0])
                z[:n] = z[:n] - ref_z[:n]
                info_parts.append(f"ref subtracted ({n} steps matched)")

        self.progress['value'] = 50

        # Stage 3: Smoothing
        smooth = self.smooth_pts.get()
        if smooth > 1:
            self.status_var.set("Stage 3/4: Smoothing...")
            self.root.update_idletasks()
            kernel = np.ones(smooth) / smooth
            for i in range(z.shape[0]):
                z[i] = np.convolve(z[i], kernel, mode='same')

        self.progress['value'] = 65

        # Stage 4: SVD background removal
        if self.svd_enabled.get() and z.shape[0] >= 3:
            self.status_var.set("Stage 4/4: SVD removal...")
            self.root.update_idletasks()
            n_rm = min(self.svd_modes.get(), z.shape[0] - 1)
            z, pct = apply_svd(z, n_rm)
            info_parts.append(f"SVD -{n_rm} ({pct:.0f}%)")

        self.progress['value'] = 85

        # Stage 5: Plot
        self.status_var.set("Rendering...")
        self.root.update_idletasks()

        self._plot(z, mode, info_parts)

        self.progress['value'] = 100
        parts_str = " | ".join(info_parts) if info_parts else "raw"
        self.info_label.config(text=parts_str)
        name = os.path.basename(self.folder) if self.folder else ""
        self.status_var.set(f"Done: {name}")

    def _plot(self, z, mode, info_parts):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        # Determine color scale
        if self.auto_color.get():
            vmax = np.percentile(np.abs(z), 98)
            vmin = -vmax
        else:
            try:
                vmin = float(self.color_min.get())
                vmax = float(self.color_max.get())
            except ValueError:
                vmin, vmax = -0.05, 0.05

        cmap = 'RdBu_r'
        if mode == "Magnitude" and not any(
            self.ref_enabled.get() or self.svd_enabled.get()
            for _ in [0]):
            cmap = 'viridis'
            vmin, vmax = None, None

        im = self.ax.pcolormesh(self.freq, self.step_vals, z,
                                shading='auto', cmap=cmap,
                                vmin=vmin, vmax=vmax)
        self.fig.colorbar(im, ax=self.ax, label=f"ΔS21 (dB)")

        self.ax.set_xlabel("Frequency (GHz)")

        # Try to determine step label from metadata
        step_label = "Step Value"
        if self.folder:
            meta_file = os.path.join(self.folder, "metadata.txt")
            if os.path.exists(meta_file):
                with open(meta_file) as f:
                    for line in f:
                        if 'Parameter:' in line and 'Step' not in line:
                            continue
                        if 'Step' in line and 'Parameter:' in line:
                            step_label = line.split(':')[1].strip()
                            break
        self.ax.set_ylabel(step_label)

        title_parts = [mode]
        title_parts.extend(info_parts)
        self.ax.set_title(" | ".join(title_parts), fontsize=10)

        self.fig.tight_layout()
        self.canvas.draw_idle()

        # Store for export
        self._last_z = z

    # ── Export ──

    def _export(self):
        if not hasattr(self, '_last_z') or self._last_z is None:
            messagebox.showinfo("Export", "Process data first.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
            initialfile="processed_map.csv")
        if not path:
            return

        z = self._last_z
        with open(path, 'w') as f:
            f.write("# VNA-FMR Post-Processed Data\n")
            f.write(f"# Source: {self.folder}\n")
            f.write(f"# Processing: gate={self.gate_enabled.get()}, "
                    f"svd={self.svd_enabled.get()}, "
                    f"ref={self.ref_enabled.get()}\n")
            # Header: step_value, freq1, freq2, ...
            f.write("StepValue," + ",".join(f"{fq:.6f}" for fq in self.freq) + "\n")
            for i, sv in enumerate(self.step_vals):
                f.write(f"{sv:.6f}," + ",".join(f"{z[i,j]:.8f}"
                        for j in range(z.shape[1])) + "\n")

        self.status_var.set(f"Exported: {os.path.basename(path)}")


# ───────────────────────────────────────────────────────────
#  Main
# ───────────────────────────────────────────────────────────

def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else None

    root = tk.Tk()
    app = PostProcessApp(root, folder)
    root.mainloop()


if __name__ == "__main__":
    main()
