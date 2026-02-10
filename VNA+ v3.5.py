"""
VNA-Based FMR Measurement System
For Copper Mountain S5180B Vector Network Analyzer

Version: 3.5 - Gate and Field Normalization (January 2026)

Features:
- Flexible sweep/step parameter selection (Frequency, B-field, Gate Voltage, Power)
- 1D sweeps or 2D sweep+step measurements
- Real-time visualization with Mag/Phase and Re/Im toggle
- Contour plots for 2D data
- Simulated data mode for GUI testing
- In-app log display and automatic log file saving
- Gate voltage normalization (reference at V_ref)
- B-field normalization (reference at B_ref)

GATE SAFETY FEATURES (v2.1):
- Never jumps gate voltage - always ramps, even on connect/reconnect
- Reads actual voltage before reset to prevent jumps
- Emergency shutdown capability
- Periodic safety checks during measurement
- Proper exception handling - always tries to safe gate on error
- Detects communication failures (no stale values)

Hardware (when connected):
- Copper Mountain S5180B VNA via TCP socket (localhost:5025)
- AMI Model 420 Magnet Power Supply Programmer via GPIB
- Keithley 2400/2450 SMU for gate voltage via GPIB

Author: Claude (Anthropic) for Scott Dietrich, Villanova University
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import queue
import os
import sys
import io
import socket
import subprocess
import json
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import Normalize, LogNorm, SymLogNorm
import matplotlib.pyplot as plt
from matplotlib import cm

# Set professional font sizes for plots
plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.titlesize': 9,
    'lines.linewidth': 0.8,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
})


class LogManager:
    """Manages logging to both GUI display and file.
    
    Captures print statements and logging output, displays them in a 
    scrolling text widget, and saves to log files alongside data files.
    """
    
    def __init__(self):
        self.text_widget = None
        self.log_queue = queue.Queue()
        self.log_buffer = []  # Buffer for file saving
        self.max_buffer_lines = 10000  # Limit memory usage
        self.current_log_file = None
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._started = False
        self._update_pending = False
        self._last_update_time = 0
        self._min_update_interval = 0.1  # Minimum 100ms between GUI updates
        
    def set_text_widget(self, widget):
        """Set the tkinter Text widget for display."""
        self.text_widget = widget
        
    def start_capture(self):
        """Start capturing stdout/stderr."""
        if self._started:
            return
        self._started = True
        
        # Create custom stream that writes to both original and our queue
        self._stdout_redirector = self._StreamRedirector(
            self._original_stdout, self.log_queue, "INFO"
        )
        self._stderr_redirector = self._StreamRedirector(
            self._original_stderr, self.log_queue, "ERROR"
        )
        
        sys.stdout = self._stdout_redirector
        sys.stderr = self._stderr_redirector
        
    def stop_capture(self):
        """Restore original stdout/stderr."""
        if not self._started:
            return
        self._started = False
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        
    def log(self, message, level="INFO"):
        """Add a message to the log."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted = f"[{timestamp}] {message}"
        
        # Add to queue for GUI update
        self.log_queue.put((level, formatted))
        
        # Add to buffer for file saving
        self.log_buffer.append(formatted)
        if len(self.log_buffer) > self.max_buffer_lines:
            self.log_buffer = self.log_buffer[-self.max_buffer_lines:]
        
        # Also print to original stdout (for console)
        self._original_stdout.write(formatted + "\n")
        self._original_stdout.flush()
        
    def update_display(self):
        """Process queued messages and update the text widget.
        
        Call this periodically from the main GUI thread.
        Throttled to prevent GUI lag.
        Returns True if any messages were processed.
        """
        if self.text_widget is None:
            # Just drain the queue if no widget
            try:
                while True:
                    self.log_queue.get_nowait()
            except queue.Empty:
                pass
            return False
        
        # Throttle updates to prevent GUI lag
        current_time = time.time()
        if current_time - self._last_update_time < self._min_update_interval:
            return False
        
        messages_processed = False
        batch_messages = []
        
        # Collect all pending messages (up to 50 at a time)
        try:
            for _ in range(50):
                level, message = self.log_queue.get_nowait()
                messages_processed = True
                batch_messages.append((level, message))
                
                # Add to buffer
                if message not in self.log_buffer:
                    self.log_buffer.append(message)
                    if len(self.log_buffer) > self.max_buffer_lines:
                        self.log_buffer = self.log_buffer[-self.max_buffer_lines:]
        except queue.Empty:
            pass
        
        # Update text widget in one batch
        if batch_messages:
            try:
                self.text_widget.config(state='normal')
                
                for level, message in batch_messages:
                    self.text_widget.insert(tk.END, message + "\n", level)
                
                # Auto-scroll to bottom
                self.text_widget.see(tk.END)
                
                # Limit display lines to prevent memory issues
                line_count = int(self.text_widget.index('end-1c').split('.')[0])
                if line_count > 500:  # Reduced from 1000
                    self.text_widget.delete('1.0', f'{line_count - 400}.0')
                
                self.text_widget.config(state='disabled')
            except tk.TclError:
                pass  # Widget may have been destroyed
            
            self._last_update_time = current_time
        
        return messages_processed
    
    def clear_display(self):
        """Clear the log display widget."""
        if self.text_widget:
            try:
                self.text_widget.config(state='normal')
                self.text_widget.delete('1.0', tk.END)
                self.text_widget.config(state='disabled')
            except tk.TclError:
                pass
    
    def clear_buffer(self):
        """Clear both display and buffer (for new measurement)."""
        self.clear_display()
        self.log_buffer = []
        
    def save_to_file(self, filepath):
        """Save log buffer to a file."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("VNA FMR Measurement Log\n")
                f.write(f"Saved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                
                for line in self.log_buffer:
                    f.write(line + "\n")
            
            self._original_stdout.write(f"Log saved to: {filepath}\n")
            return True
        except Exception as e:
            self._original_stdout.write(f"Error saving log: {e}\n")
            return False
    
    def get_log_filepath(self, data_filepath):
        """Generate log filepath from data filepath.
        
        For 1D: data_001.csv -> data_001_log.txt
        For 2D folder: data_001/ -> data_001/measurement_log.txt
        """
        if os.path.isdir(data_filepath):
            # 2D measurement folder
            return os.path.join(data_filepath, "measurement_log.txt")
        else:
            # 1D measurement file
            base, ext = os.path.splitext(data_filepath)
            return f"{base}_log.txt"
    
    class _StreamRedirector(io.StringIO):
        """Redirects a stream to both original output and a queue."""
        
        def __init__(self, original_stream, log_queue, level):
            super().__init__()
            self.original_stream = original_stream
            self.log_queue = log_queue
            self.level = level
            self.line_buffer = ""
            
        def write(self, text):
            # Write to original stream
            if self.original_stream:
                self.original_stream.write(text)
                self.original_stream.flush()
            
            # Buffer until we have complete lines
            self.line_buffer += text
            while '\n' in self.line_buffer:
                line, self.line_buffer = self.line_buffer.split('\n', 1)
                if line.strip():  # Skip empty lines
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    formatted = f"[{timestamp}] {line}"
                    self.log_queue.put((self.level, formatted))
        
        def flush(self):
            if self.original_stream:
                self.original_stream.flush()


# Global log manager instance
log_manager = LogManager()


class SimulatedDataGenerator:
    """Generates realistic-looking FMR data for GUI testing."""
    
    def __init__(self):
        # FMR parameters for simulation
        self.resonance_freq = 8e9  # 8 GHz base resonance
        self.linewidth = 200e6    # 200 MHz linewidth
        self.resonance_field = 0.3  # 300 mT
        self.field_linewidth = 0.02  # 20 mT
        
    def lorentzian_complex(self, x, x0, gamma, amplitude=1.0):
        """Generate complex Lorentzian (absorption + dispersion)."""
        # Real part: dispersion, Imag part: absorption
        delta = x - x0
        denom = delta**2 + gamma**2
        real_part = amplitude * delta / denom
        imag_part = -amplitude * gamma / denom
        return real_part + 1j * imag_part
    
    def generate_s21_vs_frequency(self, frequencies, b_field=0.0, vg=0.0, power=-10, temperature=300.0):
        """Generate S21(f) data - FMR absorption spectrum."""
        # Shift resonance with field (Kittel-like)
        f_res = self.resonance_freq + b_field * 28e9  # ~28 GHz/T for electron
        
        # Temperature affects linewidth (broader at higher T)
        temp_linewidth = self.linewidth * (1 + (temperature - 300) / 500)
        
        # Gate voltage affects amplitude slightly
        amplitude = 0.1 * (1 + 0.05 * vg)
        
        # Power affects signal amplitude
        amplitude *= 10**((power + 10) / 20)
        
        # Background transmission
        s21_background = 0.8 * np.exp(-1j * frequencies / 1e10)
        
        # FMR signal
        s21_fmr = self.lorentzian_complex(frequencies, f_res, temp_linewidth, amplitude)
        
        # Add noise (more noise at higher temperature)
        noise_level = 0.005 * (1 + temperature / 1000)
        noise = noise_level * (np.random.randn(len(frequencies)) + 1j * np.random.randn(len(frequencies)))
        
        return s21_background + s21_fmr + noise
    
    def generate_s21_vs_field(self, fields, frequency=8e9, vg=0.0, power=-10, temperature=300.0):
        """Generate S21(B) data - field-swept FMR."""
        # Find resonance field for given frequency
        b_res = (frequency - self.resonance_freq) / 28e9
        
        # Temperature affects linewidth
        temp_linewidth = self.field_linewidth * (1 + (temperature - 300) / 500)
        
        amplitude = 0.1 * (1 + 0.05 * vg)
        amplitude *= 10**((power + 10) / 20)
        
        # Background
        s21_background = 0.8 * np.ones(len(fields), dtype=complex)
        
        # FMR signal
        s21_fmr = self.lorentzian_complex(fields, b_res, temp_linewidth, amplitude)
        
        noise_level = 0.005 * (1 + temperature / 1000)
        noise = noise_level * (np.random.randn(len(fields)) + 1j * np.random.randn(len(fields)))
        
        return s21_background + s21_fmr + noise
    
    def generate_s21_vs_gate(self, vg_values, frequency=8e9, b_field=0.3, power=-10, temperature=300.0):
        """Generate S21(Vg) data - gate-dependent transmission."""
        # Gate voltage modulates transmission (e.g., graphene Dirac point)
        # Dirac point shifts slightly with temperature
        dirac_point = 2.0 + (temperature - 300) / 200
        
        amplitude = 0.1 * 10**((power + 10) / 20)
        
        # Transmission minimum at Dirac point
        conductivity = 1 + 0.3 * np.abs(vg_values - dirac_point)
        s21_background = 0.5 * conductivity / conductivity.max()
        
        # Small FMR-like feature
        s21_fmr = 0.05 * self.lorentzian_complex(vg_values, dirac_point, 1.0, amplitude)
        
        noise_level = 0.003 * (1 + temperature / 1000)
        noise = noise_level * (np.random.randn(len(vg_values)) + 1j * np.random.randn(len(vg_values)))
        
        return s21_background + s21_fmr + noise
    
    def generate_s21_vs_power(self, powers, frequency=8e9, b_field=0.3, vg=0.0, temperature=300.0):
        """Generate S21(P) data - power dependence."""
        # Linear power dependence with some saturation at high power
        linear_response = 10**(powers / 20)
        # Saturation threshold decreases at higher temperature
        sat_threshold = 100 * (1 - (temperature - 300) / 1000)
        saturation = 1 / (1 + linear_response / sat_threshold)
        
        s21_mag = 0.1 * linear_response * saturation
        s21_phase = -np.pi/4 + 0.1 * powers / 10
        
        s21 = s21_mag * np.exp(1j * s21_phase)
        
        noise_level = 0.002 * (1 + temperature / 1000)
        noise = noise_level * (np.random.randn(len(powers)) + 1j * np.random.randn(len(powers)))
        
        return s21 + noise
    
    def generate_s21_vs_temperature(self, temperatures, frequency=8e9, b_field=0.3, vg=0.0, power=-10):
        """Generate S21(T) data - temperature dependence."""
        # Resonance properties change with temperature
        # Linewidth broadens, amplitude decreases, resonance may shift
        
        amplitude = 0.1 * 10**((power + 10) / 20)
        
        # Background transmission (slight temperature dependence from cables/components)
        s21_background = 0.8 * (1 - 0.0005 * (temperatures - 300))
        
        # FMR signal amplitude decreases with temperature (thermal fluctuations)
        # and linewidth broadens
        s21_signal = []
        for T in temperatures:
            temp_amplitude = amplitude * np.exp(-(T - 300) / 500)
            temp_linewidth = self.field_linewidth * (1 + (T - 300) / 300)
            # Small resonance shift with temperature
            b_res = (frequency - self.resonance_freq) / 28e9 + 0.001 * (T - 300) / 100
            signal = temp_amplitude * self.lorentzian_complex(np.array([b_field]), b_res, temp_linewidth, 1.0)[0]
            s21_signal.append(signal)
        
        s21_signal = np.array(s21_signal)
        
        # More noise at higher temperature
        noise_level = 0.003 * (1 + temperatures / 500)
        noise = noise_level * (np.random.randn(len(temperatures)) + 1j * np.random.randn(len(temperatures)))
        
        return s21_background + s21_signal + noise


class VNAController:
    """Controller for Copper Mountain S5180B VNA.
    
    Connection: TCP socket to S2VNA software
    To enable: In S2VNA, go to System ->' Misc Setup ->' Network Setup
               Enable TCP/IP Socket Server on port 5025
    """
    
    def __init__(self):
        self.connected = False
        self.socket = None
        self.host = "127.0.0.1"
        self.port = 5025
        self.timeout = 2.0  # seconds for normal operations
        self.s_parameter = "S21"
        
    def set_port(self, port):
        """Update TCP port."""
        self.port = int(port)
        
    def connect(self):
        """Connect to VNA via TCP socket."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            self.connected = True
            
            # Verify connection with ID query
            idn = self.query("*IDN?")
            if idn:
                print(f"VNA connected: {idn}")
                return True
            else:
                self.disconnect()
                print("VNA: No response to *IDN? query")
                return False
                
        except socket.timeout:
            print("VNA connection timeout - is S2VNA running with socket server enabled?")
            self.connected = False
            return False
        except socket.error as e:
            print(f"VNA connection error: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from VNA."""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.socket = None
        self.connected = False
    
    def write(self, command):
        """Send SCPI command to VNA."""
        if not self.connected or not self.socket:
            return False
        try:
            self.socket.sendall((command + "\n").encode('utf-8'))
            return True
        except socket.error as e:
            print(f"VNA write error: {e}")
            return False
    
    def read(self, large_data=False):
        """Read response from VNA.
        
        Args:
            large_data: If True, use larger buffer and longer timeout for sweep data
        """
        if not self.connected or not self.socket:
            return None
        try:
            # For large data transfers, temporarily increase timeout
            if large_data:
                self.socket.settimeout(30.0)
            
            response = b""
            buffer_size = 65536 if large_data else 4096  # 64KB for sweep data
            
            while True:
                try:
                    chunk = self.socket.recv(buffer_size)
                    if not chunk:
                        break
                    response += chunk
                    
                    # Check for terminator - VNA typically ends with \n
                    if chunk.endswith(b"\n"):
                        break
                    
                    # Also check if we've received a complete response
                    # (some VNAs don't send \n for data queries)
                    if large_data and len(chunk) < buffer_size:
                        # Received less than buffer size, likely end of data
                        time.sleep(0.1)  # Brief pause to check if more coming
                        self.socket.setblocking(False)
                        try:
                            extra = self.socket.recv(buffer_size)
                            if extra:
                                response += extra
                        except BlockingIOError:
                            pass  # No more data
                        finally:
                            self.socket.setblocking(True)
                        break
                        
                except socket.timeout:
                    if response:
                        break  # Got some data, timeout means end
                    raise
            
            # Restore normal timeout
            if large_data:
                self.socket.settimeout(self.timeout)
            
            decoded = response.decode('utf-8').strip()
            return decoded
            
        except socket.timeout:
            print("VNA read timeout")
            self.socket.settimeout(self.timeout)  # Restore timeout
            return None
        except socket.error as e:
            print(f"VNA read error: {e}")
            return None
    
    def query(self, command, large_data=False):
        """Send command and read response."""
        if self.write(command):
            time.sleep(0.02)  # Small delay for VNA to process
            return self.read(large_data=large_data)
        return None
    
    def reset(self):
        """Reset VNA to default state."""
        self.write("*RST")
        time.sleep(1.0)
        self.write("*CLS")
    
    def setup_frequency_sweep(self, f_start, f_stop, num_points, ifbw, power, averages=1):
        """Configure VNA for frequency sweep measurement.
        
        Args:
            f_start: Start frequency in Hz
            f_stop: Stop frequency in Hz
            num_points: Number of sweep points
            ifbw: IF bandwidth in Hz
            power: Source power in dBm
            averages: Number of hardware averages (1 = no averaging)
        """
        if not self.connected:
            return False
        
        avg_str = f", {averages}avg" if averages > 1 else ""
        print(f"VNA setup: {f_start/1e9:.3f}-{f_stop/1e9:.3f} GHz, {num_points} pts, IFBW={ifbw}, P={power} dBm{avg_str}")
        
        # Abort any ongoing operation first
        self.write(":ABOR")
        time.sleep(0.1)
        
        # Set frequency range
        self.write(f":SENS:FREQ:STAR {f_start}")
        self.write(f":SENS:FREQ:STOP {f_stop}")
        
        # Set number of points
        self.write(f":SENS:SWE:POIN {num_points}")
        
        # Set IF bandwidth
        self.write(f":SENS:BAND {ifbw}")
        
        # Wait for VNA to process IFBW change
        time.sleep(0.05)
        
        # Disable smoothing
        self.write(":CALC1:SMO OFF")
        self.write(":CALC1:SMO:APER 1")
        
        # Configure hardware averaging
        if averages > 1:
            self.write(f":SENS:AVER:COUN {averages}")  # Set number of averages
            self.write(":SENS:AVER:TYP SWE")           # Sweep-by-sweep averaging
            self.write(":SENS:AVER ON")                 # Enable averaging
            print(f"VNA hardware averaging: {averages} sweeps")
        else:
            self.write(":SENS:AVER OFF")                # Disable averaging
        
        # Set source power and turn RF on
        self.write(f":SOUR:POW {power}")
        self.write(":OUTP ON")
        
        # Set sweep type to linear
        self.write(":SENS:SWE:TYPE LIN")
        
        # Set S-parameter and ensure trace is active
        self.write(f":CALC1:PAR1:DEF {self.s_parameter}")
        self.write(":CALC1:PAR1:SEL")
        
        # Set data format to real/imag pairs
        self.write(":CALC1:FORM SDAT")
        
        # Query actual settings for diagnostic
        try:
            actual_ifbw = self.query(":SENS:BAND?")
            actual_smo = self.query(":CALC1:SMO?")
            actual_avg = self.query(":SENS:AVER?")
            actual_avg_cnt = self.query(":SENS:AVER:COUN?") if averages > 1 else "1"
            print(f"VNA confirms: IFBW={actual_ifbw.strip()}, Smooth={actual_smo.strip()}, Avg={actual_avg.strip()}, AvgCnt={actual_avg_cnt.strip()}")
        except Exception as e:
            print(f"VNA diagnostic query failed: {e}")
        
        # Wait for VNA to process settings
        time.sleep(0.3)
        
        return True
    
    def setup_cw_mode(self, frequency, ifbw, power):
        """Configure VNA for CW (single frequency) measurement."""
        if not self.connected:
            return False
        
        # Store IFBW for trigger timing
        self._cw_ifbw = ifbw
        
        # Abort any ongoing operation
        self.write(":ABOR")
        
        # Single sweep mode (not continuous)
        self.write(":INIT:CONT OFF")
        
        # For CW mode: set start = stop = CW frequency, 1 point
        self.write(f":SENS:FREQ:STAR {frequency}")
        self.write(f":SENS:FREQ:STOP {frequency}")
        self.write(":SENS:SWE:POIN 1")
        
        # Set IF bandwidth
        self.write(f":SENS:BAND {ifbw}")
        
        # Wait for VNA to process IFBW change
        time.sleep(0.05)
        
        # Disable ALL processing that could cause digitization
        self.write(":CALC1:SMO OFF")           # Smoothing off
        self.write(":SENS:AVER OFF")           # Averaging off
        self.write(":CALC1:SMO:APER 1")        # Smoothing aperture to minimum
        
        # Set source power
        self.write(f":SOUR:POW {power}")
        
        # Set S-parameter
        self.write(f":CALC1:PAR1:DEF {self.s_parameter}")
        
        # Query actual settings for diagnostic
        try:
            actual_ifbw = self.query(":SENS:BAND?")
            actual_smo = self.query(":CALC1:SMO?")
            actual_avg = self.query(":SENS:AVER?")
            print(f"VNA CW setup: IFBW={actual_ifbw.strip()}, Smooth={actual_smo.strip()}, Avg={actual_avg.strip()}")
        except Exception as e:
            print(f"VNA diagnostic query failed: {e}")
        
        # Allow VNA to settle after configuration change
        time.sleep(0.1)
        
        return True
    
    def trigger_sweep(self):
        """Trigger a sweep and wait for completion (for CW mode).
        
        Uses explicit timing based on IFBW to ensure measurement is complete
        and data buffer is updated.
        """
        if not self.connected:
            return False
        
        # Trigger measurement
        self.write(":INIT:IMM")
        
        # Wait for measurement to complete based on IFBW
        # Need to wait for: measurement time + VNA processing + buffer update
        ifbw = getattr(self, '_cw_ifbw', 100)
        measurement_time = 1.0 / ifbw
        wait_time = measurement_time * 2.0 + 0.08  # 2x measurement time + 80ms buffer
        time.sleep(wait_time)
        
        return True
    
    def trigger_sweep_timed(self, num_points, ifbw, averages=1):
        """Trigger sweep and wait for completion.
        
        Args:
            num_points: Number of sweep points
            ifbw: IF bandwidth in Hz
            averages: Number of hardware averages (affects wait time)
        """
        if not self.connected:
            return False
        
        # Calculate expected sweep time with safety margin
        single_sweep_time = float(num_points) / float(ifbw)
        adjusted_single_sweep = single_sweep_time * 1.5 + 0.5  # Per-sweep overhead
        
        # Make sure RF is on
        self.write(":OUTP ON")
        
        # Set to single sweep mode (hold)
        self.write(":INIT:CONT OFF")
        time.sleep(0.1)
        
        if averages > 1:
            # Clear averaging buffer to start fresh
            self.write(":SENS:AVER:CLE")
            time.sleep(0.1)
            
            total_wait = adjusted_single_sweep * averages + 2.0
            print(f"VNA sweep: {num_points} pts x {averages} avg, est wait {total_wait:.1f}s...")
            
            # Trigger N sweeps to accumulate averaging
            for i in range(averages):
                self.write(":INIT:IMM")
                time.sleep(adjusted_single_sweep)
                
                # Brief status every few sweeps
                if (i + 1) % 5 == 0 or i == averages - 1:
                    print(f"  Averaging: {i + 1}/{averages} sweeps complete")
        else:
            # Single sweep
            self.write(":INIT:IMM")
            time.sleep(adjusted_single_sweep + 1.5)
        
        return True
    
    def trigger_sweep_blocking(self):
        """Alternative: Trigger sweep using blocking *OPC? query.
        
        This sends the command and waits for *OPC? to return "1".
        Requires the socket timeout to be long enough for the sweep.
        """
        if not self.connected:
            return False
        
        # Single sweep mode
        self.write(":INIT:CONT OFF")
        time.sleep(0.1)
        
        # Trigger and wait - *OPC? should block until complete
        # Temporarily set very long timeout
        old_timeout = self.socket.gettimeout()
        self.socket.settimeout(120.0)  # 2 minutes max
        
        try:
            self.write(":INIT:IMM")
            time.sleep(0.1)  # Small delay before query
            
            # This should block until sweep is done
            response = self.query("*OPC?")
            
            if response and response.strip() == "1":
                print("VNA sweep completed (blocking mode)")
                return True
            else:
                print(f"VNA *OPC? returned: {response}")
                return False
        finally:
            self.socket.settimeout(old_timeout)
    
    def get_sweep_data(self, expected_points=None):
        """Read sweep data as complex S-parameter array.
        
        Args:
            expected_points: Expected number of points (for logging only)
        
        Returns:
            numpy array of complex values, or None on error
        """
        if not self.connected:
            return None
        
        # Ensure any pending operations are complete before reading
        self.write("*WAI")
        
        # Get S-parameter data (real,imag pairs)
        response = self.query(":CALC:DATA:SDAT?", large_data=True)
        
        if not response:
            print("VNA get_sweep_data: No response")
            return None
        
        try:
            # Fast parsing using numpy
            values = np.fromstring(response, sep=',')
            
            if len(values) < 2:
                # Try alternate parsing
                values = [float(v.strip()) for v in response.split(',') if v.strip()]
                values = np.array(values)
            
            if len(values) < 2:
                print(f"VNA: Only got {len(values)} values")
                return None
            
            # Reshape to (N, 2) and convert to complex
            n_points = len(values) // 2
            values = values[:n_points * 2].reshape(-1, 2)
            complex_data = values[:, 0] + 1j * values[:, 1]
            
            # Log if point count doesn't match expected
            if expected_points and n_points != expected_points:
                print(f"WARNING: VNA returned {n_points} points, expected {expected_points}")
            
            return complex_data
            
        except (ValueError, IndexError) as e:
            print(f"Error parsing VNA sweep data: {e}")
            return None
    
    def get_cw_data(self):
        """Read single CW measurement point.
        
        Returns:
            Complex value, or None on error
        """
        data = self.get_sweep_data()
        if data is not None and len(data) > 0:
            return data[0]
        return None
    
    def get_frequency_list(self):
        """Get list of frequency points for current sweep."""
        if not self.connected:
            return None
        
        response = self.query(":SENS:FREQ:DATA?")
        
        if not response:
            return None
        
        try:
            frequencies = [float(f) for f in response.split(',')]
            return np.array(frequencies)
        except ValueError as e:
            print(f"Error parsing frequency data: {e}")
            return None
    
    def measure_single_point(self):
        """Take CW measurement: trigger and read single point."""
        if not self.connected:
            return None
        
        if not self.trigger_sweep():
            return None
        
        return self.get_cw_data()
    
    def measure_sweep(self):
        """Take frequency sweep: trigger and read all points."""
        if not self.connected:
            return None
        
        if not self.trigger_sweep():
            return None
        
        return self.get_sweep_data()


class MagnetController:
    """Controller for AMI Model 420 Power Supply Programmer.
    
    Controls superconducting magnet via GPIB interface.
    Reference: AMI Model 420 manual Rev. 7
    """
    
    def __init__(self, resource_manager=None):
        self.connected = False
        self.address = "GPIB0::22::INSTR"
        self.instrument = None
        self.rm = resource_manager
        self.current_field = 0.0
        self.max_field = 9.0  # Tesla - update based on your magnet
        self.field_units = 'T'  # 'T' or 'kG'
        
    def set_address(self, address):
        """Update GPIB address."""
        self.address = f"GPIB0::{address}::INSTR"
    
    def connect(self):
        """Connect to AMI 420 via GPIB."""
        try:
            if self.rm is None:
                import pyvisa
                self.rm = pyvisa.ResourceManager()
            
            self.instrument = self.rm.open_resource(self.address)
            self.instrument.timeout = 5000  # 5 second timeout
            self.instrument.read_termination = '\n'
            self.instrument.write_termination = '\n'
            
            # Query identification
            idn = self.instrument.query("*IDN?")
            print(f"AMI 420 connected: {idn.strip()}")
            
            # Get field units setting
            try:
                units = self.instrument.query("FIELD:UNITS?").strip()
                self.field_units = 'T' if 'T' in units.upper() else 'kG'
                print(f"Field units: {self.field_units}")
            except:
                pass
            
            self.connected = True
            return True
            
        except Exception as e:
            print(f"AMI 420 connection error: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from AMI 420."""
        if self.instrument:
            try:
                self.instrument.close()
            except:
                pass
        self.instrument = None
        self.connected = False
    
    def write(self, command):
        """Send command to AMI 420."""
        if not self.connected or not self.instrument:
            return False
        try:
            self.instrument.write(command)
            return True
        except Exception as e:
            print(f"AMI 420 write error: {e}")
            return False
    
    def query(self, command):
        """Query AMI 420."""
        if not self.connected or not self.instrument:
            return None
        try:
            return self.instrument.query(command).strip()
        except Exception as e:
            print(f"AMI 420 query error: {e}")
            return None
    
    def get_field(self, debug=False):
        """Get current magnetic field.
        
        Args:
            debug: If True, print raw response for debugging
        
        
        Returns field in Tesla.
        """
        if not self.connected:
            return self.current_field
        
        try:
            # Query field (returns value in configured units)
            response = self.query("FIELD:MAG?")
            if debug and response:
                print(f"  [AMI 420] Raw field response: {response}")
            if response:
                field = float(response)
                # Convert to Tesla if in kG
                if self.field_units == 'kG':
                    field = field / 10.0
                self.current_field = field
                return field
        except Exception as e:
            print(f"AMI 420 get_field error: {e}")
        
        return self.current_field
    
    def get_current(self):
        """Get magnet current in Amps."""
        response = self.query("CURR:MAG?")
        if response:
            try:
                return float(response)
            except:
                pass
        return 0.0
    
    def get_state(self):
        """Get ramping state.
        
        Returns one of:
        1 = RAMPING, 2 = HOLDING, 3 = PAUSED, 4 = MANUAL UP,
        5 = MANUAL DOWN, 6 = ZEROING, 7 = QUENCH, 8 = AT ZERO,
        9 = HEATING SWITCH, 10 = COOLING SWITCH
        """
        response = self.query("STATE?")
        if response:
            try:
                return int(response)
            except:
                pass
        return 0
    
    def get_state_string(self):
        """Get human-readable state string."""
        states = {
            1: "RAMPING",
            2: "HOLDING",
            3: "PAUSED",
            4: "MANUAL UP",
            5: "MANUAL DOWN",
            6: "ZEROING",
            7: "QUENCH",
            8: "AT ZERO",
            9: "HEATING SWITCH",
            10: "COOLING SWITCH"
        }
        return states.get(self.get_state(), "UNKNOWN")
    
    def set_field(self, target_field):
        """Set target magnetic field and initiate ramp.
        
        Args:
            target_field: Target field in Tesla
        
        Returns:
            True if command accepted
        """
        if not self.connected:
            self.current_field = target_field
            return True
        
        try:
            # Convert to configured units
            if self.field_units == 'kG':
                field_value = target_field * 10.0  # T to kG
            else:
                field_value = target_field
            
            # Set programmed field
            self.write(f"CONF:FIELD:PROG {field_value}")
            
            # Start ramping (exit PAUSE mode)
            self.write("RAMP")
            
            print(f"AMI 420: Ramping to {target_field:.4f} T")
            return True
            
        except Exception as e:
            print(f"AMI 420 set_field error: {e}")
            return False
    
    def pause(self):
        """Pause ramping."""
        return self.write("PAUSE")
    
    def ramp(self):
        """Resume/start ramping to programmed field."""
        return self.write("RAMP")
    
    def zero(self):
        """Ramp to zero field."""
        return self.write("ZERO")
    
    def set_ramp_rate(self, rate, units='T/s'):
        """Set field ramp rate.
        
        Args:
            rate: Ramp rate value
            units: 'T/s', 'T/min', 'kG/s', 'kG/min', 'A/s', 'A/min'
        """
        if 'A' in units:
            # Current ramp rate
            self.write(f"CONF:RAMP:RATE:CURR {rate}")
        else:
            # Field ramp rate
            self.write(f"CONF:RAMP:RATE:FIELD {rate}")
        return True
    
    def get_ramp_rate(self):
        """Get current ramp rate setting."""
        response = self.query("RAMP:RATE:FIELD?")
        if response:
            try:
                return float(response.split()[0])
            except:
                pass
        return 0.0
    
    def set_voltage_limit(self, voltage):
        """Set charging voltage limit."""
        return self.write(f"CONF:VOLT:LIM {voltage}")
    
    def get_supply_voltage(self):
        """Get power supply output voltage."""
        response = self.query("VOLT:SUPP?")
        if response:
            try:
                return float(response)
            except:
                pass
        return 0.0
    
    def get_magnet_voltage(self):
        """Get voltage across magnet terminals."""
        response = self.query("VOLT:MAG?")
        if response:
            try:
                return float(response)
            except:
                pass
        return 0.0
    
    def is_persistent_switch_on(self):
        """Check if persistent switch heater is on."""
        response = self.query("PS?")
        if response:
            return response.strip() == "1"
        return False
    
    def set_persistent_switch(self, on):
        """Control persistent switch heater.
        
        Args:
            on: True to turn heater on (open switch), False to turn off (close switch)
        """
        return self.write(f"PS {1 if on else 0}")
    
    def wait_for_field(self, target_field, tolerance=0.001, timeout=600):
        """Wait until field reaches target.
        
        Args:
            target_field: Target field in Tesla
            tolerance: Acceptable error in Tesla
            timeout: Maximum wait time in seconds
        
        Returns:
            True if target reached, False if timeout
        """
        import time
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            current = self.get_field()
            state = self.get_state()
            
            # Check if we're at target (HOLDING state = 2)
            if abs(current - target_field) <= tolerance and state == 2:
                return True
            
            # Check for quench (state = 7)
            if state == 7:
                print("AMI 420: QUENCH DETECTED!")
                return False
            
            time.sleep(0.5)
        
        print(f"AMI 420: Timeout waiting for field {target_field} T")
        return False
    
    def is_ramping(self):
        """Check if magnet is currently ramping."""
        state = self.get_state()
        return state in [1, 4, 5, 6]  # RAMPING, MANUAL UP/DOWN, ZEROING
    
    def is_at_field(self):
        """Check if magnet is holding at programmed field."""
        return self.get_state() == 2  # HOLDING


class CryomagneticsController:
    """Controller for Cryomagnetics 4G Magnet Power Supply.
    
    Controls superconducting magnet via GPIB interface.
    """
    
    def __init__(self, resource_manager=None):
        self.connected = False
        self.address = "GPIB0::21::INSTR"
        self.instrument = None
        self.rm = resource_manager
        self.current_field = 0.0
        self.max_field = 9.0  # Tesla
        self.field_units = 'T'
        # Coil constant for field/current conversion
        self.coil_constant_G_per_A = 2376.5  # Gauss per Amp
        self.field_per_amp = self.coil_constant_G_per_A / 10000.0  # Tesla per Amp
        
    def set_address(self, address):
        """Update GPIB address."""
        self.address = f"GPIB0::{address}::INSTR"
    
    def connect(self):
        """Connect to Cryomagnetics 4G via GPIB."""
        try:
            if self.rm is None:
                import pyvisa
                self.rm = pyvisa.ResourceManager()
            
            self.instrument = self.rm.open_resource(self.address)
            self.instrument.timeout = 30000  # 30 seconds - controller is slow during sweeps
            self.instrument.read_termination = '\r\n'
            self.instrument.write_termination = '\r\n'
            
            # Query identification
            idn = self.instrument.query("*IDN?")
            print(f"Cryomagnetics 4G connected: {idn.strip()}")
            
            # CRITICAL: Put device in REMOTE mode - required for software control
            print("  Setting REMOTE mode...")
            self.instrument.write("REMOTE")
            time.sleep(0.2)
            
            # Get current field
            time.sleep(0.1)
            try:
                field = self.instrument.query("IMAG?")
                print(f"  Current field: {field.strip()}")
            except:
                pass
            
            self.connected = True
            return True
            
        except Exception as e:
            print(f"Cryomagnetics connection error: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from Cryomagnetics 4G."""
        if self.instrument:
            try:
                # Return control to front panel
                self.instrument.write("LOCAL")
                time.sleep(0.1)
                self.instrument.close()
            except:
                pass
        self.instrument = None
        self.connected = False
    
    def write(self, command):
        """Send command to Cryomagnetics 4G."""
        if not self.connected or not self.instrument:
            return False
        try:
            self.instrument.write(command)
            return True
        except Exception as e:
            print(f"Cryomagnetics write error: {e}")
            return False
    
    def query(self, command):
        """Query Cryomagnetics 4G."""
        if not self.connected or not self.instrument:
            return None
        try:
            return self.instrument.query(command).strip()
        except Exception as e:
            print(f"Cryomagnetics query error: {e}")
            return None
    
    def get_field(self, debug=False):
        """Get current magnetic field in Tesla.
        
        Args:
            debug: If True, print raw response for debugging
        """
        if not self.connected:
            return self.current_field
        
        try:
            response = self.query("IMAG?")
            
            if debug and response:
                print(f"  [Cryomagnetics] Raw field response: {response}")
            
            if response:
                # Parse response - typically returns value with 'kG' suffix
                response = response.strip()
                
                # Handle different unit suffixes
                if response.endswith('kG'):
                    # Field in kilogauss - convert to Tesla (1 kG = 0.1 T)
                    value = float(response.replace('kG', '').strip())
                    self.current_field = value * 0.1  # kG to T
                elif response.endswith('G'):
                    # Field in gauss - convert to Tesla (1 G = 0.0001 T)
                    value = float(response.replace('G', '').strip())
                    self.current_field = value * 0.0001  # G to T
                elif response.endswith('T'):
                    # Field already in Tesla
                    value = float(response.replace('T', '').strip())
                    self.current_field = value
                elif response.endswith('A'):
                    # Current in Amps - convert using coil constant
                    current = float(response.replace('A', '').strip())
                    self.current_field = current * 0.1  # Example: 0.1 T/A
                else:
                    # Try to parse as plain number (assume kG)
                    try:
                        value = float(response)
                        self.current_field = value * 0.1  # kG to T
                    except:
                        pass
                
                return self.current_field
        except Exception as e:
            print(f"Cryomagnetics get_field error: {e}")
        
        return self.current_field
    
    def get_current(self):
        """Get magnet current in Amps."""
        response = self.query("IOUT?")
        if response:
            try:
                response = response.strip()
                # Handle different unit suffixes
                if response.endswith('kG'):
                    # Field in kilogauss - convert to approximate current
                    value = float(response.replace('kG', '').strip())
                    return value  # Return as-is, user interprets based on coil constant
                elif response.endswith('G'):
                    value = float(response.replace('G', '').strip())
                    return value / 1000.0  # Convert G to kG equivalent
                elif response.endswith('A'):
                    return float(response.replace('A', '').strip())
                else:
                    return float(response)
            except Exception as e:
                print(f"Cryomagnetics get_current parse error: {e}")
        return 0.0
    
    def get_state(self):
        """Get ramping state (simplified)."""
        # Cryomagnetics uses different status format
        # Return compatible state codes
        return 2  # HOLDING as default
    
    def get_state_string(self):
        """Get human-readable state string."""
        return "CONNECTED"
    
    def set_field(self, target_field):
        if not self.connected:
            self.current_field = target_field
            return True
        
        try:
            target_kG = target_field * 10.0
            current_field = self.get_field()
            
            if abs(target_field - current_field) < 0.001:
                print(f"Cryomagnetics: Already at {target_field:.4f} T")
                return True
            
            # CRITICAL: Use atomic compound command with semicolon
            # This prevents race condition where magnet ignores new limit
            if target_field > current_field:
                # Set upper limit and sweep up atomically
                self.write(f"ULIM {target_kG:.3f}; SWEEP UP")
                print(f"Cryomagnetics: Ramping UP to {target_field:.4f} T")
            else:
                # Set lower limit and sweep down atomically
                self.write(f"LLIM {target_kG:.3f}; SWEEP DOWN")
                print(f"Cryomagnetics: Ramping DOWN to {target_field:.4f} T")
            
            return True
            
        except Exception as e:
            print(f"Cryomagnetics set_field error: {e}")
            return False
    
    def pause(self):
        """Pause ramping."""
        return self.write("SWEEP PAUSE")
    
    def ramp(self):
        """Resume ramping."""
        return self.write("SWEEP UP")
    
    def zero(self):
        """Ramp to zero field."""
        return self.write("SWEEP ZERO")
    
    def set_ramp_rate(self, rate, units='T/min'):
        """Set field ramp rate.
        
        Args:
            rate: Ramp rate value
            units: 'T/min' (default) or 'T/s'
        """
        # Convert to T/s first
        if units == 'T/min':
            rate_T_s = rate / 60.0
        else:
            rate_T_s = rate
        
        # Convert T/s to A/s using coil constant
        rate_A_s = rate_T_s / self.field_per_amp
        
        # Clamp to safe range
        rate_A_s = max(0.001, min(1.0, rate_A_s))
        
        # Use RATE 0 format (range 0) with semicolon (firmware bug workaround)
        self.write(f"RATE 0 {rate_A_s:.4f};")
        time.sleep(0.2)
        
        print(f"Cryomagnetics: Set ramp rate to {rate:.3f} T/min ({rate_A_s:.4f} A/s)")
        
        # Verify with RATE? 0 (include range number in query)
        time.sleep(0.1)
        try:
            response = self.query("RATE? 0")
            if response:
                reported_rate = float(response.strip())
                reported_T_min = reported_rate * self.field_per_amp * 60.0
                print(f"  Confirmed rate: {reported_rate:.4f} A/s ({reported_T_min:.3f} T/min)")
                
                # If rate is not close to what we requested, warn user
                if abs(reported_rate - rate_A_s) > 0.001:
                    print(f"  WARNING: Controller reported different rate!")
                    print(f"           Requested {rate_A_s:.4f} A/s, got {reported_rate:.4f} A/s")
                    print(f"           This should not happen with semicolon workaround")
        except Exception as e:
            print(f"  Could not verify rate: {e}")
        
        return True
    
    def set_rate(self, rate):
        """Set ramp rate in T/min (convenience method)."""
        return self.set_ramp_rate(rate, units='T/min')
    
    def wait_for_field(self, target_field, tolerance=0.001, timeout=600):
        """Wait until field reaches target."""
        import time
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            current = self.get_field()
            if abs(current - target_field) <= tolerance:
                return True
            time.sleep(0.5)
        
        return False
    
    def is_ramping(self):
        """Check if magnet is currently ramping."""
        return False  # Simplified
    
    def is_at_field(self):
        """Check if magnet is holding at programmed field."""
        return True  # Simplified


class NHMFLMagnetController:
    """Controller for NHMFL SCM1 Magnet via TCP/LabVIEW interface.
    
    Communicates with LabVIEW program on the data PC via TCP socket.
    Based on UCSB Young's group code, modified by Andrew Woods.
    
    The LabVIEW program must be running on the data PC before connecting.
    """
    
    def __init__(self):
        self.connected = False
        self.address = "146.201.214.130"  # Default NHMFL data PC IP
        self.port = 6341
        self.timeout = 2
        self.client = None
        self.current_field = 0.0
        self.max_field = 18.0  # Tesla - SCM1 max field
        self.field_units = 'T'
        
    def set_address(self, address):
        """Set the TCP address (IP:port or just IP).
        
        Args:
            address: IP address string, optionally with :port
        """
        if ':' in str(address):
            parts = str(address).split(':')
            self.address = parts[0]
            self.port = int(parts[1])
        else:
            self.address = str(address)
        print(f"NHMFL magnet address set to: {self.address}:{self.port}")
    
    @staticmethod
    def _send_data(s):
        """Pack string with length prefix for LabVIEW protocol."""
        import struct
        return struct.pack('I', len(s)) + s.encode()
    
    @staticmethod
    def _get_byte_size(data):
        """Unpack length prefix from LabVIEW protocol."""
        import struct
        return int(struct.unpack('I', data)[0])
    
    def connect(self):
        """Connect to NHMFL LabVIEW magnet control.
        
        Will properly clean up any existing connection before attempting new one.
        """
        # First, clean up any existing connection
        if self.client:
            try:
                self.client.shutdown(socket.SHUT_RDWR)
            except:
                pass
            try:
                self.client.close()
            except:
                pass
            self.client = None
            time.sleep(0.5)  # Give the OS time to release the socket
        
        # Reset reconnect counter
        self._reconnect_attempts = 0
        
        print(f"NHMFL SCM1: Connecting to {self.address}:{self.port}...")
        
        try:
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.settimeout(5.0)  # Increased timeout for initial connection
            self.client.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)  # Enable keepalive
            
            # On Windows, set keepalive parameters
            if hasattr(socket, 'SIO_KEEPALIVE_VALS'):
                # keepalive time = 10s, interval = 5s, count = 3
                self.client.ioctl(socket.SIO_KEEPALIVE_VALS, (1, 10000, 5000))
            
            self.client.connect((self.address, self.port))
            
            # Restore normal timeout for operations
            self.client.settimeout(self.timeout)
            
            # Test connection by getting status
            status = self._get_status_raw_internal()  # Use internal version to avoid reconnect loop
            if status:
                # Validate the initial field reading
                raw_field = status.get('Field', 0.0)
                if 0 <= raw_field <= self.max_field * 1.1:
                    self.current_field = raw_field
                    self._last_good_field = raw_field
                else:
                    # Bad initial reading - assume 0
                    print(f"  [SCM1] Initial field {raw_field}T rejected, assuming 0T")
                    self.current_field = 0.0
                    self._last_good_field = 0.0
                
                print(f"NHMFL SCM1 connected: Field={self.current_field:.3f}T, "
                      f"Setpoint={status.get('Setpoint', 0):.3f}T")
                self.connected = True
                return True
            else:
                print("NHMFL SCM1: Connected but could not read status - check LabVIEW VI")
                # Still mark as connected since TCP connected, just status failed
                self.connected = True
                return True
                
        except socket.timeout:
            print(f"NHMFL SCM1: Connection timeout - is LabVIEW running on {self.address}?")
            self.connected = False
            if self.client:
                try:
                    self.client.close()
                except:
                    pass
                self.client = None
            return False
        except ConnectionRefusedError:
            print(f"NHMFL SCM1: Connection refused - LabVIEW may not be accepting connections")
            print(f"  Try restarting the LabVIEW VI on the data PC")
            self.connected = False
            if self.client:
                try:
                    self.client.close()
                except:
                    pass
                self.client = None
            return False
        except Exception as e:
            print(f"NHMFL SCM1 connection error: {e}")
            self.connected = False
            if self.client:
                try:
                    self.client.close()
                except:
                    pass
                self.client = None
            return False
    
    def disconnect(self):
        """Safely disconnect from NHMFL magnet control.
        
        IMPORTANT: Must call this to avoid having to restart the LabVIEW program.
        """
        if self.client:
            try:
                self.client.shutdown(socket.SHUT_RDWR)
                self.client.close()
                print("NHMFL SCM1 disconnected safely")
            except Exception as e:
                print(f"NHMFL SCM1 disconnect warning: {e}")
        self.client = None
        self.connected = False
    
    def _get_status_raw(self, debug=False):
        """Get raw status dictionary from LabVIEW.
        
        Args:
            debug: If True, print raw response for debugging
        
        Returns:
            dict with Field, Setpoint, SlewRate, Ramp, Pause, Units
            or None on error
        """
        if not self.client:
            return None
        
        try:
            # Drain any stale data from previous fire-and-forget commands
            self.client.setblocking(False)
            try:
                while True:
                    stale = self.client.recv(1024)
                    if not stale:
                        break
            except BlockingIOError:
                pass  # No stale data, good
            except:
                pass
            self.client.setblocking(True)
            
            # Now send status request
            self.client.send(self._send_data('g'))
            databytes = self.client.recv(4)  # 4 bytes containing data length
            data_len = self._get_byte_size(databytes)
            data = self.client.recv(data_len)
            decoded = data.decode()
            
            # Try to find valid CSV data in the response
            # Look for pattern: number,number,number,0or1,0or1,0or1
            import re
            # Match: optional negative, digits, optional decimal, comma repeated
            pattern = r'(-?\d+\.?\d*),(-?\d+\.?\d*),(-?\d+\.?\d*),([01]),([01]),([01])'
            match = re.search(pattern, decoded)
            
            if not match:
                if debug:
                    print(f"  [SCM1] Could not parse response: {repr(decoded)}")
                return None
            
            field = float(match.group(1))
            setpoint = float(match.group(2))
            slew_rate = float(match.group(3))
            ramp = match.group(4) == '1'
            pause = match.group(5) == '1'
            units = int(match.group(6))
            
            # Debug: print parsed values occasionally
            if debug:
                if not hasattr(self, '_debug_count'):
                    self._debug_count = 0
                self._debug_count += 1
                if self._debug_count <= 5 or self._debug_count % 20 == 0:
                    print(f"  [SCM1] Field={field:.6f}T, Setpoint={setpoint:.4f}T, Ramp={ramp}")
            
            # Reset reconnect counter on success
            self._reconnect_attempts = 0
            
            return {
                'Field': field,
                'Setpoint': setpoint,
                'SlewRate': slew_rate,
                'Ramp': ramp,
                'Pause': pause,
                'Units': units
            }
        except Exception as e:
            # Check if this is a connection-related error
            error_str = str(e).lower()
            error_code = getattr(e, 'winerror', None) or getattr(e, 'errno', None)
            
            # WinError 10054 = Connection reset by peer
            # WinError 10053 = Connection aborted
            # WinError 10057 = Not connected
            connection_errors = [10054, 10053, 10057, 104, 32]  # Windows and Unix error codes
            is_connection_error = (
                error_code in connection_errors or
                'connection' in error_str or
                'reset' in error_str or
                'closed' in error_str or
                'broken pipe' in error_str or
                isinstance(e, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError))
            )
            
            if is_connection_error:
                print(f"NHMFL SCM1 connection lost (error {error_code}): {e}")
                return self._handle_connection_lost()
            else:
                print(f"NHMFL SCM1 status error: {e}")
                return None
    
    def _handle_connection_lost(self):
        """Handle lost connection by attempting to reconnect.
        
        Returns:
            Status dict if reconnection succeeds and status is retrieved,
            None otherwise.
        """
        if not hasattr(self, '_reconnect_attempts'):
            self._reconnect_attempts = 0
        
        self._reconnect_attempts += 1
        max_attempts = 3
        
        if self._reconnect_attempts > max_attempts:
            if self._reconnect_attempts == max_attempts + 1:
                print(f"NHMFL SCM1: Max reconnection attempts ({max_attempts}) reached.")
                print(f"  The LabVIEW VI may need to be restarted on the data PC.")
                print(f"  After restarting LabVIEW, click 'Connect' in the Instrument Setup tab.")
            self.connected = False
            return None
        
        print(f"NHMFL SCM1: Attempting to reconnect (attempt {self._reconnect_attempts}/{max_attempts})...")
        
        # Properly close old socket
        if self.client:
            try:
                self.client.shutdown(socket.SHUT_RDWR)
            except:
                pass
            try:
                self.client.close()
            except:
                pass
            self.client = None
        
        # Wait longer before reconnecting - give LabVIEW time to clean up
        wait_time = 1.0 * self._reconnect_attempts  # 1s, 2s, 3s
        print(f"  Waiting {wait_time:.0f}s before retry...")
        time.sleep(wait_time)
        
        # Try to reconnect with longer timeout
        try:
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.settimeout(5.0)  # Longer timeout for reconnection
            self.client.connect((self.address, self.port))
            
            # Restore normal timeout
            self.client.settimeout(self.timeout)
            
            # Test the connection
            status = self._get_status_raw_internal()
            if status:
                print(f"NHMFL SCM1: Reconnected successfully! Field={status['Field']:.4f}T")
                self.connected = True
                self._reconnect_attempts = 0
                return status
            else:
                print("NHMFL SCM1: Reconnected but could not get status")
                # Mark as connected anyway - TCP is working
                self.connected = True
                return None
                
        except socket.timeout:
            print(f"NHMFL SCM1: Reconnection timeout")
            self.connected = False
            return None
        except ConnectionRefusedError:
            print(f"NHMFL SCM1: Connection refused - LabVIEW may need to be restarted")
            self.connected = False
            return None
        except Exception as e:
            print(f"NHMFL SCM1: Reconnection failed: {e}")
            self.connected = False
            return None
    
    def _get_status_raw_internal(self):
        """Internal status query without reconnection logic (to avoid recursion)."""
        if not self.client:
            return None
        
        try:
            self.client.send(self._send_data('g'))
            databytes = self.client.recv(4)
            data_len = self._get_byte_size(databytes)
            data = self.client.recv(data_len)
            decoded = data.decode()
            
            import re
            pattern = r'(-?\d+\.?\d*),(-?\d+\.?\d*),(-?\d+\.?\d*),([01]),([01]),([01])'
            match = re.search(pattern, decoded)
            
            if not match:
                return None
            
            return {
                'Field': float(match.group(1)),
                'Setpoint': float(match.group(2)),
                'SlewRate': float(match.group(3)),
                'Ramp': match.group(4) == '1',
                'Pause': match.group(5) == '1',
                'Units': int(match.group(6))
            }
        except:
            return None
    
    def get_field(self, debug=False):
        """Get current magnetic field in Tesla.
        
        Args:
            debug: If True, print raw response for debugging
        """
        if not self.connected:
            return self.current_field
        
        status = self._get_status_raw(debug=debug)
        if status:
            new_field = status['Field']
            
            # CRITICAL: Sanity check for obviously bad values
            # Field must be within magnet range (0 to max_field)
            # Values like 99870 are clearly garbage
            if new_field < 0 or new_field > self.max_field * 1.1:  # 10% margin
                if debug or new_field > 100:  # Always print for huge values
                    print(f"  [SCM1] Rejected bad field reading: {new_field:.4f}T (outside 0-{self.max_field}T range)")
                # Return last known good field, not current_field (which may be corrupted)
                return getattr(self, '_last_good_field', self.current_field)
            
            # Sanity check: field shouldn't jump too much between readings
            # At 0.3 T/min max, in 1 second max change is 0.005 T
            # Allow 10x margin for safety = 0.05 T max jump
            last_good = getattr(self, '_last_good_field', None)
            if last_good is not None:
                max_jump = 0.1  # T - generous margin
                if abs(new_field - last_good) > max_jump:
                    if debug:
                        print(f"  [SCM1] Rejected bad field reading: {new_field:.6f}T (jumped {abs(new_field - last_good):.4f}T from {last_good:.6f}T)")
                    return last_good
            
            # Value passed validation - save as last known good
            self._last_good_field = new_field
            self.current_field = new_field
            return new_field
        return getattr(self, '_last_good_field', self.current_field)
    
    def get_setpoint(self):
        """Get current field setpoint in Tesla."""
        status = self._get_status_raw()
        if status:
            return status['Setpoint']
        return 0.0
    
    def get_slew_rate(self):
        """Get current slew rate."""
        status = self._get_status_raw()
        if status:
            return status['SlewRate']
        return 0.0
    
    def _send_command_fire_and_forget(self, cmd):
        """Send command without waiting for response (original UCSB style).
        
        The original NHMFL code doesn't read responses after commands.
        Will attempt to reconnect if connection is lost.
        """
        # Try up to 2 times (original attempt + 1 retry after reconnect)
        for attempt in range(2):
            try:
                if not self.client or not self.connected:
                    if attempt == 0:
                        print(f"NHMFL SCM1: Not connected, attempting to reconnect...")
                        self._handle_connection_lost()
                    if not self.connected:
                        return False
                
                self.client.send(self._send_data(cmd))
                time.sleep(0.1)  # Give LabVIEW time to process
                return True
            except Exception as e:
                error_str = str(e).lower()
                error_code = getattr(e, 'winerror', None) or getattr(e, 'errno', None)
                
                # Check if connection error
                connection_errors = [10054, 10053, 10057, 104, 32]
                is_connection_error = (
                    error_code in connection_errors or
                    'connection' in error_str or
                    'reset' in error_str or
                    'closed' in error_str
                )
                
                if is_connection_error and attempt == 0:
                    print(f"NHMFL SCM1 command failed ({cmd[:1]}), connection lost. Reconnecting...")
                    self._handle_connection_lost()
                    continue  # Retry after reconnection
                else:
                    print(f"NHMFL SCM1 command error ({cmd[:1]}): {e}")
                    return False
        
        return False
    
    def set_field(self, target_field):
        """Set target magnetic field and initiate ramp.
        
        Args:
            target_field: Target field in Tesla
            
        Returns:
            True if command accepted
        """
        if not self.connected:
            self.current_field = target_field
            return True
        
        try:
            # Fire-and-forget style like original UCSB code
            # Set the setpoint
            self._send_command_fire_and_forget('s' + str(target_field))
            print(f"NHMFL SCM1: Set setpoint to {target_field:.4f} T")
            
            time.sleep(0.1)
            
            # Start ramping - LabVIEW expects 'uTrue'
            self._send_command_fire_and_forget('uTrue')
            print(f"NHMFL SCM1: Ramp command sent (uTrue)")
            
            # Check status to confirm (use validated field reading)
            time.sleep(0.3)
            field = self.get_field()  # This validates the reading
            status = self._get_status_raw()
            if status:
                # Validate field and setpoint - reject obvious garbage
                display_field = field
                if field > self.max_field * 1.1 or field < 0:
                    display_field = self.current_field if hasattr(self, '_last_good_field') else field_T
                
                setpoint = status['Setpoint']
                if setpoint < 0 or setpoint > self.max_field * 1.1:
                    setpoint = field_T  # Use our target instead
                
                print(f"NHMFL SCM1: Status: Field={display_field:.4f}T, "
                      f"Setpoint={setpoint:.4f}T, Ramp={status['Ramp']}")
            
            return True
            
        except Exception as e:
            print(f"NHMFL SCM1 set_field error: {e}")
            return False
    
    def set_rate(self, rate):
        """Set the slew rate.
        
        Args:
            rate: Slew rate value (T/min)
        """
        if not self.connected:
            return True
        
        try:
            self._send_command_fire_and_forget('r' + str(rate))
            print(f"NHMFL SCM1: Slew rate set to {rate}")
            return True
        except Exception as e:
            print(f"NHMFL SCM1 set_rate error: {e}")
            return False
    
    def ramp(self, start=True):
        """Start or stop ramping.
        
        Args:
            start: True to start ramping, False to stop
        """
        if not self.connected:
            return True
        
        try:
            # LabVIEW expects 'uTrue'/'uFalse'
            self._send_command_fire_and_forget('u' + str(bool(start)))
            print(f"NHMFL SCM1: Ramp {'started' if start else 'stopped'}")
            return True
        except Exception as e:
            print(f"NHMFL SCM1 ramp error: {e}")
            return False
    
    def pause(self, paused=True):
        """Pause or unpause ramping.
        
        Args:
            paused: True to pause, False to resume
        """
        if not self.connected:
            return True
        
        try:
            # LabVIEW expects 'pTrue'/'pFalse'
            self._send_command_fire_and_forget('p' + str(bool(paused)))
            print(f"NHMFL SCM1: {'Paused' if paused else 'Resumed'}")
            return True
        except Exception as e:
            print(f"NHMFL SCM1 pause error: {e}")
            return False
    
    def stop_ramp(self):
        """Stop ramping and hold at current field.
        
        Sets the setpoint to current field and pauses.
        """
        if not self.connected:
            return True
        
        try:
            # Get current field
            current = self.get_field()
            print(f"NHMFL SCM1: Stopping ramp at {current:.4f} T")
            
            # Set setpoint to current field
            self._send_command_fire_and_forget('s' + str(current))
            time.sleep(0.1)
            
            # Pause the ramp
            self._send_command_fire_and_forget('pTrue')
            
            print(f"NHMFL SCM1: Ramp stopped and paused at {current:.4f} T")
            return True
        except Exception as e:
            print(f"NHMFL SCM1 stop_ramp error: {e}")
            return False
    
    def get_state(self):
        """Get ramping state as integer.
        
        Returns:
            1 = RAMPING, 2 = HOLDING (compatible with AMI 420)
        """
        status = self._get_status_raw()
        if status:
            if status.get('Ramp', False):
                return 1  # RAMPING
            else:
                return 2  # HOLDING
        return 0
    
    def get_state_string(self):
        """Get human-readable state string."""
        status = self._get_status_raw()
        if status:
            if status.get('Pause', False):
                return "PAUSED"
            elif status.get('Ramp', False):
                return "RAMPING"
            else:
                return "HOLDING"
        return "UNKNOWN"
    
    def get_current(self):
        """Get magnet current (not directly available, return 0)."""
        return 0.0
    
    def is_ramping(self):
        """Check if magnet is currently ramping."""
        status = self._get_status_raw()
        if status:
            return status.get('Ramp', False)
        return False
    
    def is_at_field(self):
        """Check if magnet is holding at programmed field."""
        status = self._get_status_raw()
        if status:
            return not status.get('Ramp', False) and not status.get('Pause', False)
        return False
    
    def wait_for_field(self, target_field, tolerance=0.01, timeout=600):
        """Wait for magnet to reach target field.
        
        Args:
            target_field: Expected field value in Tesla
            tolerance: Acceptable difference from target (T)
            timeout: Maximum wait time in seconds
            
        Returns:
            True if field reached, False if timeout or error
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            current = self.get_field()
            
            if abs(current - target_field) <= tolerance:
                if not self.is_ramping():
                    return True
            
            time.sleep(0.5)
        
        print(f"NHMFL SCM1: Timeout waiting for field {target_field} T")
        return False


# Factory function to create the appropriate magnet controller
def create_magnet_controller(model='SCM1', resource_manager=None):
    """Create magnet controller based on model selection.
    
    Args:
        model: 'SCM1' or 'Cryomagnetics 4G'
        resource_manager: Optional pyvisa ResourceManager (not used for SCM1)
    
    Returns:
        NHMFLMagnetController or CryomagneticsController instance
    """
    if model == 'Cryomagnetics 4G':
        return CryomagneticsController(resource_manager)
    else:
        return NHMFLMagnetController()  # SCM1 (default)


class KeithleyController:
    """Controller for Keithley 2400/2450 SMU (gate voltage).
    
    SAFETY FEATURES (v2.1):
    - Never jumps voltage - always ramps, even on connect/reconnect
    - Reads actual voltage BEFORE reset to prevent jumps
    - Emergency shutdown capability
    - Fails loudly on communication errors (no stale values)
    
    Supports both Keithley 2400 and 2450 models with appropriate SCPI commands.
    """
    
    # Model-specific command sets
    MODELS = {
        '2400': {
            'name': 'Keithley 2400',
            'reset': '*RST',
            'source_voltage': ':SOUR:FUNC VOLT',
            'set_voltage': ':SOUR:VOLT:LEV {:.6f}',
            'get_voltage': ':SOUR:VOLT:LEV?',
            'compliance_current': ':SENS:CURR:PROT {:.9f}',
            'output_on': ':OUTP ON',
            'output_off': ':OUTP OFF',
            'output_state': ':OUTP?',
            'measure_current': ':MEAS:CURR?',
            'measure_voltage': ':MEAS:VOLT?',
        },
        '2450': {
            'name': 'Keithley 2450',
            'reset': '*RST',
            'source_voltage': ':SOUR:FUNC VOLT',
            'set_voltage': ':SOUR:VOLT {:.6f}',
            'get_voltage': ':SOUR:VOLT?',
            'compliance_current': ':SOUR:VOLT:ILIM {:.9f}',
            'output_on': ':OUTP ON',
            'output_off': ':OUTP OFF',
            'output_state': ':OUTP?',
            'measure_current': ':MEAS:CURR?',
            'measure_voltage': ':MEAS:VOLT?',
        }
    }
    
    def __init__(self, resource_manager=None, model='2450'):
        self.connected = False
        self.address = "GPIB0::24::INSTR"
        self.instrument = None
        self.rm = resource_manager
        self._current_voltage = 0.0  # Internal tracking (use get_voltage() for actual)
        self._voltage_known = False  # Flag: do we trust _current_voltage?
        self.max_voltage = 100.0
        self.compliance_current = 100e-9  # 100 nA default
        self.slew_rate = 1.0  # V/s
        self.model = model
        self.commands = self.MODELS.get(model, self.MODELS['2450'])
        
        # Safety settings
        self._safe_step_size = 1.0  # Maximum voltage step without ramping (V)
        self._emergency_stop = False
        
        # For backward compatibility
        self.current_voltage = property(lambda self: self._current_voltage)
        
    def set_model(self, model):
        """Change the SMU model (2400 or 2450)."""
        if model in self.MODELS:
            self.model = model
            self.commands = self.MODELS[model]
            print(f"Keithley model set to {self.commands['name']}")
        else:
            print(f"Unknown model: {model}. Using 2450.")
            self.model = '2450'
            self.commands = self.MODELS['2450']
        
    def set_address(self, address):
        """Update instrument address."""
        address = str(address).strip()
        
        if '::' in address:
            self.address = address
        else:
            self.address = f"GPIB0::{address}::INSTR"
        
        print(f"Keithley address set to: {self.address}")
    
    def _query_raw(self, command, timeout_ms=10000):
        """Raw query - simple and direct.
        
        Raises exception on any communication failure.
        """
        if not self.instrument:
            raise RuntimeError("Keithley not connected")
        
        old_timeout = self.instrument.timeout
        try:
            self.instrument.timeout = timeout_ms
            # Try query, with one retry on timeout
            try:
                response = self.instrument.query(command).strip()
            except Exception as e:
                if 'TMO' in str(e) or 'timeout' in str(e).lower():
                    # Timeout - try once more
                    time.sleep(0.2)
                    response = self.instrument.query(command).strip()
                else:
                    raise
            return response
        except Exception as e:
            raise RuntimeError(f"Keithley communication error: {e}")
        finally:
            self.instrument.timeout = old_timeout
    
    def _write_raw(self, command):
        """Raw write - simple and direct."""
        if not self.instrument:
            raise RuntimeError("Keithley not connected")
        
        try:
            self.instrument.write(command)
        except Exception as e:
            raise RuntimeError(f"Keithley write error: {e}")
    
    def _read_actual_voltage(self):
        """Read the ACTUAL voltage from the instrument.
        
        Returns:
            float: Current voltage setting
            
        Raises:
            RuntimeError: On communication failure
        """
        response = self._query_raw(self.commands['get_voltage'])
        try:
            voltage = float(response)
            self._current_voltage = voltage
            self._voltage_known = True
            return voltage
        except ValueError:
            raise RuntimeError(f"Invalid voltage response: {response}")
    
    def _read_output_state(self):
        """Read whether output is ON or OFF.
        
        Returns:
            bool: True if output is ON
        """
        try:
            response = self._query_raw(self.commands['output_state'])
            return response in ['1', 'ON', 'on']
        except:
            return False
    
    def _manual_ramp(self, start, target, slew_rate):
        """Low-level ramp without using ramp_to_voltage (for use during init)."""
        voltage_diff = abs(target - start)
        
        if voltage_diff < 0.01:
            return True
        
        n_steps = max(int(voltage_diff / self._safe_step_size), 2)
        step_delay = voltage_diff / slew_rate / n_steps
        voltages = np.linspace(start, target, n_steps + 1)[1:]
        
        print(f"  Ramping: {start:.2f}V -> {target:.2f}V ({n_steps} steps, {slew_rate} V/s)")
        
        for v in voltages:
            if self._emergency_stop:
                print("  EMERGENCY STOP during ramp!")
                return False
            
            self._write_raw(self.commands['set_voltage'].format(v))
            self._current_voltage = v
            time.sleep(step_delay)
        
        return True
    
    def connect(self):
        """Safely connect to Keithley SMU.
        
        SAFETY: Reads current voltage before any reset to prevent jumps.
        If voltage is non-zero, ramps to zero before resetting.
        """
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                # Fresh ResourceManager on retry
                if attempt > 0:
                    print("Retrying with fresh VISA connection...")
                    if self.rm is not None:
                        try:
                            self.rm.close()
                        except:
                            pass
                        self.rm = None
                    time.sleep(0.5)
                
                if self.rm is None:
                    import pyvisa
                    try:
                        self.rm = pyvisa.ResourceManager()
                        print(f"Using VISA backend: NI-VISA")
                    except Exception as e:
                        raise RuntimeError(f"No VISA backend available: {e}")
                
                # Close existing connection
                if self.instrument is not None:
                    try:
                        self.instrument.close()
                    except:
                        pass
                    self.instrument = None
                
                # Skip bus-wide interface clear - it affects other GPIB devices
                # and causes issues with Lakeshore 370
                
                # Open connection
                print(f"Connecting to {self.address}...")
                self.instrument = self.rm.open_resource(self.address)
                self.instrument.timeout = 5000
                self.instrument.read_termination = '\n'
                self.instrument.write_termination = '\n'
                
                # Small delay to let connection stabilize
                time.sleep(0.2)
                
                # Verify connection
                idn = self._query_raw('*IDN?')
                print(f"Connected to: {idn}")
                
                # ========== SAFETY CRITICAL SECTION ==========
                # Read current state BEFORE doing anything else
                
                output_was_on = self._read_output_state()
                
                if output_was_on:
                    current_v = self._read_actual_voltage()
                    print(f"WARNING: Output was ON at {current_v:.3f}V")
                    
                    if abs(current_v) > 0.1:
                        print(f"SAFETY: Ramping from {current_v:.3f}V to 0V before reset...")
                        
                        if not self._manual_ramp(current_v, 0.0, self.slew_rate):
                            raise RuntimeError("Failed to ramp voltage to zero")
                        
                        print(f"Gate safely ramped to 0V")
                    
                    self._write_raw(self.commands['output_off'])
                    time.sleep(0.1)
                else:
                    print("Output was OFF")
                    self._current_voltage = 0.0
                
                # ========== NOW safe to reset ==========
                print("Resetting instrument...")
                self._write_raw(self.commands['reset'])
                time.sleep(0.5)
                
                # Configure for voltage sourcing
                self._write_raw(self.commands['source_voltage'])
                self._write_raw(':SOUR:VOLT:RANG:AUTO ON')
                self._write_raw(':SYST:RSEN OFF')  # 2-wire mode
                
                # Set compliance
                self._write_raw(self.commands['compliance_current'].format(self.compliance_current))
                self._write_raw(f':SENS:CURR:RANG {self.compliance_current}')
                print(f"Compliance: {self.compliance_current*1e6:.1f} uA")
                
                # Set voltage to 0V
                self._write_raw(self.commands['set_voltage'].format(0.0))
                
                # Turn output ON
                self._write_raw(self.commands['output_on'])
                
                # Verify final state
                final_v = self._read_actual_voltage()
                if abs(final_v) > 0.01:
                    raise RuntimeError(f"Voltage not at zero after init: {final_v}V")
                
                self.connected = True
                self._voltage_known = True
                print(f"Keithley connected safely. Output ON at 0.000V")
                return True
                
            except Exception as e:
                error_str = str(e)
                print(f"Connection attempt {attempt + 1} failed: {error_str}")
                
                if self.instrument is not None:
                    try:
                        self.instrument.close()
                    except:
                        pass
                    self.instrument = None
                
                if attempt < max_retries - 1:
                    if self.rm is not None:
                        try:
                            self.rm.close()
                        except:
                            pass
                        self.rm = None
                    continue
                
                self.connected = False
                self._voltage_known = False
                return False
        
        self.connected = False
        self._voltage_known = False
        return False
    
    def disconnect(self):
        """Disconnect from Keithley (ramp to zero first)."""
        if self.connected and self.instrument:
            try:
                print("Disconnecting Keithley - ramping to zero first...")
                self.ramp_to_voltage(0.0)
                self._write_raw(self.commands['output_off'])
                self.instrument.close()
                print("Keithley disconnected safely")
            except Exception as e:
                print(f"Error during disconnect: {e}")
                try:
                    self.instrument.close()
                except:
                    pass
        
        self.connected = False
        self.instrument = None
        self._voltage_known = False
    
    def emergency_shutdown(self):
        """EMERGENCY: Immediately turn off output.
        
        Use this when something goes wrong and you need to stop immediately.
        Does NOT ramp - immediately turns off output.
        """
        self._emergency_stop = True
        print("EMERGENCY SHUTDOWN - turning output OFF immediately")
        
        if self.instrument:
            try:
                for _ in range(3):
                    try:
                        self.instrument.write(self.commands['output_off'])
                        time.sleep(0.1)
                    except:
                        pass
                
                print("Output OFF command sent")
            except Exception as e:
                print(f"Emergency shutdown error: {e}")
        
        self._voltage_known = False
    
    def clear_emergency(self):
        """Clear emergency stop flag."""
        self._emergency_stop = False
        print("Emergency stop cleared")
    
    def get_voltage(self):
        """Get current voltage setpoint.
        
        SAFETY: Returns actual queried value, raises exception on comm failure.
        """
        if self._emergency_stop:
            raise RuntimeError("Emergency stop active - call clear_emergency() first")
        
        if self.connected and self.instrument:
            try:
                return self._read_actual_voltage()
            except Exception as e:
                self._voltage_known = False
                raise RuntimeError(f"Failed to read voltage: {e}")
        else:
            # Not connected - return tracked value silently
            # Use get_voltage_safe() if you need to check reliability
            return self._current_voltage
    
    def get_voltage_safe(self):
        """Get voltage without raising exception.
        
        Returns:
            tuple: (voltage, is_reliable)
        """
        try:
            v = self.get_voltage()
            return (v, True)
        except:
            return (self._current_voltage, False)
    
    def set_voltage(self, voltage, enable_output=False, _from_ramp=False):
        """Set voltage (direct, no ramping - USE WITH CAUTION).
        
        For normal operation, use ramp_to_voltage() instead.
        
        Args:
            voltage: Target voltage
            enable_output: Also turn on output if True
            _from_ramp: Internal flag - suppresses warning when called from ramp_to_voltage
        """
        if self._emergency_stop:
            print("ERROR: Emergency stop active - cannot set voltage")
            return False
        
        if abs(voltage) > self.max_voltage:
            print(f"ERROR: Voltage {voltage}V exceeds limit {self.max_voltage}V")
            return False
        
        # Only warn about large steps if NOT called from ramp (which handles stepping)
        # And only for REALLY large steps (>5V) that are clearly problematic
        if self._voltage_known and not _from_ramp:
            diff = abs(voltage - self._current_voltage)
            if diff > 5.0:  # Only warn for >5V jumps (not 1-2V during normal operation)
                print(f"WARNING: Large voltage step ({diff:.1f}V) - consider using ramp_to_voltage()")
        
        if self.connected and self.instrument:
            try:
                self._write_raw(self.commands['set_voltage'].format(voltage))
                if enable_output:
                    self._write_raw(self.commands['output_on'])
                self._current_voltage = voltage
                self._voltage_known = True
                return True
            except Exception as e:
                print(f"Error setting voltage: {e}")
                self._voltage_known = False
                return False
        else:
            self._current_voltage = voltage
            return True
    
    def ramp_to_voltage(self, target, slew_rate=None, stop_check=None):
        """Safely ramp to target voltage.
        
        SAFETY: Always ramps, never jumps. Checks for stop requests.
        """
        if self._emergency_stop:
            print("ERROR: Emergency stop active - cannot ramp voltage")
            return False
        
        if slew_rate is None:
            slew_rate = self.slew_rate
        
        if abs(target) > self.max_voltage:
            print(f"ERROR: Target {target}V exceeds limit {self.max_voltage}V")
            return False
        
        # Get current voltage
        try:
            if self.connected and self.instrument:
                start = self._read_actual_voltage()
            else:
                start = self._current_voltage
        except Exception as e:
            print(f"ERROR: Cannot read current voltage: {e}")
            return False
        
        voltage_diff = abs(target - start)
        
        if voltage_diff < 0.001:
            return True
        
        print(f"Ramping gate: {start:.3f}V -> {target:.3f}V at {slew_rate} V/s")
        
        n_steps = max(int(voltage_diff / self._safe_step_size), 2)
        ramp_time = voltage_diff / slew_rate
        step_delay = ramp_time / n_steps
        voltages = np.linspace(start, target, n_steps + 1)[1:]
        
        for i, v in enumerate(voltages):
            if self._emergency_stop:
                print(f"EMERGENCY STOP at {self._current_voltage:.3f}V")
                return False
            
            if stop_check is not None and stop_check():
                print(f"GATE RAMP STOPPED at {self._current_voltage:.3f}V (step {i+1}/{len(voltages)})")
                return False
            
            if not self.set_voltage(v, enable_output=False, _from_ramp=True):
                print(f"Failed to set voltage to {v}V")
                return False
            
            time.sleep(step_delay)
        
        print(f"Gate ramp complete: {self._current_voltage:.3f}V")
        return True
    
    def output_on(self):
        """Enable output."""
        if self.connected and self.instrument:
            self._write_raw(self.commands['output_on'])
    
    def output_off(self):
        """Disable output."""
        if self.connected and self.instrument:
            self._write_raw(self.commands['output_off'])
    
    def measure_current(self):
        """Measure current through device."""
        if self.connected and self.instrument:
            try:
                response = self._query_raw(self.commands['measure_current'])
                return float(response.split(',')[0])
            except:
                return 0.0
        return 0.0
    
    def set_compliance(self, current_limit):
        """Set compliance (current limit) and matching sense range."""
        self.compliance_current = current_limit
        if self.connected and self.instrument:
            self._write_raw(self.commands['compliance_current'].format(current_limit))
            self._write_raw(f':SENS:CURR:RANG {current_limit}')
            print(f"Keithley compliance set to {current_limit*1e9:.0f} nA")
    
    def check_compliance(self):
        """Check if compliance (current limit) has been reached.
        
        Returns:
            bool: True if in compliance (current limited)
        """
        if self.connected and self.instrument:
            try:
                current = abs(self.measure_current())
                in_compliance = current >= (self.compliance_current * 0.95)
                if in_compliance:
                    print(f"WARNING: COMPLIANCE - Current {current*1e9:.1f} nA >= limit {self.compliance_current*1e9:.1f} nA")
                return in_compliance
            except:
                return False
        return False


class GateSafetyWrapper:
    """Wrapper to add safety checks around gate operations.
    
    Usage:
        safety = GateSafetyWrapper(keithley)
        safety.start_measurement()
        
        # During measurement:
        if safety.check_gate_ok():
            # proceed
        else:
            safety.abort_measurement()
    """
    
    def __init__(self, keithley):
        self.keithley = keithley
        self._measurement_active = False
        self._start_voltage = None
        self._last_check_time = 0
        self._check_interval = 1.0  # seconds
        
    def start_measurement(self):
        """Record state at start of measurement."""
        self._measurement_active = True
        
        # Only monitor if Keithley is actually connected
        if not self.keithley.connected:
            return
            
        try:
            self._start_voltage = self.keithley.get_voltage()
            print(f"Gate safety: Measurement started at {self._start_voltage:.3f}V")
        except:
            self._start_voltage = None
            print("Gate safety: Could not read initial voltage")
    
    def check_gate_ok(self):
        """Periodic check that gate is responding.
        
        Returns False if there's a problem.
        """
        if not self._measurement_active:
            return True
        
        # Skip checks if Keithley not connected
        if not self.keithley.connected:
            return True
        
        now = time.time()
        if now - self._last_check_time < self._check_interval:
            return True
        self._last_check_time = now
        
        v, reliable = self.keithley.get_voltage_safe()
        if not reliable:
            print("Gate safety: Lost communication with Keithley!")
            return False
        
        if self.keithley.check_compliance():
            print("Gate safety: Compliance limit reached!")
            return False
        
        return True
    
    def abort_measurement(self):
        """Handle measurement abort - try to safe the gate."""
        self._measurement_active = False
        
        # Only act if Keithley is connected
        if not self.keithley.connected:
            return
            
        print("Gate safety: Aborting measurement...")
        
        try:
            self.keithley.ramp_to_voltage(0.0)
        except Exception as e:
            print(f"Gate safety: Ramp failed ({e}), trying emergency shutdown...")
            self.keithley.emergency_shutdown()
    
    def end_measurement(self):
        """Clean end of measurement."""
        self._measurement_active = False
        
        # Only print if Keithley is connected
        if self.keithley.connected:
            print("Gate safety: Measurement ended normally")




class Lakeshore370Controller:
    """Controller for Lakeshore 370 AC Resistance Bridge.
    
    Read-only temperature monitoring for dilution refrigerator.
    Does NOT control temperature - only reads from mixing chamber sensor.
    
    GPIB communication requires ~1 second delays between commands.
    """
    
    def __init__(self, resource_manager=None):
        self.connected = False
        self.address = "GPIB0::19::INSTR"
        self.instrument = None
        self.rm = resource_manager
        self.channel = 4  # Default to mixing chamber channel
        self.last_query_time = 0
        self.min_query_interval = 1.0  # Minimum seconds between queries
        
        # Last readings cache
        self.last_temperature_k = None
        self.last_resistance_ohms = None
        self.last_read_time = 0
    
    def set_address(self, gpib_address):
        """Update GPIB address."""
        try:
            addr = int(gpib_address)
            self.address = f"GPIB0::{addr}::INSTR"
        except:
            self.address = f"GPIB0::{gpib_address}::INSTR"
    
    def set_channel(self, channel):
        """Set the channel to read from (1-16)."""
        try:
            ch = int(channel)
            if 1 <= ch <= 16:
                self.channel = ch
        except:
            pass
    
    def connect(self):
        """Connect to Lakeshore 370 via GPIB."""
        try:
            if self.rm is None:
                import pyvisa
                self.rm = pyvisa.ResourceManager()
            
            self.instrument = self.rm.open_resource(self.address)
            self.instrument.timeout = 5000  # 5 second timeout
            self.instrument.read_termination = '\r\n'
            self.instrument.write_termination = '\r\n'
            
            # Small delay to let connection stabilize
            time.sleep(0.3)
            
            # Query identification
            idn = self._safe_query("*IDN?")
            if idn and "370" in idn:
                print(f"Lakeshore 370 connected: {idn.strip()}")
                self.connected = True
                return True
            else:
                print(f"Lakeshore 370: Unexpected response: {idn}")
                return False
                
        except Exception as e:
            print(f"Lakeshore 370 connection error: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from instrument."""
        if self.instrument:
            try:
                self.instrument.close()
            except:
                pass
        self.connected = False
        self.instrument = None
    
    def _safe_query(self, cmd, delay=1.0):
        """Query with proper timing.
        
        The Lakeshore 370 requires delays between commands.
        """
        if not self.instrument:
            return None
        
        # Ensure minimum time between queries
        elapsed = time.time() - self.last_query_time
        if elapsed < self.min_query_interval:
            time.sleep(self.min_query_interval - elapsed)
        
        try:
            # Simple query with delay
            self.instrument.write(cmd)
            time.sleep(delay)
            
            # Read response
            raw = self.instrument.read_raw()
            self.last_query_time = time.time()
            
            return raw.decode('ascii', errors='replace').strip()
            
        except Exception as e:
            print(f"Lakeshore 370 query error ({cmd}): {e}")
            return None
    
    def get_temperature(self, channel=None):
        """Get temperature in Kelvin.
        
        Args:
            channel: Channel number (1-16), or None to use default
            
        Returns:
            Temperature in Kelvin, or None on error
        """
        if not self.connected:
            return None
        
        ch = channel if channel is not None else self.channel
        response = self._safe_query(f"RDGK? {ch}")
        
        if response:
            try:
                # Parse scientific notation: +XX.XXXXE+/-XX
                temp_k = float(response.replace('+', '').replace('E', 'e'))
                self.last_temperature_k = temp_k
                self.last_read_time = time.time()
                return temp_k
            except ValueError:
                print(f"Could not parse temperature: {response}")
        
        return None
    
    def get_temperature_mk(self, channel=None):
        """Get temperature in milliKelvin.
        
        Args:
            channel: Channel number (1-16), or None to use default
            
        Returns:
            Temperature in mK, or None on error
        """
        temp_k = self.get_temperature(channel)
        if temp_k is not None:
            return temp_k * 1000
        return None
    
    def get_resistance(self, channel=None):
        """Get resistance in Ohms.
        
        Args:
            channel: Channel number (1-16), or None to use default
            
        Returns:
            Resistance in Ohms, or None on error
        """
        if not self.connected:
            return None
        
        ch = channel if channel is not None else self.channel
        response = self._safe_query(f"RDGR? {ch}")
        
        if response:
            try:
                # Parse scientific notation: +XX.XXXXE+/-XX
                resistance = float(response.replace('+', '').replace('E', 'e'))
                self.last_resistance_ohms = resistance
                return resistance
            except ValueError:
                print(f"Could not parse resistance: {response}")
        
        return None
    
    def get_cached_temperature(self, max_age=60.0):
        """Get cached temperature if recent enough, otherwise read new.
        
        Args:
            max_age: Maximum age of cached value in seconds
            
        Returns:
            Temperature in Kelvin
        """
        if self.last_temperature_k is not None:
            age = time.time() - self.last_read_time
            if age < max_age:
                return self.last_temperature_k
        
        return self.get_temperature()


class MeasurementEngine:
    """Handles measurement execution with real or simulated instruments."""
    
    def __init__(self, vna, magnet, keithley, temp_controller, use_simulation=True):
        self.vna = vna
        self.magnet = magnet
        self.keithley = keithley
        self.temp_controller = temp_controller
        self.use_simulation = use_simulation
        self.sim_data = SimulatedDataGenerator()
        
        self.is_running = False
        self.should_stop = False
        self.data_queue = queue.Queue()
        self.progress_queue = queue.Queue()
        
        # Gate voltage safety settings
        self.gate_slew_rate = 10.0  # V/s
        self.gate_ramp_to_zero_after = True
        self.gate_ramp_on_stop = False
        self.gate_max_voltage = 100.0
        self.gate_compliance = 100e-9  # 100 nA
        
        # B-field settings
        self.field_ramp_rate = 0.3  # T/min (SCM1 max is 0.3)
        self.field_ramp_rate_max = 0.3  # T/min - SCM1 limit
        self.field_tolerance = 0.001  # T (10 Gauss)
        self.field_settle_time = 2.0  # seconds
        self.wait_for_field = True
        
        # VNA settings
        self.vna_settle_time = 5.0  # seconds - delay before first measurement
        
        # Gate safety wrapper for monitoring during measurements
        self.gate_safety = GateSafetyWrapper(keithley)
    
    def set_field_settings(self, ramp_rate=0.3, tolerance=0.01, settle_time=2.0, wait=True):
        """Configure B-field sweep settings.
        
        Args:
            ramp_rate: Field ramp rate in T/min (clamped to 0-0.3 for SCM1)
            tolerance: Field tolerance in T for considering target reached
            settle_time: Wait time after field reaches target (seconds)
            wait: If True, wait for field to stabilize when stepping
        """
        # Clamp ramp rate to SCM1 limits (0 to 0.3 T/min)
        ramp_rate = max(0.0, min(ramp_rate, self.field_ramp_rate_max))
        
        self.field_ramp_rate = ramp_rate
        self.field_tolerance = tolerance
        self.field_settle_time = settle_time
        self.wait_for_field = wait
        print(f"B-field settings: rate={ramp_rate} T/min, tolerance={tolerance} T, "
              f"settle={settle_time}s, wait={wait}")
    
    def set_gate_safety(self, slew_rate=10.0, ramp_to_zero_after=True, 
                        ramp_on_stop=False, max_voltage=100.0, compliance=100e-9):
        """Configure gate voltage safety settings.
        
        Settling time between gate voltage steps is auto-calculated based on:
        - Slew rate (time for voltage to change)
        - IFBW (time for VNA measurement to settle, ~3 time constants)
        """
        self.gate_slew_rate = slew_rate
        self.gate_ramp_to_zero_after = ramp_to_zero_after
        self.gate_ramp_on_stop = ramp_on_stop
        self.gate_max_voltage = max_voltage
        self.gate_compliance = compliance
        
        # Update Keithley settings
        self.keithley.slew_rate = slew_rate
        self.keithley.max_voltage = max_voltage
        self.keithley.set_compliance(compliance)  # Actually send to instrument
        print(f"Gate safety: slew={slew_rate}V/s, max={max_voltage}V, compliance={compliance*1e9:.0f}nA, "
              f"ramp_to_zero={ramp_to_zero_after}, ramp_on_stop={ramp_on_stop}")
    
    def stop_check(self):
        """Return True if measurement should stop (for use in ramp callbacks)."""
        return self.should_stop
        
    def stop(self):
        """Signal measurement to stop."""
        self.should_stop = True
        
    def run_measurement(self, config):
        """Execute measurement based on configuration."""
        self.is_running = True
        self.should_stop = False
        
        sweep_param = config['sweep_param']
        sweep_start = config['sweep_start']
        sweep_stop = config['sweep_stop']
        sweep_points = config['sweep_points']
        
        step_param = config.get('step_param', None)
        step_start = config.get('step_start', 0)
        step_stop = config.get('step_stop', 0)
        step_points = config.get('step_points', 1)
        
        fixed_values = config['fixed_values']
        s_param = config.get('s_parameter', 'S21')
        
        # Generate sweep arrays
        sweep_values = np.linspace(sweep_start, sweep_stop, sweep_points)
        
        if step_param and step_param != "None":
            step_values = np.linspace(step_start, step_stop, step_points)
            is_2d = True
            print(f"Step values array: {step_values}")
        else:
            step_values = [None]
            is_2d = False
        
        # Determine if we can use fast VNA sweep mode
        # (only for frequency sweeps when VNA is connected)
        use_vna_sweep = (
            sweep_param == "Frequency (GHz)" and 
            not self.use_simulation and 
            self.vna.connected
        )
        
        # Determine if we use continuous B-field sweep
        use_bfield_continuous = (
            sweep_param == "B-Field (T)" and
            not self.use_simulation and
            self.magnet.connected
        )
        
        # Debug output
        print(f"=== Measurement Configuration ===")
        print(f"Sweep: {sweep_param} ({sweep_start} to {sweep_stop}, {sweep_points} pts)")
        print(f"Step: {step_param} ({step_start} to {step_stop}, {step_points} pts)" if step_param else "Step: None")
        if use_vna_sweep:
            mode_str = 'VNA BATCH SWEEP'
        elif use_bfield_continuous:
            mode_str = 'B-FIELD CONTINUOUS'
        else:
            mode_str = 'POINT-BY-POINT'
        print(f"Mode: {mode_str}")
        print(f"Simulation: {self.use_simulation}, VNA connected: {self.vna.connected}")
        if use_bfield_continuous:
            print(f"Magnet connected: {self.magnet.connected}, Rate: {self.field_ramp_rate} T/min")
        print(f"================================")
        
        # Configure VNA
        if not self.use_simulation and self.vna.connected:
            self.vna.s_parameter = s_param
            
            if use_vna_sweep:
                # Configure VNA for full frequency sweep
                # If stepping power, use first step value; otherwise use fixed power
                if step_param == "Power (dBm)" and len(step_values) > 0 and step_values[0] is not None:
                    initial_power = step_values[0]
                else:
                    initial_power = fixed_values.get('power', -10)
                
                # Get number of averages for hardware averaging
                num_averages = config.get('averages', 1)
                
                print(f"Initial VNA sweep config: {sweep_start}-{sweep_stop} Hz, {sweep_points} pts, "
                      f"IFBW={fixed_values.get('ifbw', 100)}, Power={initial_power} dBm, Avg={num_averages}")
                
                self.vna.setup_frequency_sweep(
                    sweep_start, sweep_stop, sweep_points,
                    fixed_values.get('ifbw', 100),
                    initial_power,
                    num_averages
                )
            else:
                # Configure for CW mode (non-frequency sweeps)
                self.vna.setup_cw_mode(
                    fixed_values.get('frequency', 8e9),
                    fixed_values.get('ifbw', 100),
                    fixed_values.get('power', -10)
                )
        
        # Data storage
        all_data = []
        total_points = len(step_values) * len(sweep_values)
        current_point = 0
        
        # VNA settle time - wait for RF output to stabilize before first measurement
        if not self.use_simulation and self.vna.connected and self.vna_settle_time > 0:
            print(f"VNA settling for {self.vna_settle_time}s before first measurement...")
            time.sleep(self.vna_settle_time)
            # Flush stale data from VNA buffer by doing multiple dummy reads
            # This ensures any buffered/cached data from previous sweeps is discarded
            print("  Flushing VNA buffer (discarding stale readings)...")
            for i in range(3):
                try:
                    _ = self._get_real_data()
                    time.sleep(0.05)  # Small delay between flush reads
                except:
                    pass  # Ignore any errors during flush
            print("  VNA ready.")
        
        # Track if gate voltage is involved (for post-sweep ramp to zero)
        gate_involved = (sweep_param == "Gate Voltage (V)" or step_param == "Gate Voltage (V)")
        
        # Track if B-field is involved (for field normalization)
        field_involved = (sweep_param == "B-Field (T)" or step_param == "B-Field (T)")
        
        # START GATE SAFETY MONITORING
        if gate_involved:
            self.gate_safety.start_measurement()
        
        # === GATE NORMALIZATION REFERENCE MEASUREMENT ===
        # Do this FIRST, before ramping to start position, to avoid unnecessary ramping
        # (typically we're already at 0V which is the common reference voltage)
        normalization_enabled = config.get('normalization_enabled', False)
        normalization_voltage = config.get('normalization_voltage', 0.0)
        reference_s21_mag_db = None  # Single value for gate sweeps
        reference_spectrum_mag_db = None  # Full spectrum for freq sweeps with gate step
        per_step_reference = False  # Flag for gate sweep with freq step (reference taken at each step)
        step_reference_db = {}  # Dictionary to store per-step reference values
        
        if normalization_enabled and gate_involved:
            print(f"=== Taking reference measurement at V_gate = {normalization_voltage}V ===")
            
            # Check if we need to ramp to reference voltage
            if not self.use_simulation and self.keithley.connected:
                current_v, reliable = self.keithley.get_voltage_safe()
                if not reliable:
                    print("ERROR: Cannot read gate voltage - communication error")
                    self.data_queue.put({
                        'type': 'error',
                        'message': 'Cannot read gate voltage - communication error'
                    })
                    self.gate_safety.abort_measurement()
                    self.is_running = False
                    return
                
                if abs(current_v - normalization_voltage) > 0.001:
                    print(f"  Ramping gate to reference voltage {normalization_voltage}V...")
                    self.keithley.ramp_to_voltage(
                        normalization_voltage,
                        slew_rate=self.gate_slew_rate,
                        stop_check=self.stop_check
                    )
                    time.sleep(0.5)  # Settle at reference voltage
                else:
                    print(f"  Already at reference voltage {current_v:.3f}V")
            
            if self.should_stop:
                self.data_queue.put({
                    'type': 'aborted',
                    'all_data': [],
                    'config': config
                })
                self.gate_safety.end_measurement()
                self.is_running = False
                return
            
            # Configure VNA for reference measurement if needed
            if not self.use_simulation and self.vna.connected:
                if sweep_param == "Frequency (GHz)":
                    # Need full frequency sweep for reference
                    num_averages = config.get('averages', 1)
                    if step_param == "Power (dBm)" and len(step_values) > 0 and step_values[0] is not None:
                        initial_power = step_values[0]
                    else:
                        initial_power = fixed_values.get('power', -10)
                    
                    print(f"  Configuring VNA for reference sweep...")
                    self.vna.setup_frequency_sweep(
                        sweep_start, sweep_stop, sweep_points,
                        fixed_values.get('ifbw', 100),
                        initial_power,
                        num_averages
                    )
                else:
                    # CW mode for gate sweeps
                    self.vna.setup_cw_mode(
                        fixed_values.get('frequency', 8e9),
                        fixed_values.get('ifbw', 100),
                        fixed_values.get('power', -10)
                    )
            
            # Take reference measurement based on sweep type
            if sweep_param == "Gate Voltage (V)" and step_param == "Frequency (GHz)":
                # Gate sweep with frequency step: references taken per-step (see step loop below)
                # Here we just set up CW mode for the first frequency and note that per-step refs are needed
                per_step_reference = True  # Flag to take reference at each frequency step
                print(f"  Per-step reference mode: will take CW reference at {normalization_voltage}V for each frequency")
                # Dictionary to store reference for each frequency step
                step_reference_db = {}
                
            elif sweep_param == "Gate Voltage (V)":
                # Gate sweep (no freq step): take single CW measurement at reference voltage
                print(f"  Taking CW reference measurement...")
                if self.use_simulation:
                    ref_s21 = self.sim_data.generate_s21_vs_gate(
                        np.array([normalization_voltage]),
                        fixed_values.get('frequency', 8e9),
                        fixed_values.get('b_field', 0),
                        fixed_values.get('power', -10),
                        300.0
                    )[0]
                else:
                    # Trigger and read single CW point
                    self.vna.trigger_sweep()
                    ref_s21 = self.vna.get_cw_data()
                
                if ref_s21 is not None:
                    reference_s21_mag_db = 20 * np.log10(np.abs(ref_s21) + 1e-12)
                    print(f"  Reference S21 at {normalization_voltage}V: {reference_s21_mag_db:.2f} dB")
                
            elif sweep_param == "Frequency (GHz)" and step_param == "Gate Voltage (V)":
                # Freq sweep with gate step: take full spectrum at reference voltage
                print(f"  Taking reference spectrum at {normalization_voltage}V...")
                num_averages = config.get('averages', 1)
                ifbw = fixed_values.get('ifbw', 100)
                print(f"  Reference sweep params: {sweep_points} points, IFBW={ifbw}, averages={num_averages}")
                
                if self.use_simulation:
                    ref_spectrum = self.sim_data.generate_s21_vs_frequency(
                        sweep_values,
                        fixed_values.get('b_field', 0),
                        normalization_voltage,
                        fixed_values.get('power', -10),
                        300.0
                    )
                else:
                    # Trigger VNA sweep for reference (with full averaging)
                    print(f"  Triggering reference sweep with {num_averages} averages...")
                    if not self.vna.trigger_sweep_timed(sweep_points, ifbw, num_averages):
                        print("  WARNING: VNA trigger for reference sweep returned False")
                    ref_spectrum = self.vna.get_sweep_data(expected_points=sweep_points)
                
                if ref_spectrum is not None and len(ref_spectrum) > 0:
                    reference_spectrum_mag_db = 20 * np.log10(np.abs(ref_spectrum) + 1e-12)
                    print(f"  Reference spectrum: {len(reference_spectrum_mag_db)} points, "
                          f"range {np.min(reference_spectrum_mag_db):.1f} to {np.max(reference_spectrum_mag_db):.1f} dB")
                else:
                    print("  WARNING: Failed to get reference spectrum!")
            
            # Send reference data to GUI for storage/saving (skip for per-step mode - will send later)
            if not per_step_reference:
                self.data_queue.put({
                    'type': 'reference_data',
                    'voltage': normalization_voltage,
                    'single_value_db': reference_s21_mag_db,
                    'spectrum_db': reference_spectrum_mag_db.tolist() if reference_spectrum_mag_db is not None else None,
                    'frequencies': sweep_values.tolist() if reference_spectrum_mag_db is not None else None
                })
            
            print(f"=== Reference measurement complete ===")
        
        # === B-FIELD NORMALIZATION REFERENCE MEASUREMENT ===
        field_normalization_enabled = config.get('field_normalization_enabled', False)
        field_normalization_field = config.get('field_normalization_field', 0.0)
        field_reference_s21_mag_db = None  # Single value for field sweeps
        field_reference_spectrum_mag_db = None  # Full spectrum for freq sweeps with field step
        field_per_step_reference = False  # Flag for field sweep with freq step
        field_step_reference_db = {}  # Dictionary to store per-step reference values
        
        if field_normalization_enabled and field_involved:
            print(f"=== Taking reference measurement at B = {field_normalization_field}T ===")
            
            # Ramp to reference field
            if not self.use_simulation and self.magnet.connected:
                current_b = self.magnet.get_field()
                if abs(current_b - field_normalization_field) > 0.001:
                    print(f"  Ramping magnet to reference field {field_normalization_field}T...")
                    self.magnet.set_field(field_normalization_field)
                    
                    # Wait for field to stabilize
                    field_tol = 0.01  # Default tolerance
                    max_wait = 600  # 10 minute timeout for field ramp
                    start_time = time.time()
                    
                    while time.time() - start_time < max_wait:
                        if self.should_stop:
                            break
                        current_b = self.magnet.get_field()
                        if abs(current_b - field_normalization_field) <= field_tol:
                            print(f"  Field reached: {current_b:.4f}T")
                            break
                        time.sleep(0.5)
                    else:
                        print(f"  WARNING: Timeout waiting for field to reach {field_normalization_field}T")
                    
                    # Settle time after reaching field
                    time.sleep(2.0)
                else:
                    print(f"  Already at reference field {current_b:.4f}T")
            
            if self.should_stop:
                self.data_queue.put({
                    'type': 'aborted',
                    'all_data': [],
                    'config': config
                })
                if gate_involved:
                    self.gate_safety.end_measurement()
                self.is_running = False
                return
            
            # Configure VNA for reference measurement if needed
            if not self.use_simulation and self.vna.connected:
                if sweep_param == "Frequency (GHz)":
                    # Need full frequency sweep for reference
                    num_averages = config.get('averages', 1)
                    initial_power = fixed_values.get('power', -10)
                    
                    print(f"  Configuring VNA for reference sweep...")
                    self.vna.setup_frequency_sweep(
                        sweep_start, sweep_stop, sweep_points,
                        fixed_values.get('ifbw', 100),
                        initial_power,
                        num_averages
                    )
                else:
                    # CW mode for field sweeps
                    self.vna.setup_cw_mode(
                        fixed_values.get('frequency', 8e9),
                        fixed_values.get('ifbw', 100),
                        fixed_values.get('power', -10)
                    )
            
            # Take reference measurement based on sweep type
            if sweep_param == "B-Field (T)" and step_param == "Frequency (GHz)":
                # Field sweep with frequency step: references taken per-step
                field_per_step_reference = True
                print(f"  Per-step reference mode: will take CW reference at {field_normalization_field}T for each frequency")
                field_step_reference_db = {}
                
            elif sweep_param == "B-Field (T)":
                # Field sweep (no freq step): take single CW measurement at reference field
                print(f"  Taking CW reference measurement...")
                if self.use_simulation:
                    ref_s21 = self.sim_data.generate_s21_vs_field(
                        np.array([field_normalization_field]),
                        fixed_values.get('frequency', 8e9),
                        fixed_values.get('vg', 0),
                        fixed_values.get('power', -10),
                        300.0
                    )[0]
                else:
                    # Trigger and read single CW point
                    self.vna.trigger_sweep()
                    ref_s21 = self.vna.get_cw_data()
                
                if ref_s21 is not None:
                    field_reference_s21_mag_db = 20 * np.log10(np.abs(ref_s21) + 1e-12)
                    print(f"  Reference S21 at {field_normalization_field}T: {field_reference_s21_mag_db:.2f} dB")
                
            elif sweep_param == "Frequency (GHz)" and step_param == "B-Field (T)":
                # Freq sweep with field step: take full spectrum at reference field
                print(f"  Taking reference spectrum at {field_normalization_field}T...")
                num_averages = config.get('averages', 1)
                ifbw = fixed_values.get('ifbw', 100)
                print(f"  Reference sweep params: {sweep_points} points, IFBW={ifbw}, averages={num_averages}")
                
                if self.use_simulation:
                    ref_spectrum = self.sim_data.generate_s21_vs_frequency(
                        sweep_values,
                        field_normalization_field,
                        fixed_values.get('vg', 0),
                        fixed_values.get('power', -10),
                        300.0
                    )
                else:
                    # Trigger VNA sweep for reference (with full averaging)
                    print(f"  Triggering reference sweep with {num_averages} averages...")
                    if not self.vna.trigger_sweep_timed(sweep_points, ifbw, num_averages):
                        print("  WARNING: VNA trigger for reference sweep returned False")
                    ref_spectrum = self.vna.get_sweep_data(expected_points=sweep_points)
                
                if ref_spectrum is not None and len(ref_spectrum) > 0:
                    field_reference_spectrum_mag_db = 20 * np.log10(np.abs(ref_spectrum) + 1e-12)
                    print(f"  Reference spectrum: {len(field_reference_spectrum_mag_db)} points, "
                          f"range {np.min(field_reference_spectrum_mag_db):.1f} to {np.max(field_reference_spectrum_mag_db):.1f} dB")
                else:
                    print("  WARNING: Failed to get reference spectrum!")
            
            # Send field reference data to GUI for storage/saving
            if not field_per_step_reference:
                self.data_queue.put({
                    'type': 'field_reference_data',
                    'field': field_normalization_field,
                    'single_value_db': field_reference_s21_mag_db,
                    'spectrum_db': field_reference_spectrum_mag_db.tolist() if field_reference_spectrum_mag_db is not None else None,
                    'frequencies': sweep_values.tolist() if field_reference_spectrum_mag_db is not None else None
                })
            
            print(f"=== Field reference measurement complete ===")
        
        # Pre-sweep: Ramp gate voltage to start position (NEVER jump!)
        if gate_involved and (not self.use_simulation or self.keithley.connected):
            if sweep_param == "Gate Voltage (V)":
                start_voltage = sweep_start
            elif step_param == "Gate Voltage (V)":
                start_voltage = step_start
            else:
                start_voltage = fixed_values.get('vg', 0)
            
            # Use get_voltage_safe to handle communication errors gracefully
            current_voltage, reliable = self.keithley.get_voltage_safe()
            if not reliable:
                print("ERROR: Cannot read gate voltage - communication error")
                self.data_queue.put({
                    'type': 'error',
                    'message': 'Cannot read gate voltage - communication error'
                })
                self.gate_safety.abort_measurement()
                self.is_running = False
                return
            
            if abs(current_voltage - start_voltage) > 0.001:
                print(f"=== Pre-sweep: Ramping gate from {current_voltage:.3f}V to {start_voltage:.3f}V ===")
                success = self.keithley.ramp_to_voltage(
                    start_voltage,
                    slew_rate=self.gate_slew_rate,
                    stop_check=self.stop_check
                )
                if not success:
                    print("Gate ramp to start position failed or stopped")
                    if self.should_stop:
                        self.data_queue.put({
                            'type': 'aborted',
                            'all_data': [],
                            'config': config
                        })
                        self.gate_safety.end_measurement()
                        self.is_running = False
                        return
                # Small delay for sample to settle after gate reaches start position
                print("  Waiting for sample to settle at start position...")
                time.sleep(0.5)
        
        try:
            # Final flush right before starting measurements
            # This clears any data that accumulated during gate ramp
            if not self.use_simulation and self.vna.connected:
                print("Final VNA buffer flush before measurement loop...")
                for i in range(2):
                    try:
                        _ = self._get_real_data()
                        time.sleep(0.02)
                    except:
                        pass
            
            for step_idx, step_val in enumerate(step_values):
                if self.should_stop:
                    break
                
                # Set step parameter if applicable
                if is_2d and step_val is not None:
                    # Special handling: if stepping power during freq sweep, reconfigure VNA
                    if use_vna_sweep and step_param == "Power (dBm)":
                        print(f"Step {step_idx}: Power = {step_val} dBm")
                        if step_idx > 0:  # Only reconfigure after first step
                            print(f"  Reconfiguring VNA with power = {step_val} dBm")
                            num_averages = config.get('averages', 1)
                            self.vna.setup_frequency_sweep(
                                sweep_start, sweep_stop, sweep_points,
                                fixed_vaPlueslues.get('ifbw', 100),
                                step_val,  # Use stepped power value
                                num_averages
                            )
                        else:
                            print(f"  Using initial power from setup")
                        # Update fixed_values so it tracks current power
                        fixed_values['power'] = step_val
                    else:
                        # Update fixed_values to track current step value
                        # This is important when stepping frequency and sweeping power
                        if step_param == "Frequency (GHz)":
                            fixed_values['frequency'] = step_val
                            print(f"Step {step_idx}: Setting frequency to {step_val/1e9:.4f} GHz")
                        elif step_param == "Power (dBm)":
                            fixed_values['power'] = step_val
                            print(f"Step {step_idx}: Setting power to {step_val} dBm")
                        elif step_param == "B-Field (T)":
                            fixed_values['b_field'] = step_val
                            print(f"Step {step_idx}: Setting B-field to {step_val:.4f} T")
                        elif step_param == "Gate Voltage (V)":
                            fixed_values['vg'] = step_val
                            print(f"Step {step_idx}: Setting gate to {step_val} V")
                            # For step changes, use safe ramping (can be large jumps)
                            if not self.use_simulation and self.keithley.connected:
                                print(f"  Ramping gate to {step_val}V")
                                self.keithley.ramp_to_voltage(
                                    step_val,
                                    slew_rate=self.gate_slew_rate,
                                    stop_check=self.stop_check
                                )
                        
                        # Set parameter (skip gate voltage - already ramped above)
                        if step_param != "Gate Voltage (V)":
                            self._set_parameter(step_param, step_val, fixed_values, is_step=True)
                    
                    time.sleep(0.1)  # Settling time for step parameter
                
                # Temperature reading DISABLED - Lakeshore causes GPIB conflicts
                sweep_temperature_k = None
                # if self.temp_controller and self.temp_controller.connected:
                #     sweep_temperature_k = self.temp_controller.get_temperature()
                #     if sweep_temperature_k is not None:
                #         fixed_values['temperature'] = sweep_temperature_k
                #         print(f"Temperature: {sweep_temperature_k * 1000:.2f} mK")
                
                sweep_data = []
                
                # === FAST PATH: VNA native frequency sweep with hardware averaging ===
                if use_vna_sweep:
                    num_averages = config.get('averages', 1)
                    ifbw = fixed_values.get('ifbw', 100)
                    
                    if num_averages > 1:
                        print(f"Running VNA sweep {step_idx + 1}/{len(step_values)} with {num_averages}x hardware averaging...")
                    else:
                        print(f"Running VNA sweep {step_idx + 1}/{len(step_values)}...")
                    
                    # Trigger VNA sweep (VNA handles averaging internally)
                    if not self.vna.trigger_sweep_timed(sweep_points, ifbw, num_averages):
                        print("Warning: VNA trigger_sweep_timed returned False")
                    
                    # Get averaged data from VNA
                    s21_array = self.vna.get_sweep_data(expected_points=sweep_points)
                    
                    if s21_array is None:
                        print(f"Warning: VNA returned None")
                        s21_array = np.zeros(sweep_points, dtype=complex)
                    elif len(s21_array) != sweep_points:
                        print(f"Warning: VNA returned {len(s21_array)} points, expected {sweep_points}")
                        # Pad or truncate to expected size
                        if len(s21_array) < sweep_points:
                            s21_array = np.pad(s21_array, (0, sweep_points - len(s21_array)))
                        else:
                            s21_array = s21_array[:sweep_points]
                    
                    print(f"Got {len(s21_array)} points from VNA" + (f" ({num_averages}x averaged)" if num_averages > 1 else ""))
                    
                    # Build sweep_data from averaged result
                    for sweep_idx, (sweep_val, s21) in enumerate(zip(sweep_values, s21_array)):
                        if self.should_stop:
                            break
                        
                        s21_mag_db = 20 * np.log10(np.abs(s21) + 1e-12)
                        
                        # Calculate normalized value if reference spectrum available
                        # Check gate reference first, then field reference
                        s21_mag_db_norm = None
                        if reference_spectrum_mag_db is not None and sweep_idx < len(reference_spectrum_mag_db):
                            s21_mag_db_norm = s21_mag_db - reference_spectrum_mag_db[sweep_idx]
                        elif field_reference_spectrum_mag_db is not None and sweep_idx < len(field_reference_spectrum_mag_db):
                            s21_mag_db_norm = s21_mag_db - field_reference_spectrum_mag_db[sweep_idx]
                        
                        sweep_data.append({
                            'sweep_value': sweep_val,
                            'step_value': step_val,
                            's21_real': np.real(s21),
                            's21_imag': np.imag(s21),
                            's21_mag': np.abs(s21),
                            's21_phase': np.angle(s21, deg=True),
                            's21_mag_db_norm': s21_mag_db_norm  # Normalized to reference (None if no ref)
                        })
                    
                    # Update progress
                    progress = (step_idx + 1) / len(step_values) * 100
                    self.progress_queue.put(progress)
                    
                    # Send batch data to GUI
                    self.data_queue.put({
                        'type': 'batch',
                        'step_idx': step_idx,
                        'sweep_data': sweep_data.copy(),
                        'avg_num': num_averages,
                        'avg_total': num_averages
                    })
                
                # === GATE VOLTAGE SWEEP MODE ===
                # Set voltage, read it back (forces USB sync), take VNA measurement
                elif sweep_param == "Gate Voltage (V)" and not self.use_simulation and self.keithley.connected:
                    import time as time_module
                    
                    # === PER-STEP REFERENCE MEASUREMENT (for gate sweep with frequency step) ===
                    if per_step_reference and step_param == "Frequency (GHz)" and step_val is not None:
                        # Take CW reference at normalization voltage for this frequency
                        print(f"  Taking reference at {normalization_voltage}V for f={step_val/1e9:.4f} GHz...")
                        
                        # Ramp to reference voltage
                        current_v, _ = self.keithley.get_voltage_safe()
                        if abs(current_v - normalization_voltage) > 0.001:
                            self.keithley.ramp_to_voltage(
                                normalization_voltage,
                                slew_rate=self.gate_slew_rate,
                                stop_check=self.stop_check
                            )
                            time.sleep(0.2)  # Settle at reference voltage
                        
                        # Take CW measurement
                        self.vna.trigger_sweep()
                        ref_s21 = self.vna.get_cw_data()
                        
                        if ref_s21 is not None:
                            ref_db = 20 * np.log10(np.abs(ref_s21) + 1e-12)
                            step_reference_db[step_idx] = ref_db
                            print(f"    Reference S21: {ref_db:.2f} dB")
                        else:
                            print(f"    WARNING: Failed to get reference for step {step_idx}")
                            step_reference_db[step_idx] = None
                    
                    # Get number of averages
                    num_averages = config.get('averages', 1)
                    print(f"Gate voltage sweep: {sweep_start}V -> {sweep_stop}V, {sweep_points} points, {num_averages} average(s)")
                    
                    # Clear any errors and make sure we're in simple DC mode
                    self.keithley.instrument.write('*CLS')
                    self.keithley.instrument.write(':ABOR')
                    time.sleep(0.05)
                    
                    # Disable automatic measurements that slow down voltage changes
                    self.keithley.instrument.write(':OUTP OFF')
                    time.sleep(0.05)
                    self.keithley.instrument.write(':SENS:FUNC "CURR"')
                    self.keithley.instrument.write(':SENS:CURR:NPLC 0.01')
                    self.keithley.instrument.write(':SENS:CURR:RANG:AUTO OFF')  # Disable current auto-range
                    self.keithley.instrument.write(':SENS:CURR:RANG 1e-6')  # Fixed 1uA range
                    
                    # Model-specific commands
                    if self.keithley.model == '2450':
                        self.keithley.instrument.write(':SOUR:DEL 0')  # 2450: source delay
                        self.keithley.instrument.write(':SYST:AZER:STAT OFF')  # 2450: autozero
                        # 2450 doesn't have :DISP:ENAB, skip it
                    else:
                        self.keithley.instrument.write(':SOUR:VOLT:DEL 0')  # 2400: source delay
                        self.keithley.instrument.write(':SYST:AZER OFF')  # 2400: autozero
                        self.keithley.instrument.write(':DISP:ENAB OFF')  # 2400: display
                    
                    # Ensure output is on
                    self.keithley.instrument.write(':OUTP ON')
                    time.sleep(0.1)
                    
                    # Initialize accumulator arrays for averaging
                    s21_real_acc = np.zeros(len(sweep_values))
                    s21_imag_acc = np.zeros(len(sweep_values))
                    
                    t_total_start = time_module.perf_counter()
                    
                    # Loop through averages
                    for avg_idx in range(num_averages):
                        if self.should_stop:
                            break
                        
                        # Ramp to start voltage
                        self.keithley.ramp_to_voltage(sweep_start, slew_rate=self.gate_slew_rate)
                        time.sleep(0.05)
                        
                        # Flush VNA buffer before each sweep
                        if self.vna.connected:
                            self.vna.write(":INIT:IMM")
                            time.sleep(0.03)
                            _ = self.vna.get_cw_data()  # Discard stale data
                        
                        t_start = time_module.perf_counter()
                        
                        if num_averages > 1:
                            print(f"  Sweep {avg_idx + 1}/{num_averages}...", end='', flush=True)
                        
                        for sweep_idx, target_voltage in enumerate(sweep_values):
                            if self.should_stop:
                                break
                            
                            t_loop_start = time_module.perf_counter()
                            
                            current_point += 1
                            # Progress accounts for all averages
                            progress = (avg_idx * len(sweep_values) + sweep_idx + 1) / (num_averages * len(sweep_values)) * 100
                            self.progress_queue.put(progress)
                            
                            t0 = time_module.perf_counter()
                            
                            # Set voltage
                            cmd = f':SOUR:VOLT {target_voltage}\n'
                            self.keithley.instrument.write_raw(cmd.encode())
                            t1 = time_module.perf_counter()
                            self.keithley.current_voltage = target_voltage
                            
                            # Take VNA measurement
                            s21 = self._get_real_data()
                            t2 = time_module.perf_counter()
                            
                            # Store current raw measurement
                            current_raw = {
                                'sweep_value': target_voltage,
                                'step_value': step_val,
                                's21_real': np.real(s21),
                                's21_imag': np.imag(s21),
                                's21_mag': np.abs(s21),
                                's21_phase': np.angle(s21, deg=True)
                            }
                            
                            # Accumulate for averaging
                            s21_real_acc[sweep_idx] += np.real(s21)
                            s21_imag_acc[sweep_idx] += np.imag(s21)
                            
                            # Calculate running average (including current sweep)
                            n_completed = avg_idx + 1
                            avg_real = s21_real_acc[sweep_idx] / n_completed
                            avg_imag = s21_imag_acc[sweep_idx] / n_completed
                            avg_s21 = avg_real + 1j * avg_imag
                            
                            # Previous average (excluding current sweep) - for display
                            if avg_idx > 0:
                                prev_avg_real = (s21_real_acc[sweep_idx] - np.real(s21)) / avg_idx
                                prev_avg_imag = (s21_imag_acc[sweep_idx] - np.imag(s21)) / avg_idx
                                prev_avg_s21 = prev_avg_real + 1j * prev_avg_imag
                                prev_avg_data = {
                                    'sweep_value': target_voltage,
                                    's21_real': prev_avg_real,
                                    's21_imag': prev_avg_imag,
                                    's21_mag': np.abs(prev_avg_s21),
                                    's21_phase': np.angle(prev_avg_s21, deg=True)
                                }
                            else:
                                prev_avg_data = None
                            
                            # Update sweep_data with current average (final result)
                            # Calculate normalized value if reference available
                            avg_s21_mag_db = 20 * np.log10(np.abs(avg_s21) + 1e-12)
                            s21_mag_db_norm = None
                            if reference_s21_mag_db is not None:
                                s21_mag_db_norm = avg_s21_mag_db - reference_s21_mag_db
                            elif per_step_reference and step_idx in step_reference_db and step_reference_db[step_idx] is not None:
                                # Use per-step reference for gate sweep with frequency step
                                s21_mag_db_norm = avg_s21_mag_db - step_reference_db[step_idx]
                            
                            if avg_idx == 0:
                                # First sweep - append new data
                                sweep_data.append({
                                    'sweep_value': target_voltage,
                                    'step_value': step_val,
                                    's21_real': avg_real,
                                    's21_imag': avg_imag,
                                    's21_mag': np.abs(avg_s21),
                                    's21_phase': np.angle(avg_s21, deg=True),
                                    's21_mag_db_norm': s21_mag_db_norm
                                })
                            else:
                                # Subsequent sweeps - update existing data
                                sweep_data[sweep_idx] = {
                                    'sweep_value': target_voltage,
                                    'step_value': step_val,
                                    's21_real': avg_real,
                                    's21_imag': avg_imag,
                                    's21_mag': np.abs(avg_s21),
                                    's21_phase': np.angle(avg_s21, deg=True),
                                    's21_mag_db_norm': s21_mag_db_norm
                                }
                            
                            # Send data point to GUI
                            self.data_queue.put({
                                'type': 'point',
                                'sweep_idx': sweep_idx,
                                'step_idx': step_idx,
                                'data': sweep_data[sweep_idx],  # Running average
                                'current_raw': current_raw,      # Current sweep raw data
                                'prev_avg': prev_avg_data,       # Previous sweeps average (None for first sweep)
                                'avg_num': n_completed,
                                'avg_total': num_averages
                            })
                            t4 = time_module.perf_counter()
                            
                            # Print timing for first point of first sweep only
                            if sweep_idx == 0 and avg_idx == 0:
                                print(f"  Point timing: write={1000*(t1-t0):.0f}ms, VNA={1000*(t2-t1):.0f}ms, total={1000*(t4-t_loop_start):.0f}ms")
                        
                        elapsed = time_module.perf_counter() - t_start
                        if num_averages > 1:
                            print(f" {elapsed:.1f}s")
                    
                    t_total = time_module.perf_counter() - t_total_start
                    if num_averages > 1:
                        print(f"Averaging complete: {num_averages} sweeps in {t_total:.1f}s ({t_total/num_averages:.1f}s/sweep)")
                    else:
                        print(f"Gate sweep complete: {len(sweep_values)} points in {t_total:.1f}s ({1000*t_total/len(sweep_values):.0f}ms/point)")
                
                # === B-FIELD CONTINUOUS SWEEP MODE ===
                # Ramp magnet to target while continuously measuring VNA
                elif sweep_param == "B-Field (T)" and not self.use_simulation and self.magnet.connected:
                    import time as time_module
                    
                    field_start = sweep_start
                    field_stop = sweep_stop
                    
                    # Reset debug counter for this sweep
                    if hasattr(self.magnet, '_debug_count'):
                        self.magnet._debug_count = 0
                    
                    print(f"B-field continuous sweep: {field_start:.3f}T -> {field_stop:.3f}T")
                    print(f"  Ramp rate: {self.field_ramp_rate} T/min, Tolerance: {self.field_tolerance} T")
                    
                    # Get current field
                    current_field = self.magnet.get_field()
                    print(f"  Current field: {current_field:.4f} T")
                    
                    # First, ramp to start field if needed
                    if abs(current_field - field_start) > self.field_tolerance:
                        print(f"  Ramping to start field {field_start:.3f}T...")
                        if hasattr(self.magnet, 'set_rate'):
                            self.magnet.set_rate(self.field_ramp_rate)
                        self.magnet.set_field(field_start)
                        
                        # Wait for start field
                        while not self.should_stop:
                            current_field = self.magnet.get_field()
                            if abs(current_field - field_start) <= self.field_tolerance:
                                break
                            time.sleep(0.5)
                        
                        if self.should_stop:
                            # Stop the ramp on abort
                            if hasattr(self.magnet, 'stop_ramp'):
                                self.magnet.stop_ramp()
                            break
                        
                        # Settle at start
                        if self.field_settle_time > 0:
                            print(f"  Settling for {self.field_settle_time}s...")
                            time.sleep(self.field_settle_time)
                    
                    if self.should_stop:
                        if hasattr(self.magnet, 'stop_ramp'):
                            self.magnet.stop_ramp()
                        break
                    
                    # Calculate appropriate ramp rate based on VNA measurement timing
                    # For CW mode (B-field sweep), measurement is very fast
                    ifbw = fixed_values.get('ifbw', 100)
                    num_freq_points = 1  # CW mode for B-field sweeps
                    
                    # VNA measurement time: 3 time constants + minimal overhead for CW mode
                    # For full sweeps we'd use more overhead, but CW is just one point
                    vna_measurement_time = (num_freq_points / ifbw) * 3.0 + 0.05  # 3 time constants + 50ms overhead
                    
                    # Total time per B-field point = VNA measurement + settling
                    time_per_point = vna_measurement_time + self.field_settle_time
                    
                    # Calculate field step size
                    field_range = abs(field_stop - field_start)
                    field_step = field_range / max(sweep_points - 1, 1)  # Step size between points
                    
                    # Calculate required ramp rate (T/min)
                    # rate = field_step / time_per_point (T/s) × 60 (s/min)
                    calculated_ramp_rate = (field_step / time_per_point) * 60.0
                    
                    # Use the ACTUAL confirmed rate from hardware if software rate isn't working
                    # This is a workaround for hardware rate dial override
                    max_safe_rate = 0.5  # T/min - hardware limit
                    actual_ramp_rate = min(calculated_ramp_rate, max_safe_rate)
                    
                    expected_sweep_time = (field_range / actual_ramp_rate) * 60  # seconds
                    
                    print(f"  VNA measurement time: {vna_measurement_time:.3f}s (IFBW={ifbw} Hz, CW mode)")
                    print(f"  Field settle time: {self.field_settle_time:.3f}s")
                    print(f"  Time per point: {time_per_point:.3f}s")
                    print(f"  Field step: {field_step*1000:.3f} mT")
                    print(f"  Calculated ramp rate: {actual_ramp_rate:.4f} T/min")
                    print(f"  Expected sweep time: {expected_sweep_time:.1f}s for {sweep_points} points")
                    
                    # Set target and start ramping with calculated rate
                    # For continuous sweep, we need explicit control over the sequence
                    # to ensure rate is properly set before sweeping
                    
                    # 1. Pause any current sweep
                    if hasattr(self.magnet, 'pause'):
                        self.magnet.pause()
                        time.sleep(0.2)
                    
                    # 2. Set ramp rate
                    actual_hardware_rate = None
                    if hasattr(self.magnet, 'set_rate'):
                        self.magnet.set_rate(actual_ramp_rate)
                        time.sleep(0.2)
                        
                        # Check if hardware accepted the rate
                        try:
                            if hasattr(self.magnet, 'query'):
                                response = self.magnet.query("RATE? 0")
                                if response:
                                    reported_A_s = float(response.strip())
                                    actual_hardware_rate = reported_A_s * self.magnet.field_per_amp * 60.0  # Convert to T/min
                                    
                                    # If hardware rate is different, warn and recalculate timing
                                    if abs(reported_A_s - (actual_ramp_rate / 60.0 / self.magnet.field_per_amp)) > 0.001:
                                        print(f"")
                                        print(f"  ⚠️  ⚠️  ⚠️  WARNING: Rate not set correctly! ⚠️  ⚠️  ⚠️")
                                        print(f"")
                                        print(f"  Software requested: {actual_ramp_rate:.3f} T/min")
                                        print(f"  Controller reports: {actual_hardware_rate:.3f} T/min")
                                        print(f"")
                                        print(f"  NOTE: This should not happen - semicolon workaround is active")
                                        print(f"  Check if controller firmware requires different command format")
                                        print(f"")
                                        print(f"  Proceeding with controller's rate - timing adjusted...")
                                        print(f"")
                                        
                                        # Recalculate timing based on hardware rate
                                        expected_sweep_time = (field_range / actual_hardware_rate) * 60
                                        time_per_point = expected_sweep_time / sweep_points
                                        print(f"      Adjusted time per point: {time_per_point:.3f}s")
                                        print(f"      Adjusted sweep time: {expected_sweep_time:.1f}s")
                        except:
                            pass
                    
                    # 3. Set the limit (don't start sweep yet)
                    # NOTE: Semicolon required due to firmware bug in some versions
                    if hasattr(self.magnet, 'write'):
                        target_kG = field_stop * 10.0
                        if field_stop > current_field:
                            self.magnet.write(f"ULIM {target_kG:.3f};")
                            sweep_cmd = "SWEEP UP"
                        else:
                            self.magnet.write(f"LLIM {target_kG:.3f};")
                            sweep_cmd = "SWEEP DOWN"
                        time.sleep(0.2)
                    
                    # 4. Start the sweep
                    if hasattr(self.magnet, 'write'):
                        self.magnet.write(sweep_cmd)
                        print(f"Cryomagnetics: Starting continuous sweep to {field_stop:.4f} T at {actual_ramp_rate:.4f} T/min")
                    
                    # Wait for initial settling after ramp starts
                    if self.field_settle_time > 0:
                        print(f"  Waiting {self.field_settle_time:.1f}s for initial field settling...")
                        time.sleep(self.field_settle_time)
                    
                    print(f"  Starting continuous measurement...")
                    print(f"  Magnet will ramp continuously while VNA samples at {sweep_points} time intervals")
                    t_start = time_module.perf_counter()
                    
                    # Measure at regular time intervals
                    for sweep_idx in range(sweep_points):
                        if self.should_stop:
                            break
                        
                        # Wait until it's time for this measurement
                        # First point is immediate (already settled above), subsequent points wait time_per_point
                        if sweep_idx > 0:
                            target_time = t_start + sweep_idx * time_per_point
                            current_time = time_module.perf_counter()
                            wait_time = target_time - current_time
                            if wait_time > 0:
                                time.sleep(wait_time)
                        
                        # Read current field (wherever the magnet is at this time)
                        # Don't use debug mode during sweep - reduces GPIB traffic
                        # Retry on failure since controller may be busy
                        actual_field = None
                        for retry in range(3):
                            try:
                                actual_field = self.magnet.get_field(debug=False)
                                if actual_field is not None:
                                    break
                            except:
                                if retry < 2:
                                    time.sleep(0.5)
                        
                        if actual_field is None:
                            print(f"  WARNING: Could not read field at point {sweep_idx+1}, skipping")
                            continue
                        
                        # Take VNA measurement
                        s21 = self._get_real_data()
                        
                        # Calculate normalized value if field reference available
                        s21_mag_db = 20 * np.log10(np.abs(s21) + 1e-12)
                        s21_mag_db_norm = None
                        if field_reference_s21_mag_db is not None:
                            s21_mag_db_norm = s21_mag_db - field_reference_s21_mag_db
                        
                        # Record data with ACTUAL field value
                        sweep_data.append({
                            'sweep_value': actual_field,
                            'step_value': step_val,
                            's21_real': np.real(s21),
                            's21_imag': np.imag(s21),
                            's21_mag': np.abs(s21),
                            's21_phase': np.angle(s21, deg=True),
                            's21_mag_db_norm': s21_mag_db_norm
                        })
                        
                        # Send data to GUI
                        self.data_queue.put({
                            'type': 'point',
                            'sweep_idx': sweep_idx,
                            'step_idx': step_idx,
                            'data': sweep_data[-1]
                        })
                        
                        # Update progress
                        progress = (step_idx * sweep_points + sweep_idx + 1) / (len(step_values) * sweep_points) * 100
                        self.progress_queue.put(min(progress, 100))
                        
                        # Calculate expected field based on timing
                        expected_field = field_start + (sweep_idx / (sweep_points - 1)) * field_range if sweep_points > 1 else field_start
                        field_error = abs(actual_field - expected_field)
                        
                        # Only print warning if field error is large (>2 steps)
                        if field_error > 2 * field_step:
                            print(f"  Point {sweep_idx+1}/{sweep_points}: Field = {actual_field:.4f} T (expected {expected_field:.4f} T, error {field_error*1000:.1f} mT) ⚠️")
                        else:
                            print(f"  Point {sweep_idx+1}/{sweep_points}: Field = {actual_field:.4f} T")
                    
                    
                    # Handle abort - stop the magnet ramp
                    if self.should_stop:
                        print(f"  B-field sweep ABORTED")
                        if hasattr(self.magnet, 'stop_ramp'):
                            self.magnet.stop_ramp()
                    
                    t_total = time_module.perf_counter() - t_start
                    num_recorded = len(sweep_data)
                    print(f"  B-field sweep complete: {num_recorded} points in {t_total:.1f}s")
                    if num_recorded > 0:
                        print(f"  Field: {sweep_data[0]['sweep_value']:.4f}T -> {sweep_data[-1]['sweep_value']:.4f}T")
                
                # === SLOW PATH: Point-by-point measurement ===
                else:
                    # === PER-STEP REFERENCE FOR SIMULATION (gate sweep with frequency step) ===
                    if per_step_reference and self.use_simulation and step_param == "Frequency (GHz)" and step_val is not None:
                        if sweep_param == "Gate Voltage (V)":
                            # Take simulated reference at normalization voltage for this frequency
                            ref_s21 = self.sim_data.generate_s21_vs_gate(
                                np.array([normalization_voltage]),
                                step_val,  # Current frequency
                                fixed_values.get('b_field', 0),
                                fixed_values.get('power', -10),
                                300.0
                            )[0]
                            if ref_s21 is not None:
                                ref_db = 20 * np.log10(np.abs(ref_s21) + 1e-12)
                                step_reference_db[step_idx] = ref_db
                                print(f"  [SIM] Reference at {normalization_voltage}V for f={step_val/1e9:.4f} GHz: {ref_db:.2f} dB")
                    
                    for sweep_idx, sweep_val in enumerate(sweep_values):
                        if self.should_stop:
                            break
                        
                        current_point += 1
                        progress = current_point / total_points * 100
                        self.progress_queue.put(progress)
                        
                        # Set sweep parameter
                        self._set_parameter(sweep_param, sweep_val, fixed_values)
                        
                        # Get measurement
                        if self.use_simulation:
                            s21 = self._get_simulated_data(
                                sweep_param, sweep_val, step_param, step_val, fixed_values
                            )
                        else:
                            s21 = self._get_real_data()
                        
                        # Calculate normalized value if reference available
                        s21_mag_db = 20 * np.log10(np.abs(s21) + 1e-12)
                        s21_mag_db_norm = None
                        if reference_s21_mag_db is not None:
                            s21_mag_db_norm = s21_mag_db - reference_s21_mag_db
                        elif reference_spectrum_mag_db is not None and sweep_idx < len(reference_spectrum_mag_db):
                            s21_mag_db_norm = s21_mag_db - reference_spectrum_mag_db[sweep_idx]
                        elif per_step_reference and step_idx in step_reference_db and step_reference_db[step_idx] is not None:
                            # Use per-step reference for gate sweep with frequency step
                            s21_mag_db_norm = s21_mag_db - step_reference_db[step_idx]
                        
                        sweep_data.append({
                            'sweep_value': sweep_val,
                            'step_value': step_val,
                            's21_real': np.real(s21),
                            's21_imag': np.imag(s21),
                            's21_mag': np.abs(s21),
                            's21_phase': np.angle(s21, deg=True),
                            's21_mag_db_norm': s21_mag_db_norm
                        })
                        
                        # Send data point to GUI
                        self.data_queue.put({
                            'type': 'point',
                            'sweep_idx': sweep_idx,
                            'step_idx': step_idx,
                            'data': sweep_data[-1]
                        })
                        
                        # Small delay for GUI responsiveness
                        time.sleep(0.02)
                
                # Signal that this step (sweep line) is complete - include ALL data
                self.data_queue.put({
                    'type': 'step_complete',
                    'step_idx': step_idx,
                    'step_value': step_val,
                    'sweep_data': sweep_data  # Full sweep data array
                })
                
                all_data.append(sweep_data)
            
            # Send per-step reference data if collected
            if per_step_reference and step_reference_db:
                self.data_queue.put({
                    'type': 'reference_data',
                    'voltage': normalization_voltage,
                    'single_value_db': None,
                    'spectrum_db': None,
                    'frequencies': None,
                    'per_step_db': step_reference_db,
                    'step_frequencies': [step_values[i] for i in sorted(step_reference_db.keys())]
                })
            
            # Signal completion or abort
            if self.should_stop:
                self.data_queue.put({
                    'type': 'aborted',
                    'all_data': all_data,
                    'config': config
                })
            else:
                self.data_queue.put({
                    'type': 'complete',
                    'all_data': all_data,
                    'config': config
                })
            
        except Exception as e:
            # CRITICAL: Handle any exception by attempting to safe the gate
            print(f"MEASUREMENT ERROR: {e}")
            self.data_queue.put({
                'type': 'error',
                'message': str(e)
            })
            
            # Try to safe the gate on any error
            if gate_involved:
                print("Attempting to safe gate voltage after error...")
                self.gate_safety.abort_measurement()
        
        finally:
            # Post-sweep: Handle gate voltage
            if gate_involved:
                self.gate_safety.end_measurement()
                
                try:
                    if self.should_stop and self.gate_ramp_on_stop:
                        # User stopped - ramp to zero if enabled
                        print("=== Measurement stopped - ramping gate to zero ===")
                        self.keithley.ramp_to_voltage(0.0, slew_rate=self.gate_slew_rate)
                    elif not self.should_stop and self.gate_ramp_to_zero_after:
                        # Normal completion - ramp to zero if enabled
                        print("=== Post-sweep: Ramping gate voltage to zero ===")
                        self.keithley.ramp_to_voltage(0.0, slew_rate=self.gate_slew_rate)
                    else:
                        v, reliable = self.keithley.get_voltage_safe()
                        if reliable:
                            print(f"Gate voltage remains at {v:.3f}V")
                        else:
                            print("WARNING: Could not read gate voltage")
                except Exception as e:
                    print(f"ERROR in post-sweep gate handling: {e}")
                    print("Attempting emergency shutdown...")
                    self.keithley.emergency_shutdown()
            
            self.is_running = False
    
    def _set_parameter(self, param, value, fixed_values, sweep_config=None, is_step=False, averages=1):
        """Set a parameter value (real or simulated).
        
        Args:
            param: Parameter name to set
            value: Value to set
            fixed_values: Dictionary of fixed parameter values
            sweep_config: Optional dict with sweep_param, sweep_start, sweep_stop, sweep_points
                         Used to reconfigure VNA sweep when stepping power during freq sweep
            is_step: If True, this is a step parameter change (may need to wait for stabilization)
            averages: Number of hardware averages for VNA (default 1)
        """
        if param == "Frequency (GHz)":
            if not self.use_simulation and self.vna.connected:
                print(f"Setting frequency step to {value/1e9:.4f} GHz")
                self.vna.setup_cw_mode(value, fixed_values['ifbw'], fixed_values['power'])
                # Trigger TWO dummy sweeps to ensure old data is flushed
                self.vna.trigger_sweep()
                self.vna.trigger_sweep()
        elif param == "B-Field (T)":
            if not self.use_simulation and self.magnet.connected:
                # Set the ramp rate first (if the magnet supports it)
                if hasattr(self.magnet, 'set_rate'):
                    self.magnet.set_rate(self.field_ramp_rate)
                
                print(f"Setting B-field to {value:.4f} T")
                self.magnet.set_field(value)
                
                # If this is a step parameter and waiting is enabled, wait for field
                if is_step and self.wait_for_field:
                    print(f"Waiting for field to reach {value:.4f} T (tolerance: {self.field_tolerance} T)...")
                    
                    # Wait for field with stop checking
                    start_time = time.time()
                    timeout = 600  # 10 minute timeout
                    last_print_time = 0
                    
                    while time.time() - start_time < timeout:
                        if self.should_stop:
                            print("Field wait interrupted by stop request")
                            return
                        
                        current_field = self.magnet.get_field()
                        
                        # Print progress every 2 seconds
                        elapsed = time.time() - start_time
                        if elapsed - last_print_time >= 2.0:
                            print(f"  Field: {current_field:.4f} T (target: {value:.4f} T, diff: {abs(current_field - value):.4f} T)")
                            last_print_time = elapsed
                        
                        if abs(current_field - value) <= self.field_tolerance:
                            # Field reached - wait for settle time
                            if self.field_settle_time > 0:
                                print(f"Field at {current_field:.4f} T, settling for {self.field_settle_time}s...")
                                time.sleep(self.field_settle_time)
                            print(f"Field stabilized at {current_field:.4f} T")
                            break
                        
                        time.sleep(0.5)  # Check every 0.5s
                    else:
                        print(f"WARNING: Timeout waiting for field {value:.4f} T (current: {current_field:.4f} T)")
                        
        elif param == "Gate Voltage (V)":
            if not self.use_simulation and self.keithley.connected:
                # Always use ramped voltage change - respects slew rate
                # Even small steps should be rate-limited to protect sample
                self.keithley.ramp_to_voltage(
                    value,
                    slew_rate=self.gate_slew_rate,
                    stop_check=self.stop_check
                )
            elif self.use_simulation:
                # In simulation, just update the internal voltage tracking
                self.keithley.current_voltage = value
        elif param == "Power (dBm)":
            if not self.use_simulation and self.vna.connected:
                # Check if we're doing a frequency sweep (need to reconfigure full sweep)
                if sweep_config and sweep_config.get('sweep_param') == "Frequency (GHz)":
                    self.vna.setup_frequency_sweep(
                        sweep_config['sweep_start'],
                        sweep_config['sweep_stop'],
                        sweep_config['sweep_points'],
                        fixed_values.get('ifbw', 100),
                        value,  # New power
                        averages
                    )
                else:
                    # CW mode - update power and flush
                    freq = fixed_values.get('frequency', 8e9)
                    ifbw = fixed_values.get('ifbw', 100)
                    self.vna.setup_cw_mode(freq, ifbw, value)
                    # Trigger a dummy sweep to flush old data at new power
                    self.vna.trigger_sweep()
        # Note: Temperature is read-only (Lakeshore 370) - no set_parameter for temp
    
    def _get_simulated_data(self, sweep_param, sweep_val, step_param, step_val, fixed):
        """Generate simulated measurement data."""
        # Build parameter dict
        params = {
            'frequency': fixed.get('frequency', 8e9),
            'b_field': fixed.get('b_field', 0.0),
            'vg': fixed.get('vg', 0.0),
            'power': fixed.get('power', -10),
            'temperature': fixed.get('temperature', 50e-3) if fixed.get('temperature') else 50e-3  # Default 50 mK
        }
        
        # Override with sweep value
        if sweep_param == "Frequency (GHz)":
            params['frequency'] = sweep_val
        elif sweep_param == "B-Field (T)":
            params['b_field'] = sweep_val
        elif sweep_param == "Gate Voltage (V)":
            params['vg'] = sweep_val
        elif sweep_param == "Power (dBm)":
            params['power'] = sweep_val
        
        # Override with step value
        if step_param and step_val is not None:
            if step_param == "Frequency (GHz)":
                params['frequency'] = step_val
            elif step_param == "B-Field (T)":
                params['b_field'] = step_val
            elif step_param == "Gate Voltage (V)":
                params['vg'] = step_val
            elif step_param == "Power (dBm)":
                params['power'] = step_val
        
        # Generate data based on sweep parameter
        if sweep_param == "Frequency (GHz)":
            return self.sim_data.generate_s21_vs_frequency(
                np.array([params['frequency']]),
                params['b_field'], params['vg'], params['power'], params['temperature']
            )[0]
        elif sweep_param == "B-Field (T)":
            return self.sim_data.generate_s21_vs_field(
                np.array([params['b_field']]),
                params['frequency'], params['vg'], params['power'], params['temperature']
            )[0]
        elif sweep_param == "Gate Voltage (V)":
            return self.sim_data.generate_s21_vs_gate(
                np.array([params['vg']]),
                params['frequency'], params['b_field'], params['power'], params['temperature']
            )[0]
        elif sweep_param == "Power (dBm)":
            return self.sim_data.generate_s21_vs_power(
                np.array([params['power']]),
                params['frequency'], params['b_field'], params['vg'], params['temperature']
            )[0]
        
        return 0.5 + 0j
    
    def _get_real_data(self):
        """Get real measurement from VNA."""
        if not self.vna.connected:
            print("Warning: VNA not connected, returning zero")
            return 0.0 + 0.0j
        
        # Trigger measurement and get data
        data = self.vna.measure_single_point()
        
        if data is None:
            print("Warning: VNA measurement failed, returning zero")
            return 0.0 + 0.0j
        
        return data


class ToolTip:
    """Simple tooltip class for tkinter widgets."""
    
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)
    
    def show_tip(self, event=None):
        if self.tip_window:
            return
        x, y, _, _ = self.widget.bbox("insert") if hasattr(self.widget, 'bbox') else (0, 0, 0, 0)
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        
        label = tk.Label(tw, text=self.text, justify='left',
                        background="#ffffe0", relief='solid', borderwidth=1,
                        font=("Arial", 9))
        label.pack()
    
    def hide_tip(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


class VNAMeasurementApp:
    """Main GUI application for VNA-based FMR measurements."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("VNA FMR Measurement System - Villanova University")
        self.root.geometry("1400x900")
        
        # Maximize window on startup (platform-specific)
        try:
            self.root.state('zoomed')  # Windows
        except:
            try:
                self.root.attributes('-zoomed', True)  # Linux
            except:
                pass  # Fall back to default geometry
        
        # Settings file path (in same directory as script)
        self.settings_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.vna_fmr_settings.json')
        
        # Initialize instruments (simulation mode by default)
        # Each controller manages its own VISA ResourceManager to avoid
        # termination character conflicts between devices
        self.vna = VNAController()
        self.magnet = MagnetController()
        self.keithley = KeithleyController()
        # Temperature controller DISABLED - causes GPIB bus conflicts with Keithley
        self.temp_controller = None  # Lakeshore370Controller()
        self.measurement_engine = MeasurementEngine(
            self.vna, self.magnet, self.keithley, self.temp_controller, use_simulation=False
        )
        
        # Data storage
        self.sweep_data_1d = []
        self.sweep_data_2d = []
        self.current_config = None
        self.current_step_index = 0
        
        # Averaging display data
        self.current_sweep_raw = []
        self.prev_avg_data = []
        self.current_avg_num = 1
        self.current_avg_total = 1
        
        # Gate normalization reference data
        self.reference_data = None  # Will hold reference S21 (single value or spectrum)
        self.reference_voltage = None  # The voltage at which reference was taken
        
        # Field normalization reference data
        self.field_reference_data = None  # Will hold reference S21 (single value or spectrum)
        self.reference_field = None  # The field at which reference was taken
        
        # Temperature monitoring
        self.current_temperature_k = None  # Last read temperature from Lakeshore 370
        
        # Sample parameters for density/filling factor
        self.sample_density_per_volt = None
        
        # GUI variables
        self.setup_variables()
        
        # Build interface
        self.build_interface()
        
        # Load saved settings
        self.load_settings()
        
        # Set up window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start update loop
        self.update_gui()
    
    def setup_variables(self):
        """Initialize all GUI variables."""
        # Instrument connection
        self.simulation_mode = tk.BooleanVar(value=False)
        self.vna_connected = tk.BooleanVar(value=False)
        self.magnet_connected = tk.BooleanVar(value=False)
        self.keithley_connected = tk.BooleanVar(value=False)
        self.temp_connected = tk.BooleanVar(value=False)
        
        self.vna_port = tk.StringVar(value="5025")
        self.magnet_addr = tk.StringVar(value="scm1datapc.ad.magnet.fsu.edu")
        self.magnet_model = tk.StringVar(value="SCM1")
        self.keithley_addr = tk.StringVar(value="18")
        self.keithley_model = tk.StringVar(value="2450")
        self.temp_gpib_address = tk.StringVar(value="19")
        self.temp_channel = tk.StringVar(value="4")
        
        # Sweep parameters
        self.sweep_param = tk.StringVar(value="Frequency (GHz)")
        self.sweep_start = tk.StringVar(value="0.0001")
        self.sweep_stop = tk.StringVar(value="18")
        self.sweep_points = tk.StringVar(value="1001")
        self.sweep_averages = tk.StringVar(value="1")  # Number of sweeps to average
        
        # Step parameters
        self.step_param = tk.StringVar(value="None")
        self.step_start = tk.StringVar(value="0")
        self.step_stop = tk.StringVar(value="1")
        self.step_points = tk.StringVar(value="11")
        
        # Fixed parameters
        self.fixed_frequency = tk.StringVar(value="8")
        self.fixed_field = tk.StringVar(value="0")
        self.fixed_gate = tk.StringVar(value="0")
        self.fixed_power = tk.StringVar(value="-50")
        self.ifbw = tk.StringVar(value="100")
        self.vna_settle_time = tk.StringVar(value="5")  # Delay before first measurement
        self.input_attenuation = tk.StringVar(value="6")  # dB attenuation before probe
        self.output_attenuation = tk.StringVar(value="6")  # dB attenuation after probe
        
        # S-parameter selection
        self.s_parameter = tk.StringVar(value="S21")
        
        # Gate voltage normalization
        self.gate_normalization_enabled = tk.BooleanVar(value=True)
        self.gate_normalization_voltage = tk.StringVar(value="0")
        
        # B-field normalization
        self.field_normalization_enabled = tk.BooleanVar(value=False)
        self.field_normalization_field = tk.StringVar(value="0")
        
        # Sample parameters (for density/filling factor)
        self.hbn_thickness = tk.StringVar(value="80")
        self.v_cnp = tk.StringVar(value="0")
        
        # CPW geometry for conductivity calculation
        self.cpw_slot_width = tk.StringVar(value="10")  # um
        self.cpw_slot_length = tk.StringVar(value="100")  # um
        
        # Display options
        self.display_mode = tk.StringVar(value="Mag/Phase")
        self.trace_display_mode = tk.StringVar(value="Magnitude")
        self.contour_mode = tk.StringVar(value="Magnitude")
        
        # File settings
        self.data_directory = tk.StringVar(value=os.path.expanduser("~/Desktop/fmr_data"))
        self.filename = tk.StringVar(value="fmr_data")  # Base name without extension
        self.auto_save = tk.BooleanVar(value=True)
        self.current_2d_folder = None  # Track current 2D measurement folder
        
        # Progress
        self.progress_var = tk.DoubleVar(value=0)
        self.status_var = tk.StringVar(value="Ready")
        
        # Parameter options (5 parameters now)
        self.param_options = ["Frequency (GHz)", "B-Field (T)", "Gate Voltage (V)", "Power (dBm)"]
    
    def build_interface(self):
        """Build the main GUI interface."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create tabs
        setup_frame = ttk.Frame(self.notebook)
        sweep_frame = ttk.Frame(self.notebook)
        plot_frame = ttk.Frame(self.notebook)
        
        self.notebook.add(setup_frame, text="Instrument Setup")
        self.notebook.add(sweep_frame, text="Measurement Control")
        self.notebook.add(plot_frame, text="Data Visualization")
        
        self.create_setup_tab(setup_frame)
        self.create_sweep_tab(sweep_frame)
        self.create_plot_tab(plot_frame)
        
        # Status bar at bottom
        self.create_status_bar()
    
    def create_setup_tab(self, parent):
        """Create instrument setup interface."""
        # Simulation mode toggle
        sim_frame = ttk.LabelFrame(parent, text="Operation Mode")
        sim_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Checkbutton(
            sim_frame, text="Simulation Mode (use fake data for GUI testing)",
            variable=self.simulation_mode, command=self.toggle_simulation
        ).pack(pady=5, padx=10, anchor='w')
        
        ttk.Label(
            sim_frame, 
            text="[!] When simulation mode is OFF, ensure all instruments are connected before running measurements",
            foreground='orange'
        ).pack(pady=2, padx=10, anchor='w')
        
        # Instrument connections
        conn_frame = ttk.LabelFrame(parent, text="Instrument Connections")
        conn_frame.pack(fill='x', padx=10, pady=5)
        
        # Headers
        headers = ["Instrument", "Address/Port", "Status", "Action", ""]
        for col, header in enumerate(headers):
            ttk.Label(conn_frame, text=header, font=('Arial', 9, 'bold')).grid(
                row=0, column=col, padx=10, pady=5
            )
        
        # VNA
        ttk.Label(conn_frame, text="VNA (S5180B)").grid(row=1, column=0, padx=10, pady=5, sticky='w')
        vna_port_frame = ttk.Frame(conn_frame)
        vna_port_frame.grid(row=1, column=1, padx=10, pady=5, sticky='w')
        ttk.Label(vna_port_frame, text="localhost:").pack(side='left')
        ttk.Entry(vna_port_frame, textvariable=self.vna_port, width=25).pack(side='left')
        
        self.vna_status = ttk.Label(conn_frame, text="o", foreground='gray', font=('Arial', 14))
        self.vna_status.grid(row=1, column=2, padx=10, pady=5)
        ttk.Button(conn_frame, text="Connect", command=self.connect_vna).grid(row=1, column=3, padx=10, pady=5)
        ttk.Button(conn_frame, text="Launch S2VNA", command=self.launch_s2vna).grid(row=1, column=4, padx=10, pady=5)
        
        # Magnet
        magnet_label_frame = ttk.Frame(conn_frame)
        magnet_label_frame.grid(row=2, column=0, padx=10, pady=5, sticky='w')
        ttk.Label(magnet_label_frame, text="Magnet:").pack(side='left')
        magnet_model_combo = ttk.Combobox(
            magnet_label_frame, textvariable=self.magnet_model,
            values=["SCM1", "Cryomagnetics 4G"], width=14, state='readonly'
        )
        magnet_model_combo.pack(side='left', padx=5)
        magnet_model_combo.bind('<<ComboboxSelected>>', self.on_magnet_model_change)
        
        magnet_addr_frame = ttk.Frame(conn_frame)
        magnet_addr_frame.grid(row=2, column=1, padx=10, pady=5, sticky='w')
        self.magnet_addr_label = ttk.Label(magnet_addr_frame, text="IP:")
        self.magnet_addr_label.pack(side='left')
        self.magnet_addr_entry = ttk.Entry(magnet_addr_frame, textvariable=self.magnet_addr, width=25)
        self.magnet_addr_entry.pack(side='left')
        
        self.magnet_status = ttk.Label(conn_frame, text="o", foreground='gray', font=('Arial', 14))
        self.magnet_status.grid(row=2, column=2, padx=10, pady=5)
        ttk.Button(conn_frame, text="Connect", command=self.connect_magnet).grid(row=2, column=3, padx=10, pady=5)
        
        # Keithley SMU
        keithley_label_frame = ttk.Frame(conn_frame)
        keithley_label_frame.grid(row=3, column=0, padx=10, pady=5, sticky='w')
        ttk.Label(keithley_label_frame, text="Gate SMU:").pack(side='left')
        keithley_model_combo = ttk.Combobox(
            keithley_label_frame, textvariable=self.keithley_model,
            values=["2400", "2450"], width=5, state='readonly'
        )
        keithley_model_combo.pack(side='left', padx=5)
        
        keithley_addr_frame = ttk.Frame(conn_frame)
        keithley_addr_frame.grid(row=3, column=1, padx=10, pady=5, sticky='w')
        ttk.Label(keithley_addr_frame, text="Addr:").pack(side='left')
        keithley_addr_entry = ttk.Entry(keithley_addr_frame, textvariable=self.keithley_addr, width=25)
        keithley_addr_entry.pack(side='left')
        ToolTip(keithley_addr_entry, "GPIB: Enter number (e.g., 18)\nUSB: Enter full address (e.g., USB0::0x05E6::0x2450::04102170::INSTR)")
        
        self.keithley_status = ttk.Label(conn_frame, text="o", foreground='gray', font=('Arial', 14))
        self.keithley_status.grid(row=3, column=2, padx=10, pady=5)
        ttk.Button(conn_frame, text="Connect", command=self.connect_keithley).grid(row=3, column=3, padx=10, pady=5)
        
        # Temperature Monitor (Lakeshore 370) - DISABLED due to GPIB conflicts
        temp_label = ttk.Label(conn_frame, text="Lakeshore 370 (DISABLED)", foreground='gray')
        temp_label.grid(row=4, column=0, padx=10, pady=5, sticky='w')
        temp_frame = ttk.Frame(conn_frame)
        temp_frame.grid(row=4, column=1, padx=10, pady=5, sticky='w')
        ttk.Label(temp_frame, text="GPIB:", foreground='gray').pack(side='left')
        temp_gpib_entry = ttk.Entry(temp_frame, textvariable=self.temp_gpib_address, width=5, state='disabled')
        temp_gpib_entry.pack(side='left')
        ttk.Label(temp_frame, text="Ch:", foreground='gray').pack(side='left', padx=(5, 0))
        temp_ch_entry = ttk.Entry(temp_frame, textvariable=self.temp_channel, width=3, state='disabled')
        temp_ch_entry.pack(side='left')
        
        self.temp_status = ttk.Label(conn_frame, text="o", foreground='gray', font=('Arial', 14))
        self.temp_status.grid(row=4, column=2, padx=10, pady=5)
        temp_connect_btn = ttk.Button(conn_frame, text="Connect", command=self.connect_temp, state='disabled')
        temp_connect_btn.grid(row=4, column=3, padx=10, pady=5)
        
        # Temperature display (read-only) - show disabled
        self.temp_display = ttk.Label(conn_frame, text="DISABLED", font=('Arial', 10), foreground='gray')
        self.temp_display.grid(row=4, column=4, padx=10, pady=5)
        
        # Settings container - organize in two columns
        settings_container = ttk.Frame(parent)
        settings_container.pack(fill='x', padx=10, pady=5)
        
        left_settings = ttk.Frame(settings_container)
        left_settings.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        right_settings = ttk.Frame(settings_container)
        right_settings.pack(side='left', fill='both', expand=True, padx=(5, 0))
        
        # === LEFT COLUMN: VNA Settings + Data File Settings ===
        
        # VNA Settings
        vna_settings = ttk.LabelFrame(left_settings, text="VNA Settings")
        vna_settings.pack(fill='x', pady=2)
        
        # Row 0: S-parameter, IFBW, settle time
        ttk.Label(vna_settings, text="S-Param:").grid(row=0, column=0, padx=5, pady=3, sticky='w')
        ttk.Combobox(
            vna_settings, textvariable=self.s_parameter, 
            values=["S21", "S11"], width=5, state='readonly'
        ).grid(row=0, column=1, padx=2, pady=3, sticky='w')
        
        ttk.Label(vna_settings, text="IFBW (Hz):").grid(row=0, column=2, padx=5, pady=3, sticky='w')
        ttk.Entry(vna_settings, textvariable=self.ifbw, width=7).grid(row=0, column=3, padx=2, pady=3, sticky='w')
        
        ttk.Label(vna_settings, text="Settle (s):").grid(row=0, column=4, padx=5, pady=3, sticky='w')
        vna_settle_entry = ttk.Entry(vna_settings, textvariable=self.vna_settle_time, width=5)
        vna_settle_entry.grid(row=0, column=5, padx=2, pady=3, sticky='w')
        ToolTip(vna_settle_entry, "Delay before first measurement after\nstarting a sweep. Allows RF to stabilize.")
        
        # Row 1: Probe attenuation settings
        ttk.Label(vna_settings, text="In Atten (dB):").grid(row=1, column=0, padx=5, pady=3, sticky='w')
        input_atten_entry = ttk.Entry(vna_settings, textvariable=self.input_attenuation, width=5)
        input_atten_entry.grid(row=1, column=1, padx=2, pady=3, sticky='w')
        ToolTip(input_atten_entry, "Total attenuation between VNA output\nand probe input (cables, attenuators, etc.)")
        
        ttk.Label(vna_settings, text="Out Atten (dB):").grid(row=1, column=2, padx=5, pady=3, sticky='w')
        output_atten_entry = ttk.Entry(vna_settings, textvariable=self.output_attenuation, width=5)
        output_atten_entry.grid(row=1, column=3, padx=2, pady=3, sticky='w')
        ToolTip(output_atten_entry, "Total attenuation between probe output\nand VNA input (cables, attenuators, etc.)")
        
        ttk.Label(vna_settings, text="-> Probe:").grid(row=1, column=4, padx=5, pady=3, sticky='w')
        self.power_at_probe_label = ttk.Label(vna_settings, text="-- dBm", font=('Arial', 9, 'bold'))
        self.power_at_probe_label.grid(row=1, column=5, padx=2, pady=3, sticky='w')
        
        self.fixed_power.trace_add('write', lambda *args: self.update_power_at_probe())
        self.input_attenuation.trace_add('write', lambda *args: self.update_power_at_probe())
        self.update_power_at_probe()
        
        # Data File Settings
        file_frame = ttk.LabelFrame(left_settings, text="Data File Settings")
        file_frame.pack(fill='x', pady=2)
        
        ttk.Label(file_frame, text="Directory:").grid(row=0, column=0, padx=5, pady=3, sticky='w')
        ttk.Entry(file_frame, textvariable=self.data_directory, width=35).grid(row=0, column=1, padx=2, pady=3, sticky='w')
        ttk.Button(file_frame, text="Browse", command=self.browse_directory).grid(row=0, column=2, padx=5, pady=3)
        
        ttk.Label(file_frame, text="Filename:").grid(row=1, column=0, padx=5, pady=3, sticky='w')
        ttk.Entry(file_frame, textvariable=self.filename, width=35).grid(row=1, column=1, padx=2, pady=3, sticky='w')
        
        auto_save_check = ttk.Checkbutton(file_frame, text="Auto-save", variable=self.auto_save)
        auto_save_check.grid(row=1, column=2, padx=5, pady=3)
        ToolTip(auto_save_check, "Automatically save data after each measurement.\n"
                                  "1D sweep: saves as filename_001.csv, _002.csv, etc.\n"
                                  "2D sweep: creates folder with individual sweep files.")
        
        # === RIGHT COLUMN: Gate Safety + B-Field Settings ===
        
        # Gate Voltage Safety Settings
        safety_frame = ttk.LabelFrame(right_settings, text="Gate Voltage Safety")
        safety_frame.pack(fill='x', pady=2)
        
        # Row 0: Slew rate, max voltage, compliance
        ttk.Label(safety_frame, text="Slew (V/s):").grid(row=0, column=0, padx=5, pady=3, sticky='w')
        self.gate_slew_rate = tk.StringVar(value="10.0")
        slew_entry = ttk.Entry(safety_frame, textvariable=self.gate_slew_rate, width=6)
        slew_entry.grid(row=0, column=1, padx=2, pady=3, sticky='w')
        ToolTip(slew_entry, "Rate at which gate voltage changes (V/s).")
        
        ttk.Label(safety_frame, text="Max (V):").grid(row=0, column=2, padx=5, pady=3, sticky='w')
        self.max_gate_voltage = tk.StringVar(value="100")
        max_entry = ttk.Entry(safety_frame, textvariable=self.max_gate_voltage, width=6)
        max_entry.grid(row=0, column=3, padx=2, pady=3, sticky='w')
        
        ttk.Label(safety_frame, text="Compl (nA):").grid(row=0, column=4, padx=5, pady=3, sticky='w')
        self.gate_compliance = tk.StringVar(value="100")
        compliance_entry = ttk.Entry(safety_frame, textvariable=self.gate_compliance, width=6)
        compliance_entry.grid(row=0, column=5, padx=2, pady=3, sticky='w')
        
        # Row 1: Checkboxes
        self.ramp_gate_to_zero = tk.BooleanVar(value=True)
        ttk.Checkbutton(safety_frame, text="Ramp to 0 after sweep", variable=self.ramp_gate_to_zero).grid(
            row=1, column=0, columnspan=3, padx=5, pady=2, sticky='w')
        
        self.ramp_gate_on_stop = tk.BooleanVar(value=False)
        ttk.Checkbutton(safety_frame, text="Ramp to 0 on Stop", variable=self.ramp_gate_on_stop).grid(
            row=1, column=3, columnspan=3, padx=5, pady=2, sticky='w')
        
        # Row 2: Current gate display and manual control
        ttk.Label(safety_frame, text="Current:").grid(row=2, column=0, padx=5, pady=3, sticky='w')
        self.current_gate_display = ttk.Label(safety_frame, text="0.000 V", font=('Arial', 9, 'bold'))
        self.current_gate_display.grid(row=2, column=1, padx=2, pady=3, sticky='w')
        
        ttk.Label(safety_frame, text="Set:").grid(row=2, column=2, padx=5, pady=3, sticky='w')
        self.manual_gate_entry = ttk.Entry(safety_frame, width=6)
        self.manual_gate_entry.insert(0, "0")
        self.manual_gate_entry.grid(row=2, column=3, padx=2, pady=3, sticky='w')
        
        ttk.Button(safety_frame, text="Ramp", command=self.manual_gate_ramp, width=5).grid(row=2, column=4, padx=2, pady=3)
        ttk.Button(safety_frame, text="-> 0", command=self.ramp_gate_to_zero_now, width=4).grid(row=2, column=5, padx=2, pady=3)
        
        # B-Field Settings
        bfield_frame = ttk.LabelFrame(right_settings, text="B-Field Settings")
        bfield_frame.pack(fill='x', pady=2)
        
        # Row 0: Ramp rate, tolerance, settle time
        ttk.Label(bfield_frame, text="Rate (T/min):").grid(row=0, column=0, padx=5, pady=3, sticky='w')
        self.field_ramp_rate = tk.StringVar(value="0.3")
        rate_entry = ttk.Entry(bfield_frame, textvariable=self.field_ramp_rate, width=6)
        rate_entry.grid(row=0, column=1, padx=2, pady=3, sticky='w')
        ToolTip(rate_entry, "B-field ramp rate in Tesla/minute.\nSCM1 max: 0.3 T/min")
        
        ttk.Label(bfield_frame, text="Tol (T):").grid(row=0, column=2, padx=5, pady=3, sticky='w')
        self.field_tolerance = tk.StringVar(value="0.001")
        tol_entry = ttk.Entry(bfield_frame, textvariable=self.field_tolerance, width=6)
        tol_entry.grid(row=0, column=3, padx=2, pady=3, sticky='w')
        ToolTip(tol_entry, "Field tolerance. 0.001 T = 10 Gauss")
        
        ttk.Label(bfield_frame, text="Settle (s):").grid(row=0, column=4, padx=5, pady=3, sticky='w')
        self.field_settle_time = tk.StringVar(value="2.0")
        settle_entry = ttk.Entry(bfield_frame, textvariable=self.field_settle_time, width=6)
        settle_entry.grid(row=0, column=5, padx=2, pady=3, sticky='w')
        
        # Row 1: Checkbox and current field
        self.wait_for_field = tk.BooleanVar(value=True)
        ttk.Checkbutton(bfield_frame, text="Wait for field (2D)", variable=self.wait_for_field).grid(
            row=1, column=0, columnspan=3, padx=5, pady=2, sticky='w')
        
        ttk.Label(bfield_frame, text="Current:").grid(row=1, column=3, padx=5, pady=3, sticky='w')
        self.current_field_display = ttk.Label(bfield_frame, text="0.000 T", font=('Arial', 9, 'bold'))
        self.current_field_display.grid(row=1, column=4, columnspan=2, padx=2, pady=3, sticky='w')
        
        # Row 2: Time estimation
        ttk.Label(bfield_frame, text="Est. Time:").grid(row=2, column=0, padx=5, pady=3, sticky='w')
        self.time_estimate_display = ttk.Label(bfield_frame, text="--", font=('Arial', 9))
        self.time_estimate_display.grid(row=2, column=1, columnspan=3, padx=2, pady=3, sticky='w')
        ttk.Button(bfield_frame, text="Calculate", command=self.calculate_time_estimate).grid(row=2, column=4, columnspan=2, padx=2, pady=3)
        
        # SCM1 warning at bottom
        scm1_warning = ttk.Label(
            right_settings, 
            text="[!] SCM1: Ensure LabVIEW is in 'Ramp to Setpoint' mode",
            foreground='#CC6600',
            font=('Arial', 8)
        )
        scm1_warning.pack(pady=2)
    
    def create_sweep_tab(self, parent):
        """Create measurement control interface."""
        # Main container with three sections: left params, right params, bottom plot
        top_frame = ttk.Frame(parent)
        top_frame.pack(fill='x', padx=10, pady=5)
        
        left_frame = ttk.Frame(top_frame)
        left_frame.pack(side='left', fill='both', expand=True)
        
        right_frame = ttk.Frame(top_frame)
        right_frame.pack(side='right', fill='both', expand=True, padx=(10, 0))
        
        # Sweep parameter selection
        sweep_frame = ttk.LabelFrame(left_frame, text="Sweep Parameter (Primary)")
        sweep_frame.pack(fill='x', pady=5)
        
        ttk.Label(sweep_frame, text="Parameter:").grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.sweep_combo = ttk.Combobox(
            sweep_frame, textvariable=self.sweep_param,
            values=self.param_options, width=20, state='readonly'
        )
        self.sweep_combo.grid(row=0, column=1, padx=10, pady=5, sticky='w')
        self.sweep_combo.bind('<<ComboboxSelected>>', self.on_sweep_param_changed)
        
        ttk.Label(sweep_frame, text="Start:").grid(row=1, column=0, padx=10, pady=5, sticky='w')
        ttk.Entry(sweep_frame, textvariable=self.sweep_start, width=15).grid(row=1, column=1, padx=10, pady=5, sticky='w')
        
        ttk.Label(sweep_frame, text="Stop:").grid(row=2, column=0, padx=10, pady=5, sticky='w')
        ttk.Entry(sweep_frame, textvariable=self.sweep_stop, width=15).grid(row=2, column=1, padx=10, pady=5, sticky='w')
        
        ttk.Label(sweep_frame, text="Points:").grid(row=3, column=0, padx=10, pady=5, sticky='w')
        ttk.Entry(sweep_frame, textvariable=self.sweep_points, width=15).grid(row=3, column=1, padx=10, pady=5, sticky='w')
        
        # Averages control
        ttk.Label(sweep_frame, text="Averages:").grid(row=4, column=0, padx=10, pady=5, sticky='w')
        avg_frame = ttk.Frame(sweep_frame)
        avg_frame.grid(row=4, column=1, padx=10, pady=5, sticky='w')
        ttk.Entry(avg_frame, textvariable=self.sweep_averages, width=5).pack(side='left')
        ttk.Label(avg_frame, text="sweeps").pack(side='left', padx=5)
        # Quick buttons for common values
        for n in [1, 4, 10, 25]:
            ttk.Button(avg_frame, text=str(n), width=3,
                       command=lambda x=n: self.sweep_averages.set(str(x))).pack(side='left', padx=1)
        
        # Scan time estimate label
        self.scan_time_label = ttk.Label(sweep_frame, text="", foreground='blue')
        self.scan_time_label.grid(row=5, column=0, columnspan=3, padx=10, pady=5, sticky='w')
        
        # Add tooltip to explain calculation
        self.scan_time_tooltip = ToolTip(
            self.scan_time_label,
            "Estimated scan time per sweep:\n"
            "  time = (points / IFBW) x 1.5 + 2s\n\n"
            "The 1.5x multiplier accounts for VNA\n"
            "processing overhead per point.\n"
            "The +2s is fixed overhead.\n\n"
            "Adjust IFBW in Setup tab to change speed.\n\n"
            "Limits (auto-clamped):\n"
            "  Frequency: 0.0001 - 18 GHz\n"
            "  Power: -50 to +10 dBm"
        )
        
        # Set up variable traces to update scan time
        self.sweep_points.trace_add('write', self.update_scan_time_display)
        self.sweep_averages.trace_add('write', self.update_scan_time_display)
        self.ifbw.trace_add('write', self.update_scan_time_display)
        
        # Single sweep progress bar (below sweep settings)
        sweep_progress_frame = ttk.Frame(sweep_frame)
        sweep_progress_frame.grid(row=6, column=0, columnspan=2, padx=10, pady=5, sticky='ew')
        
        ttk.Label(sweep_progress_frame, text="Sweep:").pack(side='left')
        self.progress_bar = ttk.Progressbar(
            sweep_progress_frame, variable=self.progress_var, 
            maximum=100, length=200
        )
        self.progress_bar.pack(side='left', padx=5, fill='x', expand=True)
        self.progress_label = ttk.Label(sweep_progress_frame, text="0%", width=5)
        self.progress_label.pack(side='left')
        
        # Step parameter selection (optional)
        step_frame = ttk.LabelFrame(left_frame, text="Step Parameter (Optional - for 2D scans)")
        step_frame.pack(fill='x', pady=5)
        
        ttk.Label(step_frame, text="Parameter:").grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.step_combo = ttk.Combobox(
            step_frame, textvariable=self.step_param,
            values=["None"] + self.param_options, width=20, state='readonly'
        )
        self.step_combo.grid(row=0, column=1, padx=10, pady=5, sticky='w')
        self.step_combo.bind('<<ComboboxSelected>>', self.on_step_param_changed)
        
        ttk.Label(step_frame, text="Start:").grid(row=1, column=0, padx=10, pady=5, sticky='w')
        self.step_start_entry = ttk.Entry(step_frame, textvariable=self.step_start, width=15, state='disabled')
        self.step_start_entry.grid(row=1, column=1, padx=10, pady=5, sticky='w')
        
        ttk.Label(step_frame, text="Stop:").grid(row=2, column=0, padx=10, pady=5, sticky='w')
        self.step_stop_entry = ttk.Entry(step_frame, textvariable=self.step_stop, width=15, state='disabled')
        self.step_stop_entry.grid(row=2, column=1, padx=10, pady=5, sticky='w')
        
        ttk.Label(step_frame, text="Steps:").grid(row=3, column=0, padx=10, pady=5, sticky='w')
        self.step_points_entry = ttk.Entry(step_frame, textvariable=self.step_points, width=15, state='disabled')
        self.step_points_entry.grid(row=3, column=1, padx=10, pady=5, sticky='w')
        
        # Add traces for step parameters to update time estimate
        self.step_param.trace_add('write', self.update_scan_time_display)
        self.step_points.trace_add('write', self.update_scan_time_display)
        
        # 2D sweep progress bar (below step settings)
        self.step_progress_var = tk.DoubleVar(value=0)
        step_progress_frame = ttk.Frame(step_frame)
        step_progress_frame.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky='ew')
        
        ttk.Label(step_progress_frame, text="2D Step:").pack(side='left')
        self.step_progress_bar = ttk.Progressbar(
            step_progress_frame, variable=self.step_progress_var, 
            maximum=100, length=200
        )
        self.step_progress_bar.pack(side='left', padx=5, fill='x', expand=True)
        self.step_progress_label = ttk.Label(step_progress_frame, text="0%", width=5)
        self.step_progress_label.pack(side='left')
        
        # Gate Normalization settings
        self.gate_norm_frame = ttk.LabelFrame(left_frame, text="Gate Voltage Normalization")
        self.gate_norm_frame.pack(fill='x', pady=5)
        
        gate_norm_row1 = ttk.Frame(self.gate_norm_frame)
        gate_norm_row1.pack(fill='x', padx=5, pady=2)
        
        self.gate_norm_checkbox = ttk.Checkbutton(
            gate_norm_row1, text="Enable normalization to reference voltage",
            variable=self.gate_normalization_enabled,
            command=self.on_gate_normalization_changed
        )
        self.gate_norm_checkbox.pack(side='left')
        
        gate_norm_row2 = ttk.Frame(self.gate_norm_frame)
        gate_norm_row2.pack(fill='x', padx=5, pady=2)
        
        ttk.Label(gate_norm_row2, text="Reference V_gate:").pack(side='left', padx=(20, 5))
        self.gate_norm_voltage_entry = ttk.Entry(gate_norm_row2, textvariable=self.gate_normalization_voltage, width=8)
        self.gate_norm_voltage_entry.pack(side='left')
        ttk.Label(gate_norm_row2, text="V").pack(side='left', padx=2)
        
        self.gate_norm_info_label = ttk.Label(
            self.gate_norm_frame, 
            text="Will measure S21 at reference voltage first, then subtract from all measurements",
            font=('Arial', 8), foreground='gray'
        )
        self.gate_norm_info_label.pack(padx=5, pady=2, anchor='w')
        
        # B-Field Normalization settings
        self.field_norm_frame = ttk.LabelFrame(left_frame, text="B-Field Normalization")
        self.field_norm_frame.pack(fill='x', pady=5)
        
        field_norm_row1 = ttk.Frame(self.field_norm_frame)
        field_norm_row1.pack(fill='x', padx=5, pady=2)
        
        self.field_norm_checkbox = ttk.Checkbutton(
            field_norm_row1, text="Enable normalization to reference field",
            variable=self.field_normalization_enabled,
            command=self.on_field_normalization_changed
        )
        self.field_norm_checkbox.pack(side='left')
        
        field_norm_row2 = ttk.Frame(self.field_norm_frame)
        field_norm_row2.pack(fill='x', padx=5, pady=2)
        
        ttk.Label(field_norm_row2, text="Reference B_field:").pack(side='left', padx=(20, 5))
        self.field_norm_entry = ttk.Entry(field_norm_row2, textvariable=self.field_normalization_field, width=8)
        self.field_norm_entry.pack(side='left')
        ttk.Label(field_norm_row2, text="T").pack(side='left', padx=2)
        
        self.field_norm_info_label = ttk.Label(
            self.field_norm_frame, 
            text="Will measure S21 at reference field first, then subtract from all measurements",
            font=('Arial', 8), foreground='gray'
        )
        self.field_norm_info_label.pack(padx=5, pady=2, anchor='w')
        
        # Update normalization UI visibility based on sweep/step params
        self.update_normalization_visibility()
        
        # Control buttons (below sweep settings)
        control_frame = ttk.LabelFrame(left_frame, text="Measurement Control")
        control_frame.pack(fill='x', pady=5)
        
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(pady=10)
        
        self.start_button = ttk.Button(
            button_frame, text="> Start Measurement", 
            command=self.start_measurement, width=20
        )
        self.start_button.pack(side='left', padx=5)
        
        # Make stop button more prominent with red styling
        stop_style = ttk.Style()
        stop_style.configure('Stop.TButton', foreground='red')
        
        self.stop_button = ttk.Button(
            button_frame, text="[STOP] ABORT", 
            command=self.stop_measurement, width=15, 
            state='disabled', style='Stop.TButton'
        )
        self.stop_button.pack(side='left', padx=5)
        
        # Status display
        status_frame = ttk.Frame(control_frame)
        status_frame.pack(fill='x', padx=10, pady=5)
        ttk.Label(status_frame, text="Status:").pack(side='left')
        self.control_status_label = ttk.Label(status_frame, text="Ready", font=('Arial', 9))
        self.control_status_label.pack(side='left', padx=10)
        
        # Fixed parameters (right side)
        fixed_frame = ttk.LabelFrame(right_frame, text="Fixed Parameters")
        fixed_frame.pack(fill='x', pady=5)
        
        # Create fixed parameter entries (will be enabled/disabled based on sweep/step selection)
        self.fixed_entries = {}
        self.fixed_labels = {}  # Store labels so we can show/hide them
        
        row = 0
        self.fixed_labels['frequency'] = ttk.Label(fixed_frame, text="Frequency (GHz):")
        self.fixed_labels['frequency'].grid(row=row, column=0, padx=10, pady=5, sticky='w')
        self.fixed_entries['frequency'] = ttk.Entry(fixed_frame, textvariable=self.fixed_frequency, width=15)
        self.fixed_entries['frequency'].grid(row=row, column=1, padx=10, pady=5, sticky='w')
        ToolTip(self.fixed_entries['frequency'], "Range: 0.0001 - 18 GHz\n(auto-clamped if out of range)")
        
        row += 1
        # B-field row - will be hidden if magnet not connected
        self.fixed_labels['b_field'] = ttk.Label(fixed_frame, text="B-Field (T):")
        self.fixed_labels['b_field'].grid(row=row, column=0, padx=10, pady=5, sticky='w')
        self.fixed_entries['b_field'] = ttk.Entry(fixed_frame, textvariable=self.fixed_field, width=15)
        self.fixed_entries['b_field'].grid(row=row, column=1, padx=10, pady=5, sticky='w')
        ToolTip(self.fixed_entries['b_field'], "Range: -18 to +18 T\n(auto-clamped if out of range)")
        self.bfield_row = row  # Store row number for show/hide
        
        row += 1
        self.fixed_labels['vg'] = ttk.Label(fixed_frame, text="Gate Voltage (V):")
        self.fixed_labels['vg'].grid(row=row, column=0, padx=10, pady=5, sticky='w')
        self.fixed_entries['vg'] = ttk.Entry(fixed_frame, textvariable=self.fixed_gate, width=15)
        self.fixed_entries['vg'].grid(row=row, column=1, padx=10, pady=5, sticky='w')
        
        row += 1
        self.fixed_labels['power'] = ttk.Label(fixed_frame, text="VNA Power (dBm):")
        self.fixed_labels['power'].grid(row=row, column=0, padx=10, pady=5, sticky='w')
        self.fixed_entries['power'] = ttk.Entry(fixed_frame, textvariable=self.fixed_power, width=15)
        self.fixed_entries['power'].grid(row=row, column=1, padx=10, pady=5, sticky='w')
        ToolTip(self.fixed_entries['power'], "VNA output power setting.\nRange: -50 to +10 dBm\n\n"
                                             "Power at probe = VNA power - input attenuation\n"
                                             "(See VNA Settings for calculated probe power)")
        
        row += 1
        self.fixed_labels['temperature'] = ttk.Label(fixed_frame, text="Temperature:", foreground='gray')
        self.fixed_labels['temperature'].grid(row=row, column=0, padx=10, pady=5, sticky='w')
        # Temperature display DISABLED - Lakeshore causes GPIB conflicts
        self.fixed_temp_display = ttk.Label(fixed_frame, text="DISABLED", font=('Arial', 9), foreground='gray')
        self.fixed_temp_display.grid(row=row, column=1, padx=10, pady=5, sticky='w')
        ToolTip(self.fixed_labels['temperature'], "Temperature monitoring DISABLED\n"
                                                   "Lakeshore 370 causes GPIB conflicts with Keithley")
        
        # Update fixed parameter states
        self.update_fixed_params_state()
        
        # Initially hide B-field if magnet not connected
        self.update_bfield_visibility()
        
        # Measurement summary
        summary_frame = ttk.LabelFrame(right_frame, text="Measurement Summary")
        summary_frame.pack(fill='x', pady=5)
        
        self.summary_text = tk.Text(summary_frame, height=8, width=40, state='disabled')
        self.summary_text.pack(fill='x', padx=10, pady=5)
        
        # Log display
        log_frame = ttk.LabelFrame(right_frame, text="Log Output")
        log_frame.pack(fill='both', expand=True, pady=5)
        
        # Log control buttons
        log_buttons = ttk.Frame(log_frame)
        log_buttons.pack(fill='x', padx=5, pady=2)
        
        ttk.Button(log_buttons, text="Clear Log", command=self.clear_log_display, width=10).pack(side='left', padx=2)
        ttk.Button(log_buttons, text="Save Log", command=self.save_log_manually, width=10).pack(side='left', padx=2)
        
        # Auto-scroll checkbox
        self.log_autoscroll = tk.BooleanVar(value=True)
        ttk.Checkbutton(log_buttons, text="Auto-scroll", variable=self.log_autoscroll).pack(side='right', padx=5)
        
        # Log text widget with scrollbar
        log_text_frame = ttk.Frame(log_frame)
        log_text_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        log_scrollbar = ttk.Scrollbar(log_text_frame)
        log_scrollbar.pack(side='right', fill='y')
        
        self.log_text = tk.Text(
            log_text_frame, 
            height=10, 
            width=40, 
            state='disabled',
            wrap='word',
            font=('Consolas', 8),
            yscrollcommand=log_scrollbar.set
        )
        self.log_text.pack(fill='both', expand=True)
        log_scrollbar.config(command=self.log_text.yview)
        
        # Configure log text tags for coloring
        self.log_text.tag_configure("INFO", foreground="black")
        self.log_text.tag_configure("ERROR", foreground="red")
        self.log_text.tag_configure("WARNING", foreground="orange")
        self.log_text.tag_configure("SUCCESS", foreground="green")
        
        # Connect log manager to this text widget
        log_manager.set_text_widget(self.log_text)
        log_manager.start_capture()
        
        self.update_step_options()  # Initialize step options based on default sweep
        self.update_summary()
        self.update_scan_time_display()
    
    def create_plot_tab(self, parent):
        """Create data visualization interface with dynamic 1D/2D layout."""
        # Main container that will hold everything
        self.plot_main_frame = ttk.Frame(parent)
        self.plot_main_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create shared control panel at top (always visible)
        self.plot_control_frame = ttk.Frame(self.plot_main_frame)
        self.plot_control_frame.pack(fill='x', padx=5, pady=5)
        
        # Data type selection (Mag/Phase/Real/Imag)
        type_frame = ttk.Frame(self.plot_control_frame)
        type_frame.pack(fill='x', pady=2)
        
        ttk.Label(type_frame, text="Display:").pack(side='left', padx=5)
        self.trace_display_mode = tk.StringVar(value="Magnitude")
        for mode in ["Magnitude", "Normalized", "Phase", "Real", "Imaginary", "Conductivity"]:
            rb = ttk.Radiobutton(
                type_frame, text=mode,
                variable=self.trace_display_mode, value=mode,
                command=self.on_display_mode_changed
            )
            rb.pack(side='left', padx=3)
            if mode == "Normalized":
                ToolTip(rb, "Delta S21 (dB) = S21 - S21_ref\n\n"
                           "Subtracts reference measurement taken at\n"
                           "the normalization voltage (default 0V).\n\n"
                           "Enable normalization in Measurement Control tab.")
            elif mode == "Conductivity":
                ToolTip(rb, "Estimated conductivity from S21:\nsigma = -ln(S21/S21_max) x w / (2 x L x Z0)\n\n"
                           "Requires CPW slot geometry (w, L).\n"
                           "Uses max S21 as reference (sigma=0).\n\n"
                           "[!] This is an approximation - see literature\n"
                           "for proper transmission line analysis.")
        
        # Smoothing control
        ttk.Label(type_frame, text="    Smoothing:").pack(side='left', padx=5)
        self.smoothing_window = tk.IntVar(value=1)
        smooth_spinbox = ttk.Spinbox(
            type_frame, from_=1, to=101, increment=2,
            textvariable=self.smoothing_window, width=5,
            command=self._update_all_plots
        )
        smooth_spinbox.pack(side='left', padx=2)
        smooth_spinbox.bind('<Return>', lambda e: self._update_all_plots())
        ttk.Label(type_frame, text="pts").pack(side='left')
        
        for n in [1, 5, 11, 21]:
            text = "Off" if n == 1 else str(n)
            ttk.Button(type_frame, text=text, width=3,
                       command=lambda x=n: self._set_smoothing(x)).pack(side='left', padx=1)
        
        # Buttons on right
        ttk.Button(type_frame, text="Clear All", command=self.clear_plots).pack(side='right', padx=5)
        ttk.Button(type_frame, text="Save Data", command=self.save_data).pack(side='right', padx=5)
        
        # Second row: X-axis mode for gate sweeps
        xaxis_frame = ttk.Frame(self.plot_control_frame)
        xaxis_frame.pack(fill='x', pady=2)
        
        ttk.Label(xaxis_frame, text="X-axis:").pack(side='left', padx=5)
        self.xaxis_mode = tk.StringVar(value="Gate Voltage (V)")
        for mode in ["Gate Voltage (V)", "Density (cm^-2)", "Filling Factor (nu)"]:
            ttk.Radiobutton(
                xaxis_frame, text=mode,
                variable=self.xaxis_mode, value=mode,
                command=self.on_xaxis_mode_changed
            ).pack(side='left', padx=3)
        
        # V_CNP adjustment
        ttk.Label(xaxis_frame, text="    V_CNP:").pack(side='left', padx=2)
        vcnp_entry_plot = ttk.Entry(xaxis_frame, textvariable=self.v_cnp, width=6)
        vcnp_entry_plot.pack(side='left', padx=2)
        vcnp_entry_plot.bind('<Return>', lambda e: self.on_xaxis_mode_changed())
        ttk.Label(xaxis_frame, text="V").pack(side='left')
        
        # hBN thickness
        ttk.Label(xaxis_frame, text="    hBN:").pack(side='left', padx=2)
        hbn_entry = ttk.Entry(xaxis_frame, textvariable=self.hbn_thickness, width=5)
        hbn_entry.pack(side='left', padx=2)
        hbn_entry.bind('<Return>', lambda e: self.on_xaxis_mode_changed())
        ttk.Label(xaxis_frame, text="nm").pack(side='left')
        
        # Calculated density per volt display
        ttk.Label(xaxis_frame, text="    ->'").pack(side='left', padx=2)
        self.density_per_volt_display = ttk.Label(xaxis_frame, text="--", font=('Arial', 8))
        self.density_per_volt_display.pack(side='left', padx=2)
        
        # Set up trace for hBN changes
        self.hbn_thickness.trace_add('write', self.update_sample_calculations)
        self.update_sample_calculations()
        
        # Third row: CPW geometry for conductivity calculation
        cpw_frame = ttk.Frame(self.plot_control_frame)
        cpw_frame.pack(fill='x', pady=2)
        
        ttk.Label(cpw_frame, text="CPW Geometry (for sigma):").pack(side='left', padx=5)
        
        ttk.Label(cpw_frame, text="Slot w:").pack(side='left', padx=2)
        slot_w_entry = ttk.Entry(cpw_frame, textvariable=self.cpw_slot_width, width=5)
        slot_w_entry.pack(side='left', padx=2)
        slot_w_entry.bind('<Return>', lambda e: self._update_all_plots())
        ToolTip(slot_w_entry, "CPW slot width in um\n(gap between center conductor and ground)")
        ttk.Label(cpw_frame, text="um").pack(side='left')
        
        ttk.Label(cpw_frame, text="    Length L:").pack(side='left', padx=2)
        slot_l_entry = ttk.Entry(cpw_frame, textvariable=self.cpw_slot_length, width=5)
        slot_l_entry.pack(side='left', padx=2)
        slot_l_entry.bind('<Return>', lambda e: self._update_all_plots())
        ToolTip(slot_l_entry, "CPW channel length in um\n(length of graphene over slot)")
        ttk.Label(cpw_frame, text="um").pack(side='left')
        
        # Autoscale checkbox
        self.trace_autoscale = True  # Start with autoscale on
        self.trace_autoscale_var = tk.BooleanVar(value=True)
        autoscale_cb = ttk.Checkbutton(cpw_frame, text="Autoscale", 
                                        variable=self.trace_autoscale_var,
                                        command=self.on_autoscale_changed)
        autoscale_cb.pack(side='right', padx=10)
        
        ttk.Button(cpw_frame, text="Reset Zoom", command=self.reset_trace_zoom).pack(side='right', padx=5)
        
        # Container for plots (will be reconfigured for 1D vs 2D)
        self.plot_container = ttk.Frame(self.plot_main_frame)
        self.plot_container.pack(fill='both', expand=True)
        
        # ===== Create 1D-only frame (full width single trace) =====
        self.frame_1d = ttk.Frame(self.plot_container)
        
        # Step selector (for viewing multi-step data after measurement)
        step_frame_1d = ttk.Frame(self.frame_1d)
        step_frame_1d.pack(fill='x', padx=5, pady=2)
        
        ttk.Label(step_frame_1d, text="Step:").pack(side='left', padx=5)
        self.prev_step_btn = ttk.Button(step_frame_1d, text="<", width=3, command=self.prev_step, state='disabled')
        self.prev_step_btn.pack(side='left', padx=2)
        
        self.step_slider_var = tk.IntVar(value=0)
        self.step_slider = ttk.Scale(step_frame_1d, from_=0, to=0, orient='horizontal', 
                                      variable=self.step_slider_var, command=self.on_step_slider_changed)
        self.step_slider.pack(side='left', fill='x', expand=True, padx=5)
        self.step_slider.config(state='disabled')
        
        self.next_step_btn = ttk.Button(step_frame_1d, text=">", width=3, command=self.next_step, state='disabled')
        self.next_step_btn.pack(side='left', padx=2)
        
        self.step_value_label = ttk.Label(step_frame_1d, text="", font=('Arial', 9))
        self.step_value_label.pack(side='left', padx=10)
        
        self.show_all_traces = tk.BooleanVar(value=False)
        ttk.Checkbutton(step_frame_1d, text="Overlay all", variable=self.show_all_traces, 
                        command=self.update_single_trace).pack(side='right', padx=5)
        
        # 1D figure (large)
        self.fig_trace = Figure(figsize=(10, 6))
        self.ax_trace = self.fig_trace.add_subplot(111)
        self.canvas_trace = FigureCanvasTkAgg(self.fig_trace, self.frame_1d)
        
        toolbar_frame_1d = ttk.Frame(self.frame_1d)
        toolbar_frame_1d.pack(side='bottom', fill='x')
        self.toolbar_trace = NavigationToolbar2Tk(self.canvas_trace, toolbar_frame_1d)
        self.toolbar_trace.update()
        
        self.canvas_trace.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
        
        # ===== Create 2D frame (map large, trace small) =====
        self.frame_2d = ttk.Frame(self.plot_container)
        
        # Use PanedWindow for resizable split
        self.paned_2d = ttk.PanedWindow(self.frame_2d, orient='horizontal')
        self.paned_2d.pack(fill='both', expand=True)
        
        # Left side: 2D map (larger)
        map_frame = ttk.LabelFrame(self.paned_2d, text="2D Map")
        self.paned_2d.add(map_frame, weight=3)
        
        # 2D color scale controls
        scale_frame = ttk.Frame(map_frame)
        scale_frame.pack(fill='x', padx=5, pady=2)
        
        ttk.Label(scale_frame, text="Color:").pack(side='left', padx=2)
        self.auto_scale = tk.BooleanVar(value=True)
        ttk.Checkbutton(scale_frame, text="Auto", variable=self.auto_scale, 
                        command=self.on_scale_mode_changed).pack(side='left', padx=2)
        
        self.log_scale = tk.BooleanVar(value=False)
        ttk.Checkbutton(scale_frame, text="Log", variable=self.log_scale, 
                        command=self.update_2d_plot).pack(side='left', padx=2)
        
        # V_norm normalization controls
        self.normalize_at_v = tk.BooleanVar(value=False)
        norm_cb = ttk.Checkbutton(scale_frame, text="Norm at V:", variable=self.normalize_at_v, 
                        command=self._on_normalization_changed)
        norm_cb.pack(side='left', padx=(5, 0))
        ToolTip(norm_cb, "Normalize data by subtracting values at V_norm.\n\n"
                         "For gate sweeps: S21(V) - S21(V_norm)\n"
                         "For 2D maps with gate as sweep: each curve normalized\n"
                         "For 2D maps with gate as step: spectrum at V_norm\n"
                         "  is subtracted from all other spectra.\n\n"
                         "Works with all display modes (Mag, Phase, sigma, etc.)")
        
        self.v_norm = tk.StringVar(value="0")
        v_norm_entry = ttk.Entry(scale_frame, textvariable=self.v_norm, width=6)
        v_norm_entry.pack(side='left', padx=2)
        v_norm_entry.bind('<Return>', lambda e: self._on_normalization_changed())
        ToolTip(v_norm_entry, "Gate voltage for normalization reference.\n"
                              "Data at this voltage will be subtracted.\n"
                              "Typically set to V_CNP (charge neutrality point)\n"
                              "where graphene conductivity is minimum.")
        ttk.Label(scale_frame, text="V").pack(side='left')
        
        # Status label for normalization (shows when waiting for V_norm data)
        self.norm_status_label = ttk.Label(scale_frame, text="", foreground='orange', font=('Arial', 8))
        self.norm_status_label.pack(side='left', padx=5)
        
        ttk.Label(scale_frame, text="Min:").pack(side='left', padx=(5, 2))
        self.color_min = tk.StringVar(value="-60")
        self.color_min_entry = ttk.Entry(scale_frame, textvariable=self.color_min, width=6)
        self.color_min_entry.pack(side='left', padx=2)
        self.color_min_entry.bind('<Return>', lambda e: self.update_2d_plot())
        
        ttk.Label(scale_frame, text="Max:").pack(side='left', padx=(5, 2))
        self.color_max = tk.StringVar(value="0")
        self.color_max_entry = ttk.Entry(scale_frame, textvariable=self.color_max, width=6)
        self.color_max_entry.pack(side='left', padx=2)
        self.color_max_entry.bind('<Return>', lambda e: self.update_2d_plot())
        
        ttk.Button(scale_frame, text="Apply", command=self.update_2d_plot).pack(side='left', padx=5)
        self.on_scale_mode_changed()
        
        # 2D figure
        self.fig_2d = Figure(figsize=(7, 6))
        self.ax_2d = self.fig_2d.add_subplot(111)
        self.canvas_2d = FigureCanvasTkAgg(self.fig_2d, map_frame)
        
        # Colorbar interaction state
        self.colorbar_2d = None
        self.colorbar_mappable = None
        self.colorbar_data_range = (0, 1)
        self.colorbar_dragging = False
        self.colorbar_drag_start_y = None
        self.colorbar_drag_start_vmin = None
        self.colorbar_drag_start_vmax = None
        self.contour_mode = tk.StringVar(value="Magnitude")
        
        # Connect mouse events for interactive colorbar
        self.canvas_2d.mpl_connect('button_press_event', self.on_colorbar_press)
        self.canvas_2d.mpl_connect('button_release_event', self.on_colorbar_release)
        self.canvas_2d.mpl_connect('motion_notify_event', self.on_colorbar_motion)
        self.canvas_2d.mpl_connect('scroll_event', self.on_colorbar_scroll)
        
        toolbar_frame_2d = ttk.Frame(map_frame)
        toolbar_frame_2d.pack(side='bottom', fill='x')
        self.toolbar_2d = NavigationToolbar2Tk(self.canvas_2d, toolbar_frame_2d)
        self.toolbar_2d.update()
        
        self.canvas_2d.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
        
        # Right side: Current trace (smaller)
        trace_frame_2d = ttk.LabelFrame(self.paned_2d, text="Current Trace")
        self.paned_2d.add(trace_frame_2d, weight=1)
        
        # Step selector for 2D mode
        step_frame_2d = ttk.Frame(trace_frame_2d)
        step_frame_2d.pack(fill='x', padx=2, pady=2)
        
        self.prev_step_btn_2d = ttk.Button(step_frame_2d, text="<", width=2, command=self.prev_step, state='disabled')
        self.prev_step_btn_2d.pack(side='left', padx=1)
        
        self.step_slider_2d = ttk.Scale(step_frame_2d, from_=0, to=0, orient='horizontal',
                                         variable=self.step_slider_var, command=self.on_step_slider_changed)
        self.step_slider_2d.pack(side='left', fill='x', expand=True, padx=2)
        self.step_slider_2d.config(state='disabled')
        
        self.next_step_btn_2d = ttk.Button(step_frame_2d, text=">", width=2, command=self.next_step, state='disabled')
        self.next_step_btn_2d.pack(side='left', padx=1)
        
        self.step_value_label_2d = ttk.Label(step_frame_2d, text="", font=('Arial', 8))
        self.step_value_label_2d.pack(side='left', padx=5)
        
        # Secondary trace figure (smaller, for 2D mode)
        self.fig_trace_2d = Figure(figsize=(4, 4))
        self.ax_trace_2d = self.fig_trace_2d.add_subplot(111)
        self.canvas_trace_2d = FigureCanvasTkAgg(self.fig_trace_2d, trace_frame_2d)
        self.canvas_trace_2d.get_tk_widget().pack(fill='both', expand=True, padx=2, pady=2)
        
        # Initialize in 1D mode
        self.current_plot_mode = "1D"
        self.set_plot_layout("1D")
        
        # Initialize plots
        self.current_step_index = 0
        self.init_plots()
    
    def set_plot_layout(self, mode):
        """Switch between 1D (single trace full) and 2D (map + small trace) layouts."""
        self.current_plot_mode = mode
        
        if mode == "1D":
            self.frame_2d.pack_forget()
            self.frame_1d.pack(fill='both', expand=True)
        else:  # 2D
            self.frame_1d.pack_forget()
            self.frame_2d.pack(fill='both', expand=True)
    
    def on_display_mode_changed(self):
        """Handle display mode change - update both trace views."""
        # Sync contour mode with trace display mode
        self.contour_mode.set(self.trace_display_mode.get())
        self.update_single_trace()
        if self.current_plot_mode == "2D":
            self.update_2d_plot()
    
    def on_xaxis_mode_changed(self):
        """Handle x-axis mode change - update plots.
        
        For 2D plots:
        - Filling Factor mode is DISABLED (requires per-row B-field, too complex for large datasets)
        - V to Density is a linear transform (can be fast)
        - Large dataset updates are throttled
        """
        mode = self.xaxis_mode.get()
        sweep_param = self.current_config.get('sweep_param') if self.current_config else None
        
        # Always update single trace (relatively fast, and supports all modes)
        self.update_single_trace()
        
        # For 2D plot, handle special cases
        if self.current_plot_mode == "2D" and len(self.sweep_data_2d) >= 2:
            # BLOCK Filling Factor for 2D plots - it's too complex and causes GUI freezes
            if mode == "Filling Factor (nu)" and sweep_param == "Gate Voltage (V)":
                # Revert to previous mode (Density if available, else Voltage)
                self.xaxis_mode.set("Gate Voltage (V)")
                self.status_var.set("[!] Filling Factor disabled for 2D maps (use single trace)")
                # Don't update 2D plot - keep current display
                return
            
            # Check dataset size before updating
            total_points = sum(len(t) for t in self.sweep_data_2d if t)
            if total_points > 30000:
                # For very large datasets, warn user and skip update
                # They can manually click "Apply" in color scale to force update
                self.status_var.set(f"Large dataset ({total_points} pts) - 2D plot axis unchanged")
                return
            elif total_points > 10000:
                # Medium datasets - show progress
                self.status_var.set(f"Updating 2D plot ({total_points} points)...")
                self.root.update_idletasks()
            
            self.update_2d_plot()
            
            if total_points > 10000:
                self.status_var.set("Ready")
    
    def on_autoscale_changed(self):
        """Handle autoscale checkbox change."""
        self.trace_autoscale = self.trace_autoscale_var.get()
        if self.trace_autoscale:
            # Re-enable autoscale by resetting zoom
            self.reset_trace_zoom()
    
    def reset_trace_zoom(self):
        """Reset trace plot to full data range."""
        self.trace_autoscale = True
        self.trace_autoscale_var.set(True)
        self._trace_has_data = False  # Force re-autoscale
        self.update_single_trace()
    
    def create_status_bar(self):
        """Create status bar at bottom of window."""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side='bottom', fill='x')
        
        ttk.Label(status_frame, textvariable=self.status_var).pack(side='left', padx=10)
        
        # Simulation indicator
        self.sim_indicator = ttk.Label(status_frame, text="[SIMULATION MODE]", foreground='blue')
        self.sim_indicator.pack(side='right', padx=10)
    
    def init_plots(self):
        """Initialize empty plots."""
        # Single trace plot (1D mode, full size)
        self.ax_trace.set_xlabel("Sweep Parameter")
        self.ax_trace.set_ylabel("Signal")
        self.ax_trace.set_title("Single Trace Viewer")
        self.ax_trace.grid(True, alpha=0.3)
        self.ax_trace.text(0.5, 0.5, "Run a measurement to see data",
                          ha='center', va='center', transform=self.ax_trace.transAxes,
                          fontsize=8, color='gray')
        
        # 2D plot
        self.ax_2d.set_xlabel("Sweep Parameter")
        self.ax_2d.set_ylabel("Step Parameter")
        self.ax_2d.set_title("2D Map")
        self.ax_2d.text(0.5, 0.5, "Run a 2D scan to see contour plot",
                       ha='center', va='center', transform=self.ax_2d.transAxes,
                       fontsize=8, color='gray')
        
        # Secondary trace plot (2D mode, small)
        self.ax_trace_2d.set_xlabel("Sweep")
        self.ax_trace_2d.set_ylabel("Signal")
        self.ax_trace_2d.grid(True, alpha=0.3)
        self.ax_trace_2d.tick_params(labelsize=7)
        
        self.fig_trace.tight_layout()
        self.fig_2d.tight_layout()
        self.fig_trace_2d.tight_layout()
        
        self.canvas_trace.draw()
        self.canvas_2d.draw()
        self.canvas_trace_2d.draw()
    
    def update_step_slider(self):
        """Update step slider range based on available data."""
        n_steps = len(self.sweep_data_2d)
        
        if n_steps > 0:
            # Update both sliders (1D and 2D mode)
            self.step_slider.config(from_=0, to=max(0, n_steps - 1), state='normal')
            self.step_slider_2d.config(from_=0, to=max(0, n_steps - 1), state='normal')
            self.prev_step_btn.config(state='normal')
            self.next_step_btn.config(state='normal')
            self.prev_step_btn_2d.config(state='normal')
            self.next_step_btn_2d.config(state='normal')
            
            # Clamp current index to valid range
            if self.current_step_index >= n_steps:
                self.current_step_index = n_steps - 1
            self.step_slider_var.set(self.current_step_index)
            
            # Update label
            self.update_step_label()
        else:
            self.step_slider.config(from_=0, to=0, state='disabled')
            self.step_slider_2d.config(from_=0, to=0, state='disabled')
            self.prev_step_btn.config(state='disabled')
            self.next_step_btn.config(state='disabled')
            self.prev_step_btn_2d.config(state='disabled')
            self.next_step_btn_2d.config(state='disabled')
            self.step_value_label.config(text="")
            self.step_value_label_2d.config(text="")
    
    def update_step_label(self):
        """Update the step value label."""
        if not self.sweep_data_2d or self.current_step_index >= len(self.sweep_data_2d):
            self.step_value_label.config(text="")
            self.step_value_label_2d.config(text="")
            return
        
        trace_data = self.sweep_data_2d[self.current_step_index]
        if trace_data and trace_data[0]['step_value'] is not None:
            step_val = trace_data[0]['step_value']
            step_param = self.current_config.get('step_param', 'Step') if self.current_config else 'Step'
            label_text = f"{step_param} = {step_val:.4g}  ({self.current_step_index + 1}/{len(self.sweep_data_2d)})"
            self.step_value_label.config(text=label_text)
            self.step_value_label_2d.config(text=f"{step_val:.4g}")
        else:
            self.step_value_label.config(text=f"Sweep ({self.current_step_index + 1}/{len(self.sweep_data_2d)})")
            self.step_value_label_2d.config(text="")
    
    def on_step_slider_changed(self, value):
        """Handle step slider movement."""
        new_index = int(float(value))
        if new_index != self.current_step_index:
            self.current_step_index = new_index
            self.update_step_label()
            self.update_single_trace()
    
    def prev_step(self):
        """Go to previous step."""
        if self.current_step_index > 0:
            self.current_step_index -= 1
            self.step_slider_var.set(self.current_step_index)
            self.update_step_label()
            self.update_single_trace()
    
    def next_step(self):
        """Go to next step."""
        if self.current_step_index < len(self.sweep_data_2d) - 1:
            self.current_step_index += 1
            self.step_slider_var.set(self.current_step_index)
            self.update_step_label()
            self.update_single_trace()
    
    def update_single_trace(self):
        """Update the single trace viewer."""
        # Save current axis limits if user has zoomed
        save_xlim = self.ax_trace.get_xlim() if hasattr(self, '_trace_has_data') and self._trace_has_data else None
        save_ylim = self.ax_trace.get_ylim() if hasattr(self, '_trace_has_data') and self._trace_has_data else None
        
        self.ax_trace.clear()
        
        if not self.sweep_data_2d:
            self._trace_has_data = False
            self.ax_trace.set_title("Single Trace Viewer")
            self.ax_trace.text(0.5, 0.5, "No data available",
                              ha='center', va='center', transform=self.ax_trace.transAxes,
                              fontsize=8, color='gray')
            self.ax_trace.grid(True, alpha=0.3)
            self.fig_trace.tight_layout()
            self.canvas_trace.draw_idle()
            return
        
        self._trace_has_data = True
        sweep_param = self.current_config['sweep_param'] if self.current_config else "Sweep"
        step_param = self.current_config.get('step_param') if self.current_config else None
        mode = self.trace_display_mode.get()
        xaxis_mode = self.xaxis_mode.get()
        
        # Determine y-axis label based on mode
        if mode == "Magnitude":
            ylabel = "|S21| (dB)"
        elif mode == "Phase":
            ylabel = "Phase (deg)"
        elif mode == "Real":
            ylabel = "Re(S21)"
        elif mode == "Conductivity":
            ylabel = "sigma (e^2/h)"
        else:
            ylabel = "Im(S21)"
        
        # Get frequency scaling for x-axis if sweeping frequency
        freq_scale, freq_unit = self._get_frequency_scale()
        
        # Get x-axis conversion function and label
        x_convert, xlabel = self._get_xaxis_conversion(sweep_param, freq_scale, freq_unit, xaxis_mode)
        
        if self.show_all_traces.get():
            # Show all traces with color gradient
            n_traces = len(self.sweep_data_2d)
            colors = cm.viridis(np.linspace(0, 1, max(n_traces, 1)))
            
            for trace_idx, trace_data in enumerate(self.sweep_data_2d):
                if not trace_data:
                    continue
                
                sweep_vals = [x_convert(d['sweep_value']) for d in trace_data]
                y = self._get_trace_y_data(trace_data, mode)
                
                # Highlight current step
                alpha = 1.0 if trace_idx == self.current_step_index else 0.3
                lw = 1.0 if trace_idx == self.current_step_index else 0.4
                
                self.ax_trace.plot(sweep_vals, y, '-', color=colors[trace_idx], 
                                  alpha=alpha, linewidth=lw)
            
            self.ax_trace.set_title(f"All Traces ({n_traces} total, current highlighted)")
        else:
            # Show only selected trace
            if self.current_step_index < len(self.sweep_data_2d):
                trace_data = self.sweep_data_2d[self.current_step_index]
                
                if trace_data:
                    sweep_vals = [x_convert(d['sweep_value']) for d in trace_data]
                    y = self._get_trace_y_data(trace_data, mode)
                    
                    # Check if we're in active averaging mode
                    avg_total = getattr(self, 'current_avg_total', 1)
                    avg_num = getattr(self, 'current_avg_num', 1)
                    is_averaging = avg_total > 1 and avg_num > 0
                    
                    if is_averaging and avg_num > 1:
                        # During averaging: show full averaged data + current sweep
                        
                        # Plot full averaged data (use sweep_data which has running average)
                        self.ax_trace.plot(sweep_vals, y, 'b-', linewidth=0.5, 
                                          label=f'Avg (1-{avg_num})', alpha=0.9)
                        
                        # Plot current sweep raw data (thinner, orange)
                        if hasattr(self, 'current_sweep_raw') and self.current_sweep_raw:
                            raw_data = [d for d in self.current_sweep_raw if d is not None]
                            if raw_data:
                                raw_sweep_vals = [x_convert(d['sweep_value']) for d in raw_data]
                                raw_y = self._get_trace_y_data(raw_data, mode)
                                self.ax_trace.plot(raw_sweep_vals, raw_y, '-', color='orange', 
                                                  linewidth=0.4, label=f'Sweep {avg_num}', alpha=0.6)
                        
                        self.ax_trace.legend(loc='upper right', fontsize=8)
                    else:
                        # Normal display (single sweep or first sweep of averaging)
                        self.ax_trace.plot(sweep_vals, y, 'b-', linewidth=0.5)
                    
                    # Build title with sweep details
                    title_parts = []
                    step_val = trace_data[0].get('step_value')
                    if step_val is not None and step_param:
                        title_parts.append(f"{step_param} = {step_val:.4g}")
                    
                    # Add fixed parameter info from config
                    if self.current_config:
                        fixed = self.current_config.get('fixed_values', {})
                        details = []
                        
                        # Show relevant fixed parameters (not the sweep parameter)
                        if sweep_param != "Frequency (GHz)":
                            freq = fixed.get('frequency', 0)
                            if freq >= 1e9:
                                details.append(f"f={freq/1e9:.2f}GHz")
                            elif freq >= 1e6:
                                details.append(f"f={freq/1e6:.1f}MHz")
                        
                        if sweep_param != "B-Field (T)":
                            b_field = fixed.get('b_field', 0)
                            if b_field != 0:
                                details.append(f"B={b_field:.2f}T")
                        
                        if sweep_param != "Power (dBm)":
                            power = fixed.get('power', -50)
                            # Show power at probe (adjusted for input attenuation)
                            try:
                                input_atten = float(self.input_attenuation.get())
                                power_at_probe = power - input_atten
                                details.append(f"P={power_at_probe:.0f}dBm@probe")
                            except:
                                details.append(f"P={power:.0f}dBm")
                        
                        if sweep_param != "Gate Voltage (V)":
                            vg = fixed.get('vg', 0)
                            if vg != 0:
                                details.append(f"Vg={vg:.1f}V")
                        
                        ifbw = fixed.get('ifbw', 100)
                        details.append(f"IFBW={ifbw:.0f}Hz")
                        
                        if details:
                            title_parts.append(", ".join(details))
                    
                    # Add averaging info if active
                    averages = self.current_config.get('averages', 1) if self.current_config else 1
                    if averages > 1:
                        if is_averaging:
                            title_parts.append(f"avg {avg_num}/{averages}")
                        else:
                            title_parts.append(f"avg={averages}")
                    
                    # Add smoothing info if active
                    smooth_window = self.smoothing_window.get()
                    if smooth_window > 1:
                        title_parts.append(f"smooth={smooth_window}pt")
                    
                    title = " | ".join(title_parts) if title_parts else "Single Trace"
                    self.ax_trace.set_title(title, fontsize=9)
        
        # xlabel was set by _get_xaxis_conversion
        self.ax_trace.set_xlabel(xlabel)
        self.ax_trace.set_ylabel(ylabel)
        self.ax_trace.grid(True, alpha=0.3)
        
        # Custom tick formatting for filling factor mode
        if xaxis_mode == "Filling Factor (nu)" and sweep_param == "Gate Voltage (V)":
            self._set_filling_factor_ticks(self.ax_trace)
            # Custom cursor format for filling factor
            self.ax_trace.format_coord = lambda x, y: f'nu = {x:.2f}, S21 = {y:.4f} dB'
        elif xaxis_mode == "Density (cm^-2)" and sweep_param == "Gate Voltage (V)":
            self.ax_trace.format_coord = lambda x, y: f'n = {x:.3e} cm^-2, S21 = {y:.4f} dB'
        elif sweep_param == "Gate Voltage (V)":
            self.ax_trace.format_coord = lambda x, y: f'Vg = {x:.2f} V, S21 = {y:.4f} dB'
        else:
            # Default format
            self.ax_trace.format_coord = lambda x, y: f'x = {x:.4f}, y = {y:.4f}'
        
        # Restore zoom if user had zoomed (check autoscale setting)
        if not getattr(self, 'trace_autoscale', True) and save_xlim and save_ylim:
            self.ax_trace.set_xlim(save_xlim)
            self.ax_trace.set_ylim(save_ylim)
        
        self.fig_trace.tight_layout()
        self.canvas_trace.draw_idle()
        
        # Also update the secondary trace in 2D mode
        if self.current_plot_mode == "2D" and self.current_step_index < len(self.sweep_data_2d):
            self.ax_trace_2d.clear()
            trace_data = self.sweep_data_2d[self.current_step_index]
            
            if trace_data:
                sweep_vals = [x_convert(d['sweep_value']) for d in trace_data]
                y = self._get_trace_y_data(trace_data, mode)
                self.ax_trace_2d.plot(sweep_vals, y, 'b-', linewidth=0.8)
                self.ax_trace_2d.set_xlabel(xlabel, fontsize=8)
                self.ax_trace_2d.set_ylabel(ylabel, fontsize=8)
                self.ax_trace_2d.tick_params(labelsize=7)
                self.ax_trace_2d.grid(True, alpha=0.3)
                
                # Custom tick formatting for filling factor
                if xaxis_mode == "Filling Factor (nu)" and sweep_param == "Gate Voltage (V)":
                    self._set_filling_factor_ticks(self.ax_trace_2d, fontsize=7)
                    self.ax_trace_2d.format_coord = lambda x, y: f'nu = {x:.2f}, S21 = {y:.4f} dB'
            
            self.fig_trace_2d.tight_layout()
            self.canvas_trace_2d.draw_idle()
    
    def _get_frequency_scale(self):
        """Determine appropriate frequency scale and unit based on data range.
        
        Returns:
            (scale_factor, unit_string) where scale_factor converts Hz to display units
        """
        if not self.current_config:
            return 1.0, None
        
        sweep_param = self.current_config.get('sweep_param', '')
        if sweep_param != "Frequency (GHz)":
            return 1.0, None
        
        # Get frequency range in Hz
        f_start = self.current_config.get('sweep_start', 0)
        f_stop = self.current_config.get('sweep_stop', 0)
        f_max = max(abs(f_start), abs(f_stop))
        f_min = min(abs(f_start), abs(f_stop))
        
        # Choose unit based on the range
        if f_max >= 1e9:  # 1 GHz or above -> use GHz
            return 1e-9, "GHz"
        elif f_max >= 1e6:  # 1 MHz or above -> use MHz
            return 1e-6, "MHz"
        elif f_max >= 1e3:  # 1 kHz or above -> use kHz
            return 1e-3, "kHz"
        else:
            return 1.0, "Hz"
    
    def _get_param_display_label(self, param):
        """Convert parameter name to display label.
        
        For power, shows 'Power at Probe (dBm)' to indicate it's adjusted for attenuation.
        """
        if param == "Power (dBm)":
            return "Power at Probe (dBm)"
        return param
    
    def _get_power_at_probe_offset(self):
        """Get the input attenuation for converting VNA power to power at probe."""
        try:
            return float(self.input_attenuation.get())
        except:
            return 0.0
    
    def _get_xaxis_conversion(self, sweep_param, freq_scale, freq_unit, xaxis_mode):
        """Get x-axis conversion function and label based on sweep type and display mode.
        
        Returns:
            (convert_function, xlabel_string)
        """
        # For frequency sweeps, always use frequency
        if sweep_param == "Frequency (GHz)":
            xlabel = f"Frequency ({freq_unit})" if freq_unit else "Frequency"
            return lambda v: v * freq_scale, xlabel
        
        # For power sweeps, convert to power at probe
        if sweep_param == "Power (dBm)":
            input_atten = self._get_power_at_probe_offset()
            return lambda v: v - input_atten, "Power at Probe (dBm)"
        
        # For gate voltage sweeps, check x-axis mode
        if sweep_param == "Gate Voltage (V)":
            if xaxis_mode == "Density (cm^-2)":
                # Convert to density
                if hasattr(self, 'sample_density_per_volt') and self.sample_density_per_volt is not None:
                    try:
                        v_cnp = float(self.v_cnp.get())
                        density_per_v = self.sample_density_per_volt
                        # Format label with appropriate scale
                        if abs(density_per_v) > 1e11:
                            return lambda v: density_per_v * (v - v_cnp) / 1e12, "Density (x10^12 cm^-2)"
                        else:
                            return lambda v: density_per_v * (v - v_cnp) / 1e10, "Density (x10^10 cm^-2)"
                    except:
                        pass
                # Fall back to gate voltage
                return lambda v: v, "Gate Voltage (V) [set hBN params]"
            
            elif xaxis_mode == "Filling Factor (nu)":
                # Convert to filling factor - need B field
                if hasattr(self, 'sample_density_per_volt') and self.sample_density_per_volt is not None:
                    try:
                        v_cnp = float(self.v_cnp.get())
                        b_field = self.current_config.get('fixed_values', {}).get('b_field', 0)
                        if b_field != 0:
                            density_per_v = self.sample_density_per_volt
                            # nu = n * h / (e * B)
                            h = 6.626e-34
                            e = 1.602e-19
                            # n in m^-2 = density_per_v * 1e4 * (v - v_cnp)
                            # nu = n * h / (e * B)
                            factor = density_per_v * 1e4 * h / (e * b_field)
                            return lambda v, f=factor, vc=v_cnp: f * (v - vc), f"Filling Factor nu (B={b_field:.1f}T)"
                    except:
                        pass
                # Fall back to gate voltage
                return lambda v: v, "Gate Voltage (V) [need B>0]"
        
        # Default: use raw sweep value with display label conversion
        return lambda v: v * freq_scale, self._get_param_display_label(sweep_param)
    
    def _get_trace_y_data(self, trace_data, mode, apply_normalization=True):
        """Extract y-axis data from trace based on display mode.
        
        Args:
            trace_data: List of data point dictionaries
            mode: Display mode (Magnitude, Normalized, Phase, Real, Imaginary, Conductivity)
            apply_normalization: If True, apply V_norm normalization if enabled
            
        Returns:
            List of y values
        """
        if mode == "Magnitude":
            y = [20 * np.log10(d['s21_mag'] + 1e-12) for d in trace_data]
        elif mode == "Normalized":
            # Use pre-measured reference normalization (from reference measurement at start)
            # Check if normalized data is available in the data points
            if trace_data and 's21_mag_db_norm' in trace_data[0] and trace_data[0]['s21_mag_db_norm'] is not None:
                y = [d.get('s21_mag_db_norm', 0) for d in trace_data]
            else:
                # Fall back to raw magnitude if no normalization data available
                y = [20 * np.log10(d['s21_mag'] + 1e-12) for d in trace_data]
                # Show warning in status
                if hasattr(self, 'status_var'):
                    self.status_var.set("No normalization reference - showing raw magnitude")
        elif mode == "Phase":
            y = [d['s21_phase'] for d in trace_data]
        elif mode == "Real":
            y = [d['s21_real'] for d in trace_data]
        elif mode == "Conductivity":
            y = self._calculate_conductivity(trace_data)
        else:  # Imaginary
            y = [d['s21_imag'] for d in trace_data]
        
        # Apply V_norm normalization if enabled (separate from "Normalized" mode)
        if apply_normalization and mode != "Normalized" and self.normalize_at_v.get():
            ref_value, success = self._get_normalization_reference(trace_data, mode)
            if success and ref_value != 0.0:
                y = [val - ref_value for val in y]
            elif not success:
                # Update status to show normalization pending
                if hasattr(self, 'norm_status_label'):
                    try:
                        v_norm = float(self.v_norm.get())
                        self.norm_status_label.config(text=f"(waiting for V={v_norm:.1f})")
                    except:
                        pass
        
        # Apply smoothing if enabled
        window = self.smoothing_window.get()
        if window > 1 and len(y) >= window:
            y = self._smooth_data(y, window)
        
        return y
    
    def _calculate_conductivity(self, trace_data):
        """Calculate conductivity from S21 data.
        
        Uses formula: sigma = -ln(S21/S21_max) x w / (2 x L x Z0)
        
        Where:
            S21_max is used as reference (sigma=0 point)
            w = slot width (um, converted to m)
            L = slot length (um, converted to m)
            Z0 = 50Ohm characteristic impedance
        
        Returns conductivity in units of e^2/h (quantum conductance units)
        """
        if not trace_data:
            return []
        
        try:
            w_um = float(self.cpw_slot_width.get())
            L_um = float(self.cpw_slot_length.get())
        except ValueError:
            w_um = 10.0
            L_um = 100.0
        
        # Convert to meters
        w = w_um * 1e-6
        L = L_um * 1e-6
        Z0 = 50.0  # Ohms
        
        # Quantum of conductance e^2/h ~ 3.874e-5 S
        e2_over_h = 3.87405e-5  # Siemens
        
        # Get S21 magnitudes (linear)
        s21_mags = np.array([d['s21_mag'] for d in trace_data])
        
        # Use maximum S21 as reference (P0 where sigma~0)
        s21_max = np.max(s21_mags)
        
        # Avoid division by zero or log of zero
        s21_mags = np.clip(s21_mags, 1e-12, None)
        s21_max = max(s21_max, 1e-12)
        
        # Calculate conductivity: sigma = -ln(S21/S21_max) x w / (2 x L x Z0)
        # Negative sign because S21 < S21_max gives negative ln, but sigma should be positive
        # The formula gives sheet conductivity in Siemens
        sigma_S = -np.log(s21_mags / s21_max) * w / (2 * L * Z0)
        
        # Convert to units of e^2/h
        sigma_e2h = sigma_S / e2_over_h
        
        return sigma_e2h.tolist()
    
    def _smooth_data(self, data, window):
        """Apply moving average smoothing to data.
        
        Args:
            data: List or array of values
            window: Window size (should be odd)
        
        Returns:
            Smoothed data array
        """
        # Ensure window is odd
        if window % 2 == 0:
            window += 1
        
        # Use numpy convolution for efficient moving average
        data = np.array(data)
        kernel = np.ones(window) / window
        
        # 'same' mode keeps output same length, but edges will be affected
        # Use 'valid' and pad, or just use 'same' for simplicity
        smoothed = np.convolve(data, kernel, mode='same')
        
        # Fix edge effects by using original values at edges
        half = window // 2
        smoothed[:half] = data[:half]
        smoothed[-half:] = data[-half:]
        
        return smoothed.tolist()
    
    def _set_smoothing(self, value):
        """Set smoothing window and update plots."""
        self.smoothing_window.set(value)
        self._update_all_plots()
    
    def _update_all_plots(self):
        """Update both single trace and 2D plot."""
        self.update_single_trace()
        if self.current_plot_mode == "2D":
            self.update_2d_plot()
    
    def _set_filling_factor_ticks(self, ax, fontsize=10):
        """Set integer ticks for filling factor axis with QH sequence (0, +/-2, +/-6, +/-10,...) in bold."""
        # Get current x limits
        xlim = ax.get_xlim()
        xmin, xmax = min(xlim), max(xlim)
        
        # Generate integer ticks - all integers in range
        tick_min = int(np.floor(xmin))
        tick_max = int(np.ceil(xmax))
        
        # Show all integers by default
        x_range = tick_max - tick_min
        if x_range <= 30:
            step = 1  # Every integer
        elif x_range <= 60:
            step = 2  # Every 2
        else:
            step = 4  # Every 4
        
        # Generate ticks
        ticks = list(range(tick_min, tick_max + 1, step))
        
        # Make sure we have 0 if in range
        if tick_min <= 0 <= tick_max and 0 not in ticks:
            ticks.append(0)
            ticks.sort()
        
        ax.set_xticks(ticks)
        
        # QH sequence for graphene: 0, +/-2, +/-6, +/-10, +/-14, +/-18, +/-22, ... (4n+2 and 0)
        qh_values = {0}
        for n in range(20):
            val = 4 * n + 2
            qh_values.add(val)
            qh_values.add(-val)
        
        # Create tick labels - bold only for QH values
        labels = []
        for t in ticks:
            if t in qh_values:
                labels.append(f'$\\mathbf{{{t}}}$')  # Bold via LaTeX
            else:
                labels.append(str(t))
        
        ax.set_xticklabels(labels, fontsize=fontsize)
    
    # ===== Event Handlers =====
    
    def toggle_simulation(self):
        """Toggle simulation mode."""
        is_sim = self.simulation_mode.get()
        self.measurement_engine.use_simulation = is_sim
        
        if is_sim:
            self.sim_indicator.config(text="[SIMULATION MODE]", foreground='blue')
        else:
            self.sim_indicator.config(text="[LIVE MODE]", foreground='green')
    
    def launch_s2vna(self):
        """Launch the S2VNA software."""
        # Common installation paths for S2VNA
        possible_paths = [
            r"C:\Program Files\Copper Mountain Technologies\S2VNA\S2VNA.exe",
            r"C:\Program Files (x86)\Copper Mountain Technologies\S2VNA\S2VNA.exe",
            r"C:\CMT\S2VNA\S2VNA.exe",
        ]
        
        # Check which path exists
        s2vna_path = None
        for path in possible_paths:
            if os.path.exists(path):
                s2vna_path = path
                break
        
        if s2vna_path is None:
            # Ask user to locate it
            s2vna_path = filedialog.askopenfilename(
                title="Locate S2VNA.exe",
                filetypes=[("Executable", "*.exe"), ("All files", "*.*")],
                initialdir=r"C:\Program Files"
            )
            if not s2vna_path:
                return
        
        try:
            subprocess.Popen([s2vna_path], shell=False)
            self.status_var.set("S2VNA launched - enable socket server then connect")
            messagebox.showinfo(
                "S2VNA Launched",
                "S2VNA is starting.\n\n"
                "To enable remote control:\n"
                "1. In S2VNA, go to System ->' Misc Setup\n"
                "2. Click 'Network Setup'\n"
                "3. Check 'Enable Socket Server'\n"
                "4. Port should be 5025\n"
                "5. Click OK, then Connect in this program"
            )
        except Exception as e:
            messagebox.showerror("Launch Error", f"Failed to launch S2VNA: {e}")
    
    def connect_vna(self):
        """Connect to VNA."""
        if self.simulation_mode.get():
            self.vna_connected.set(True)
            self.vna_status.config(foreground='green')
            self.status_var.set("VNA connected (simulation)")
        else:
            try:
                # Update port from GUI
                self.vna.set_port(self.vna_port.get())
                
                self.status_var.set("Connecting to VNA...")
                self.root.update_idletasks()
                
                if self.vna.connect():
                    self.vna_connected.set(True)
                    self.vna_status.config(foreground='green')
                    self.status_var.set("VNA connected")
                else:
                    self.vna_status.config(foreground='red')
                    messagebox.showerror(
                        "Connection Error", 
                        "Failed to connect to VNA.\n\n"
                        "Make sure:\n"
                        "1. S2VNA software is running\n"
                        "2. Socket server is enabled (System ->' Misc Setup ->' Network Setup)\n"
                        "3. Port matches (default: 5025)"
                    )
            except Exception as e:
                self.vna_status.config(foreground='red')
                messagebox.showerror("Connection Error", f"VNA connection failed: {e}")
    
    def connect_magnet(self):
        """Connect to magnet controller."""
        # Create appropriate controller based on model selection
        model = self.magnet_model.get()
        self.magnet = create_magnet_controller(model)
        
        if self.simulation_mode.get():
            self.magnet_connected.set(True)
            self.magnet_status.config(foreground='green')
            self.status_var.set(f"{model} connected (simulation)")
            # Update measurement engine reference
            self.measurement_engine.magnet = self.magnet
            # Show B-field in fixed params
            self.update_bfield_visibility()
        else:
            try:
                self.magnet.set_address(self.magnet_addr.get())
                if self.magnet.connect():
                    self.magnet_connected.set(True)
                    self.magnet_status.config(foreground='green')
                    self.status_var.set(f"{model} connected")
                    # Update measurement engine reference
                    self.measurement_engine.magnet = self.magnet
                    # Show B-field in fixed params
                    self.update_bfield_visibility()
                    
                    # For SCM1, show reminder about LabVIEW mode
                    if model == "SCM1":
                        messagebox.showinfo(
                            "SCM1 Connected",
                            "SCM1 magnet controller connected.\n\n"
                            "IMPORTANT: Ensure the LabVIEW VI on the data PC is set to:\n"
                            "* Mode: 'Ramp to Setpoint' (not 'Hold' or 'Ramp to Zero')\n"
                            "* Pause: OFF\n\n"
                            "The software will send ramp commands, but the LabVIEW mode\n"
                            "must be correct for the magnet to actually ramp."
                        )
                else:
                    messagebox.showerror("Connection Error", f"Failed to connect to {model}")
            except Exception as e:
                messagebox.showerror("Connection Error", f"{model} connection failed: {e}")
    
    def on_magnet_model_change(self, event=None):
        """Handle magnet model selection change."""
        model = self.magnet_model.get()
        
        # Update address label and default based on model
        if model == "SCM1":
            self.magnet_addr_label.config(text="IP:")
            self.magnet_addr_entry.config(width=26)
            self.magnet_addr.set("scm1datapc.ad.magnet.fsu.edu")
        elif model == "Cryomagnetics 4G":
            self.magnet_addr_label.config(text="GPIB::")
            self.magnet_addr_entry.config(width=4)
            self.magnet_addr.set("21")
        
        # Disconnect existing if connected
        if self.magnet_connected.get():
            self.magnet.disconnect()
            self.magnet_connected.set(False)
            self.magnet_status.config(foreground='gray')
            self.status_var.set(f"Switched to {model} - reconnect required")
    
    def connect_keithley(self):
        """Connect to Keithley SMU."""
        # Set the model first
        self.keithley.set_model(self.keithley_model.get())
        
        # Get compliance from GUI
        try:
            compliance = float(self.gate_compliance.get()) * 1e-9  # nA to A
        except ValueError:
            compliance = 100e-9
        self.keithley.compliance_current = compliance
        
        if self.simulation_mode.get():
            self.keithley_connected.set(True)
            self.keithley_status.config(foreground='green')
            self.status_var.set(f"Keithley {self.keithley_model.get()} connected (simulation)")
        else:
            try:
                self.keithley.set_address(self.keithley_addr.get())
                if self.keithley.connect():
                    self.keithley_connected.set(True)
                    self.keithley_status.config(foreground='green')
                    self.status_var.set(f"Keithley {self.keithley_model.get()} connected")
                    print(f"Compliance set to {compliance*1e9:.0f} nA")
                    
                    # Force 2-wire mode and matching sense range (critical for speed!)
                    self.keithley.instrument.write(':SYST:RSEN OFF')
                    self.keithley.instrument.write(f':SENS:CURR:RANG {compliance}')
                    print(f"2-wire mode enabled, sense range = {compliance*1e9:.0f} nA")
                    
                    # Clear any errors that accumulated during setup
                    try:
                        self.keithley.instrument.write('*CLS')  # Clear status
                        time.sleep(0.1)
                        # Read and discard any errors in queue
                        for _ in range(5):  # Quick check
                            err = self.keithley.instrument.query(':SYST:ERR?')
                            if '0,' in err or 'No error' in err:
                                break
                            print(f"  Cleared Keithley error: {err.strip()}")
                    except Exception as e:
                        print(f"  Error queue clear failed: {e}")
                    print("Keithley ready")
                else:
                    messagebox.showerror("Connection Error", 
                        f"Failed to connect to Keithley {self.keithley_model.get()}\n\n"
                        "Check:\n"
                        "1. GPIB address is correct\n"
                        "2. Instrument is powered on\n"
                        "3. GPIB cable is connected")
            except Exception as e:
                error_msg = str(e)
                if "VISA" in error_msg or "library" in error_msg.lower():
                    messagebox.showerror("VISA Backend Missing", 
                        "No VISA backend found.\n\n"
                        "Please install one of:\n"
                        "1. NI-VISA from ni.com/visa\n"
                        "2. pyvisa-py: pip install pyvisa-py gpib-ctypes\n\n"
                        "For GPIB support, you also need a GPIB adapter driver.")
                else:
                    messagebox.showerror("Connection Error", f"Keithley connection failed:\n{e}")
    
    def connect_temp(self):
        """Connect to Lakeshore 370 temperature monitor - DISABLED."""
        # Temperature controller disabled due to GPIB bus conflicts with Keithley
        messagebox.showinfo("Disabled", 
            "Temperature monitoring is currently disabled.\n\n"
            "The Lakeshore 370 causes GPIB bus conflicts with the Keithley.\n"
            "Please physically disconnect the Lakeshore 370 GPIB cable\n"
            "when using the Keithley for gate sweeps.")
    
    def update_temp_display(self):
        """Update temperature display - DISABLED."""
        # Temperature controller disabled
        if hasattr(self, 'temp_display'):
            self.temp_display.config(text="DISABLED", foreground='gray')
        if hasattr(self, 'fixed_temp_display'):
            self.fixed_temp_display.config(text="N/A", foreground='gray')
    
    def browse_directory(self):
        """Browse for data directory."""
        directory = filedialog.askdirectory(initialdir=self.data_directory.get())
        if directory:
            self.data_directory.set(directory)
    
    def manual_gate_ramp(self):
        """Manually ramp gate voltage to specified value."""
        try:
            target = float(self.manual_gate_entry.get())
            max_v = float(self.max_gate_voltage.get())
            slew = float(self.gate_slew_rate.get())
            
            # Clamp to max voltage
            if abs(target) > max_v:
                target = max_v if target > 0 else -max_v
                self.manual_gate_entry.delete(0, tk.END)
                self.manual_gate_entry.insert(0, str(target))
                print(f"Target clamped to +/-{max_v}V")
            
            # Update Keithley settings
            self.keithley.max_voltage = max_v
            self.keithley.slew_rate = slew
            
            # Perform ramp in background thread
            def ramp_thread():
                self.status_var.set(f"Ramping gate to {target}V...")
                success = self.keithley.ramp_to_voltage(target, slew)
                if success:
                    self.status_var.set(f"Gate voltage set to {target}V")
                else:
                    self.status_var.set("Gate ramp failed or interrupted")
                self.update_gate_display()
            
            threading.Thread(target=ramp_thread, daemon=True).start()
            
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid voltage value.")
    
    def ramp_gate_to_zero_now(self):
        """Immediately ramp gate voltage to zero."""
        try:
            slew = float(self.gate_slew_rate.get())
            self.keithley.slew_rate = slew
            
            def ramp_thread():
                current = self.keithley.get_voltage()
                self.status_var.set(f"Ramping gate from {current:.3f}V to 0V...")
                success = self.keithley.ramp_to_voltage(0.0, slew)
                if success:
                    self.status_var.set("Gate voltage at 0V")
                else:
                    self.status_var.set(f"Gate ramp stopped at {self.keithley.get_voltage():.3f}V")
                self.update_gate_display()
            
            threading.Thread(target=ramp_thread, daemon=True).start()
            
        except ValueError:
            messagebox.showerror("Invalid Input", "Invalid slew rate setting.")
    
    def calculate_time_estimate(self):
        """Calculate and display estimated measurement time."""
        try:
            sweep_param = self.sweep_param.get()
            step_param = self.step_param.get()
            
            sweep_start = float(self.sweep_start.get())
            sweep_stop = float(self.sweep_stop.get())
            sweep_points = int(self.sweep_points.get())
            
            step_points = 1
            step_start = 0
            step_stop = 0
            if step_param != "None":
                step_start = float(self.step_start.get())
                step_stop = float(self.step_stop.get())
                step_points = int(self.step_points.get())
            
            # Get timing parameters
            ifbw = float(self.ifbw.get())
            field_rate = float(self.field_ramp_rate.get())  # T/min
            field_rate = max(0.001, min(field_rate, 0.3))  # Clamp to SCM1 limits (0.001 min to avoid div by 0)
            gate_slew = float(self.gate_slew_rate.get())  # V/s
            field_settle = float(self.field_settle_time.get())  # s
            
            total_time = 0.0
            
            # VNA measurement time per point (rough estimate: 1/IFBW + overhead)
            time_per_vna_point = 1.0 / ifbw + 0.05  # seconds
            
            # Calculate based on sweep type
            if sweep_param == "Frequency (GHz)":
                # Frequency sweep is fast (VNA handles it)
                vna_sweep_time = sweep_points * time_per_vna_point
                total_time = vna_sweep_time * step_points
                
                # Add time for stepping parameter
                if step_param == "B-Field (T)":
                    field_range = abs(step_stop - step_start)
                    field_time = (field_range / field_rate) * 60  # Convert T/min to seconds
                    total_time += field_time + field_settle * step_points
                elif step_param == "Gate Voltage (V)":
                    gate_range = abs(step_stop - step_start)
                    gate_time = gate_range / gate_slew
                    total_time += gate_time
                    
            elif sweep_param == "B-Field (T)":
                # B-field continuous sweep - time is dominated by field ramp
                field_range = abs(sweep_stop - sweep_start)
                field_time = (field_range / field_rate) * 60  # seconds
                # Total time is just the field ramp time (VNA measures continuously during ramp)
                total_time = field_time * step_points
                
                # Add settle time at each step
                total_time += field_settle * step_points
                
                # Add time for stepping parameter
                if step_param == "Gate Voltage (V)":
                    gate_range = abs(step_stop - step_start)
                    gate_time = gate_range / gate_slew
                    total_time += gate_time
                    
            elif sweep_param == "Gate Voltage (V)":
                # Gate sweep
                gate_range = abs(sweep_stop - sweep_start)
                gate_time = gate_range / gate_slew
                vna_time = sweep_points * time_per_vna_point
                total_time = (gate_time + vna_time) * step_points
                
                # Add time for stepping B-field
                if step_param == "B-Field (T)":
                    field_range = abs(step_stop - step_start)
                    field_time = (field_range / field_rate) * 60
                    total_time += field_time + field_settle * step_points
            else:
                # Generic estimate
                total_time = sweep_points * step_points * time_per_vna_point
            
            # Add 10% overhead for communication, etc.
            total_time *= 1.1
            
            # Format nicely
            if total_time < 60:
                time_str = f"{total_time:.0f} seconds"
            elif total_time < 3600:
                minutes = total_time / 60
                time_str = f"{minutes:.1f} minutes"
            else:
                hours = total_time / 3600
                time_str = f"{hours:.1f} hours"
            
            self.time_estimate_display.config(text=time_str)
            self.status_var.set(f"Estimated time: {time_str}")
            
        except Exception as e:
            self.time_estimate_display.config(text="Error")
            self.status_var.set(f"Time estimate error: {e}")
    
    def update_field_display(self):
        """Update the current B-field display."""
        try:
            if self.magnet_connected.get():
                field = self.magnet.get_field()
                self.current_field_display.config(text=f"{field:.3f} T")
        except:
            pass
    
    def update_gate_display(self):
        """Update the current gate voltage display."""
        try:
            voltage = self.keithley.get_voltage()
            self.current_gate_display.config(text=f"{voltage:.3f} V")
        except:
            pass
    
    def clear_log_display(self):
        """Clear the log display widget."""
        log_manager.clear_display()
    
    def save_log_manually(self):
        """Manually save log to a file."""
        # Ask user for location
        filepath = filedialog.asksaveasfilename(
            title="Save Log File",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialdir=self.data_directory.get(),
            initialfile=f"measurement_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        if filepath:
            if log_manager.save_to_file(filepath):
                self.status_var.set(f"Log saved to {os.path.basename(filepath)}")
            else:
                messagebox.showerror("Save Error", "Failed to save log file")
    
    def update_log_display(self):
        """Process queued log messages and update display.
        
        Call this from the main update loop.
        """
        log_manager.update_display()
    
    def update_power_at_probe(self):
        """Update the displayed power at probe based on VNA power and input attenuation."""
        try:
            vna_power = float(self.fixed_power.get())
            input_atten = float(self.input_attenuation.get())
            power_at_probe = vna_power - input_atten
            self.power_at_probe_label.config(text=f"{power_at_probe:.1f} dBm")
        except (ValueError, AttributeError):
            if hasattr(self, 'power_at_probe_label'):
                self.power_at_probe_label.config(text="-- dBm")
    
    def update_sample_calculations(self, *args):
        """Update calculated sample parameters from hBN thickness."""
        try:
            d_nm = float(self.hbn_thickness.get())
            if d_nm <= 0:
                raise ValueError("Thickness must be positive")
            
            # Constants
            epsilon_0 = 8.854e-12  # F/m
            epsilon_hbn = 3.5  # relative permittivity of hBN
            e = 1.602e-19  # C
            
            # Capacitance per area: C/A = epsilon0 * epsilon_r / d
            d_m = d_nm * 1e-9  # convert to meters
            cap_per_area = epsilon_0 * epsilon_hbn / d_m  # F/m^2
            
            # Density per volt: n = C/A * V / e
            # n [m^-2] = cap_per_area * 1V / e
            # n [cm^-2] = n [m^-2] / 1e4
            density_per_volt = cap_per_area / e / 1e4  # cm^-2 per volt
            
            # Display compact info
            density_str = f"{density_per_volt/1e10:.2f}x10^10 cm^-2/V"
            self.density_per_volt_display.config(text=density_str)
            
            # Store for use in plotting
            self.sample_density_per_volt = density_per_volt
            
            # Trigger replot if in density/filling mode
            if hasattr(self, 'xaxis_mode') and self.xaxis_mode.get() != "Gate Voltage (V)":
                self.on_xaxis_mode_changed()
            
        except (ValueError, ZeroDivisionError):
            self.density_per_volt_display.config(text="--")
            self.sample_density_per_volt = None
    
    def gate_to_density(self, v_gate):
        """Convert gate voltage to carrier density in cm^-2."""
        try:
            v_cnp = float(self.v_cnp.get())
            if self.sample_density_per_volt is not None:
                return self.sample_density_per_volt * (v_gate - v_cnp)
        except:
            pass
        return None
    
    def gate_to_filling_factor(self, v_gate, b_field):
        """Convert gate voltage to filling factor at given B field."""
        density = self.gate_to_density(v_gate)
        if density is None or b_field == 0:
            return None
        
        # nu = n * h / (e * B)
        # where n is in m^-2, B in Tesla
        h = 6.626e-34  # J*s
        e = 1.602e-19  # C
        
        # Convert density from cm^-2 to m^-2
        n_m2 = density * 1e4
        
        nu = n_m2 * h / (e * b_field)
        return nu
    
    def on_sweep_param_changed(self, event=None):
        """Handle sweep parameter selection change."""
        sweep = self.sweep_param.get()
        
        # Set sensible defaults based on parameter type
        if sweep == "Frequency (GHz)":
            self.sweep_start.set("0.0001")
            self.sweep_stop.set("18")
            self.sweep_points.set("1001")
        elif sweep == "Power (dBm)":
            self.sweep_start.set("-50")
            self.sweep_stop.set("0")
            self.sweep_points.set("51")
        elif sweep == "B-Field (T)":
            self.sweep_start.set("0")
            self.sweep_stop.set("0.5")
            self.sweep_points.set("101")
        elif sweep == "Gate Voltage (V)":
            self.sweep_start.set("0")
            self.sweep_stop.set("10")
            self.sweep_points.set("101")
        
        self.update_fixed_params_state()
        self.update_step_options()
        self.update_normalization_visibility()
        self.update_summary()
        self.update_scan_time_display()
    
    def update_scan_time_display(self, *args):
        """Update the estimated scan time display based on sweep type, points, IFBW, and averages."""
        try:
            points = int(float(self.sweep_points.get()))
            ifbw = float(self.ifbw.get())
            sweep_param = self.sweep_param.get()
            averages = max(1, int(float(self.sweep_averages.get())))
            step_param = self.step_param.get()
            
            if points <= 0 or ifbw <= 0:
                self.scan_time_label.config(text="")
                return
            
            # Different timing for different sweep types
            if sweep_param == "Frequency (GHz)":
                # VNA batch sweep mode - fast
                sweep_time = points / ifbw
                single_sweep_time = sweep_time * 1.5 + 2.0
            elif sweep_param == "Gate Voltage (V)":
                # ~2ms Keithley + VNA time per point (based on IFBW)
                # VNA CW measurement takes ~1/IFBW + overhead
                vna_time_per_point = (1.0 / ifbw) + 0.040  # 1/IFBW + 40ms overhead
                time_per_point = 0.002 + vna_time_per_point  # 2ms Keithley + VNA
                single_sweep_time = points * time_per_point + 2.0
            else:
                # Other point-by-point sweeps (B-field, Power, Temp)
                time_per_point = 0.10 + (1.0 / ifbw)
                single_sweep_time = points * time_per_point + 2.0
            
            # Total time with averages for single sweep
            sweep_with_avg = single_sweep_time * averages
            
            # Check if 2D map
            if step_param and step_param != "None":
                try:
                    step_points = int(float(self.step_points.get()))
                    if step_points > 0:
                        total_time = sweep_with_avg * step_points
                        
                        # Format display for 2D
                        if total_time < 60:
                            time_str = f"~{total_time:.1f}s"
                        elif total_time < 3600:
                            minutes = int(total_time // 60)
                            seconds = total_time % 60
                            time_str = f"~{minutes}m {seconds:.0f}s"
                        else:
                            hours = total_time / 3600
                            time_str = f"~{hours:.1f}h"
                        
                        sweep_str = f"{sweep_with_avg:.1f}s/step" if averages == 1 else f"{sweep_with_avg:.1f}s/step ({averages}avg)"
                        self.scan_time_label.config(text=f"2D Est: {time_str} ({step_points} steps x {sweep_str})")
                        return
                except ValueError:
                    pass
            
            # 1D sweep display
            if sweep_with_avg < 60:
                time_str = f"~{sweep_with_avg:.1f}s"
            else:
                minutes = int(sweep_with_avg // 60)
                seconds = sweep_with_avg % 60
                time_str = f"~{minutes}m {seconds:.0f}s"
            
            if averages > 1:
                self.scan_time_label.config(text=f"Est: {time_str} ({averages} sweeps x {single_sweep_time:.1f}s)")
            else:
                self.scan_time_label.config(text=f"Est: {time_str}")
            
        except (ValueError, ZeroDivisionError):
            self.scan_time_label.config(text="")
    
    def on_step_param_changed(self, event=None):
        """Handle step parameter selection change."""
        step = self.step_param.get()
        
        # Enable/disable step entries
        state = 'normal' if step != "None" else 'disabled'
        self.step_start_entry.config(state=state)
        self.step_stop_entry.config(state=state)
        self.step_points_entry.config(state=state)
        
        # Set sensible defaults based on parameter type
        if step == "Power (dBm)":
            self.step_start.set("-50")
            self.step_stop.set("0")
            self.step_points.set("6")
        elif step == "Frequency (GHz)":
            self.step_start.set("0.0001")
            self.step_stop.set("18")
            self.step_points.set("5")
        elif step == "B-Field (T)":
            self.step_start.set("0")
            self.step_stop.set("0.5")
            self.step_points.set("5")
        elif step == "Gate Voltage (V)":
            self.step_start.set("0")
            self.step_stop.set("10")
            self.step_points.set("5")
        
        self.update_fixed_params_state()
        self.update_normalization_visibility()
        self.update_summary()
    
    def update_step_options(self):
        """Update step parameter options based on sweep selection."""
        sweep = self.sweep_param.get()
        options = ["None"] + [p for p in self.param_options if p != sweep]
        self.step_combo['values'] = options
        print(f"Step options updated: sweep={sweep}, options={options}")
        
        if self.step_param.get() == sweep:
            self.step_param.set("None")
            self.on_step_param_changed()
    
    def update_bfield_visibility(self):
        """Show/hide B-field in fixed parameters based on magnet connection."""
        if hasattr(self, 'fixed_labels') and 'b_field' in self.fixed_labels:
            if self.magnet_connected.get():
                # Show B-field row
                self.fixed_labels['b_field'].grid()
                self.fixed_entries['b_field'].grid()
            else:
                # Hide B-field row
                self.fixed_labels['b_field'].grid_remove()
                self.fixed_entries['b_field'].grid_remove()
    
    def update_normalization_visibility(self):
        """Show/hide gate and field normalization options based on sweep/step parameters.
        
        Gate normalization is available when:
        - Gate voltage is the sweep parameter, OR
        - Gate voltage is the step parameter
        
        Field normalization is available when:
        - B-Field is the sweep parameter, OR
        - B-Field is the step parameter
        """
        sweep = self.sweep_param.get()
        step = self.step_param.get()
        
        # Check if gate voltage is involved
        gate_is_sweep = (sweep == "Gate Voltage (V)")
        gate_is_step = (step == "Gate Voltage (V)")
        
        # Check if B-field is involved
        field_is_sweep = (sweep == "B-Field (T)")
        field_is_step = (step == "B-Field (T)")
        
        # Gate normalization visibility
        if hasattr(self, 'gate_norm_frame'):
            if gate_is_sweep or gate_is_step:
                # Show gate normalization options
                self.gate_norm_frame.pack(fill='x', pady=5)
                
                # Update info text based on mode
                if gate_is_sweep:
                    self.gate_norm_info_label.config(
                        text="Will measure S21 at V_ref first, then subtract from each gate voltage point"
                    )
                else:  # gate_is_step
                    self.gate_norm_info_label.config(
                        text="Will take full spectrum at V_ref first, then subtract from each gate step spectrum"
                    )
            else:
                # Hide gate normalization options
                self.gate_norm_frame.pack_forget()
        
        # Field normalization visibility
        if hasattr(self, 'field_norm_frame'):
            if field_is_sweep or field_is_step:
                # Show field normalization options
                self.field_norm_frame.pack(fill='x', pady=5)
                
                # Update info text based on mode
                if field_is_sweep:
                    self.field_norm_info_label.config(
                        text="Will measure S21 at B_ref first, then subtract from each field point"
                    )
                else:  # field_is_step
                    self.field_norm_info_label.config(
                        text="Will take full spectrum at B_ref first, then subtract from each field step spectrum"
                    )
            else:
                # Hide field normalization options
                self.field_norm_frame.pack_forget()
    
    def on_gate_normalization_changed(self):
        """Handle gate normalization checkbox change."""
        enabled = self.gate_normalization_enabled.get()
        state = 'normal' if enabled else 'disabled'
        self.gate_norm_voltage_entry.config(state=state)
    
    def on_field_normalization_changed(self):
        """Handle field normalization checkbox change."""
        enabled = self.field_normalization_enabled.get()
        state = 'normal' if enabled else 'disabled'
        self.field_norm_entry.config(state=state)
    
    def is_gate_normalization_applicable(self):
        """Check if gate normalization is applicable to current sweep/step configuration."""
        sweep = self.sweep_param.get()
        step = self.step_param.get()
        return (sweep == "Gate Voltage (V)") or (step == "Gate Voltage (V)")
    
    def is_field_normalization_applicable(self):
        """Check if field normalization is applicable to current sweep/step configuration."""
        sweep = self.sweep_param.get()
        step = self.step_param.get()
        return (sweep == "B-Field (T)") or (step == "B-Field (T)")
    
    def update_fixed_params_state(self):
        """Enable/disable fixed parameter entries based on sweep/step selection."""
        sweep = self.sweep_param.get()
        step = self.step_param.get()
        
        param_map = {
            "Frequency (GHz)": 'frequency',
            "B-Field (T)": 'b_field',
            "Gate Voltage (V)": 'vg',
            "Power (dBm)": 'power'
        }
        
        for param_name, entry_key in param_map.items():
            if entry_key in self.fixed_entries:
                if param_name == sweep or param_name == step:
                    self.fixed_entries[entry_key].config(state='disabled')
                else:
                    self.fixed_entries[entry_key].config(state='normal')
    
    def update_summary(self):
        """Update measurement summary text."""
        sweep = self.sweep_param.get()
        step = self.step_param.get()
        
        # Convert parameter names for display
        sweep_display = self._get_param_display_label(sweep) if hasattr(self, '_get_param_display_label') else sweep
        step_display = self._get_param_display_label(step) if hasattr(self, '_get_param_display_label') and step != "None" else step
        
        try:
            sweep_pts = int(self.sweep_points.get())
            step_pts = int(self.step_points.get()) if step != "None" else 1
            total_pts = sweep_pts * step_pts
            
            # Estimate time (rough: 100ms per point)
            est_time = total_pts * 0.1
            
            summary = f"Sweep: {sweep_display}\n"
            summary += f"  Range: {self.sweep_start.get()} to {self.sweep_stop.get()}\n"
            summary += f"  Points: {sweep_pts}\n\n"
            
            if step != "None":
                summary += f"Step: {step_display}\n"
                summary += f"  Range: {self.step_start.get()} to {self.step_stop.get()}\n"
                summary += f"  Steps: {step_pts}\n\n"
                summary += f"Measurement Type: 2D Scan\n"
            else:
                summary += f"Measurement Type: 1D Sweep\n"
            
            summary += f"\nTotal Points: {total_pts}\n"
            summary += f"Est. Time: {est_time:.1f} seconds"
            
        except ValueError:
            summary = "Invalid parameters"
        
        self.summary_text.config(state='normal')
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, summary)
        self.summary_text.config(state='disabled')
    
    def start_measurement(self):
        """Start the measurement."""
        # Clear log buffer for new measurement
        log_manager.clear_buffer()
        print("=" * 50)
        print(f"Starting new measurement - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50)
        
        # Validate parameters
        try:
            sweep_param = self.sweep_param.get()
            sweep_start = float(self.sweep_start.get())
            sweep_stop = float(self.sweep_stop.get())
            
            step_param = self.step_param.get() if self.step_param.get() != "None" else None
            step_start = float(self.step_start.get()) if step_param else 0
            step_stop = float(self.step_stop.get()) if step_param else 0
            
            # Parse fixed values with defaults for empty/disabled fields
            freq_str = self.fixed_frequency.get().strip()
            fixed_freq = float(freq_str) if freq_str else 8.0  # Default 8 GHz
            
            power_str = self.fixed_power.get().strip()
            fixed_power = float(power_str) if power_str else -30.0  # Default -30 dBm
            
            # Frequency limits
            freq_min_ghz = 0.0001  # 100 kHz
            freq_max_ghz = 18.0    # 18 GHz
            
            # Power limits for S5180B VNA
            power_min = -50  # dBm
            power_max = 10   # dBm
            
            # B-field limits for SCM1
            field_min = -18.0  # Tesla
            field_max = 18.0   # Tesla
            
            clamped = False  # Track if any values were clamped
            
            # Clamp and convert sweep frequency
            if sweep_param == "Frequency (GHz)":
                new_start = max(freq_min_ghz, min(freq_max_ghz, sweep_start))
                new_stop = max(freq_min_ghz, min(freq_max_ghz, sweep_stop))
                if new_start != sweep_start or new_stop != sweep_stop:
                    clamped = True
                    self.sweep_start.set(str(new_start))
                    self.sweep_stop.set(str(new_stop))
                sweep_start = new_start * 1e9  # Convert to Hz
                sweep_stop = new_stop * 1e9
            
            # Clamp sweep power
            if sweep_param == "Power (dBm)":
                new_start = max(power_min, min(power_max, sweep_start))
                new_stop = max(power_min, min(power_max, sweep_stop))
                if new_start != sweep_start or new_stop != sweep_stop:
                    clamped = True
                    self.sweep_start.set(str(new_start))
                    self.sweep_stop.set(str(new_stop))
                sweep_start = new_start
                sweep_stop = new_stop
            
            # Clamp and convert step frequency
            if step_param == "Frequency (GHz)":
                new_start = max(freq_min_ghz, min(freq_max_ghz, step_start))
                new_stop = max(freq_min_ghz, min(freq_max_ghz, step_stop))
                if new_start != step_start or new_stop != step_stop:
                    clamped = True
                    self.step_start.set(str(new_start))
                    self.step_stop.set(str(new_stop))
                step_start = new_start * 1e9
                step_stop = new_stop * 1e9
            
            # Clamp step power
            if step_param == "Power (dBm)":
                new_start = max(power_min, min(power_max, step_start))
                new_stop = max(power_min, min(power_max, step_stop))
                if new_start != step_start or new_stop != step_stop:
                    clamped = True
                    self.step_start.set(str(new_start))
                    self.step_stop.set(str(new_stop))
                step_start = new_start
                step_stop = new_stop
            
            # Clamp sweep B-field
            if sweep_param == "B-Field (T)":
                new_start = max(field_min, min(field_max, sweep_start))
                new_stop = max(field_min, min(field_max, sweep_stop))
                if new_start != sweep_start or new_stop != sweep_stop:
                    clamped = True
                    self.sweep_start.set(str(new_start))
                    self.sweep_stop.set(str(new_stop))
                sweep_start = new_start
                sweep_stop = new_stop
            
            # Clamp step B-field
            if step_param == "B-Field (T)":
                new_start = max(field_min, min(field_max, step_start))
                new_stop = max(field_min, min(field_max, step_stop))
                if new_start != step_start or new_stop != step_stop:
                    clamped = True
                    self.step_start.set(str(new_start))
                    self.step_stop.set(str(new_stop))
                step_start = new_start
                step_stop = new_stop
            
            # Clamp fixed frequency (always, regardless of sweep param)
            new_freq = max(freq_min_ghz, min(freq_max_ghz, fixed_freq))
            if new_freq != fixed_freq:
                clamped = True
                self.fixed_frequency.set(str(new_freq))
            fixed_freq = new_freq * 1e9  # Convert to Hz
            
            # Clamp fixed power (always, regardless of sweep param)
            new_power = max(power_min, min(power_max, fixed_power))
            if new_power != fixed_power:
                clamped = True
                self.fixed_power.set(str(new_power))
            fixed_power = new_power
            
            # Clamp fixed B-field (always, regardless of sweep param)
            field_str = self.fixed_field.get().strip()
            fixed_field = float(field_str) if field_str else 0.0  # Default 0 T
            new_field = max(field_min, min(field_max, fixed_field))
            if new_field != fixed_field:
                clamped = True
                self.fixed_field.set(str(new_field))
            fixed_field = new_field
            
            # Parse fixed gate voltage with default
            gate_str = self.fixed_gate.get().strip()
            fixed_gate = float(gate_str) if gate_str else 0.0  # Default 0 V
            
            # Force GUI update if values were clamped
            if clamped:
                self.root.update_idletasks()
                print("Values were clamped to valid ranges")
            
            config = {
                'sweep_param': sweep_param,
                'sweep_start': sweep_start,
                'sweep_stop': sweep_stop,
                'sweep_points': int(self.sweep_points.get()),
                'averages': max(1, int(float(self.sweep_averages.get()))),
                'step_param': step_param,
                'step_start': step_start,
                'step_stop': step_stop,
                'step_points': int(self.step_points.get()) if step_param else 1,
                'fixed_values': {
                    'frequency': fixed_freq,
                    'b_field': fixed_field,
                    'vg': fixed_gate,
                    'power': fixed_power,
                    'temperature': self.current_temperature_k,  # From Lakeshore 370 (may be None)
                    'ifbw': float(self.ifbw.get())
                },
                's_parameter': self.s_parameter.get(),
                # Gate normalization settings
                'gate_normalization_enabled': self.gate_normalization_enabled.get() and self.is_gate_normalization_applicable(),
                'gate_normalization_voltage': float(self.gate_normalization_voltage.get()) if self.gate_normalization_enabled.get() else 0.0,
                # Field normalization settings
                'field_normalization_enabled': self.field_normalization_enabled.get() and self.is_field_normalization_applicable(),
                'field_normalization_field': float(self.field_normalization_field.get()) if self.field_normalization_enabled.get() else 0.0,
                # Legacy keys for backwards compatibility
                'normalization_enabled': self.gate_normalization_enabled.get() and self.is_gate_normalization_applicable(),
                'normalization_voltage': float(self.gate_normalization_voltage.get()) if self.gate_normalization_enabled.get() else 0.0
            }
        except ValueError as e:
            messagebox.showerror("Invalid Parameters", f"Please check parameter values: {e}")
            return
        
        # Check if we need to ramp to a fixed B-field before starting
        if (self.magnet_connected.get() and 
            sweep_param != "B-Field (T)" and 
            (step_param is None or step_param != "B-Field (T)")):
            
            # B-field is a fixed parameter - check if we need to ramp
            current_field = self.magnet.get_field()
            target_field = fixed_field
            field_tolerance = float(self.field_tolerance.get()) if hasattr(self, 'field_tolerance') else 0.001
            
            if abs(current_field - target_field) > field_tolerance:
                # Field needs to change - confirm with user
                field_diff = target_field - current_field
                ramp_time = abs(field_diff) / 0.3 * 60  # seconds at max rate
                
                confirm = messagebox.askyesno(
                    "B-Field Change Required",
                    f"The fixed B-field is set to {target_field:.4f} T\n"
                    f"but the current field is {current_field:.4f} T.\n\n"
                    f"This will require ramping {abs(field_diff):.4f} T\n"
                    f"(approximately {ramp_time:.0f} seconds at 0.3 T/min)\n\n"
                    f"Do you want to ramp to {target_field:.4f} T and start the measurement?"
                )
                
                if not confirm:
                    return
                
                # Ramp to fixed field
                self.status_var.set(f"Ramping to {target_field:.4f} T...")
                self.root.update_idletasks()
                
                # Set rate and start ramp
                if hasattr(self.magnet, 'set_rate'):
                    self.magnet.set_rate(0.3)  # Use max safe rate
                self.magnet.set_field(target_field)
                
                # Wait for field with progress updates
                start_time = time.time()
                timeout = 600  # 10 minute timeout
                
                while time.time() - start_time < timeout:
                    current = self.magnet.get_field()
                    self.status_var.set(f"Ramping to {target_field:.4f} T... (currently {current:.4f} T)")
                    self.root.update_idletasks()
                    
                    if abs(current - target_field) <= field_tolerance:
                        # Field reached - wait for settle time
                        settle_time = float(self.field_settle_time.get()) if hasattr(self, 'field_settle_time') else 2.0
                        if settle_time > 0:
                            self.status_var.set(f"Field at {current:.4f} T, settling...")
                            self.root.update_idletasks()
                            time.sleep(settle_time)
                        print(f"Fixed B-field reached: {current:.4f} T")
                        break
                    
                    time.sleep(0.5)
                else:
                    messagebox.showerror("Timeout", f"Timeout waiting for field to reach {target_field:.4f} T")
                    return
        
        # Clear previous data
        self.sweep_data_1d = []
        self.sweep_data_2d = []
        self.current_step_index = 0
        self.current_config = config
        self.current_2d_folder = None  # Reset for new measurement
        
        # Clear reference data for normalization
        self.reference_data = None
        self.reference_voltage = None
        
        # Clear field reference data for normalization
        self.field_reference_data = None
        self.reference_field = None
        
        # Invalidate Z data cache
        self._z_cache = None
        self._z_cache_key = None
        
        # Clear averaging display data
        self.current_sweep_raw = []
        self.prev_avg_data = []
        self.current_avg_num = 1
        self.current_avg_total = 1
        
        # Set plot layout based on 1D vs 2D measurement
        is_2d = step_param is not None and step_param != "None"
        self.set_plot_layout("2D" if is_2d else "1D")
        
        # Reset step slider
        self.step_slider_var.set(0)
        self.update_step_slider()
        
        # Configure gate voltage safety settings
        try:
            gate_slew = float(self.gate_slew_rate.get())
            gate_max = float(self.max_gate_voltage.get())
            gate_compliance = float(self.gate_compliance.get()) * 1e-9  # nA to A
        except ValueError:
            gate_slew = 10.0
            gate_max = 100.0
            gate_compliance = 100e-9
        
        self.measurement_engine.set_gate_safety(
            slew_rate=gate_slew,
            ramp_to_zero_after=self.ramp_gate_to_zero.get(),
            ramp_on_stop=self.ramp_gate_on_stop.get(),
            max_voltage=gate_max,
            compliance=gate_compliance
        )
        
        # Apply B-field settings
        try:
            field_rate = float(self.field_ramp_rate.get())
            field_tol = float(self.field_tolerance.get())
            field_settle = float(self.field_settle_time.get())
        except ValueError:
            field_rate = 0.3
            field_tol = 0.01
            field_settle = 2.0
        
        self.measurement_engine.set_field_settings(
            ramp_rate=field_rate,
            tolerance=field_tol,
            settle_time=field_settle,
            wait=self.wait_for_field.get()
        )
        
        # Apply VNA settle time
        try:
            self.measurement_engine.vna_settle_time = float(self.vna_settle_time.get())
        except ValueError:
            self.measurement_engine.vna_settle_time = 5.0
        
        # Update UI
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.progress_var.set(0)
        self.step_progress_var.set(0)
        self.progress_label.config(text="0%")
        self.step_progress_label.config(text="0%")
        self.control_status_label.config(text="Running...")
        self.status_var.set("Measurement in progress...")
        
        # Start measurement in thread
        self.measurement_thread = threading.Thread(
            target=self.measurement_engine.run_measurement,
            args=(config,)
        )
        self.measurement_thread.start()
    
    def stop_measurement(self):
        """Stop the measurement immediately."""
        self.measurement_engine.stop()
        self.status_var.set("ABORTING - Please wait...")
        self.stop_button.config(state='disabled')
        # Force GUI update so user sees the abort message
        self.root.update_idletasks()
    
    def emergency_stop(self):
        """EMERGENCY: Immediately stop measurement and turn off gate output.
        
        Use this when something goes wrong and you need to stop immediately.
        Does NOT ramp - immediately turns off output.
        """
        # Stop the measurement engine
        self.measurement_engine.stop()
        
        # Immediately turn off gate output
        self.keithley.emergency_shutdown()
        
        # Update GUI
        self.status_var.set("EMERGENCY STOP - Gate output OFF")
        self.stop_button.config(state='disabled')
        self.root.update_idletasks()
        
        # Show warning
        messagebox.showwarning("Emergency Stop", 
            "Gate output has been turned OFF.\n\n"
            "The measurement has been stopped.\n"
            "Check your sample before continuing.\n\n"
            "You may need to reconnect to the Keithley.")
    
    def update_gui(self):
        """Periodic GUI update (called every 50ms)."""
        # Check for progress updates (process all available)
        try:
            while True:
                progress = self.measurement_engine.progress_queue.get_nowait()
                self.progress_var.set(progress)
                self.progress_label.config(text=f"{progress:.1f}%")
        except queue.Empty:
            pass
        
        # Check for data updates (process all available to avoid queue buildup)
        updates_processed = 0
        max_updates_per_cycle = 50  # Increased from 10 to handle rapid step completions
        try:
            while updates_processed < max_updates_per_cycle:
                data = self.measurement_engine.data_queue.get_nowait()
                self.process_data_update(data)
                updates_processed += 1
        except queue.Empty:
            pass
        
        # Update log display (process queued log messages) - throttled internally
        self.update_log_display()
        
        # Throttle instrument display updates to every 500ms to avoid GUI lag
        # These involve slow GPIB/TCP communication
        current_time = time.time()
        if not hasattr(self, '_last_instrument_update'):
            self._last_instrument_update = 0
        
        if current_time - self._last_instrument_update > 0.5:
            self._last_instrument_update = current_time
            # Update gate voltage display periodically
            self.update_gate_display()
            # Update B-field display periodically
            self.update_field_display()
        
        # Schedule next update
        self.root.after(50, self.update_gui)
    
    def process_data_update(self, data):
        """Process data update from measurement engine."""
        if data['type'] == 'reference_data':
            # Store reference data for gate normalization
            self.reference_voltage = data.get('voltage', 0.0)
            self.reference_data = {
                'voltage': self.reference_voltage,
                'single_value_db': data.get('single_value_db'),
                'spectrum_db': data.get('spectrum_db'),
                'frequencies': data.get('frequencies'),
                'per_step_db': data.get('per_step_db'),  # Dict of step_idx -> reference_db
                'step_frequencies': data.get('step_frequencies')  # List of frequencies for each step
            }
            print(f"[GUI] Gate reference data received for V={self.reference_voltage}V")
            if self.reference_data['single_value_db'] is not None:
                print(f"  Single CW reference: {self.reference_data['single_value_db']:.2f} dB")
            if self.reference_data['spectrum_db'] is not None:
                print(f"  Reference spectrum: {len(self.reference_data['spectrum_db'])} points")
            if self.reference_data['per_step_db'] is not None:
                print(f"  Per-step references: {len(self.reference_data['per_step_db'])} frequencies")
            return
        
        elif data['type'] == 'field_reference_data':
            # Store reference data for field normalization
            self.reference_field = data.get('field', 0.0)
            self.field_reference_data = {
                'field': self.reference_field,
                'single_value_db': data.get('single_value_db'),
                'spectrum_db': data.get('spectrum_db'),
                'frequencies': data.get('frequencies'),
                'per_step_db': data.get('per_step_db'),  # Dict of step_idx -> reference_db
                'step_frequencies': data.get('step_frequencies')  # List of frequencies for each step
            }
            print(f"[GUI] Field reference data received for B={self.reference_field}T")
            if self.field_reference_data['single_value_db'] is not None:
                print(f"  Single CW reference: {self.field_reference_data['single_value_db']:.2f} dB")
            if self.field_reference_data['spectrum_db'] is not None:
                print(f"  Reference spectrum: {len(self.field_reference_data['spectrum_db'])} points")
            if self.field_reference_data['per_step_db'] is not None:
                print(f"  Per-step references: {len(self.field_reference_data['per_step_db'])} frequencies")
            return
        
        elif data['type'] == 'point':
            # Track current sweep for live plot
            sweep_idx = data['sweep_idx']
            step_idx = data['step_idx']
            
            # Check if we're in averaging mode
            avg_num = data.get('avg_num', 1)
            avg_total = data.get('avg_total', 1)
            current_raw = data.get('current_raw')
            prev_avg = data.get('prev_avg')
            
            # Store for 2D array (running average goes here)
            while len(self.sweep_data_2d) <= step_idx:
                self.sweep_data_2d.append([])
            
            # For averaging: update or append
            if avg_total > 1 and avg_num > 1:
                # Update existing point with new average
                if sweep_idx < len(self.sweep_data_2d[step_idx]):
                    self.sweep_data_2d[step_idx][sweep_idx] = data['data']
                else:
                    self.sweep_data_2d[step_idx].append(data['data'])
            else:
                self.sweep_data_2d[step_idx].append(data['data'])
            
            # Store current sweep raw data for dual display
            if current_raw is not None:
                if not hasattr(self, 'current_sweep_raw') or sweep_idx == 0:
                    self.current_sweep_raw = []
                while len(self.current_sweep_raw) <= sweep_idx:
                    self.current_sweep_raw.append(None)
                self.current_sweep_raw[sweep_idx] = current_raw
            
            # Store previous average for dual display
            if prev_avg is not None:
                if not hasattr(self, 'prev_avg_data') or sweep_idx == 0:
                    self.prev_avg_data = []
                while len(self.prev_avg_data) <= sweep_idx:
                    self.prev_avg_data.append(None)
                self.prev_avg_data[sweep_idx] = prev_avg
            
            # Store averaging info for plot title
            self.current_avg_num = avg_num
            self.current_avg_total = avg_total
            
            # Update step slider range
            self.update_step_slider()
            
            # Keep current step at latest if we're at the end
            if self.current_step_index == step_idx or self.current_step_index == step_idx - 1:
                self.current_step_index = step_idx
                self.step_slider_var.set(self.current_step_index)
            
            # Update single trace view periodically (every 5 points)
            sweep_points = self.current_config.get('sweep_points', 100) if self.current_config else 100
            if sweep_idx % 5 == 0 or sweep_idx == sweep_points - 1:
                self.update_single_trace()
        
        elif data['type'] == 'batch':
            # Batch data from VNA sweep - entire sweep line at once
            step_idx = data['step_idx']
            sweep_data = data['sweep_data']
            
            # Check for averaging info
            avg_num = data.get('avg_num', 1)
            avg_total = data.get('avg_total', 1)
            current_raw = data.get('current_raw')
            
            # Store averaging info for plot display
            self.current_avg_num = avg_num
            self.current_avg_total = avg_total
            
            # Store current sweep raw data for dual display during averaging
            if current_raw is not None:
                self.current_sweep_raw = current_raw
            
            # Ensure we have enough rows
            while len(self.sweep_data_2d) <= step_idx:
                self.sweep_data_2d.append([])
            
            # Store complete sweep data (running average)
            self.sweep_data_2d[step_idx] = sweep_data
            
            # Log with averaging info
            if avg_total > 1:
                print(f"Batch received: step {step_idx}, avg {avg_num}/{avg_total}, {len(sweep_data)} points")
                # Update control status to show averaging progress
                total_steps = self.current_config.get('step_points', 1) if self.current_config else 1
                self.control_status_label.config(text=f"Step {step_idx + 1}/{total_steps}, Avg {avg_num}/{avg_total}")
            else:
                print(f"Batch received: step {step_idx} with {len(sweep_data)} points")
            
            # Update step tracking
            self.current_step_index = step_idx
            self.update_step_slider()
            self.step_slider_var.set(self.current_step_index)
            
            # Update plots - always update during averaging to show progress
            self.update_single_trace()
        
        elif data['type'] == 'step_complete':
            # A full sweep line just finished
            step_idx = data['step_idx']
            step_value = data.get('step_value')
            
            # Update 2D step progress bar
            if self.current_config and self.current_config.get('step_param'):
                total_steps = self.current_config.get('step_points', 1)
                step_progress = ((step_idx + 1) / total_steps) * 100
                self.step_progress_var.set(step_progress)
                self.step_progress_label.config(text=f"{step_progress:.0f}%")
                # Also update control status
                self.control_status_label.config(text=f"Step {step_idx + 1}/{total_steps}")
            
            # If sweep_data is included (batch mode), use it to replace/set the full data
            if 'sweep_data' in data and data['sweep_data']:
                # Ensure we have enough rows
                while len(self.sweep_data_2d) <= step_idx:
                    self.sweep_data_2d.append([])
                # Replace with complete sweep data
                self.sweep_data_2d[step_idx] = data['sweep_data']
                print(f"Step {step_idx} complete: stored {len(data['sweep_data'])} points")
                
                # Auto-save this sweep (only for 2D measurements)
                is_2d = self.current_config and self.current_config.get('step_param')
                if is_2d:
                    self.auto_save_sweep(data['sweep_data'], step_idx, step_value)
            
            # Update plots after EVERY step - user wants to see progress
            # Update step slider first (fast)
            self.update_step_slider()
            
            # Update single trace view
            self.update_single_trace()
            
            # Update 2D map - need at least 2 rows for a 2D plot
            if len(self.sweep_data_2d) >= 2:
                self.update_2d_plot()
            
            # Force GUI to process the updates NOW
            self.root.update_idletasks()
        
        elif data['type'] == 'complete':
            # CRITICAL: First, process any remaining queued step_complete messages
            # to ensure all sweeps are saved before final processing
            try:
                while True:
                    queued_data = self.measurement_engine.data_queue.get_nowait()
                    if queued_data['type'] == 'step_complete':
                        # Process this step - but skip plot updates for speed
                        step_idx = queued_data['step_idx']
                        step_value = queued_data.get('step_value')
                        if 'sweep_data' in queued_data and queued_data['sweep_data']:
                            while len(self.sweep_data_2d) <= step_idx:
                                self.sweep_data_2d.append([])
                            self.sweep_data_2d[step_idx] = queued_data['sweep_data']
                            print(f"Step {step_idx} complete: stored {len(queued_data['sweep_data'])} points")
                            is_2d = self.current_config and self.current_config.get('step_param')
                            if is_2d:
                                self.auto_save_sweep(queued_data['sweep_data'], step_idx, step_value)
            except queue.Empty:
                pass  # All queued messages processed
            
            self.status_var.set("Measurement complete - saving data...")
            self.control_status_label.config(text="Saving...")
            self.start_button.config(state='normal')
            self.stop_button.config(state='disabled')
            
            # Reset averaging display state for clean final view
            self.current_avg_num = 1
            self.current_avg_total = 1
            self.current_sweep_raw = []
            self.prev_avg_data = []
            
            # CRITICAL: For 2D measurements, ensure ALL sweeps are saved
            # Some may have been queued but not processed yet
            is_2d = self.current_config and self.current_config.get('step_param')
            if is_2d and self.current_2d_folder:
                # Count how many sweep files exist
                import glob
                existing_files = glob.glob(os.path.join(self.current_2d_folder, "sweep_*.csv"))
                n_saved = len(existing_files)
                n_sweeps = len(self.sweep_data_2d)
                
                # Save any missing sweeps
                if n_saved < n_sweeps:
                    print(f"Saving remaining sweeps ({n_saved} of {n_sweeps} saved)...")
                    for idx, sweep_data in enumerate(self.sweep_data_2d):
                        sweep_num = idx + 1
                        expected_file = os.path.join(self.current_2d_folder, f"sweep_{sweep_num:03d}.csv")
                        if not os.path.exists(expected_file) and sweep_data:
                            step_value = sweep_data[0].get('step_value', 0) if sweep_data else 0
                            self.auto_save_sweep(sweep_data, idx, step_value)
                
                # Save 2D plot images at end of 2D measurement
                self.save_plot_images(os.path.join(self.current_2d_folder, "plot"))
                
                # Save log file for 2D measurement
                log_filepath = log_manager.get_log_filepath(self.current_2d_folder)
                log_manager.save_to_file(log_filepath)
                print(f"Measurement log saved to: {log_filepath}")
                
            elif not is_2d and self.sweep_data_2d and self.sweep_data_2d[0]:
                # For 1D measurements, auto-save at completion
                self.auto_save_sweep(self.sweep_data_2d[0])
            
            self.status_var.set("Measurement complete!")
            self.control_status_label.config(text="Complete")
            
            # Update slider first (fast)
            self.update_step_slider()
            
            # Defer expensive plot updates to avoid GUI freeze
            # Use after() so GUI stays responsive
            self.root.after(100, self._deferred_plot_update)
            
            # Reset 2D folder tracker for next measurement
            self.current_2d_folder = None
        
        elif data['type'] == 'aborted':
            self.status_var.set("Measurement ABORTED by user - saving plots...")
            self.control_status_label.config(text="Aborted")
            self.start_button.config(state='normal')
            self.stop_button.config(state='disabled')
            
            # Reset averaging display state
            self.current_avg_num = 1
            self.current_avg_total = 1
            self.current_sweep_raw = []
            self.prev_avg_data = []
            
            self.update_step_slider()
            self.update_single_trace()
            self.update_2d_plot()
            
            # Auto-save plot screenshots for aborted measurements
            try:
                if self.current_2d_folder:
                    # 2D measurement with folder - save plots there with "aborted" suffix
                    plot_base = os.path.join(self.current_2d_folder, "plot_ABORTED")
                    self.save_plot_images(plot_base)
                    print(f"Aborted 2D measurement plots saved to: {self.current_2d_folder}")
                elif self.sweep_data_2d and self.sweep_data_2d[0]:
                    # 1D measurement or no folder yet - create aborted file in data directory
                    data_dir = self.data_directory.get()
                    if not os.path.exists(data_dir):
                        os.makedirs(data_dir)
                    
                    # Generate filename with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    base_name = self.filename.get().strip() or "fmr_data"
                    if base_name.endswith('.csv'):
                        base_name = base_name[:-4]
                    
                    plot_base = os.path.join(data_dir, f"{base_name}_ABORTED_{timestamp}")
                    self.save_plot_images(plot_base)
                    print(f"Aborted measurement plots saved: {plot_base}_*.png")
            except Exception as e:
                print(f"Error saving aborted measurement plots: {e}")
            
            # Save log for aborted measurement if we have a folder
            if self.current_2d_folder:
                log_filepath = log_manager.get_log_filepath(self.current_2d_folder)
                log_manager.save_to_file(log_filepath)
                print(f"Aborted measurement log saved to: {log_filepath}")
            
            self.status_var.set("Measurement ABORTED - plots saved")
            
            # Reset 2D folder tracker (partial data already saved)
            self.current_2d_folder = None
        
        elif data['type'] == 'error':
            self.status_var.set(f"Error: {data['message']}")
            self.start_button.config(state='normal')
            self.stop_button.config(state='disabled')
            
            # Save log for error case
            if self.current_2d_folder:
                log_filepath = log_manager.get_log_filepath(self.current_2d_folder)
                log_manager.save_to_file(log_filepath)
            
            messagebox.showerror("Measurement Error", data['message'])
    
    def _update_plots_async(self):
        """Update plots without blocking the GUI.
        
        Called via root.after_idle() during measurement to allow
        the GUI to remain responsive.
        """
        try:
            # Only update single trace during measurement (fast)
            self.update_single_trace()
            
            # Skip 2D plot during measurement - it's too slow
            # It will be updated at the end
        except Exception as e:
            print(f"Error in async plot update: {e}")
    
    def _deferred_plot_update(self):
        """Deferred plot update after measurement completion.
        
        Called via root.after() to keep GUI responsive during heavy plot updates.
        """
        try:
            self.status_var.set("Updating plots...")
            self.root.update_idletasks()
            
            # Update single trace (usually fast)
            self.update_single_trace()
            
            # Update 2D plot (can be slow for large datasets)
            total_points = sum(len(t) for t in self.sweep_data_2d if t)
            if total_points > 50000:
                self.status_var.set(f"Rendering 2D map ({total_points} points)...")
                self.root.update_idletasks()
            
            self.update_2d_plot()
            self.status_var.set("Measurement complete!")
            
        except Exception as e:
            print(f"Error in deferred plot update: {e}")
            self.status_var.set("Complete (plot update had errors)")
    
    def update_2d_plot(self, force_rebuild=False):
        """Update 2D contour plot.
        
        Args:
            force_rebuild: If True, rebuild Z data even if cached
        """
        step_param = self.current_config.get('step_param') if self.current_config else None
        
        # Clear entire figure and recreate axis (cleanest way to handle colorbar)
        self.fig_2d.clear()
        self.ax_2d = self.fig_2d.add_subplot(111)
        
        # Only show 2D plot if we have step data with at least 2 steps
        if not step_param or len(self.sweep_data_2d) < 2:
            self.ax_2d.set_title("2D Map (requires step parameter)")
            self.ax_2d.text(0.5, 0.5, "Run a 2D scan to see contour plot",
                          ha='center', va='center', transform=self.ax_2d.transAxes,
                          fontsize=8, color='gray')
            self.fig_2d.tight_layout()
            self.canvas_2d.draw_idle()
            return
        
        sweep_param = self.current_config['sweep_param'] if self.current_config else "Sweep"
        
        # Build 2D arrays - only use complete sweep lines
        complete_traces = [t for t in self.sweep_data_2d if len(t) > 0]
        if len(complete_traces) < 2:
            self.fig_2d.tight_layout()
            self.canvas_2d.draw_idle()
            return
        
        # Get dimensions from first complete trace
        n_sweep = len(complete_traces[0])
        n_steps = len(complete_traces)
        
        # Filter to traces with matching sweep points
        valid_traces = [t for t in complete_traces if len(t) == n_sweep]
        if len(valid_traces) < 2:
            self.fig_2d.tight_layout()
            self.canvas_2d.draw_idle()
            return
        
        # === Z DATA CACHING ===
        # Z data only depends on: mode, normalize, smoothing, and data count
        # X-axis mode does NOT affect Z data
        mode = self.contour_mode.get()
        smooth_window = self.smoothing_window.get()
        normalize_at_v = self.normalize_at_v.get()
        try:
            v_norm_val = float(self.v_norm.get()) if normalize_at_v else 0
        except:
            v_norm_val = 0
        data_hash = (len(valid_traces), n_sweep, mode, smooth_window, normalize_at_v, v_norm_val)
        
        # Check if we can use cached Z data
        use_cache = (
            not force_rebuild and
            hasattr(self, '_z_cache') and
            self._z_cache is not None and
            hasattr(self, '_z_cache_key') and
            self._z_cache_key == data_hash
        )
        
        if use_cache:
            z = self._z_cache
            zlabel = self._z_label_cache
            cmap = self._z_cmap_cache
        else:
            # Need to rebuild Z data
            if mode == "Magnitude":
                z = np.array([[20 * np.log10(d['s21_mag'] + 1e-12) for d in trace] 
                             for trace in valid_traces])
                zlabel = "|S21| (dB)"
                cmap = 'viridis'
            elif mode == "Normalized":
                # Use pre-measured reference normalization
                # Check if normalized data is available
                has_norm = (valid_traces and valid_traces[0] and 
                           's21_mag_db_norm' in valid_traces[0][0] and 
                           valid_traces[0][0]['s21_mag_db_norm'] is not None)
                if has_norm:
                    z = np.array([[d.get('s21_mag_db_norm', 0) for d in trace] 
                                 for trace in valid_traces])
                    zlabel = "Delta S21 (dB)"
                else:
                    # Fall back to raw magnitude
                    z = np.array([[20 * np.log10(d['s21_mag'] + 1e-12) for d in trace] 
                                 for trace in valid_traces])
                    zlabel = "|S21| (dB) [no ref]"
                cmap = 'RdBu_r'  # Diverging colormap for difference
            elif mode == "Phase":
                z = np.array([[d['s21_phase'] for d in trace] for trace in valid_traces])
                zlabel = "Phase (deg)"
                cmap = 'RdBu_r'
            elif mode == "Real":
                z = np.array([[d['s21_real'] for d in trace] for trace in valid_traces])
                zlabel = "Re(S21)"
                cmap = 'RdBu_r'
            elif mode == "Conductivity":
                z_list = []
                for trace in valid_traces:
                    sigma = self._calculate_conductivity(trace)
                    z_list.append(sigma)
                z = np.array(z_list)
                zlabel = "sigma (e^2/h)"
                cmap = 'viridis'
            else:  # Imaginary
                z = np.array([[d['s21_imag'] for d in trace] for trace in valid_traces])
                zlabel = "Im(S21)"
                cmap = 'RdBu_r'
            
            # Apply smoothing if enabled
            if smooth_window > 1 and z.shape[1] >= smooth_window:
                z_smoothed = np.zeros_like(z)
                for i in range(z.shape[0]):
                    z_smoothed[i] = self._smooth_data(z[i], smooth_window)
                z = z_smoothed
            
            # Apply V_norm normalization if enabled
            if normalize_at_v:
                ref_data, norm_type, success = self._get_2d_normalization_reference(valid_traces, mode)
                
                if success and ref_data is not None:
                    if norm_type == 'per_trace':
                        # Gate is sweep parameter - subtract reference value from each row
                        for i, ref_val in enumerate(ref_data):
                            z[i] = z[i] - ref_val
                        zlabel = "Delta " + zlabel
                        cmap = 'RdBu_r'
                    elif norm_type == 'spectrum':
                        # Gate is step parameter - subtract reference spectrum from all rows
                        ref_spectrum = ref_data
                        if len(ref_spectrum) == z.shape[1]:
                            for i in range(z.shape[0]):
                                z[i] = z[i] - ref_spectrum
                            zlabel = "Delta " + zlabel
                            cmap = 'RdBu_r'
                    # Clear status message on success
                    if hasattr(self, 'norm_status_label'):
                        self.norm_status_label.config(text="")
                elif not success and norm_type == 'spectrum':
                    # V_norm not yet measured - show status
                    if hasattr(self, 'norm_status_label'):
                        try:
                            v_norm = float(self.v_norm.get())
                            self.norm_status_label.config(text=f"(waiting for V={v_norm:.1f})")
                        except:
                            pass

            # Cache the Z data
            self._z_cache = z
            self._z_cache_key = data_hash
            self._z_label_cache = zlabel
            self._z_cmap_cache = cmap
        
        # Check for large datasets - warn user
        total_points = z.size
        if total_points > 50000 and not hasattr(self, '_large_dataset_warned'):
            self._large_dataset_warned = True
            print(f"Note: Large 2D dataset ({total_points} points) - display updates may be slow")
        
        # Get frequency scaling for axes
        freq_scale, freq_unit = self._get_frequency_scale()
        step_freq_scale, step_freq_unit = self._get_step_frequency_scale()
        
        # Get x-axis conversion function based on display mode
        xaxis_mode = self.xaxis_mode.get()
        
        # For 2D plots, disable filling factor mode (too complex with per-row B-field)
        if xaxis_mode == "Filling Factor (nu)" and sweep_param == "Gate Voltage (V)":
            xaxis_mode = "Gate Voltage (V)"  # Fall back to gate voltage
        
        x_convert, xlabel_from_mode = self._get_xaxis_conversion(sweep_param, freq_scale, freq_unit, xaxis_mode)
        
        # Apply x-axis conversion to sweep values
        raw_sweep_vals = np.array([d['sweep_value'] for d in valid_traces[0]])
        sweep_vals = np.array([x_convert(v) for v in raw_sweep_vals])
        
        # Get step values, converting power to power at probe
        raw_step_vals = np.array([t[0]['step_value'] * step_freq_scale for t in valid_traces])
        if step_param == "Power (dBm)":
            # Convert to power at probe
            try:
                input_atten = float(self.input_attenuation.get())
                step_vals = raw_step_vals - input_atten
            except:
                step_vals = raw_step_vals
        else:
            step_vals = raw_step_vals
        
        # Create meshgrid
        X, Y = np.meshgrid(sweep_vals, step_vals)
        
        # Determine color scale limits and normalization
        if self.auto_scale.get():
            vmin, vmax = None, None
            # Update the entry fields with actual data range for reference
            try:
                self.color_min.set(f"{np.nanmin(z):.1f}")
                self.color_max.set(f"{np.nanmax(z):.1f}")
            except:
                pass
        else:
            try:
                vmin = float(self.color_min.get())
                vmax = float(self.color_max.get())
            except ValueError:
                vmin, vmax = None, None
        
        # Set up normalization (log or linear)
        norm = None
        if self.log_scale.get():
            # For log scale, need to handle the data appropriately
            if mode == "Magnitude":
                # dB values can be negative, use SymLogNorm
                # linthresh is the range around zero that is linear
                z_range = np.nanmax(np.abs(z)) if np.nanmax(np.abs(z)) > 0 else 1
                linthresh = z_range * 0.01  # 1% of range is linear
                if vmin is not None and vmax is not None:
                    norm = SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)
                else:
                    norm = SymLogNorm(linthresh=linthresh)
            else:
                # For Real/Imag/Phase, also use SymLogNorm since they can be negative
                z_range = np.nanmax(np.abs(z)) if np.nanmax(np.abs(z)) > 0 else 1
                linthresh = z_range * 0.01
                if vmin is not None and vmax is not None:
                    norm = SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)
                else:
                    norm = SymLogNorm(linthresh=linthresh)
        
        # Plot contour
        if norm is not None:
            im = self.ax_2d.pcolormesh(X, Y, z, shading='auto', cmap=cmap, norm=norm)
        else:
            im = self.ax_2d.pcolormesh(X, Y, z, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        
        self.colorbar_2d = self.fig_2d.colorbar(im, ax=self.ax_2d, label=zlabel)
        self.colorbar_mappable = im
        self.colorbar_data_range = (np.nanmin(z), np.nanmax(z))
        
        # Update axis labels - use x-axis mode for sweep axis
        xlabel = xlabel_from_mode  # From _get_xaxis_conversion
            
        if step_param == "Frequency (GHz)" and step_freq_unit:
            ylabel = f"Frequency ({step_freq_unit})"
        else:
            ylabel = self._get_param_display_label(step_param)
        
        self.ax_2d.set_xlabel(xlabel)
        self.ax_2d.set_ylabel(ylabel)
        
        # Build title with fixed parameters that aren't on axes
        title_parts = [f"2D Map - {zlabel}"]
        
        if self.current_config:
            fixed = self.current_config.get('fixed_values', {})
            details = []
            
            # Add frequency if not on axes
            if sweep_param != "Frequency (GHz)" and step_param != "Frequency (GHz)":
                freq = fixed.get('frequency', 0)
                if freq > 0:
                    freq_ghz = freq / 1e9
                    details.append(f"f={freq_ghz:.3f}GHz")
            
            # Add B-field if not on axes
            if sweep_param != "B-Field (T)" and step_param != "B-Field (T)":
                b_field = fixed.get('b_field', 0)
                if b_field != 0:
                    details.append(f"B={b_field:.2f}T")
            
            # Add power at probe if not on axes
            if sweep_param != "Power (dBm)" and step_param != "Power (dBm)":
                power = fixed.get('power', -50)
                try:
                    input_atten = float(self.input_attenuation.get())
                    power_at_probe = power - input_atten
                    details.append(f"P={power_at_probe:.0f}dBm@probe")
                except:
                    details.append(f"P={power:.0f}dBm")
            
            # Add gate voltage if not on axes
            if sweep_param != "Gate Voltage (V)" and step_param != "Gate Voltage (V)":
                vg = fixed.get('vg', 0)
                if vg != 0:
                    details.append(f"Vg={vg:.1f}V")
            
            # Add temperature (always shown - measured from Lakeshore 370)
            temp = fixed.get('temperature')
            if temp is not None and temp > 0:
                if temp < 1:  # Less than 1K, show in mK
                    details.append(f"T={temp*1000:.1f}mK")
                else:
                    details.append(f"T={temp:.2f}K")
            
            if details:
                title_parts.append(" | ".join(details))
        
        self.ax_2d.set_title("\n".join(title_parts), fontsize=9)
        
        # Custom tick formatting for filling factor mode on x-axis
        if xaxis_mode == "Filling Factor (nu)" and sweep_param == "Gate Voltage (V)":
            self._set_filling_factor_ticks(self.ax_2d, fontsize=9)
        
        self.fig_2d.tight_layout()
        self.canvas_2d.draw_idle()  # Non-blocking draw
    
    def _get_step_frequency_scale(self):
        """Determine appropriate frequency scale for step parameter.
        
        Returns:
            (scale_factor, unit_string) where scale_factor converts Hz to display units
        """
        if not self.current_config:
            return 1.0, None
        
        step_param = self.current_config.get('step_param', '')
        if step_param != "Frequency (GHz)":
            return 1.0, None
        
        # Get frequency range in Hz
        f_start = self.current_config.get('step_start', 0)
        f_stop = self.current_config.get('step_stop', 0)
        f_max = max(abs(f_start), abs(f_stop))
        
        # Choose unit based on the range
        if f_max >= 1e9:
            return 1e-9, "GHz"
        elif f_max >= 1e6:
            return 1e-6, "MHz"
        elif f_max >= 1e3:
            return 1e-3, "kHz"
        else:
            return 1.0, "Hz"
    
    def on_scale_mode_changed(self):
        """Handle auto/manual scale mode change for 2D plot."""
        if self.auto_scale.get():
            self.color_min_entry.config(state='disabled')
            self.color_max_entry.config(state='disabled')
        else:
            self.color_min_entry.config(state='normal')
            self.color_max_entry.config(state='normal')
        # Only update plot if it exists (not during initialization)
        if hasattr(self, 'fig_2d'):
            self.update_2d_plot()
    
    def _on_normalization_changed(self):
        """Handle normalization checkbox or V_norm value change."""
        # Clear status message
        if hasattr(self, 'norm_status_label'):
            self.norm_status_label.config(text="")
        
        # Update both plots
        self.update_single_trace()
        if self.current_plot_mode == "2D":
            self.update_2d_plot()
    
    def _find_nearest_index(self, values, target):
        """Find index of value nearest to target.
        
        Args:
            values: List or array of values
            target: Target value to find
            
        Returns:
            Index of nearest value, or None if values is empty
        """
        if not values or len(values) == 0:
            return None
        
        values = np.array(values)
        idx = np.argmin(np.abs(values - target))
        return idx
    
    def _get_normalization_reference(self, trace_data, mode):
        """Get the normalization reference value for a single trace (gate sweep).
        
        Args:
            trace_data: List of data points from sweep
            mode: Display mode (Magnitude, Phase, etc.)
            
        Returns:
            (reference_value, success) tuple
            reference_value: The y-value at V_norm to subtract
            success: True if reference was found, False if V_norm not in data range
        """
        if not self.normalize_at_v.get():
            return 0.0, True
        
        if not trace_data:
            return 0.0, False
        
        # Check if this is a gate voltage sweep
        sweep_param = self.current_config.get('sweep_param', '') if self.current_config else ''
        if sweep_param != "Gate Voltage (V)":
            return 0.0, True  # Not a gate sweep, no normalization needed
        
        try:
            v_norm = float(self.v_norm.get())
        except ValueError:
            return 0.0, False
        
        # Get gate voltages from trace
        gate_voltages = [d['sweep_value'] for d in trace_data]
        
        # Check if V_norm is within the data range
        v_min, v_max = min(gate_voltages), max(gate_voltages)
        if v_norm < v_min or v_norm > v_max:
            return 0.0, False  # V_norm outside range
        
        # Find nearest index
        idx = self._find_nearest_index(gate_voltages, v_norm)
        if idx is None:
            return 0.0, False
        
        # Get y data and extract reference value
        y_data = self._get_trace_y_data_raw(trace_data, mode)
        if idx < len(y_data):
            return y_data[idx], True
        
        return 0.0, False
    
    def _get_trace_y_data_raw(self, trace_data, mode):
        """Get raw y data without normalization or smoothing.
        
        Used internally for normalization calculations.
        """
        if mode == "Magnitude":
            y = [20 * np.log10(d['s21_mag'] + 1e-12) for d in trace_data]
        elif mode == "Phase":
            y = [d['s21_phase'] for d in trace_data]
        elif mode == "Real":
            y = [d['s21_real'] for d in trace_data]
        elif mode == "Conductivity":
            y = self._calculate_conductivity(trace_data)
        else:  # Imaginary
            y = [d['s21_imag'] for d in trace_data]
        return y
    
    def _get_2d_normalization_reference(self, all_traces, mode):
        """Get normalization reference for 2D data.
        
        For gate as sweep parameter: returns per-trace references (one per step)
        For gate as step parameter: returns reference spectrum to subtract from all
        
        Args:
            all_traces: List of traces (each trace is list of data points)
            mode: Display mode
            
        Returns:
            (reference_data, norm_type, success) tuple
            reference_data: Either list of scalars (per-trace) or array (reference spectrum)
            norm_type: 'per_trace' or 'spectrum'
            success: True if reference was found
        """
        if not self.normalize_at_v.get():
            return None, None, True
        
        if not all_traces or not self.current_config:
            return None, None, False
        
        sweep_param = self.current_config.get('sweep_param', '')
        step_param = self.current_config.get('step_param', '')
        
        try:
            v_norm = float(self.v_norm.get())
        except ValueError:
            return None, None, False
        
        if sweep_param == "Gate Voltage (V)":
            # Gate is sweep parameter - normalize each trace at V_norm
            references = []
            all_success = True
            
            for trace in all_traces:
                if not trace:
                    references.append(0.0)
                    continue
                
                gate_voltages = [d['sweep_value'] for d in trace]
                v_min, v_max = min(gate_voltages), max(gate_voltages)
                
                if v_norm < v_min or v_norm > v_max:
                    all_success = False
                    references.append(0.0)
                    continue
                
                idx = self._find_nearest_index(gate_voltages, v_norm)
                y_data = self._get_trace_y_data_raw(trace, mode)
                
                if idx is not None and idx < len(y_data):
                    references.append(y_data[idx])
                else:
                    references.append(0.0)
            
            return references, 'per_trace', all_success
            
        elif step_param == "Gate Voltage (V)":
            # Gate is step parameter - find reference spectrum and subtract from all
            # Find the trace closest to V_norm
            step_values = []
            for trace in all_traces:
                if trace and len(trace) > 0:
                    step_val = trace[0].get('step_value')
                    if step_val is not None:
                        step_values.append(step_val)
                    else:
                        step_values.append(0)
                else:
                    step_values.append(0)
            
            if not step_values:
                return None, None, False
            
            v_min, v_max = min(step_values), max(step_values)
            
            if v_norm < v_min or v_norm > v_max:
                # V_norm not yet in data - check if it will be
                step_start = self.current_config.get('step_start', 0)
                step_stop = self.current_config.get('step_stop', 0)
                
                if min(step_start, step_stop) <= v_norm <= max(step_start, step_stop):
                    # V_norm will be measured eventually
                    return None, 'spectrum', False
                else:
                    # V_norm outside measurement range
                    return None, None, False
            
            # Find nearest step to V_norm
            ref_idx = self._find_nearest_index(step_values, v_norm)
            
            if ref_idx is None or ref_idx >= len(all_traces):
                return None, None, False
            
            ref_trace = all_traces[ref_idx]
            if not ref_trace:
                return None, None, False
            
            # Get y data for reference trace
            ref_y = self._get_trace_y_data_raw(ref_trace, mode)
            
            return np.array(ref_y), 'spectrum', True
        
        # Gate voltage not involved
        return None, None, True

    def _is_on_colorbar(self, event):
        """Check if mouse event is on the colorbar."""
        if self.colorbar_2d is None or event.inaxes is None:
            return False
        # Check if the event is in the colorbar axes
        return event.inaxes == self.colorbar_2d.ax
    
    def on_colorbar_press(self, event):
        """Handle mouse press on colorbar for interactive adjustment."""
        if not self._is_on_colorbar(event):
            return
        if self.colorbar_mappable is None:
            return
        
        # Disable auto-scale when user starts interacting
        self.auto_scale.set(False)
        self.color_min_entry.config(state='normal')
        self.color_max_entry.config(state='normal')
        
        self.colorbar_dragging = True
        self.colorbar_drag_start_y = event.y
        self.colorbar_drag_start_vmin = self.colorbar_mappable.get_clim()[0]
        self.colorbar_drag_start_vmax = self.colorbar_mappable.get_clim()[1]
        self.colorbar_drag_button = event.button
    
    def on_colorbar_release(self, event):
        """Handle mouse release after colorbar drag."""
        self.colorbar_dragging = False
    
    def on_colorbar_motion(self, event):
        """Handle mouse drag on colorbar to adjust color scale."""
        if not self.colorbar_dragging or self.colorbar_mappable is None:
            return
        if self.colorbar_drag_start_y is None:
            return
        
        # Calculate drag distance (in pixels)
        dy = event.y - self.colorbar_drag_start_y
        
        # Get current range
        vmin = self.colorbar_drag_start_vmin
        vmax = self.colorbar_drag_start_vmax
        vrange = vmax - vmin
        
        # Scale factor: drag 100 pixels = shift by full range
        scale_factor = vrange / 100.0
        
        if self.colorbar_drag_button == 1:  # Left click: shift range (brightness)
            shift = -dy * scale_factor
            new_vmin = vmin + shift
            new_vmax = vmax + shift
        elif self.colorbar_drag_button == 3:  # Right click: adjust contrast
            # Drag up = more contrast (smaller range), drag down = less contrast
            contrast_factor = 1.0 + dy / 100.0
            contrast_factor = max(0.1, min(10.0, contrast_factor))  # Limit range
            center = (vmin + vmax) / 2
            new_range = vrange * contrast_factor
            new_vmin = center - new_range / 2
            new_vmax = center + new_range / 2
        else:
            return
        
        # Apply new limits
        self.colorbar_mappable.set_clim(new_vmin, new_vmax)
        self.color_min.set(f"{new_vmin:.1f}")
        self.color_max.set(f"{new_vmax:.1f}")
        self.canvas_2d.draw_idle()
    
    def on_colorbar_scroll(self, event):
        """Handle scroll wheel on colorbar to adjust contrast."""
        if not self._is_on_colorbar(event):
            return
        if self.colorbar_mappable is None:
            return
        
        # Disable auto-scale
        self.auto_scale.set(False)
        self.color_min_entry.config(state='normal')
        self.color_max_entry.config(state='normal')
        
        # Get current limits
        vmin, vmax = self.colorbar_mappable.get_clim()
        center = (vmin + vmax) / 2
        vrange = vmax - vmin
        
        # Scroll up = zoom in (more contrast), scroll down = zoom out (less contrast)
        if event.button == 'up':
            new_range = vrange * 0.8
        else:
            new_range = vrange * 1.25
        
        # Limit minimum range
        data_range = self.colorbar_data_range[1] - self.colorbar_data_range[0]
        new_range = max(data_range * 0.01, new_range)  # At least 1% of data range
        
        new_vmin = center - new_range / 2
        new_vmax = center + new_range / 2
        
        # Apply new limits
        self.colorbar_mappable.set_clim(new_vmin, new_vmax)
        self.color_min.set(f"{new_vmin:.1f}")
        self.color_max.set(f"{new_vmax:.1f}")
        self.canvas_2d.draw_idle()
    
    def clear_plots(self):
        """Clear all plots and data."""
        self.sweep_data_1d = []
        self.sweep_data_2d = []
        self.current_step_index = 0
        
        # Invalidate Z data cache
        self._z_cache = None
        self._z_cache_key = None
        
        # Reset step sliders (both 1D and 2D mode)
        self.step_slider_var.set(0)
        self.step_slider.config(from_=0, to=0, state='disabled')
        self.step_slider_2d.config(from_=0, to=0, state='disabled')
        self.prev_step_btn.config(state='disabled')
        self.next_step_btn.config(state='disabled')
        self.prev_step_btn_2d.config(state='disabled')
        self.next_step_btn_2d.config(state='disabled')
        self.step_value_label.config(text="")
        self.step_value_label_2d.config(text="")
        
        # Clear averaging state
        self.current_sweep_raw = []
        self.prev_avg_data = []
        self.current_avg_num = 1
        self.current_avg_total = 1
        
        # Clear and recreate trace figures
        self.ax_trace.clear()
        self.ax_trace_2d.clear()
        
        # Clear and recreate 2D figure
        self.fig_2d.clear()
        self.ax_2d = self.fig_2d.add_subplot(111)
        
        self.init_plots()
    
    def get_next_filename(self, base_name, extension=".csv"):
        """Get the next available filename with incremental suffix.
        
        Returns filename like base_001.csv, base_002.csv, etc.
        Checks for both files and folders to avoid conflicts.
        """
        directory = self.data_directory.get()
        counter = 1
        while True:
            filename = f"{base_name}_{counter:03d}{extension}"
            filepath = os.path.join(directory, filename)
            # Also check if a folder with same number exists
            folderpath = os.path.join(directory, f"{base_name}_{counter:03d}")
            if not os.path.exists(filepath) and not os.path.exists(folderpath):
                return filename, filepath
            counter += 1
            if counter > 9999:
                raise RuntimeError("Too many files with same base name")
    
    def get_next_folder(self, base_name):
        """Get the next available folder name with incremental suffix.
        
        Returns folder like base_001/, base_002/, etc.
        Checks for both files and folders to avoid conflicts.
        """
        directory = self.data_directory.get()
        counter = 1
        while True:
            folder_name = f"{base_name}_{counter:03d}"
            folder_path = os.path.join(directory, folder_name)
            # Also check if a file with same number exists
            filepath = os.path.join(directory, f"{base_name}_{counter:03d}.csv")
            if not os.path.exists(folder_path) and not os.path.exists(filepath):
                return folder_name, folder_path
            counter += 1
            if counter > 9999:
                raise RuntimeError("Too many folders with same base name")
    
    def save_single_sweep(self, sweep_data, filepath, step_value=None):
        """Save a single sweep to a CSV file."""
        try:
            with open(filepath, 'w') as f:
                # Write metadata header
                f.write("# VNA FMR Measurement Data\n")
                f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# S-Parameter: {self.s_parameter.get()}\n")
                
                # VNA attenuation info
                try:
                    input_atten = float(self.input_attenuation.get())
                    output_atten = float(self.output_attenuation.get())
                    vna_power = float(self.fixed_power.get())
                    power_at_probe = vna_power - input_atten
                    f.write(f"# Input Attenuation: {input_atten} dB\n")
                    f.write(f"# Output Attenuation: {output_atten} dB\n")
                    f.write(f"# Power at Probe: {power_at_probe} dBm\n")
                except:
                    pass
                
                sweep_param = self.current_config['sweep_param'] if self.current_config else "Sweep"
                is_freq_sweep = sweep_param == "Frequency (GHz)"
                is_power_sweep = sweep_param == "Power (dBm)"
                
                if self.current_config:
                    # Convert sweep range to display units
                    sweep_start = self.current_config['sweep_start']
                    sweep_stop = self.current_config['sweep_stop']
                    if is_freq_sweep:
                        sweep_start = sweep_start / 1e9  # Hz to GHz
                        sweep_stop = sweep_stop / 1e9
                    elif is_power_sweep:
                        # Convert to power at probe
                        try:
                            input_atten = float(self.input_attenuation.get())
                            sweep_start = sweep_start - input_atten
                            sweep_stop = sweep_stop - input_atten
                        except:
                            pass
                    
                    # Write sweep parameter with display label
                    sweep_param_display = self._get_param_display_label(sweep_param)
                    f.write(f"# Sweep Parameter: {sweep_param_display}\n")
                    f.write(f"# Sweep Range: {sweep_start} to {sweep_stop}\n")
                    f.write(f"# Sweep Points: {self.current_config['sweep_points']}\n")
                    f.write(f"# Averages: {self.current_config.get('averages', 1)}\n")
                    
                    if step_value is not None:
                        step_param = self.current_config['step_param']
                        # Convert step value to display units
                        step_display = step_value
                        if step_param == "Frequency (GHz)":
                            step_display = step_value / 1e9
                            f.write(f"# Step Parameter: {step_param}\n")
                            f.write(f"# Step Value: {step_display}\n")
                        elif step_param == "Power (dBm)":
                            # Show power at probe
                            try:
                                input_atten = float(self.input_attenuation.get())
                                step_display = step_value - input_atten
                                f.write(f"# Step Parameter: Power at Probe (dBm)\n")
                                f.write(f"# Step Value: {step_display}\n")
                                f.write(f"# VNA Power: {step_value}\n")
                            except:
                                f.write(f"# Step Parameter: {step_param}\n")
                                f.write(f"# Step Value: {step_display}\n")
                        else:
                            f.write(f"# Step Parameter: {step_param}\n")
                            f.write(f"# Step Value: {step_display}\n")
                    
                    f.write(f"# Fixed Values: {self.current_config['fixed_values']}\n")
                    
                    # Add normalization info if applicable
                    if self.current_config.get('normalization_enabled', False):
                        norm_voltage = self.current_config.get('normalization_voltage', 0.0)
                        f.write(f"# Normalization Enabled: True\n")
                        f.write(f"# Normalization Reference Voltage: {norm_voltage} V\n")
                        if self.reference_data:
                            if self.reference_data.get('single_value_db') is not None:
                                f.write(f"# Reference S21: {self.reference_data['single_value_db']:.4f} dB\n")
                            elif self.reference_data.get('spectrum_db') is not None:
                                f.write(f"# Reference Spectrum Points: {len(self.reference_data['spectrum_db'])}\n")
                
                f.write("#\n")
                
                # Check if we have normalized data
                has_normalized = sweep_data and 's21_mag_db_norm' in sweep_data[0] and sweep_data[0]['s21_mag_db_norm'] is not None
                
                # Column header with proper units
                if is_freq_sweep:
                    header = "# Frequency_GHz, S21_Real, S21_Imag, S21_Mag_dB, S21_Phase_deg"
                elif is_power_sweep:
                    header = "# Power_at_Probe_dBm, S21_Real, S21_Imag, S21_Mag_dB, S21_Phase_deg"
                else:
                    header = f"# {sweep_param.replace(' ', '_')}, S21_Real, S21_Imag, S21_Mag_dB, S21_Phase_deg"
                
                if has_normalized:
                    header += ", S21_Mag_dB_Normalized"
                f.write(header + "\n")
                
                # Write data
                for d in sweep_data:
                    # Convert sweep value to display units
                    sweep_val = d['sweep_value']
                    if is_freq_sweep:
                        sweep_val = sweep_val / 1e9  # Hz to GHz
                    elif is_power_sweep:
                        # Convert to power at probe
                        try:
                            input_atten = float(self.input_attenuation.get())
                            sweep_val = sweep_val - input_atten
                        except:
                            pass
                    
                    # Convert magnitude to dB
                    mag_db = 20 * np.log10(d['s21_mag']) if d['s21_mag'] > 0 else -200
                    f.write(f"{sweep_val}, {d['s21_real']}, {d['s21_imag']}, ")
                    f.write(f"{mag_db}, {d['s21_phase']}")
                    
                    if has_normalized:
                        norm_val = d.get('s21_mag_db_norm')
                        if norm_val is not None:
                            f.write(f", {norm_val}")
                        else:
                            f.write(", ")
                    f.write("\n")
            
            return True
        except Exception as e:
            print(f"Error saving sweep: {e}")
            return False
    
    def save_2d_metadata(self, folder_path):
        """Save metadata text file for a 2D measurement."""
        filepath = os.path.join(folder_path, "metadata.txt")
        try:
            with open(filepath, 'w') as f:
                f.write("VNA FMR 2D Measurement Metadata\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"S-Parameter: {self.s_parameter.get()}\n\n")
                
                # VNA Settings
                f.write("VNA Settings\n")
                f.write("-" * 20 + "\n")
                try:
                    input_atten = float(self.input_attenuation.get())
                    output_atten = float(self.output_attenuation.get())
                    f.write(f"  Input Attenuation: {input_atten} dB\n")
                    f.write(f"  Output Attenuation: {output_atten} dB\n")
                except:
                    pass
                try:
                    vna_settle = float(self.vna_settle_time.get())
                    f.write(f"  VNA Settle Time: {vna_settle} s\n")
                except:
                    pass
                f.write("\n")
                
                # Sweep parameter info (convert Hz to GHz if frequency)
                sweep_param = self.current_config['sweep_param']
                sweep_start = self.current_config['sweep_start']
                sweep_stop = self.current_config['sweep_stop']
                if sweep_param == "Frequency (GHz)":
                    sweep_start = sweep_start / 1e9
                    sweep_stop = sweep_stop / 1e9
                
                f.write("Sweep Parameter\n")
                f.write("-" * 20 + "\n")
                f.write(f"  Parameter: {sweep_param}\n")
                f.write(f"  Start: {sweep_start}\n")
                f.write(f"  Stop: {sweep_stop}\n")
                f.write(f"  Points: {self.current_config['sweep_points']}\n")
                f.write(f"  Averages: {self.current_config.get('averages', 1)}\n\n")
                
                # Step parameter info (convert Hz to GHz if frequency)
                step_param = self.current_config['step_param']
                step_start = self.current_config['step_start']
                step_stop = self.current_config['step_stop']
                if step_param == "Frequency (GHz)":
                    step_start = step_start / 1e9
                    step_stop = step_stop / 1e9
                
                f.write("Step Parameter\n")
                f.write("-" * 20 + "\n")
                f.write(f"  Parameter: {step_param}\n")
                f.write(f"  Start: {step_start}\n")
                f.write(f"  Stop: {step_stop}\n")
                f.write(f"  Points: {self.current_config['step_points']}\n\n")
                
                # Fixed values (convert frequency to GHz)
                f.write("Fixed Values\n")
                f.write("-" * 20 + "\n")
                fixed = self.current_config['fixed_values']
                f.write(f"  Frequency: {fixed['frequency'] / 1e9} GHz\n")
                f.write(f"  B-Field: {fixed['b_field']} T\n")
                f.write(f"  Gate Voltage: {fixed['vg']} V\n")
                f.write(f"  VNA Power: {fixed['power']} dBm\n")
                # Calculate and show power at probe
                try:
                    input_atten = float(self.input_attenuation.get())
                    power_at_probe = fixed['power'] - input_atten
                    f.write(f"  Power at Probe: {power_at_probe} dBm\n")
                except:
                    pass
                # Temperature from Lakeshore 370
                temp = fixed.get('temperature')
                if temp is not None:
                    if temp < 1:  # Less than 1K, show in mK
                        f.write(f"  Temperature: {temp*1000:.2f} mK (from Lakeshore 370)\n")
                    else:
                        f.write(f"  Temperature: {temp:.3f} K (from Lakeshore 370)\n")
                else:
                    f.write(f"  Temperature: Not measured\n")
                f.write(f"  IF Bandwidth: {fixed['ifbw']} Hz\n\n")
                
                # Gate Safety Settings
                f.write("Gate Safety Settings\n")
                f.write("-" * 20 + "\n")
                try:
                    f.write(f"  Slew Rate: {self.gate_slew_rate.get()} V/s\n")
                    f.write(f"  Max Voltage: {self.max_gate_voltage.get()} V\n")
                    f.write(f"  Compliance: {self.gate_compliance.get()} nA\n")
                    f.write(f"  Ramp to Zero After: {self.ramp_gate_to_zero.get()}\n")
                    f.write(f"  Ramp to Zero on Stop: {self.ramp_gate_on_stop.get()}\n")
                except:
                    pass
                f.write("\n")
                
                # B-Field Settings
                f.write("B-Field Settings\n")
                f.write("-" * 20 + "\n")
                try:
                    f.write(f"  Ramp Rate: {self.field_ramp_rate.get()} T/min\n")
                    f.write(f"  Tolerance: {self.field_tolerance.get()} T\n")
                    f.write(f"  Settle Time: {self.field_settle_time.get()} s\n")
                    f.write(f"  Wait for Field: {self.wait_for_field.get()}\n")
                except:
                    pass
                f.write("\n")
                
                # Sample Parameters
                f.write("Sample Parameters\n")
                f.write("-" * 20 + "\n")
                try:
                    f.write(f"  hBN Thickness: {self.hbn_thickness.get()} nm\n")
                    f.write(f"  V_CNP: {self.v_cnp.get()} V\n")
                    if self.sample_density_per_volt:
                        f.write(f"  Density per Volt: {self.sample_density_per_volt:.3e} cm^-2/V\n")
                except:
                    pass
                try:
                    f.write(f"  CPW Slot Width: {self.cpw_slot_width.get()} um\n")
                    f.write(f"  CPW Slot Length: {self.cpw_slot_length.get()} um\n")
                except:
                    pass
                f.write("\n")
                
                # Normalization info
                f.write("Normalization\n")
                f.write("-" * 20 + "\n")
                if self.current_config.get('normalization_enabled', False):
                    norm_v = self.current_config.get('normalization_voltage', 0.0)
                    f.write(f"  Enabled: True\n")
                    f.write(f"  Reference Voltage: {norm_v} V\n")
                    if self.reference_data:
                        if self.reference_data.get('single_value_db') is not None:
                            f.write(f"  Reference S21: {self.reference_data['single_value_db']:.4f} dB\n")
                        elif self.reference_data.get('spectrum_db') is not None:
                            f.write(f"  Reference Type: Frequency spectrum ({len(self.reference_data['spectrum_db'])} points)\n")
                        elif self.reference_data.get('per_step_db') is not None:
                            per_step = self.reference_data['per_step_db']
                            f.write(f"  Reference Type: Per-frequency ({len(per_step)} points)\n")
                            step_freqs = self.reference_data.get('step_frequencies', [])
                            for i, freq in enumerate(step_freqs):
                                if i in per_step and per_step[i] is not None:
                                    f.write(f"    {freq/1e9:.4f} GHz: {per_step[i]:.2f} dB\n")
                    f.write(f"  Reference File: reference.csv\n")
                else:
                    f.write(f"  Enabled: False\n")
                f.write("\n")
                
                # Instrument Connection Info
                f.write("Instruments\n")
                f.write("-" * 20 + "\n")
                f.write(f"  VNA Port: {self.vna_port.get()}\n")
                f.write(f"  Magnet Model: {self.magnet_model.get()}\n")
                f.write(f"  Magnet Address: {self.magnet_addr.get()}\n")
                f.write(f"  Keithley Model: {self.keithley_model.get()}\n")
                f.write(f"  Keithley Address: {self.keithley_addr.get()}\n")
            
            return True
        except Exception as e:
            print(f"Error saving metadata: {e}")
            return False
    
    def save_reference_data(self, folder_path):
        """Save reference measurement data to a CSV file."""
        if not self.reference_data:
            return False
        
        filepath = os.path.join(folder_path, "reference.csv")
        try:
            with open(filepath, 'w') as f:
                f.write("# VNA FMR Reference Measurement\n")
                f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Reference Voltage: {self.reference_data.get('voltage', 0)} V\n")
                f.write(f"# S-Parameter: {self.s_parameter.get()}\n")
                
                if self.reference_data.get('single_value_db') is not None:
                    # Single CW reference (for gate sweeps)
                    f.write(f"# Type: Single CW measurement\n")
                    f.write(f"# S21_Mag_dB: {self.reference_data['single_value_db']:.6f}\n")
                
                elif self.reference_data.get('spectrum_db') is not None:
                    # Full spectrum reference (for freq sweep with gate step)
                    spectrum = self.reference_data['spectrum_db']
                    frequencies = self.reference_data.get('frequencies', [])
                    
                    f.write(f"# Type: Frequency spectrum\n")
                    f.write(f"# Points: {len(spectrum)}\n")
                    f.write("#\n")
                    f.write("# Frequency_GHz, S21_Mag_dB\n")
                    
                    for i, s21_db in enumerate(spectrum):
                        freq_ghz = frequencies[i] / 1e9 if i < len(frequencies) else i
                        f.write(f"{freq_ghz}, {s21_db}\n")
                
                elif self.reference_data.get('per_step_db') is not None:
                    # Per-frequency reference (for gate sweep with freq step)
                    per_step = self.reference_data['per_step_db']
                    step_freqs = self.reference_data.get('step_frequencies', [])
                    
                    f.write(f"# Type: Per-frequency CW measurements\n")
                    f.write(f"# Points: {len(per_step)}\n")
                    f.write("#\n")
                    f.write("# Frequency_GHz, S21_Mag_dB\n")
                    
                    for i, freq in enumerate(step_freqs):
                        if i in per_step and per_step[i] is not None:
                            f.write(f"{freq/1e9}, {per_step[i]}\n")
            
            print(f"Saved reference data: {filepath}")
            return True
        except Exception as e:
            print(f"Error saving reference data: {e}")
            return False
    
    def auto_save_sweep(self, sweep_data, step_idx=None, step_value=None):
        """Auto-save a sweep after completion."""
        print(f"[AUTO-SAVE] Called with step_idx={step_idx}, step_value={step_value}, data_len={len(sweep_data) if sweep_data else 0}")
        
        if not self.auto_save.get():
            print("[AUTO-SAVE] Auto-save is disabled")
            return
        
        # Ensure data directory exists
        data_dir = self.data_directory.get()
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"Created data directory: {data_dir}")
        
        base_name = self.filename.get().strip()
        if not base_name:
            base_name = "fmr_data"
        
        # Remove extension if user included one
        if base_name.endswith('.csv'):
            base_name = base_name[:-4]
        
        # Append sweep type to filename
        sweep_type_map = {
            "Frequency (GHz)": "Freq",
            "Gate Voltage (V)": "Gate",
            "B-Field (T)": "Field",
            "Power (dBm)": "Power"
        }
        if self.current_config:
            sweep_param = self.current_config.get('sweep_param', '')
            sweep_suffix = sweep_type_map.get(sweep_param, '')
            if sweep_suffix:
                base_name = f"{base_name}_{sweep_suffix}"
        
        is_2d = self.current_config and self.current_config.get('step_param')
        
        try:
            if is_2d:
                # 2D measurement: save to folder
                if self.current_2d_folder is None:
                    # First sweep of 2D measurement - create folder
                    folder_name, folder_path = self.get_next_folder(base_name)
                    os.makedirs(folder_path)
                    self.current_2d_folder = folder_path
                    self.save_2d_metadata(folder_path)
                    
                    # Save reference data if available
                    if self.reference_data and self.current_config.get('normalization_enabled', False):
                        self.save_reference_data(folder_path)
                    
                    print(f"Created 2D measurement folder: {folder_path}")
                
                # Save individual sweep
                sweep_filename = f"sweep_{step_idx+1:03d}.csv"
                filepath = os.path.join(self.current_2d_folder, sweep_filename)
                
                if self.save_single_sweep(sweep_data, filepath, step_value):
                    self.status_var.set(f"Auto-saved: {sweep_filename}")
                    print(f"Auto-saved sweep {step_idx+1}: {filepath}")
            else:
                # 1D measurement: save with incremental suffix
                filename, filepath = self.get_next_filename(base_name)
                
                if self.save_single_sweep(sweep_data, filepath):
                    self.status_var.set(f"Auto-saved: {filename}")
                    print(f"Auto-saved: {filepath}")
                    
                    # Save plot image
                    self.save_plot_images(filepath)
                    
                    # Save log file alongside data
                    log_filepath = log_manager.get_log_filepath(filepath)
                    log_manager.save_to_file(log_filepath)
        
        except Exception as e:
            print(f"Auto-save error: {e}")
            self.status_var.set(f"Auto-save failed: {e}")
    
    def save_plot_images(self, data_filepath):
        """Save plot images alongside data file."""
        try:
            # Get base path without extension
            base_path = os.path.splitext(data_filepath)[0]
            
            # Save single trace plot
            trace_path = f"{base_path}_single.png"
            self.fig_trace.savefig(trace_path, dpi=150, bbox_inches='tight', 
                                   facecolor='white', edgecolor='none')
            print(f"Saved trace plot: {trace_path}")
            
            # Save 2D map if we have 2D data
            is_2d = self.current_config and self.current_config.get('step_param')
            if is_2d and len(self.sweep_data_2d) > 1:
                map_path = f"{base_path}_map.png"
                self.fig_2d.savefig(map_path, dpi=150, bbox_inches='tight',
                                    facecolor='white', edgecolor='none')
                print(f"Saved 2D map: {map_path}")
                
        except Exception as e:
            print(f"Error saving plot images: {e}")
    
    def save_data(self):
        """Save measurement data to file (manual save button)."""
        if not self.sweep_data_2d:
            messagebox.showwarning("No Data", "No data to save!")
            return
        
        # Ensure data directory exists
        data_dir = self.data_directory.get()
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"Created data directory: {data_dir}")
        
        base_name = self.filename.get().strip()
        if not base_name:
            base_name = "fmr_data"
        if base_name.endswith('.csv'):
            base_name = base_name[:-4]
        
        # Append sweep type to filename
        sweep_type_map = {
            "Frequency (GHz)": "Freq",
            "Gate Voltage (V)": "Gate",
            "B-Field (T)": "Field",
            "Power (dBm)": "Power"
        }
        if self.current_config:
            sweep_param = self.current_config.get('sweep_param', '')
            sweep_suffix = sweep_type_map.get(sweep_param, '')
            if sweep_suffix:
                base_name = f"{base_name}_{sweep_suffix}"
        
        is_2d = self.current_config and self.current_config.get('step_param')
        
        try:
            if is_2d:
                # Save as folder with individual sweeps
                folder_name, folder_path = self.get_next_folder(base_name)
                os.makedirs(folder_path)
                
                # Save metadata
                self.save_2d_metadata(folder_path)
                
                # Save each sweep
                for step_idx, sweep_data in enumerate(self.sweep_data_2d):
                    if sweep_data:
                        step_value = sweep_data[0].get('step_value') if sweep_data else None
                        sweep_filename = f"sweep_{step_idx+1:03d}.csv"
                        filepath = os.path.join(folder_path, sweep_filename)
                        self.save_single_sweep(sweep_data, filepath, step_value)
                
                # Save plot images in the folder
                self.save_plot_images(os.path.join(folder_path, "plot"))
                
                self.status_var.set(f"Data saved to {folder_name}/")
                messagebox.showinfo("Save Complete", f"2D data saved to:\n{folder_path}")
            else:
                # Single 1D sweep
                filename, filepath = self.get_next_filename(base_name)
                
                if self.sweep_data_2d and self.sweep_data_2d[0]:
                    self.save_single_sweep(self.sweep_data_2d[0], filepath)
                    self.save_plot_images(filepath)
                    self.status_var.set(f"Data saved to {filename}")
                    messagebox.showinfo("Save Complete", f"Data saved to:\n{filepath}")
        
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save data: {e}")
    
    def save_settings(self):
        """Save current settings to file."""
        settings = {
            # Instrument settings
            'vna_port': self.vna_port.get(),
            'magnet_addr': self.magnet_addr.get(),
            'magnet_model': self.magnet_model.get(),
            'keithley_addr': self.keithley_addr.get(),
            'keithley_model': self.keithley_model.get(),
            'temp_gpib_address': self.temp_gpib_address.get(),
            'temp_channel': self.temp_channel.get(),
            
            # Sweep parameters
            'sweep_param': self.sweep_param.get(),
            'sweep_start': self.sweep_start.get(),
            'sweep_stop': self.sweep_stop.get(),
            'sweep_points': self.sweep_points.get(),
            'sweep_averages': self.sweep_averages.get(),
            
            # Step parameters
            'step_param': self.step_param.get(),
            'step_start': self.step_start.get(),
            'step_stop': self.step_stop.get(),
            'step_points': self.step_points.get(),
            
            # Fixed parameters
            'fixed_frequency': self.fixed_frequency.get(),
            'fixed_field': self.fixed_field.get(),
            'fixed_gate': self.fixed_gate.get(),
            'fixed_power': self.fixed_power.get(),
            'ifbw': self.ifbw.get(),
            'vna_settle_time': self.vna_settle_time.get(),
            'input_attenuation': self.input_attenuation.get(),
            'output_attenuation': self.output_attenuation.get(),
            
            # S-parameter
            's_parameter': self.s_parameter.get(),
            
            # Sample parameters
            'hbn_thickness': self.hbn_thickness.get(),
            'v_cnp': self.v_cnp.get(),
            'cpw_slot_width': self.cpw_slot_width.get(),
            'cpw_slot_length': self.cpw_slot_length.get(),
            
            # Gate safety
            'gate_slew_rate': self.gate_slew_rate.get(),
            'max_gate_voltage': self.max_gate_voltage.get(),
            'gate_compliance': self.gate_compliance.get(),
            'ramp_gate_to_zero': self.ramp_gate_to_zero.get(),
            'ramp_gate_on_stop': self.ramp_gate_on_stop.get(),
            
            # B-field settings
            'field_ramp_rate': self.field_ramp_rate.get(),
            'field_tolerance': self.field_tolerance.get(),
            'field_settle_time': self.field_settle_time.get(),
            'wait_for_field': self.wait_for_field.get(),
            
            # File settings
            'data_directory': self.data_directory.get(),
            'filename': self.filename.get(),
            'auto_save': self.auto_save.get(),
            
            # Display settings
            'xaxis_mode': self.xaxis_mode.get(),
            'trace_display_mode': self.trace_display_mode.get(),
            'smoothing_window': self.smoothing_window.get(),
            'normalize_at_v': self.normalize_at_v.get(),
            'v_norm': self.v_norm.get(),
            
            # CNP normalization settings
            'gate_normalization_enabled': self.gate_normalization_enabled.get(),
            'gate_normalization_voltage': self.gate_normalization_voltage.get(),
            
            # Field normalization settings
            'field_normalization_enabled': self.field_normalization_enabled.get(),
            'field_normalization_field': self.field_normalization_field.get(),
        }
        
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save settings: {e}")
    
    def load_settings(self):
        """Load settings from file."""
        if not os.path.exists(self.settings_file):
            return
        
        try:
            with open(self.settings_file, 'r') as f:
                settings = json.load(f)
            
            # Instrument settings
            if 'vna_port' in settings: self.vna_port.set(settings['vna_port'])
            if 'magnet_addr' in settings: self.magnet_addr.set(settings['magnet_addr'])
            if 'magnet_model' in settings: self.magnet_model.set(settings['magnet_model'])
            if 'keithley_addr' in settings: self.keithley_addr.set(settings['keithley_addr'])
            if 'keithley_model' in settings: self.keithley_model.set(settings['keithley_model'])
            if 'temp_gpib_address' in settings: self.temp_gpib_address.set(settings['temp_gpib_address'])
            if 'temp_channel' in settings: self.temp_channel.set(settings['temp_channel'])
            
            # Sweep parameters
            if 'sweep_param' in settings: self.sweep_param.set(settings['sweep_param'])
            if 'sweep_start' in settings: self.sweep_start.set(settings['sweep_start'])
            if 'sweep_stop' in settings: self.sweep_stop.set(settings['sweep_stop'])
            if 'sweep_points' in settings: self.sweep_points.set(settings['sweep_points'])
            if 'sweep_averages' in settings: self.sweep_averages.set(settings['sweep_averages'])
            
            # Step parameters
            if 'step_param' in settings: self.step_param.set(settings['step_param'])
            if 'step_start' in settings: self.step_start.set(settings['step_start'])
            if 'step_stop' in settings: self.step_stop.set(settings['step_stop'])
            if 'step_points' in settings: self.step_points.set(settings['step_points'])
            
            # Fixed parameters
            if 'fixed_frequency' in settings: self.fixed_frequency.set(settings['fixed_frequency'])
            if 'fixed_field' in settings: self.fixed_field.set(settings['fixed_field'])
            if 'fixed_gate' in settings: self.fixed_gate.set(settings['fixed_gate'])
            if 'fixed_power' in settings: self.fixed_power.set(settings['fixed_power'])
            if 'ifbw' in settings: self.ifbw.set(settings['ifbw'])
            if 'vna_settle_time' in settings: self.vna_settle_time.set(settings['vna_settle_time'])
            if 'input_attenuation' in settings: self.input_attenuation.set(settings['input_attenuation'])
            if 'output_attenuation' in settings: self.output_attenuation.set(settings['output_attenuation'])
            
            # S-parameter
            if 's_parameter' in settings: self.s_parameter.set(settings['s_parameter'])
            
            # Sample parameters
            if 'hbn_thickness' in settings: self.hbn_thickness.set(settings['hbn_thickness'])
            if 'v_cnp' in settings: self.v_cnp.set(settings['v_cnp'])
            if 'cpw_slot_width' in settings: self.cpw_slot_width.set(settings['cpw_slot_width'])
            if 'cpw_slot_length' in settings: self.cpw_slot_length.set(settings['cpw_slot_length'])
            
            # Gate safety
            if 'gate_slew_rate' in settings: self.gate_slew_rate.set(settings['gate_slew_rate'])
            if 'max_gate_voltage' in settings: self.max_gate_voltage.set(settings['max_gate_voltage'])
            if 'gate_compliance' in settings: self.gate_compliance.set(settings['gate_compliance'])
            if 'ramp_gate_to_zero' in settings: self.ramp_gate_to_zero.set(settings['ramp_gate_to_zero'])
            if 'ramp_gate_on_stop' in settings: self.ramp_gate_on_stop.set(settings['ramp_gate_on_stop'])
            
            # B-field settings
            if 'field_ramp_rate' in settings: self.field_ramp_rate.set(settings['field_ramp_rate'])
            if 'field_tolerance' in settings: self.field_tolerance.set(settings['field_tolerance'])
            if 'field_settle_time' in settings: self.field_settle_time.set(settings['field_settle_time'])
            if 'wait_for_field' in settings: self.wait_for_field.set(settings['wait_for_field'])
            
            # File settings
            if 'data_directory' in settings: self.data_directory.set(settings['data_directory'])
            if 'filename' in settings: self.filename.set(settings['filename'])
            if 'auto_save' in settings: self.auto_save.set(settings['auto_save'])
            
            # Display settings
            if 'xaxis_mode' in settings: self.xaxis_mode.set(settings['xaxis_mode'])
            if 'trace_display_mode' in settings: self.trace_display_mode.set(settings['trace_display_mode'])
            if 'smoothing_window' in settings: self.smoothing_window.set(settings['smoothing_window'])
            if 'normalize_at_v' in settings: self.normalize_at_v.set(settings['normalize_at_v'])
            if 'v_norm' in settings: self.v_norm.set(settings['v_norm'])
            
            # CNP normalization settings
            if 'gate_normalization_enabled' in settings: self.gate_normalization_enabled.set(settings['gate_normalization_enabled'])
            if 'gate_normalization_voltage' in settings: self.gate_normalization_voltage.set(settings['gate_normalization_voltage'])
            
            # Field normalization settings
            if 'field_normalization_enabled' in settings: self.field_normalization_enabled.set(settings['field_normalization_enabled'])
            if 'field_normalization_field' in settings: self.field_normalization_field.set(settings['field_normalization_field'])
            
            # Update step parameter UI state
            self.on_step_param_changed()
            
            # Update normalization visibility
            self.update_normalization_visibility()
            
            # Update sample calculations
            self.update_sample_calculations()
            
            print(f"Settings loaded from {self.settings_file}")
            
        except Exception as e:
            print(f"Warning: Could not load settings: {e}")
    
    def on_closing(self):
        """Handle window close - save settings and cleanup."""
        # Save settings
        self.save_settings()
        
        # Stop log capture
        log_manager.stop_capture()
        
        # Stop any running measurement
        self.measurement_engine.stop_flag = True
        
        # Close the window
        self.root.destroy()


def main():
    """Main entry point."""
    print("=" * 60)
    print("VNA FMR Measurement System v3.5")
    print("Villanova University - Dietrich Lab")
    print("=" * 60)
    print("\nFeatures:")
    print("- Flexible sweep/step parameter selection")
    print("- 1D and 2D measurement modes")
    print("- Real-time visualization")
    print("- Gate voltage normalization for gate sweeps/steps")
    print("- B-field normalization for field sweeps/steps")
    print("- Simulation mode for GUI testing")
    print("- In-app log display with auto-save")
    print("\nStarting GUI...")
    
    root = tk.Tk()
    app = VNAMeasurementApp(root)
    
    def on_closing():
        # Try to get gate voltage safely
        gate_voltage, reliable = app.keithley.get_voltage_safe()
        
        if not reliable:
            # Communication problem - warn user
            if messagebox.askyesno("Warning", 
                "Cannot communicate with Keithley.\n\n"
                "The gate voltage state is UNKNOWN.\n\n"
                "Do you want to attempt emergency shutdown?"):
                app.keithley.emergency_shutdown()
            # Still disconnect other instruments
            _cleanup_instruments()
            root.destroy()
            return
        
        if app.measurement_engine.is_running:
            if messagebox.askokcancel("Quit", "Measurement in progress. Stop and quit?"):
                app.measurement_engine.stop()
                time.sleep(0.5)  # Give measurement time to stop
                
                # Ramp gate to zero if not already
                if abs(gate_voltage) > 0.01:
                    app.status_var.set("Ramping gate to zero before exit...")
                    root.update()
                    try:
                        app.keithley.ramp_to_voltage(0.0)
                    except Exception as e:
                        print(f"Ramp failed: {e}")
                        if messagebox.askyesno("Error", 
                            f"Ramp to zero failed: {e}\n\nAttempt emergency shutdown?"):
                            app.keithley.emergency_shutdown()
                
                _cleanup_instruments()
                root.destroy()
        else:
            # Not running, but check gate voltage
            if abs(gate_voltage) > 0.01:
                if messagebox.askyesno("Gate Voltage", 
                    f"Gate voltage is at {gate_voltage:.3f}V.\n\nRamp to zero before exiting?"):
                    app.status_var.set("Ramping gate to zero...")
                    root.update()
                    try:
                        app.keithley.ramp_to_voltage(0.0)
                    except Exception as e:
                        print(f"Ramp failed: {e}")
                        if messagebox.askyesno("Error", 
                            f"Ramp to zero failed: {e}\n\nAttempt emergency shutdown?"):
                            app.keithley.emergency_shutdown()
            _cleanup_instruments()
            root.destroy()
    
    def _cleanup_instruments():
        """Properly disconnect all instruments before exit."""
        print("Cleaning up instrument connections...")
        
        # Save settings first
        try:
            app.save_settings()
            print("Settings saved.")
        except Exception as e:
            print(f"Settings save error: {e}")
        
        # Disconnect magnet (important for SCM1 TCP connection!)
        try:
            if app.magnet and app.magnet.connected:
                print("Disconnecting magnet...")
                app.magnet.disconnect()
        except Exception as e:
            print(f"Magnet disconnect error: {e}")
        
        # Disconnect VNA
        try:
            if app.vna and app.vna.connected:
                print("Disconnecting VNA...")
                app.vna.disconnect()
        except Exception as e:
            print(f"VNA disconnect error: {e}")
        
        # Disconnect Keithley
        try:
            if app.keithley and app.keithley.connected:
                print("Disconnecting Keithley...")
                app.keithley.disconnect()
        except Exception as e:
            print(f"Keithley disconnect error: {e}")
        
        # Disconnect temperature controller
        try:
            if app.temp_controller and app.temp_controller.connected:
                print("Disconnecting temperature controller...")
                app.temp_controller.disconnect()
        except Exception as e:
            print(f"Temperature controller disconnect error: {e}")
        
        print("Cleanup complete.")
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()