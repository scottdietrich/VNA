"""
VNA-Based FMR Measurement System
For Copper Mountain S5180B Vector Network Analyzer

Features:
- Flexible sweep/step parameter selection (Frequency, B-field, Gate Voltage, Power)
- 1D sweeps or 2D sweep+step measurements
- Real-time visualization with Mag/Phase and Re/Im toggle
- Contour plots for 2D data
- Simulated data mode for GUI testing

Hardware (when connected):
- Copper Mountain S5180B VNA via TCP socket (localhost:5025)
- Cryomagnetics 4G Magnet Power Supply via GPIB
- Keithley 2400 SMU for gate voltage via GPIB

Author: Claude (Anthropic) for Scott Dietrich, Villanova University
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import queue
import os
import socket
import subprocess
import json
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
    To enable: In S2VNA, go to System → Misc Setup → Network Setup
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
    
    def setup_frequency_sweep(self, f_start, f_stop, num_points, ifbw, power):
        """Configure VNA for frequency sweep measurement."""
        if not self.connected:
            return False
        
        print(f"VNA setup: {f_start/1e9:.3f}-{f_stop/1e9:.3f} GHz, {num_points} pts, IFBW={ifbw}, P={power} dBm, {self.s_parameter}")
        
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
        
        # Set source power
        self.write(f":SOUR:POW {power}")
        
        # Set sweep type to linear
        self.write(":SENS:SWE:TYPE LIN")
        
        # Set S-parameter
        self.write(f":CALC1:PAR1:DEF {self.s_parameter}")
        
        # Wait for VNA to process settings
        time.sleep(0.2)
        
        # Verify the point count was accepted
        actual_points = self.query(":SENS:SWE:POIN?")
        if actual_points:
            try:
                actual = int(float(actual_points))
                if actual != num_points:
                    print(f"WARNING: VNA has {actual} points, expected {num_points}")
                else:
                    print(f"VNA configured: {actual} points confirmed")
            except:
                pass
        
        return True
    
    def setup_cw_mode(self, frequency, ifbw, power):
        """Configure VNA for CW (single frequency) measurement."""
        if not self.connected:
            return False
        
        # Abort any ongoing operation
        self.write(":ABOR")
        
        # For CW mode: set start = stop = CW frequency, 1 point
        self.write(f":SENS:FREQ:STAR {frequency}")
        self.write(f":SENS:FREQ:STOP {frequency}")
        self.write(":SENS:SWE:POIN 1")
        
        # Set IF bandwidth
        self.write(f":SENS:BAND {ifbw}")
        
        # Set source power
        self.write(f":SOUR:POW {power}")
        
        # Set S-parameter
        self.write(f":CALC1:PAR1:DEF {self.s_parameter}")
        
        # Allow VNA to settle after configuration change
        time.sleep(0.1)
        
        return True
    
    def trigger_sweep(self):
        """Trigger a sweep and wait for completion (for CW mode)."""
        if not self.connected:
            return False
        
        # Abort any ongoing sweep
        self.write(":ABOR")
        
        # Single sweep mode
        self.write(":INIT:CONT OFF")
        
        # Trigger the sweep
        self.write(":INIT:IMM")
        
        # For CW mode (1 point), wait a fixed time
        # 1 point / IFBW + overhead = ~0.5 seconds should be plenty
        time.sleep(0.5)
        
        return True
    
    def trigger_sweep_timed(self, num_points, ifbw):
        """Trigger sweep and wait for completion.
        
        Args:
            num_points: Number of sweep points
            ifbw: IF bandwidth in Hz
        """
        if not self.connected:
            return False
        
        # Calculate expected sweep time with safety margin
        # Base sweep time plus 50% for VNA internal processing overhead
        sweep_time = float(num_points) / float(ifbw)
        adjusted_sweep_time = sweep_time * 1.5  # 50% extra for VNA overhead
        
        # Add fixed overhead for data processing
        overhead = 2.0
        wait_time = adjusted_sweep_time + overhead
        
        print(f"Triggering sweep: {num_points} pts, IFBW={ifbw} Hz, waiting {wait_time:.1f}s")
        
        # Abort any ongoing sweep
        self.write(":ABOR")
        time.sleep(0.1)
        
        # Ensure single sweep mode
        self.write(":INIT:CONT OFF")
        time.sleep(0.1)
        
        # Trigger the sweep
        self.write(":INIT:IMM")
        
        # Wait for sweep to complete
        start = time.time()
        time.sleep(wait_time)
        elapsed = time.time() - start
        
        print(f"Sweep wait complete (waited {elapsed:.1f}s)")
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
    """Controller for Cryomagnetics 4G Magnet Power Supply.
    
    Copied from existing cryostat code with same interface.
    """
    
    def __init__(self, resource_manager=None):
        self.connected = False
        self.address = "GPIB0::21::INSTR"
        self.instrument = None
        self.rm = resource_manager
        self.current_field = 0.0
        self.max_field = 6.0
        self.ramp_rate = 0.01  # T/s
        
    def set_address(self, address):
        """Update GPIB address."""
        self.address = f"GPIB0::{address}::INSTR"
    
    def connect(self):
        """Connect to magnet controller."""
        # TODO: Implement real connection using pyvisa
        self.connected = True
        return True
    
    def disconnect(self):
        """Disconnect from magnet controller."""
        self.connected = False
    
    def get_field(self):
        """Get current magnetic field in Tesla."""
        return self.current_field
    
    def set_field(self, target_field):
        """Set magnetic field (initiates ramp)."""
        # TODO: Implement real field setting
        self.current_field = target_field
        return True
    
    def wait_for_field(self, target_field, tolerance=0.001):
        """Wait until field reaches target."""
        # TODO: Implement real wait
        pass


class KeithleyController:
    """Controller for Keithley 2400 SMU (gate voltage).
    
    Includes safe voltage ramping from existing code.
    """
    
    def __init__(self, resource_manager=None):
        self.connected = False
        self.address = "GPIB0::24::INSTR"
        self.instrument = None
        self.rm = resource_manager
        self.current_voltage = 0.0
        self.max_voltage = 100.0
        self.compliance_current = 100e-9
        self.slew_rate = 1.0  # V/s
        
    def set_address(self, address):
        """Update GPIB address."""
        self.address = f"GPIB0::{address}::INSTR"
    
    def connect(self):
        """Connect to Keithley."""
        # TODO: Implement real connection
        self.connected = True
        return True
    
    def disconnect(self):
        """Disconnect from Keithley."""
        self.connected = False
    
    def get_voltage(self):
        """Get current voltage."""
        return self.current_voltage
    
    def set_voltage(self, voltage):
        """Set voltage (with ramping for safety)."""
        # TODO: Implement real voltage setting with ramping
        self.current_voltage = voltage
        return True
    
    def ramp_to_voltage(self, target, slew_rate=None):
        """Safely ramp to target voltage."""
        # TODO: Implement ramped voltage change
        self.current_voltage = target
        return True


class TemperatureController:
    """Controller for CTC100 or similar temperature controller.
    
    Handles temperature setpoint and readback for sample stage.
    """
    
    def __init__(self, resource_manager=None):
        self.connected = False
        self.port = ""  # Serial port like COM3 or /dev/ttyUSB0
        self.instrument = None
        self.rm = resource_manager
        self.current_temperature = 300.0  # Room temperature default
        self.setpoint = 300.0
        self.max_temperature = 400.0  # K
        self.min_temperature = 4.0    # K (for cryogenic systems)
        self.ramp_rate = 1.0  # K/min
        
    def set_port(self, port):
        """Update serial port."""
        self.port = port
    
    def connect(self):
        """Connect to temperature controller."""
        # TODO: Implement real connection
        # For CTC100, this would be serial/USB connection
        # import serial
        # self.instrument = serial.Serial(self.port, 9600, timeout=1)
        self.connected = True
        return True
    
    def disconnect(self):
        """Disconnect from temperature controller."""
        if self.instrument:
            try:
                self.instrument.close()
            except:
                pass
        self.connected = False
    
    def get_temperature(self):
        """Get current temperature reading."""
        # TODO: Implement real temperature readback
        # For CTC100: self.instrument.query("getOutput.01?")
        return self.current_temperature
    
    def set_temperature(self, target):
        """Set temperature setpoint."""
        if target > self.max_temperature:
            print(f"Target {target}K exceeds maximum {self.max_temperature}K")
            return False
        if target < self.min_temperature:
            print(f"Target {target}K below minimum {self.min_temperature}K")
            return False
        
        # TODO: Implement real temperature setting
        # For CTC100: self.instrument.write(f"setOutput.01 {target}")
        self.setpoint = target
        self.current_temperature = target  # In simulation, instant change
        return True
    
    def wait_for_temperature(self, target, tolerance=0.5, timeout=600):
        """Wait for temperature to stabilize at target."""
        # TODO: Implement real wait with polling
        # In simulation, assume instant
        return True


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
        
        # Debug output
        print(f"=== Measurement Configuration ===")
        print(f"Sweep: {sweep_param} ({sweep_start} to {sweep_stop}, {sweep_points} pts)")
        print(f"Step: {step_param} ({step_start} to {step_stop}, {step_points} pts)" if step_param else "Step: None")
        print(f"Mode: {'VNA BATCH SWEEP' if use_vna_sweep else 'POINT-BY-POINT'}")
        print(f"Simulation: {self.use_simulation}, VNA connected: {self.vna.connected}")
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
                
                print(f"Initial VNA sweep config: {sweep_start}-{sweep_stop} Hz, {sweep_points} pts, "
                      f"IFBW={fixed_values.get('ifbw', 100)}, Power={initial_power} dBm")
                
                self.vna.setup_frequency_sweep(
                    sweep_start, sweep_stop, sweep_points,
                    fixed_values.get('ifbw', 100),
                    initial_power
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
        
        try:
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
                            self.vna.setup_frequency_sweep(
                                sweep_start, sweep_stop, sweep_points,
                                fixed_values.get('ifbw', 100),
                                step_val  # Use stepped power value
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
                        elif step_param == "Power (dBm)":
                            fixed_values['power'] = step_val
                        elif step_param == "B-Field (T)":
                            fixed_values['b_field'] = step_val
                        elif step_param == "Gate Voltage (V)":
                            fixed_values['vg'] = step_val
                        elif step_param == "Temperature (K)":
                            fixed_values['temperature'] = step_val
                        
                        self._set_parameter(step_param, step_val, fixed_values)
                    
                    time.sleep(0.1)  # Settling time for step parameter
                
                sweep_data = []
                
                # === FAST PATH: VNA native frequency sweep ===
                if use_vna_sweep:
                    print(f"Running VNA sweep {step_idx + 1}/{len(step_values)}...")
                    
                    # Trigger single VNA sweep with calculated wait time
                    ifbw = fixed_values.get('ifbw', 100)
                    if not self.vna.trigger_sweep_timed(sweep_points, ifbw):
                        print("Warning: VNA trigger_sweep_timed returned False")
                    
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
                    
                    print(f"Got {len(s21_array)} points from VNA")
                    
                    # Process all points into sweep_data
                    for sweep_idx, (sweep_val, s21) in enumerate(zip(sweep_values, s21_array)):
                        if self.should_stop:
                            break
                        
                        current_point += 1
                        progress = current_point / total_points * 100
                        self.progress_queue.put(progress)
                        
                        sweep_data.append({
                            'sweep_value': sweep_val,
                            'step_value': step_val,
                            's21_real': np.real(s21),
                            's21_imag': np.imag(s21),
                            's21_mag': np.abs(s21),
                            's21_phase': np.angle(s21, deg=True)
                        })
                    
                    # Send ALL data points to GUI at once using batch message
                    self.data_queue.put({
                        'type': 'batch',
                        'step_idx': step_idx,
                        'sweep_data': sweep_data.copy()
                    })
                
                # === SLOW PATH: Point-by-point measurement ===
                else:
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
                        
                        sweep_data.append({
                            'sweep_value': sweep_val,
                            'step_value': step_val,
                            's21_real': np.real(s21),
                            's21_imag': np.imag(s21),
                            's21_mag': np.abs(s21),
                            's21_phase': np.angle(s21, deg=True)
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
            self.data_queue.put({
                'type': 'error',
                'message': str(e)
            })
        
        finally:
            self.is_running = False
    
    def _set_parameter(self, param, value, fixed_values, sweep_config=None):
        """Set a parameter value (real or simulated).
        
        Args:
            param: Parameter name to set
            value: Value to set
            fixed_values: Dictionary of fixed parameter values
            sweep_config: Optional dict with sweep_param, sweep_start, sweep_stop, sweep_points
                         Used to reconfigure VNA sweep when stepping power during freq sweep
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
                self.magnet.set_field(value)
        elif param == "Gate Voltage (V)":
            if not self.use_simulation and self.keithley.connected:
                self.keithley.set_voltage(value)
        elif param == "Power (dBm)":
            if not self.use_simulation and self.vna.connected:
                # Check if we're doing a frequency sweep (need to reconfigure full sweep)
                if sweep_config and sweep_config.get('sweep_param') == "Frequency (GHz)":
                    self.vna.setup_frequency_sweep(
                        sweep_config['sweep_start'],
                        sweep_config['sweep_stop'],
                        sweep_config['sweep_points'],
                        fixed_values.get('ifbw', 100),
                        value  # New power
                    )
                else:
                    # CW mode - update power and flush
                    freq = fixed_values.get('frequency', 8e9)
                    ifbw = fixed_values.get('ifbw', 100)
                    self.vna.setup_cw_mode(freq, ifbw, value)
                    # Trigger a dummy sweep to flush old data at new power
                    self.vna.trigger_sweep()
        elif param == "Temperature (K)":
            if not self.use_simulation and self.temp_controller.connected:
                self.temp_controller.set_temperature(value)
    
    def _get_simulated_data(self, sweep_param, sweep_val, step_param, step_val, fixed):
        """Generate simulated measurement data."""
        # Build parameter dict
        params = {
            'frequency': fixed.get('frequency', 8e9),
            'b_field': fixed.get('b_field', 0.0),
            'vg': fixed.get('vg', 0.0),
            'power': fixed.get('power', -10),
            'temperature': fixed.get('temperature', 300.0)
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
        elif sweep_param == "Temperature (K)":
            params['temperature'] = sweep_val
        
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
            elif step_param == "Temperature (K)":
                params['temperature'] = step_val
        
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
        elif sweep_param == "Temperature (K)":
            return self.sim_data.generate_s21_vs_temperature(
                np.array([params['temperature']]),
                params['frequency'], params['b_field'], params['vg'], params['power']
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
        
        # Initialize instruments (simulation mode by default)
        self.vna = VNAController()
        self.magnet = MagnetController()
        self.keithley = KeithleyController()
        self.temp_controller = TemperatureController()
        self.measurement_engine = MeasurementEngine(
            self.vna, self.magnet, self.keithley, self.temp_controller, use_simulation=True
        )
        
        # Data storage
        self.sweep_data_1d = []
        self.sweep_data_2d = []
        self.current_config = None
        self.current_step_index = 0
        
        # GUI variables
        self.setup_variables()
        
        # Build interface
        self.build_interface()
        
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
        self.magnet_addr = tk.StringVar(value="21")
        self.keithley_addr = tk.StringVar(value="24")
        self.temp_port = tk.StringVar(value="COM3")
        
        # Sweep parameters
        self.sweep_param = tk.StringVar(value="Frequency (GHz)")
        self.sweep_start = tk.StringVar(value="0.0001")
        self.sweep_stop = tk.StringVar(value="18")
        self.sweep_points = tk.StringVar(value="1001")
        
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
        self.fixed_temp = tk.StringVar(value="300")
        self.ifbw = tk.StringVar(value="100")
        
        # S-parameter selection
        self.s_parameter = tk.StringVar(value="S21")
        
        # Display options
        self.display_mode = tk.StringVar(value="Mag/Phase")
        self.trace_display_mode = tk.StringVar(value="Magnitude")
        self.contour_mode = tk.StringVar(value="Magnitude")
        
        # File settings
        self.data_directory = tk.StringVar(value=os.path.expanduser("~/Desktop"))
        self.filename = tk.StringVar(value="fmr_data")  # Base name without extension
        self.auto_save = tk.BooleanVar(value=True)
        self.current_2d_folder = None  # Track current 2D measurement folder
        
        # Progress
        self.progress_var = tk.DoubleVar(value=0)
        self.status_var = tk.StringVar(value="Ready")
        
        # Parameter options (5 parameters now)
        self.param_options = ["Frequency (GHz)", "B-Field (T)", "Gate Voltage (V)", "Power (dBm)", "Temperature (K)"]
    
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
            text="⚠ When simulation mode is OFF, ensure all instruments are connected before running measurements",
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
        vna_port_frame.grid(row=1, column=1, padx=10, pady=5)
        ttk.Label(vna_port_frame, text="localhost:").pack(side='left')
        ttk.Entry(vna_port_frame, textvariable=self.vna_port, width=6).pack(side='left')
        
        self.vna_status = ttk.Label(conn_frame, text="●", foreground='gray', font=('Arial', 14))
        self.vna_status.grid(row=1, column=2, padx=10, pady=5)
        ttk.Button(conn_frame, text="Connect", command=self.connect_vna).grid(row=1, column=3, padx=10, pady=5)
        ttk.Button(conn_frame, text="Launch S2VNA", command=self.launch_s2vna).grid(row=1, column=4, padx=10, pady=5)
        
        # Magnet
        ttk.Label(conn_frame, text="Magnet (Cryomagnetics 4G)").grid(row=2, column=0, padx=10, pady=5, sticky='w')
        magnet_addr_frame = ttk.Frame(conn_frame)
        magnet_addr_frame.grid(row=2, column=1, padx=10, pady=5)
        ttk.Label(magnet_addr_frame, text="GPIB::").pack(side='left')
        ttk.Entry(magnet_addr_frame, textvariable=self.magnet_addr, width=4).pack(side='left')
        
        self.magnet_status = ttk.Label(conn_frame, text="●", foreground='gray', font=('Arial', 14))
        self.magnet_status.grid(row=2, column=2, padx=10, pady=5)
        ttk.Button(conn_frame, text="Connect", command=self.connect_magnet).grid(row=2, column=3, padx=10, pady=5)
        
        # Keithley
        ttk.Label(conn_frame, text="Gate SMU (Keithley 2400)").grid(row=3, column=0, padx=10, pady=5, sticky='w')
        keithley_addr_frame = ttk.Frame(conn_frame)
        keithley_addr_frame.grid(row=3, column=1, padx=10, pady=5)
        ttk.Label(keithley_addr_frame, text="GPIB::").pack(side='left')
        ttk.Entry(keithley_addr_frame, textvariable=self.keithley_addr, width=4).pack(side='left')
        
        self.keithley_status = ttk.Label(conn_frame, text="●", foreground='gray', font=('Arial', 14))
        self.keithley_status.grid(row=3, column=2, padx=10, pady=5)
        ttk.Button(conn_frame, text="Connect", command=self.connect_keithley).grid(row=3, column=3, padx=10, pady=5)
        
        # Temperature Controller
        ttk.Label(conn_frame, text="Temperature (CTC100/Lakeshore)").grid(row=4, column=0, padx=10, pady=5, sticky='w')
        temp_port_frame = ttk.Frame(conn_frame)
        temp_port_frame.grid(row=4, column=1, padx=10, pady=5)
        ttk.Label(temp_port_frame, text="Port:").pack(side='left')
        ttk.Entry(temp_port_frame, textvariable=self.temp_port, width=8).pack(side='left')
        
        self.temp_status = ttk.Label(conn_frame, text="●", foreground='gray', font=('Arial', 14))
        self.temp_status.grid(row=4, column=2, padx=10, pady=5)
        ttk.Button(conn_frame, text="Connect", command=self.connect_temp).grid(row=4, column=3, padx=10, pady=5)
        
        # Future magnet option placeholder
        ttk.Separator(conn_frame, orient='horizontal').grid(row=5, column=0, columnspan=4, sticky='ew', pady=10)
        ttk.Label(
            conn_frame, 
            text="NHMFL Magnet (LabVIEW VI) - Coming Soon",
            foreground='gray'
        ).grid(row=6, column=0, columnspan=4, pady=5)
        
        # VNA Settings
        vna_settings = ttk.LabelFrame(parent, text="VNA Settings")
        vna_settings.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(vna_settings, text="S-Parameter:").grid(row=0, column=0, padx=10, pady=5, sticky='w')
        ttk.Combobox(
            vna_settings, textvariable=self.s_parameter, 
            values=["S21", "S11"], width=8, state='readonly'
        ).grid(row=0, column=1, padx=10, pady=5, sticky='w')
        
        ttk.Label(vna_settings, text="IF Bandwidth (Hz):").grid(row=0, column=2, padx=10, pady=5, sticky='w')
        ttk.Entry(vna_settings, textvariable=self.ifbw, width=10).grid(row=0, column=3, padx=10, pady=5, sticky='w')
        
        # Safety settings
        safety_frame = ttk.LabelFrame(parent, text="Safety Settings")
        safety_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(safety_frame, text="Gate Voltage Slew Rate (V/s):").grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.gate_slew_rate = ttk.Entry(safety_frame, width=10)
        self.gate_slew_rate.insert(0, "1.0")
        self.gate_slew_rate.grid(row=0, column=1, padx=10, pady=5, sticky='w')
        
        ttk.Label(safety_frame, text="Max Gate Voltage (V):").grid(row=0, column=2, padx=10, pady=5, sticky='w')
        self.max_gate = ttk.Entry(safety_frame, width=10)
        self.max_gate.insert(0, "100")
        self.max_gate.grid(row=0, column=3, padx=10, pady=5, sticky='w')
        
        # File settings
        file_frame = ttk.LabelFrame(parent, text="Data File Settings")
        file_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(file_frame, text="Directory:").grid(row=0, column=0, padx=10, pady=5, sticky='w')
        ttk.Entry(file_frame, textvariable=self.data_directory, width=40).grid(row=0, column=1, padx=10, pady=5, sticky='w')
        ttk.Button(file_frame, text="Browse", command=self.browse_directory).grid(row=0, column=2, padx=10, pady=5)
        
        ttk.Label(file_frame, text="Base Filename:").grid(row=1, column=0, padx=10, pady=5, sticky='w')
        ttk.Entry(file_frame, textvariable=self.filename, width=40).grid(row=1, column=1, padx=10, pady=5, sticky='w')
        
        auto_save_check = ttk.Checkbutton(file_frame, text="Auto-save", variable=self.auto_save)
        auto_save_check.grid(row=1, column=2, padx=10, pady=5)
        ToolTip(auto_save_check, "Automatically save data after each measurement.\n"
                                  "1D sweep: saves as filename_001.csv, _002.csv, etc.\n"
                                  "2D sweep: creates folder with individual sweep files.")
    
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
        
        # Scan time estimate label
        self.scan_time_label = ttk.Label(sweep_frame, text="", foreground='blue')
        self.scan_time_label.grid(row=3, column=2, padx=10, pady=5, sticky='w')
        
        # Add tooltip to explain calculation
        self.scan_time_tooltip = ToolTip(
            self.scan_time_label,
            "Estimated scan time per sweep:\n"
            "  time = (points / IFBW) × 1.5 + 2s\n\n"
            "The 1.5× multiplier accounts for VNA\n"
            "processing overhead per point.\n"
            "The +2s is fixed overhead.\n\n"
            "Adjust IFBW in Setup tab to change speed.\n\n"
            "Limits (auto-clamped):\n"
            "  Frequency: 0.0001 - 18 GHz\n"
            "  Power: -50 to +10 dBm"
        )
        
        # Set up variable traces to update scan time
        self.sweep_points.trace_add('write', self.update_scan_time_display)
        self.ifbw.trace_add('write', self.update_scan_time_display)
        
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
        
        # Fixed parameters
        fixed_frame = ttk.LabelFrame(right_frame, text="Fixed Parameters")
        fixed_frame.pack(fill='x', pady=5)
        
        # Create fixed parameter entries (will be enabled/disabled based on sweep/step selection)
        self.fixed_entries = {}
        
        row = 0
        ttk.Label(fixed_frame, text="Frequency (GHz):").grid(row=row, column=0, padx=10, pady=5, sticky='w')
        self.fixed_entries['frequency'] = ttk.Entry(fixed_frame, textvariable=self.fixed_frequency, width=15)
        self.fixed_entries['frequency'].grid(row=row, column=1, padx=10, pady=5, sticky='w')
        ToolTip(self.fixed_entries['frequency'], "Range: 0.0001 - 18 GHz\n(auto-clamped if out of range)")
        
        row += 1
        ttk.Label(fixed_frame, text="B-Field (T):").grid(row=row, column=0, padx=10, pady=5, sticky='w')
        self.fixed_entries['b_field'] = ttk.Entry(fixed_frame, textvariable=self.fixed_field, width=15)
        self.fixed_entries['b_field'].grid(row=row, column=1, padx=10, pady=5, sticky='w')
        
        row += 1
        ttk.Label(fixed_frame, text="Gate Voltage (V):").grid(row=row, column=0, padx=10, pady=5, sticky='w')
        self.fixed_entries['vg'] = ttk.Entry(fixed_frame, textvariable=self.fixed_gate, width=15)
        self.fixed_entries['vg'].grid(row=row, column=1, padx=10, pady=5, sticky='w')
        
        row += 1
        ttk.Label(fixed_frame, text="Power (dBm):").grid(row=row, column=0, padx=10, pady=5, sticky='w')
        self.fixed_entries['power'] = ttk.Entry(fixed_frame, textvariable=self.fixed_power, width=15)
        self.fixed_entries['power'].grid(row=row, column=1, padx=10, pady=5, sticky='w')
        ToolTip(self.fixed_entries['power'], "Range: -50 to +10 dBm\n(auto-clamped if out of range)")
        
        row += 1
        ttk.Label(fixed_frame, text="Temperature (K):").grid(row=row, column=0, padx=10, pady=5, sticky='w')
        self.fixed_entries['temperature'] = ttk.Entry(fixed_frame, textvariable=self.fixed_temp, width=15)
        self.fixed_entries['temperature'].grid(row=row, column=1, padx=10, pady=5, sticky='w')
        
        # Update fixed parameter states
        self.update_fixed_params_state()
        
        # Control buttons
        control_frame = ttk.LabelFrame(right_frame, text="Measurement Control")
        control_frame.pack(fill='x', pady=5)
        
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(pady=10)
        
        self.start_button = ttk.Button(
            button_frame, text="▶ Start Measurement", 
            command=self.start_measurement, width=20
        )
        self.start_button.pack(side='left', padx=5)
        
        # Make stop button more prominent with red styling
        stop_style = ttk.Style()
        stop_style.configure('Stop.TButton', foreground='red')
        
        self.stop_button = ttk.Button(
            button_frame, text="⬛ ABORT", 
            command=self.stop_measurement, width=15, 
            state='disabled', style='Stop.TButton'
        )
        self.stop_button.pack(side='left', padx=5)
        
        # Progress
        progress_frame = ttk.Frame(control_frame)
        progress_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(progress_frame, text="Progress:").pack(side='left')
        self.progress_bar = ttk.Progressbar(
            progress_frame, variable=self.progress_var, 
            maximum=100, length=300
        )
        self.progress_bar.pack(side='left', padx=10, fill='x', expand=True)
        
        self.progress_label = ttk.Label(progress_frame, text="0%")
        self.progress_label.pack(side='left')
        
        # Measurement summary
        summary_frame = ttk.LabelFrame(right_frame, text="Measurement Summary")
        summary_frame.pack(fill='x', pady=5)
        
        self.summary_text = tk.Text(summary_frame, height=8, width=40, state='disabled')
        self.summary_text.pack(fill='x', padx=10, pady=5)
        
        self.update_step_options()  # Initialize step options based on default sweep
        self.update_summary()
        self.update_scan_time_display()
    
    def create_plot_tab(self, parent):
        """Create data visualization interface."""
        # Main container
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Left panel: Single trace viewer
        left_frame = ttk.LabelFrame(main_frame, text="Single Trace Viewer")
        left_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        # Control panel for trace selection
        trace_control = ttk.Frame(left_frame)
        trace_control.pack(fill='x', padx=5, pady=5)
        
        # Data type selection (Mag/Phase/Real/Imag)
        type_frame = ttk.Frame(trace_control)
        type_frame.pack(fill='x', pady=2)
        
        ttk.Label(type_frame, text="Display:").pack(side='left', padx=5)
        self.trace_display_mode = tk.StringVar(value="Magnitude")
        for mode in ["Magnitude", "Phase", "Real", "Imaginary"]:
            ttk.Radiobutton(
                type_frame, text=mode,
                variable=self.trace_display_mode, value=mode,
                command=self.update_single_trace
            ).pack(side='left', padx=3)
        
        # Step selector frame
        step_select_frame = ttk.Frame(trace_control)
        step_select_frame.pack(fill='x', pady=5)
        
        ttk.Label(step_select_frame, text="Step Selection:").pack(side='left', padx=5)
        
        # Previous button
        self.prev_step_btn = ttk.Button(
            step_select_frame, text="◀ Prev", width=8,
            command=self.prev_step, state='disabled'
        )
        self.prev_step_btn.pack(side='left', padx=2)
        
        # Slider for step selection
        self.step_slider_var = tk.IntVar(value=0)
        self.step_slider = ttk.Scale(
            step_select_frame, from_=0, to=0,
            orient='horizontal', variable=self.step_slider_var,
            command=self.on_step_slider_changed
        )
        self.step_slider.pack(side='left', fill='x', expand=True, padx=5)
        self.step_slider.config(state='disabled')
        
        # Next button
        self.next_step_btn = ttk.Button(
            step_select_frame, text="Next ▶", width=8,
            command=self.next_step, state='disabled'
        )
        self.next_step_btn.pack(side='left', padx=2)
        
        # Current step value display
        step_value_frame = ttk.Frame(trace_control)
        step_value_frame.pack(fill='x', pady=2)
        
        ttk.Label(step_value_frame, text="Current:").pack(side='left', padx=5)
        self.step_value_label = ttk.Label(step_value_frame, text="No data", font=('Arial', 10, 'bold'))
        self.step_value_label.pack(side='left', padx=5)
        
        self.step_index_label = ttk.Label(step_value_frame, text="", foreground='gray')
        self.step_index_label.pack(side='left', padx=5)
        
        # Show all traces checkbox
        self.show_all_traces = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            step_value_frame, text="Show all traces (overlay)",
            variable=self.show_all_traces, command=self.update_single_trace
        ).pack(side='right', padx=10)
        
        # Matplotlib figure for single trace
        self.fig_trace = Figure(figsize=(6, 5))
        self.ax_trace = self.fig_trace.add_subplot(111)
        
        self.canvas_trace = FigureCanvasTkAgg(self.fig_trace, left_frame)
        
        # Toolbar for trace plot (zoom, pan, save) - pack at bottom FIRST
        toolbar_frame_trace = ttk.Frame(left_frame)
        toolbar_frame_trace.pack(side='bottom', fill='x')
        self.toolbar_trace = NavigationToolbar2Tk(self.canvas_trace, toolbar_frame_trace)
        self.toolbar_trace.update()
        
        # Now pack the canvas (it will fill remaining space)
        self.canvas_trace.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
        
        # Right panel: 2D contour
        right_frame = ttk.LabelFrame(main_frame, text="2D Contour Map")
        right_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        # Control panel for 2D plot
        contour_control = ttk.Frame(right_frame)
        contour_control.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(contour_control, text="Display:").pack(side='left', padx=5)
        self.contour_mode = tk.StringVar(value="Magnitude")
        for mode in ["Magnitude", "Phase", "Real", "Imaginary"]:
            ttk.Radiobutton(
                contour_control, text=mode,
                variable=self.contour_mode, value=mode,
                command=self.update_2d_plot
            ).pack(side='left', padx=3)
        
        # Buttons
        ttk.Button(contour_control, text="Clear All", command=self.clear_plots).pack(side='right', padx=5)
        ttk.Button(contour_control, text="Save Data", command=self.save_data).pack(side='right', padx=5)
        
        # Color scale controls
        scale_frame = ttk.Frame(right_frame)
        scale_frame.pack(fill='x', padx=5, pady=2)
        
        ttk.Label(scale_frame, text="Color Scale:").pack(side='left', padx=5)
        
        self.auto_scale = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            scale_frame, text="Auto",
            variable=self.auto_scale, command=self.on_scale_mode_changed
        ).pack(side='left', padx=3)
        
        self.log_scale = tk.BooleanVar(value=False)
        log_check = ttk.Checkbutton(
            scale_frame, text="Log",
            variable=self.log_scale, command=self.update_2d_plot
        )
        log_check.pack(side='left', padx=3)
        ToolTip(log_check, "Use logarithmic color scale.\nBest for data with large dynamic range.")
        
        ttk.Label(scale_frame, text="Min:").pack(side='left', padx=(8, 2))
        self.color_min = tk.StringVar(value="-60")
        self.color_min_entry = ttk.Entry(scale_frame, textvariable=self.color_min, width=7)
        self.color_min_entry.pack(side='left', padx=2)
        self.color_min_entry.bind('<Return>', lambda e: self.update_2d_plot())
        
        ttk.Label(scale_frame, text="Max:").pack(side='left', padx=(8, 2))
        self.color_max = tk.StringVar(value="0")
        self.color_max_entry = ttk.Entry(scale_frame, textvariable=self.color_max, width=7)
        self.color_max_entry.pack(side='left', padx=2)
        self.color_max_entry.bind('<Return>', lambda e: self.update_2d_plot())
        
        ttk.Button(scale_frame, text="Apply", command=self.update_2d_plot).pack(side='left', padx=8)
        
        # Help label for colorbar interaction
        colorbar_help = ttk.Label(scale_frame, text="(?)", foreground='blue', cursor='question_arrow')
        colorbar_help.pack(side='left', padx=3)
        ToolTip(colorbar_help, "Interactive colorbar:\n"
                               "• Left-drag on colorbar: shift range (brightness)\n"
                               "• Right-drag on colorbar: adjust contrast\n"
                               "• Scroll wheel on colorbar: zoom contrast")
        
        # Set initial state (disabled when auto)
        self.on_scale_mode_changed()
        
        # Matplotlib figure for 2D contour
        self.fig_2d = Figure(figsize=(6, 5))
        self.ax_2d = self.fig_2d.add_subplot(111)
        
        self.canvas_2d = FigureCanvasTkAgg(self.fig_2d, right_frame)
        
        # Colorbar interaction state
        self.colorbar_2d = None
        self.colorbar_mappable = None
        self.colorbar_data_range = (0, 1)
        self.colorbar_dragging = False
        self.colorbar_drag_start_y = None
        self.colorbar_drag_start_vmin = None
        self.colorbar_drag_start_vmax = None
        
        # Connect mouse events for interactive colorbar
        self.canvas_2d.mpl_connect('button_press_event', self.on_colorbar_press)
        self.canvas_2d.mpl_connect('button_release_event', self.on_colorbar_release)
        self.canvas_2d.mpl_connect('motion_notify_event', self.on_colorbar_motion)
        self.canvas_2d.mpl_connect('scroll_event', self.on_colorbar_scroll)
        
        # Toolbar for 2D plot - pack at bottom FIRST
        toolbar_frame_2d = ttk.Frame(right_frame)
        toolbar_frame_2d.pack(side='bottom', fill='x')
        self.toolbar_2d = NavigationToolbar2Tk(self.canvas_2d, toolbar_frame_2d)
        self.toolbar_2d.update()
        
        # Now pack the canvas (it will fill remaining space)
        self.canvas_2d.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
        
        # Initialize plots
        self.current_step_index = 0
        self.init_plots()
    
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
        # Single trace plot
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
        
        self.fig_trace.tight_layout()
        self.fig_2d.tight_layout()
        
        self.canvas_trace.draw()
        self.canvas_2d.draw()
    
    def update_step_slider(self):
        """Update step slider range based on available data."""
        n_steps = len(self.sweep_data_2d)
        
        if n_steps > 0:
            self.step_slider.config(from_=0, to=max(0, n_steps - 1), state='normal')
            self.prev_step_btn.config(state='normal')
            self.next_step_btn.config(state='normal')
            
            # Clamp current index to valid range
            if self.current_step_index >= n_steps:
                self.current_step_index = n_steps - 1
            self.step_slider_var.set(self.current_step_index)
            
            # Update label
            self.update_step_label()
        else:
            self.step_slider.config(from_=0, to=0, state='disabled')
            self.prev_step_btn.config(state='disabled')
            self.next_step_btn.config(state='disabled')
            self.step_value_label.config(text="No data")
            self.step_index_label.config(text="")
    
    def update_step_label(self):
        """Update the step value label."""
        if not self.sweep_data_2d or self.current_step_index >= len(self.sweep_data_2d):
            self.step_value_label.config(text="No data")
            self.step_index_label.config(text="")
            return
        
        trace_data = self.sweep_data_2d[self.current_step_index]
        if trace_data and trace_data[0]['step_value'] is not None:
            step_val = trace_data[0]['step_value']
            step_param = self.current_config.get('step_param', 'Step') if self.current_config else 'Step'
            self.step_value_label.config(text=f"{step_param} = {step_val:.4g}")
            self.step_index_label.config(text=f"(index {self.current_step_index + 1} of {len(self.sweep_data_2d)})")
        else:
            self.step_value_label.config(text="1D Sweep")
            self.step_index_label.config(text="(no step parameter)")
    
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
        self.ax_trace.clear()
        
        if not self.sweep_data_2d:
            self.ax_trace.set_title("Single Trace Viewer")
            self.ax_trace.text(0.5, 0.5, "No data available",
                              ha='center', va='center', transform=self.ax_trace.transAxes,
                              fontsize=8, color='gray')
            self.ax_trace.grid(True, alpha=0.3)
            self.fig_trace.tight_layout()
            self.canvas_trace.draw_idle()
            return
        
        sweep_param = self.current_config['sweep_param'] if self.current_config else "Sweep"
        step_param = self.current_config.get('step_param') if self.current_config else None
        mode = self.trace_display_mode.get()
        
        # Determine y-axis label based on mode
        if mode == "Magnitude":
            ylabel = "|S21| (dB)"
        elif mode == "Phase":
            ylabel = "Phase (deg)"
        elif mode == "Real":
            ylabel = "Re(S21)"
        else:
            ylabel = "Im(S21)"
        
        # Get frequency scaling for x-axis if sweeping frequency
        freq_scale, freq_unit = self._get_frequency_scale()
        
        if self.show_all_traces.get():
            # Show all traces with color gradient
            n_traces = len(self.sweep_data_2d)
            colors = cm.viridis(np.linspace(0, 1, max(n_traces, 1)))
            
            for trace_idx, trace_data in enumerate(self.sweep_data_2d):
                if not trace_data:
                    continue
                
                sweep_vals = [d['sweep_value'] * freq_scale for d in trace_data]
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
                    sweep_vals = [d['sweep_value'] * freq_scale for d in trace_data]
                    y = self._get_trace_y_data(trace_data, mode)
                    
                    self.ax_trace.plot(sweep_vals, y, 'b-', linewidth=0.8)
                    
                    # Title with step info
                    step_val = trace_data[0].get('step_value')
                    if step_val is not None and step_param:
                        self.ax_trace.set_title(f"{step_param} = {step_val:.4g}")
                    else:
                        self.ax_trace.set_title("Single Trace")
        
        # Update x-axis label with appropriate unit
        if sweep_param == "Frequency (GHz)" and freq_unit:
            xlabel = f"Frequency ({freq_unit})"
        else:
            xlabel = sweep_param
        
        self.ax_trace.set_xlabel(xlabel)
        self.ax_trace.set_ylabel(ylabel)
        self.ax_trace.grid(True, alpha=0.3)
        
        self.fig_trace.tight_layout()
        self.canvas_trace.draw_idle()
    
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
    
    def _get_trace_y_data(self, trace_data, mode):
        """Extract y-axis data from trace based on display mode."""
        if mode == "Magnitude":
            return [20 * np.log10(d['s21_mag'] + 1e-12) for d in trace_data]
        elif mode == "Phase":
            return [d['s21_phase'] for d in trace_data]
        elif mode == "Real":
            return [d['s21_real'] for d in trace_data]
        else:  # Imaginary
            return [d['s21_imag'] for d in trace_data]
    
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
                "1. In S2VNA, go to System → Misc Setup\n"
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
                        "2. Socket server is enabled (System → Misc Setup → Network Setup)\n"
                        "3. Port matches (default: 5025)"
                    )
            except Exception as e:
                self.vna_status.config(foreground='red')
                messagebox.showerror("Connection Error", f"VNA connection failed: {e}")
    
    def connect_magnet(self):
        """Connect to magnet controller."""
        if self.simulation_mode.get():
            self.magnet_connected.set(True)
            self.magnet_status.config(foreground='green')
            self.status_var.set("Magnet connected (simulation)")
        else:
            try:
                self.magnet.set_address(self.magnet_addr.get())
                if self.magnet.connect():
                    self.magnet_connected.set(True)
                    self.magnet_status.config(foreground='green')
                    self.status_var.set("Magnet connected")
                else:
                    messagebox.showerror("Connection Error", "Failed to connect to magnet")
            except Exception as e:
                messagebox.showerror("Connection Error", f"Magnet connection failed: {e}")
    
    def connect_keithley(self):
        """Connect to Keithley SMU."""
        if self.simulation_mode.get():
            self.keithley_connected.set(True)
            self.keithley_status.config(foreground='green')
            self.status_var.set("Keithley connected (simulation)")
        else:
            try:
                self.keithley.set_address(self.keithley_addr.get())
                if self.keithley.connect():
                    self.keithley_connected.set(True)
                    self.keithley_status.config(foreground='green')
                    self.status_var.set("Keithley connected")
                else:
                    messagebox.showerror("Connection Error", "Failed to connect to Keithley")
            except Exception as e:
                messagebox.showerror("Connection Error", f"Keithley connection failed: {e}")
    
    def connect_temp(self):
        """Connect to temperature controller."""
        if self.simulation_mode.get():
            self.temp_connected.set(True)
            self.temp_status.config(foreground='green')
            self.status_var.set("Temperature controller connected (simulation)")
        else:
            try:
                self.temp_controller.set_port(self.temp_port.get())
                if self.temp_controller.connect():
                    self.temp_connected.set(True)
                    self.temp_status.config(foreground='green')
                    self.status_var.set("Temperature controller connected")
                else:
                    messagebox.showerror("Connection Error", "Failed to connect to temperature controller")
            except Exception as e:
                messagebox.showerror("Connection Error", f"Temperature controller connection failed: {e}")
    
    def browse_directory(self):
        """Browse for data directory."""
        directory = filedialog.askdirectory(initialdir=self.data_directory.get())
        if directory:
            self.data_directory.set(directory)
    
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
        elif sweep == "Temperature (K)":
            self.sweep_start.set("4")
            self.sweep_stop.set("300")
            self.sweep_points.set("30")
        
        self.update_fixed_params_state()
        self.update_step_options()
        self.update_summary()
        self.update_scan_time_display()
    
    def update_scan_time_display(self, *args):
        """Update the estimated scan time display based on points and IFBW."""
        try:
            points = int(float(self.sweep_points.get()))
            ifbw = float(self.ifbw.get())
            
            if points <= 0 or ifbw <= 0:
                self.scan_time_label.config(text="")
                return
            
            # Calculate time using same formula as trigger_sweep_timed
            sweep_time = points / ifbw
            adjusted_time = sweep_time * 1.5 + 2.0
            
            # Format display
            if adjusted_time < 60:
                time_str = f"~{adjusted_time:.1f}s"
            else:
                minutes = int(adjusted_time // 60)
                seconds = adjusted_time % 60
                time_str = f"~{minutes}m {seconds:.0f}s"
            
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
        elif step == "Temperature (K)":
            self.step_start.set("4")
            self.step_stop.set("300")
            self.step_points.set("5")
        
        self.update_fixed_params_state()
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
    
    def update_fixed_params_state(self):
        """Enable/disable fixed parameter entries based on sweep/step selection."""
        sweep = self.sweep_param.get()
        step = self.step_param.get()
        
        param_map = {
            "Frequency (GHz)": 'frequency',
            "B-Field (T)": 'b_field',
            "Gate Voltage (V)": 'vg',
            "Power (dBm)": 'power',
            "Temperature (K)": 'temperature'
        }
        
        for param_name, entry_key in param_map.items():
            if param_name == sweep or param_name == step:
                self.fixed_entries[entry_key].config(state='disabled')
            else:
                self.fixed_entries[entry_key].config(state='normal')
    
    def update_summary(self):
        """Update measurement summary text."""
        sweep = self.sweep_param.get()
        step = self.step_param.get()
        
        try:
            sweep_pts = int(self.sweep_points.get())
            step_pts = int(self.step_points.get()) if step != "None" else 1
            total_pts = sweep_pts * step_pts
            
            # Estimate time (rough: 100ms per point)
            est_time = total_pts * 0.1
            
            summary = f"Sweep: {sweep}\n"
            summary += f"  Range: {self.sweep_start.get()} to {self.sweep_stop.get()}\n"
            summary += f"  Points: {sweep_pts}\n\n"
            
            if step != "None":
                summary += f"Step: {step}\n"
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
        # Validate parameters
        try:
            sweep_param = self.sweep_param.get()
            sweep_start = float(self.sweep_start.get())
            sweep_stop = float(self.sweep_stop.get())
            
            step_param = self.step_param.get() if self.step_param.get() != "None" else None
            step_start = float(self.step_start.get()) if step_param else 0
            step_stop = float(self.step_stop.get()) if step_param else 0
            
            fixed_freq = float(self.fixed_frequency.get())
            fixed_power = float(self.fixed_power.get())
            
            # Frequency limits
            freq_min_ghz = 0.0001  # 100 kHz
            freq_max_ghz = 18.0    # 18 GHz
            
            # Power limits for S5180B VNA
            power_min = -50  # dBm
            power_max = 10   # dBm
            
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
            
            # Force GUI update if values were clamped
            if clamped:
                self.root.update_idletasks()
                print("Values were clamped to valid ranges")
            
            config = {
                'sweep_param': sweep_param,
                'sweep_start': sweep_start,
                'sweep_stop': sweep_stop,
                'sweep_points': int(self.sweep_points.get()),
                'step_param': step_param,
                'step_start': step_start,
                'step_stop': step_stop,
                'step_points': int(self.step_points.get()) if step_param else 1,
                'fixed_values': {
                    'frequency': fixed_freq,
                    'b_field': float(self.fixed_field.get()),
                    'vg': float(self.fixed_gate.get()),
                    'power': fixed_power,
                    'temperature': float(self.fixed_temp.get()),
                    'ifbw': float(self.ifbw.get())
                },
                's_parameter': self.s_parameter.get()
            }
        except ValueError as e:
            messagebox.showerror("Invalid Parameters", f"Please check parameter values: {e}")
            return
        
        # Clear previous data
        self.sweep_data_1d = []
        self.sweep_data_2d = []
        self.current_step_index = 0
        self.current_config = config
        self.current_2d_folder = None  # Reset for new measurement
        
        # Reset step slider
        self.step_slider_var.set(0)
        self.update_step_slider()
        
        # Update UI
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.progress_var.set(0)
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
    
    def update_gui(self):
        """Periodic GUI update (called every 50ms)."""
        # Process pending GUI events to keep interface responsive
        self.root.update_idletasks()
        
        # Check for progress updates (process all available)
        try:
            while True:
                progress = self.measurement_engine.progress_queue.get_nowait()
                self.progress_var.set(progress)
                self.progress_label.config(text=f"{progress:.1f}%")
        except queue.Empty:
            pass
        
        # Check for data updates (limit to a few per cycle to stay responsive)
        updates_processed = 0
        max_updates_per_cycle = 10
        try:
            while updates_processed < max_updates_per_cycle:
                data = self.measurement_engine.data_queue.get_nowait()
                self.process_data_update(data)
                updates_processed += 1
        except queue.Empty:
            pass
        
        # Schedule next update
        self.root.after(50, self.update_gui)
    
    def process_data_update(self, data):
        """Process data update from measurement engine."""
        if data['type'] == 'point':
            # Track current sweep for live plot
            sweep_idx = data['sweep_idx']
            step_idx = data['step_idx']
            
            # Store for 2D array
            while len(self.sweep_data_2d) <= step_idx:
                self.sweep_data_2d.append([])
            self.sweep_data_2d[step_idx].append(data['data'])
            
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
            
            # Ensure we have enough rows
            while len(self.sweep_data_2d) <= step_idx:
                self.sweep_data_2d.append([])
            
            # Store complete sweep data
            self.sweep_data_2d[step_idx] = sweep_data
            print(f"Batch received: step {step_idx} with {len(sweep_data)} points")
            
            # Update step tracking
            self.current_step_index = step_idx
            self.update_step_slider()
            self.step_slider_var.set(self.current_step_index)
            
            # Update plots
            self.update_single_trace()
        
        elif data['type'] == 'step_complete':
            # A full sweep line just finished
            step_idx = data['step_idx']
            step_value = data.get('step_value')
            
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
            
            # Update plots
            self.update_step_slider()
            self.update_single_trace()
            self.update_2d_plot()
        
        elif data['type'] == 'complete':
            self.status_var.set("Measurement complete!")
            self.start_button.config(state='normal')
            self.stop_button.config(state='disabled')
            self.update_step_slider()
            self.update_single_trace()
            self.update_2d_plot()
            
            # For 1D measurements, auto-save at completion
            # (2D measurements auto-save each sweep in step_complete)
            is_2d = self.current_config and self.current_config.get('step_param')
            if not is_2d and self.sweep_data_2d and self.sweep_data_2d[0]:
                self.auto_save_sweep(self.sweep_data_2d[0])
            
            # Reset 2D folder tracker for next measurement
            self.current_2d_folder = None
        
        elif data['type'] == 'aborted':
            self.status_var.set("Measurement ABORTED by user")
            self.start_button.config(state='normal')
            self.stop_button.config(state='disabled')
            self.update_step_slider()
            self.update_single_trace()
            self.update_2d_plot()
            
            # Reset 2D folder tracker (partial data already saved)
            self.current_2d_folder = None
        
        elif data['type'] == 'error':
            self.status_var.set(f"Error: {data['message']}")
            self.start_button.config(state='normal')
            self.stop_button.config(state='disabled')
            messagebox.showerror("Measurement Error", data['message'])
    
    def update_2d_plot(self):
        """Update 2D contour plot."""
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
        
        # Get frequency scaling for axes
        freq_scale, freq_unit = self._get_frequency_scale()
        step_freq_scale, step_freq_unit = self._get_step_frequency_scale()
        
        sweep_vals = np.array([d['sweep_value'] * freq_scale for d in valid_traces[0]])
        step_vals = np.array([t[0]['step_value'] * step_freq_scale for t in valid_traces])
        
        # Get data based on contour mode selection
        mode = self.contour_mode.get()
        
        if mode == "Magnitude":
            z = np.array([[20 * np.log10(d['s21_mag'] + 1e-12) for d in trace] 
                         for trace in valid_traces])
            zlabel = "|S21| (dB)"
            cmap = 'viridis'
        elif mode == "Phase":
            z = np.array([[d['s21_phase'] for d in trace] for trace in valid_traces])
            zlabel = "Phase (deg)"
            cmap = 'RdBu_r'
        elif mode == "Real":
            z = np.array([[d['s21_real'] for d in trace] for trace in valid_traces])
            zlabel = "Re(S21)"
            cmap = 'RdBu_r'
        else:  # Imaginary
            z = np.array([[d['s21_imag'] for d in trace] for trace in valid_traces])
            zlabel = "Im(S21)"
            cmap = 'RdBu_r'
        
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
        
        # Update axis labels with appropriate units
        if sweep_param == "Frequency (GHz)" and freq_unit:
            xlabel = f"Frequency ({freq_unit})"
        else:
            xlabel = sweep_param
            
        if step_param == "Frequency (GHz)" and step_freq_unit:
            ylabel = f"Frequency ({step_freq_unit})"
        else:
            ylabel = step_param
        
        self.ax_2d.set_xlabel(xlabel)
        self.ax_2d.set_ylabel(ylabel)
        self.ax_2d.set_title(f"2D Map - {zlabel}")
        
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
        
        # Reset step slider
        self.step_slider_var.set(0)
        self.step_slider.config(from_=0, to=0, state='disabled')
        self.prev_step_btn.config(state='disabled')
        self.next_step_btn.config(state='disabled')
        self.step_value_label.config(text="No data")
        self.step_index_label.config(text="")
        
        # Clear and recreate trace figure
        self.ax_trace.clear()
        
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
                
                sweep_param = self.current_config['sweep_param'] if self.current_config else "Sweep"
                is_freq_sweep = sweep_param == "Frequency (GHz)"
                
                if self.current_config:
                    # Convert sweep range to display units
                    sweep_start = self.current_config['sweep_start']
                    sweep_stop = self.current_config['sweep_stop']
                    if is_freq_sweep:
                        sweep_start = sweep_start / 1e9  # Hz to GHz
                        sweep_stop = sweep_stop / 1e9
                    
                    f.write(f"# Sweep Parameter: {sweep_param}\n")
                    f.write(f"# Sweep Range: {sweep_start} to {sweep_stop}\n")
                    f.write(f"# Sweep Points: {self.current_config['sweep_points']}\n")
                    
                    if step_value is not None:
                        step_param = self.current_config['step_param']
                        # Convert step value to display units
                        step_display = step_value
                        if step_param == "Frequency (GHz)":
                            step_display = step_value / 1e9
                        f.write(f"# Step Parameter: {step_param}\n")
                        f.write(f"# Step Value: {step_display}\n")
                    
                    f.write(f"# Fixed Values: {self.current_config['fixed_values']}\n")
                
                f.write("#\n")
                
                # Column header with proper units
                if is_freq_sweep:
                    f.write("# Frequency_GHz, S21_Real, S21_Imag, S21_Mag_dB, S21_Phase_deg\n")
                else:
                    f.write(f"# {sweep_param.replace(' ', '_')}, S21_Real, S21_Imag, S21_Mag_dB, S21_Phase_deg\n")
                
                # Write data
                for d in sweep_data:
                    # Convert sweep value to display units
                    sweep_val = d['sweep_value']
                    if is_freq_sweep:
                        sweep_val = sweep_val / 1e9  # Hz to GHz
                    
                    # Convert magnitude to dB
                    mag_db = 20 * np.log10(d['s21_mag']) if d['s21_mag'] > 0 else -200
                    f.write(f"{sweep_val}, {d['s21_real']}, {d['s21_imag']}, ")
                    f.write(f"{mag_db}, {d['s21_phase']}\n")
            
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
                f.write(f"  Points: {self.current_config['sweep_points']}\n\n")
                
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
                f.write(f"  Power: {fixed['power']} dBm\n")
                f.write(f"  Temperature: {fixed['temperature']} K\n")
                f.write(f"  IF Bandwidth: {fixed['ifbw']} Hz\n")
            
            return True
        except Exception as e:
            print(f"Error saving metadata: {e}")
            return False
    
    def auto_save_sweep(self, sweep_data, step_idx=None, step_value=None):
        """Auto-save a sweep after completion."""
        if not self.auto_save.get():
            return
        
        base_name = self.filename.get().strip()
        if not base_name:
            base_name = "fmr_data"
        
        # Remove extension if user included one
        if base_name.endswith('.csv'):
            base_name = base_name[:-4]
        
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
        
        except Exception as e:
            print(f"Auto-save error: {e}")
            self.status_var.set(f"Auto-save failed: {e}")
    
    def save_data(self):
        """Save measurement data to file (manual save button)."""
        if not self.sweep_data_2d:
            messagebox.showwarning("No Data", "No data to save!")
            return
        
        base_name = self.filename.get().strip()
        if not base_name:
            base_name = "fmr_data"
        if base_name.endswith('.csv'):
            base_name = base_name[:-4]
        
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
                
                self.status_var.set(f"Data saved to {folder_name}/")
                messagebox.showinfo("Save Complete", f"2D data saved to:\n{folder_path}")
            else:
                # Single 1D sweep
                filename, filepath = self.get_next_filename(base_name)
                
                if self.sweep_data_2d and self.sweep_data_2d[0]:
                    self.save_single_sweep(self.sweep_data_2d[0], filepath)
                    self.status_var.set(f"Data saved to {filename}")
                    messagebox.showinfo("Save Complete", f"Data saved to:\n{filepath}")
        
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save data: {e}")


def main():
    """Main entry point."""
    print("=" * 60)
    print("VNA FMR Measurement System")
    print("Villanova University - Dietrich Lab")
    print("=" * 60)
    print("\nFeatures:")
    print("- Flexible sweep/step parameter selection")
    print("- 1D and 2D measurement modes")
    print("- Real-time visualization")
    print("- Simulation mode for GUI testing")
    print("\nStarting GUI...")
    
    root = tk.Tk()
    app = VNAMeasurementApp(root)
    
    def on_closing():
        if app.measurement_engine.is_running:
            if messagebox.askokcancel("Quit", "Measurement in progress. Stop and quit?"):
                app.measurement_engine.stop()
                root.destroy()
        else:
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()