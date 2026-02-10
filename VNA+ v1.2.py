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
- AMI Model 420 Magnet Power Supply Programmer via GPIB
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
        
        print(f"VNA setup: {f_start/1e9:.3f}-{f_stop/1e9:.3f} GHz, {num_points} pts, IFBW={ifbw}, P={power} dBm")
        
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
        
        # Wait for VNA to process settings
        time.sleep(0.3)
        
        return True
    
    def setup_cw_mode(self, frequency, ifbw, power):
        """Configure VNA for CW (single frequency) measurement."""
        if not self.connected:
            return False
        
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
        
        # Set source power
        self.write(f":SOUR:POW {power}")
        
        # Set S-parameter
        self.write(f":CALC1:PAR1:DEF {self.s_parameter}")
        
        # Allow VNA to settle after configuration change
        time.sleep(0.1)
        
        return True
    
    def trigger_sweep(self):
        """Trigger a sweep and wait for completion (for CW mode).
        
        Assumes VNA is already configured in single sweep mode (setup_cw_mode does this).
        Only sends the trigger command for speed.
        """
        if not self.connected:
            return False
        
        # Just trigger - setup already done in setup_cw_mode
        self.write(":INIT:IMM")
        
        # Minimal wait - data query will block until ready
        time.sleep(0.02)  # 20ms
        
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
        sweep_time = float(num_points) / float(ifbw)
        adjusted_sweep_time = sweep_time * 1.5  # 50% extra for VNA overhead
        wait_time = adjusted_sweep_time + 2.0  # Fixed overhead
        
        # Make sure RF is on
        self.write(":OUTP ON")
        
        # Set to single sweep mode
        self.write(":INIT:CONT OFF")
        time.sleep(0.1)
        
        # Trigger the sweep
        self.write(":INIT:IMM")
        
        # Wait for sweep to complete
        time.sleep(wait_time)
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
    
    def get_field(self):
        """Get current magnetic field.
        
        Returns field in Tesla.
        """
        if not self.connected:
            return self.current_field
        
        try:
            # Query field (returns value in configured units)
            response = self.query("FIELD:MAG?")
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
            self.instrument.timeout = 5000
            self.instrument.read_termination = '\r\n'
            self.instrument.write_termination = '\r\n'
            
            # Query identification
            idn = self.instrument.query("*IDN?")
            print(f"Cryomagnetics 4G connected: {idn.strip()}")
            
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
    
    def get_field(self):
        """Get current magnetic field in Tesla."""
        if not self.connected:
            return self.current_field
        
        try:
            # Cryomagnetics uses IOUT? for current, need coil constant to convert
            response = self.query("IOUT?")
            if response:
                # Parse current and convert to field using coil constant
                current = float(response.replace('A', '').strip())
                # Assuming coil constant is set - this would need calibration
                self.current_field = current * 0.1  # Example: 0.1 T/A
                return self.current_field
        except Exception as e:
            print(f"Cryomagnetics get_field error: {e}")
        
        return self.current_field
    
    def get_current(self):
        """Get magnet current in Amps."""
        response = self.query("IOUT?")
        if response:
            try:
                return float(response.replace('A', '').strip())
            except:
                pass
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
        """Set target magnetic field.
        
        Args:
            target_field: Target field in Tesla
        """
        if not self.connected:
            self.current_field = target_field
            return True
        
        try:
            # Convert field to current (need coil constant)
            target_current = target_field / 0.1  # Example: 0.1 T/A
            
            # Set upper limit
            self.write(f"ULIM {target_current:.4f}")
            
            # Start ramping up
            self.write("SWEEP UP")
            
            print(f"Cryomagnetics: Ramping to {target_field:.4f} T ({target_current:.2f} A)")
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
    
    def set_ramp_rate(self, rate, units='A/s'):
        """Set current ramp rate."""
        # Cryomagnetics uses A/sec
        self.write(f"RATE 0 {rate:.4f}")
        return True
    
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


# Factory function to create the appropriate magnet controller
def create_magnet_controller(model='AMI 420', resource_manager=None):
    """Create magnet controller based on model selection.
    
    Args:
        model: 'AMI 420' or 'Cryomagnetics 4G'
        resource_manager: Optional pyvisa ResourceManager
    
    Returns:
        MagnetController or CryomagneticsController instance
    """
    if model == 'Cryomagnetics 4G':
        return CryomagneticsController(resource_manager)
    else:
        return MagnetController(resource_manager)  # AMI 420


class KeithleyController:
    """Controller for Keithley 2400/2450 SMU (gate voltage).
    
    Supports both Keithley 2400 and 2450 models with appropriate SCPI commands.
    Includes safe voltage ramping from existing code.
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
            'measure_current': ':MEAS:CURR?',
            'measure_voltage': ':MEAS:VOLT?',
        }
    }
    
    def __init__(self, resource_manager=None, model='2450'):
        self.connected = False
        self.address = "GPIB0::24::INSTR"
        self.instrument = None
        self.rm = resource_manager
        self.current_voltage = 0.0
        self.max_voltage = 100.0
        self.compliance_current = 100e-9  # 100 nA default
        self.slew_rate = 1.0  # V/s
        self.model = model
        self.commands = self.MODELS.get(model, self.MODELS['2450'])
        
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
        """Update instrument address.
        
        Args:
            address: Either a GPIB address number (e.g., "18") 
                     or a full VISA resource string (e.g., "USB0::0x05E6::0x2450::04102170::INSTR")
        """
        address = str(address).strip()
        
        # Check if it's a full resource string
        if '::' in address:
            self.address = address
        else:
            # Assume it's a GPIB address number
            self.address = f"GPIB0::{address}::INSTR"
        
        print(f"Keithley address set to: {self.address}")
    
    def connect(self):
        """Connect to Keithley SMU."""
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                # On retry, force a fresh ResourceManager
                if attempt > 0:
                    print("Retrying with fresh VISA connection...")
                    if self.rm is not None:
                        try:
                            self.rm.close()
                        except:
                            pass
                        self.rm = None
                    time.sleep(0.5)  # Give VISA time to clean up
                
                if self.rm is None:
                    import pyvisa
                    
                    # Try default NI-VISA backend
                    try:
                        self.rm = pyvisa.ResourceManager()
                        print(f"Using VISA backend: NI-VISA")
                    except Exception as e:
                        raise RuntimeError(
                            f"No VISA backend available.\n\n"
                            f"For GPIB instruments, install NI-VISA from ni.com/visa\n"
                            f"Error: {e}"
                        )
                
                # Close any existing connection first
                if self.instrument is not None:
                    try:
                        self.instrument.close()
                    except:
                        pass
                    self.instrument = None
                
                # Try to send interface clear to GPIB bus before connecting
                # This can help reset stuck instruments
                try:
                    gpib_interface = self.address.split('::')[0]  # e.g., "GPIB0"
                    intfc = self.rm.open_resource(f"{gpib_interface}::INTFC")
                    intfc.send_ifc()
                    intfc.close()
                    print(f"Sent interface clear to {gpib_interface}")
                    time.sleep(0.2)
                except Exception as e:
                    # Interface clear not supported or failed - continue anyway
                    print(f"Interface clear skipped: {e}")
                
                print(f"Attempting to connect to {self.address}...")
                self.instrument = self.rm.open_resource(self.address)
                self.instrument.timeout = 5000  # 5 second timeout
                
                # Clear any pending operations
                try:
                    self.instrument.clear()
                except:
                    pass
                time.sleep(0.1)
                
                # Query ID to verify connection
                idn = self.instrument.query('*IDN?').strip()
                print(f"Connected to: {idn}")
                
                # Initialize for voltage sourcing
                self.instrument.write(self.commands['reset'])
                time.sleep(0.5)
                self.instrument.write(self.commands['source_voltage'])
                self.instrument.write(':SOUR:VOLT:RANG 20')  # Fixed 20V range
                self.instrument.write(':SYST:RSEN OFF')  # 2-wire mode (not 4-wire)
                
                # Set compliance and matching current sense range
                self.instrument.write(self.commands['compliance_current'].format(self.compliance_current))
                self.instrument.write(f':SENS:CURR:RANG {self.compliance_current}')  # Match sense range to compliance
                print(f"Compliance: {self.compliance_current*1e6:.1f} µA, Sense range matched")
                
                self.instrument.write(self.commands['set_voltage'].format(0.0))
                self.instrument.write(self.commands['output_on'])
                
                self.connected = True
                self.current_voltage = 0.0
                return True
                
            except Exception as e:
                error_str = str(e)
                print(f"Connection attempt {attempt + 1} failed: {error_str}")
                
                # Clean up failed connection
                if self.instrument is not None:
                    try:
                        self.instrument.close()
                    except:
                        pass
                    self.instrument = None
                
                # If connection lost error, retry with fresh ResourceManager
                if "CONN_LOST" in error_str or "connection" in error_str.lower():
                    if attempt < max_retries - 1:
                        # Force fresh ResourceManager on next attempt
                        if self.rm is not None:
                            try:
                                self.rm.close()
                            except:
                                pass
                            self.rm = None
                        continue
                
                # Final failure
                self.connected = False
                return False
        
        self.connected = False
        return False
    
    def disconnect(self):
        """Disconnect from Keithley (ramp to zero first)."""
        if self.connected and self.instrument:
            try:
                # Ramp to zero before disconnecting
                self.ramp_to_voltage(0.0)
                self.instrument.write(self.commands['output_off'])
                self.instrument.close()
            except Exception as e:
                print(f"Error during disconnect: {e}")
        self.connected = False
        self.instrument = None  # Clear so reconnect works
    
    def get_voltage(self):
        """Get current voltage setpoint."""
        if self.connected and self.instrument:
            try:
                response = self.instrument.query(self.commands['get_voltage'])
                self.current_voltage = float(response)
            except:
                pass
        return self.current_voltage
    
    def set_voltage(self, voltage, enable_output=False):
        """Set voltage (direct, no ramping).
        
        Args:
            voltage: Target voltage in V
            enable_output: If True, also sends output_on command (slower)
        """
        if abs(voltage) > self.max_voltage:
            print(f"Voltage {voltage}V exceeds limit {self.max_voltage}V")
            return False
        
        if self.connected and self.instrument:
            try:
                self.instrument.write(self.commands['set_voltage'].format(voltage))
                if enable_output:
                    self.instrument.write(self.commands['output_on'])
                self.current_voltage = voltage
                return True
            except Exception as e:
                print(f"Error setting voltage: {e}")
                return False
        else:
            # Simulation mode
            self.current_voltage = voltage
            return True
    
    def ramp_to_voltage(self, target, slew_rate=None, stop_check=None):
        """Safely ramp to target voltage with optional stop checking.
        
        For small voltage changes (<1V), sets directly (safe, fast).
        For large changes, ramps in steps to prevent damage.
        
        Args:
            target: Target voltage in V
            slew_rate: Ramp rate in V/s (uses default if None)
            stop_check: Optional callable that returns True if ramp should stop
            
        Returns:
            True if ramp completed, False if stopped or failed
        """
        if slew_rate is None:
            slew_rate = self.slew_rate
        
        if abs(target) > self.max_voltage:
            print(f"Target {target}V exceeds limit {self.max_voltage}V")
            return False
        
        start = self.current_voltage
        voltage_diff = abs(target - start)
        
        if voltage_diff < 0.001:  # Already at target
            return True
        
        # For small changes (<1V), just set directly - it's safe and fast
        # Output is already on from connection, so don't send redundant command
        if voltage_diff < 1.0:
            return self.set_voltage(target, enable_output=False)
        
        # For large changes, ramp to prevent sample damage
        print(f"Ramping gate: {start:.3f}V → {target:.3f}V at {slew_rate} V/s")
        
        # Calculate ramp parameters - ~1V steps, limited by slew rate
        n_steps = max(int(voltage_diff), 2)  # At least 2 steps, ~1V per step
        ramp_time = voltage_diff / slew_rate
        step_delay = ramp_time / n_steps
        voltages = np.linspace(start, target, n_steps + 1)[1:]  # Skip start
        
        for i, v in enumerate(voltages):
            # Check for stop request
            if stop_check is not None and stop_check():
                print(f"GATE RAMP STOPPED at {self.current_voltage:.3f}V (step {i}/{len(voltages)})")
                return False
            
            if not self.set_voltage(v, enable_output=False):  # Don't re-enable output each step
                print(f"Failed to set voltage to {v}V")
                return False
            time.sleep(step_delay)
        
        print(f"Gate ramp complete: now at {self.current_voltage:.3f}V")
        return True
    
    def output_on(self):
        """Enable output."""
        if self.connected and self.instrument:
            self.instrument.write(self.commands['output_on'])
    
    def output_off(self):
        """Disable output."""
        if self.connected and self.instrument:
            self.instrument.write(self.commands['output_off'])
    
    def measure_current(self):
        """Measure current through device."""
        if self.connected and self.instrument:
            try:
                response = self.instrument.query(self.commands['measure_current'])
                # Response format may include multiple values, take first
                return float(response.split(',')[0])
            except:
                return 0.0
        return 0.0
    
    def set_compliance(self, current_limit):
        """Set compliance (current limit) and matching sense range."""
        self.compliance_current = current_limit
        if self.connected and self.instrument:
            self.instrument.write(self.commands['compliance_current'].format(current_limit))
            self.instrument.write(f':SENS:CURR:RANG {current_limit}')  # Match sense range
            print(f"Keithley compliance set to {current_limit*1e9:.0f} nA")


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
        
        # Gate voltage safety settings
        self.gate_slew_rate = 10.0  # V/s
        self.gate_ramp_to_zero_after = True
        self.gate_ramp_on_stop = False
        self.gate_max_voltage = 100.0
        self.gate_compliance = 100e-9  # 100 nA
    
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
        
        # Track if gate voltage is involved (for post-sweep ramp to zero)
        gate_involved = (sweep_param == "Gate Voltage (V)" or step_param == "Gate Voltage (V)")
        
        # Pre-sweep: Ramp gate voltage to start position (NEVER jump!)
        if gate_involved and (not self.use_simulation or self.keithley.connected):
            if sweep_param == "Gate Voltage (V)":
                start_voltage = sweep_start
            elif step_param == "Gate Voltage (V)":
                start_voltage = step_start
            else:
                start_voltage = fixed_values.get('vg', 0)
            
            current_voltage = self.keithley.get_voltage()
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
                        self.is_running = False
                        return
        
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
                            # For step changes, use safe ramping (can be large jumps)
                            if not self.use_simulation and self.keithley.connected:
                                print(f"Step {step_idx}: Ramping gate to {step_val}V")
                                self.keithley.ramp_to_voltage(
                                    step_val,
                                    slew_rate=self.gate_slew_rate,
                                    stop_check=self.stop_check
                                )
                        elif step_param == "Temperature (K)":
                            fixed_values['temperature'] = step_val
                        
                        # Set parameter (skip gate voltage - already ramped above)
                        if step_param != "Gate Voltage (V)":
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
                
                # === GATE VOLTAGE SWEEP MODE ===
                # Set voltage, read it back (forces USB sync), take VNA measurement
                elif sweep_param == "Gate Voltage (V)" and not self.use_simulation and self.keithley.connected:
                    import time as time_module
                    
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
                    self.keithley.instrument.write(':SENS:CURR:RANG 1e-6')  # Fixed 1µA range
                    self.keithley.instrument.write(':SOUR:VOLT:DEL 0')  # Zero source delay
                    self.keithley.instrument.write(':SYST:AZER OFF')
                    self.keithley.instrument.write(':DISP:ENAB OFF')
                    
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
                            
                            # Accumulate for averaging
                            s21_real_acc[sweep_idx] += np.real(s21)
                            s21_imag_acc[sweep_idx] += np.imag(s21)
                            
                            # Calculate running average
                            n_completed = avg_idx + 1
                            avg_real = s21_real_acc[sweep_idx] / n_completed
                            avg_imag = s21_imag_acc[sweep_idx] / n_completed
                            avg_s21 = avg_real + 1j * avg_imag
                            
                            # Update sweep_data with averaged value
                            if avg_idx == 0:
                                # First sweep - append new data
                                sweep_data.append({
                                    'sweep_value': target_voltage,
                                    'step_value': step_val,
                                    's21_real': avg_real,
                                    's21_imag': avg_imag,
                                    's21_mag': np.abs(avg_s21),
                                    's21_phase': np.angle(avg_s21, deg=True)
                                })
                            else:
                                # Subsequent sweeps - update existing data
                                sweep_data[sweep_idx] = {
                                    'sweep_value': target_voltage,
                                    'step_value': step_val,
                                    's21_real': avg_real,
                                    's21_imag': avg_imag,
                                    's21_mag': np.abs(avg_s21),
                                    's21_phase': np.angle(avg_s21, deg=True)
                                }
                            
                            # Send data point to GUI (shows running average)
                            self.data_queue.put({
                                'type': 'point',
                                'sweep_idx': sweep_idx,
                                'step_idx': step_idx,
                                'data': sweep_data[sweep_idx],
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
            # Post-sweep: Handle gate voltage
            if gate_involved:
                if self.should_stop and self.gate_ramp_on_stop:
                    # User stopped - ramp to zero if enabled
                    print("=== Measurement stopped - ramping gate to zero ===")
                    self.keithley.ramp_to_voltage(0.0, slew_rate=self.gate_slew_rate)
                elif not self.should_stop and self.gate_ramp_to_zero_after:
                    # Normal completion - ramp to zero if enabled
                    print("=== Post-sweep: Ramping gate voltage to zero ===")
                    self.keithley.ramp_to_voltage(0.0, slew_rate=self.gate_slew_rate)
                else:
                    current_v = self.keithley.get_voltage()
                    print(f"Gate voltage remains at {current_v:.3f}V")
            
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
            self.vna, self.magnet, self.keithley, self.temp_controller, use_simulation=False
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
        self.magnet_addr = tk.StringVar(value="22")
        self.magnet_model = tk.StringVar(value="AMI 420")
        self.keithley_addr = tk.StringVar(value="18")
        self.keithley_model = tk.StringVar(value="2450")
        self.temp_port = tk.StringVar(value="COM3")
        
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
        self.fixed_temp = tk.StringVar(value="300")
        self.ifbw = tk.StringVar(value="100")
        
        # S-parameter selection
        self.s_parameter = tk.StringVar(value="S21")
        
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
        magnet_label_frame = ttk.Frame(conn_frame)
        magnet_label_frame.grid(row=2, column=0, padx=10, pady=5, sticky='w')
        ttk.Label(magnet_label_frame, text="Magnet:").pack(side='left')
        magnet_model_combo = ttk.Combobox(
            magnet_label_frame, textvariable=self.magnet_model,
            values=["AMI 420", "Cryomagnetics 4G"], width=14, state='readonly'
        )
        magnet_model_combo.pack(side='left', padx=5)
        magnet_model_combo.bind('<<ComboboxSelected>>', self.on_magnet_model_change)
        
        magnet_addr_frame = ttk.Frame(conn_frame)
        magnet_addr_frame.grid(row=2, column=1, padx=10, pady=5)
        ttk.Label(magnet_addr_frame, text="GPIB::").pack(side='left')
        ttk.Entry(magnet_addr_frame, textvariable=self.magnet_addr, width=4).pack(side='left')
        
        self.magnet_status = ttk.Label(conn_frame, text="●", foreground='gray', font=('Arial', 14))
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
        keithley_addr_frame.grid(row=3, column=1, padx=10, pady=5)
        ttk.Label(keithley_addr_frame, text="Addr:").pack(side='left')
        keithley_addr_entry = ttk.Entry(keithley_addr_frame, textvariable=self.keithley_addr, width=40)
        keithley_addr_entry.pack(side='left')
        ToolTip(keithley_addr_entry, "GPIB: Enter number (e.g., 18)\nUSB: Enter full address (e.g., USB0::0x05E6::0x2450::04102170::INSTR)")
        
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
        
        # Gate Voltage Safety Settings
        safety_frame = ttk.LabelFrame(parent, text="Gate Voltage Safety")
        safety_frame.pack(fill='x', padx=10, pady=5)
        
        # Row 0: Slew rate and max voltage
        ttk.Label(safety_frame, text="Slew Rate (V/s):").grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.gate_slew_rate = tk.StringVar(value="10.0")
        slew_entry = ttk.Entry(safety_frame, textvariable=self.gate_slew_rate, width=8)
        slew_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        ToolTip(slew_entry, "Rate at which gate voltage changes (V/s).\nPrevents abrupt voltage jumps that can damage samples.")
        
        ttk.Label(safety_frame, text="Max Voltage (V):").grid(row=0, column=2, padx=10, pady=5, sticky='w')
        self.max_gate_voltage = tk.StringVar(value="100")
        max_entry = ttk.Entry(safety_frame, textvariable=self.max_gate_voltage, width=8)
        max_entry.grid(row=0, column=3, padx=5, pady=5, sticky='w')
        ToolTip(max_entry, "Maximum allowed gate voltage magnitude.\nVoltages beyond this will be clamped.")
        
        ttk.Label(safety_frame, text="Compliance (nA):").grid(row=0, column=4, padx=10, pady=5, sticky='w')
        self.gate_compliance = tk.StringVar(value="100")
        compliance_entry = ttk.Entry(safety_frame, textvariable=self.gate_compliance, width=8)
        compliance_entry.grid(row=0, column=5, padx=5, pady=5, sticky='w')
        ToolTip(compliance_entry, "Current limit for gate (nA).\nProtects against gate leakage/shorts.")
        
        # Row 1: Safety checkboxes
        self.ramp_gate_to_zero = tk.BooleanVar(value=True)
        ramp_zero_cb = ttk.Checkbutton(
            safety_frame, text="Ramp gate to zero after sweep",
            variable=self.ramp_gate_to_zero
        )
        ramp_zero_cb.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky='w')
        ToolTip(ramp_zero_cb, "After measurement completes, safely ramp\ngate voltage back to 0V.")
        
        self.ramp_gate_on_stop = tk.BooleanVar(value=False)
        ramp_stop_cb = ttk.Checkbutton(
            safety_frame, text="Ramp to zero on Stop",
            variable=self.ramp_gate_on_stop
        )
        ramp_stop_cb.grid(row=1, column=2, columnspan=2, padx=10, pady=5, sticky='w')
        ToolTip(ramp_stop_cb, "When Stop is pressed, ramp gate to 0V.\nIf unchecked, gate freezes at current value.")
        
        # Row 2: Current gate voltage display and manual control
        ttk.Label(safety_frame, text="Current Gate:").grid(row=2, column=0, padx=10, pady=5, sticky='w')
        self.current_gate_display = ttk.Label(safety_frame, text="0.000 V", font=('Arial', 10, 'bold'))
        self.current_gate_display.grid(row=2, column=1, padx=5, pady=5, sticky='w')
        
        ttk.Label(safety_frame, text="Set Gate (V):").grid(row=2, column=2, padx=10, pady=5, sticky='w')
        self.manual_gate_entry = ttk.Entry(safety_frame, width=8)
        self.manual_gate_entry.insert(0, "0")
        self.manual_gate_entry.grid(row=2, column=3, padx=5, pady=5, sticky='w')
        
        ttk.Button(safety_frame, text="Ramp To", command=self.manual_gate_ramp).grid(row=2, column=4, padx=5, pady=5)
        ttk.Button(safety_frame, text="Ramp to Zero", command=self.ramp_gate_to_zero_now).grid(row=2, column=5, padx=5, pady=5)
        
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
        self.sweep_averages.trace_add('write', self.update_scan_time_display)
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
        
        # Smoothing control frame
        smooth_frame = ttk.Frame(trace_control)
        smooth_frame.pack(fill='x', pady=2)
        
        ttk.Label(smooth_frame, text="Smoothing:").pack(side='left', padx=5)
        self.smoothing_window = tk.IntVar(value=1)
        smooth_spinbox = ttk.Spinbox(
            smooth_frame, from_=1, to=101, increment=2,
            textvariable=self.smoothing_window, width=5,
            command=self.update_single_trace
        )
        smooth_spinbox.pack(side='left', padx=2)
        smooth_spinbox.bind('<Return>', lambda e: self.update_single_trace())
        ttk.Label(smooth_frame, text="pts (1=off, odd values only)").pack(side='left', padx=2)
        
        # Quick smoothing buttons
        ttk.Button(smooth_frame, text="Off", width=4,
                   command=lambda: self._set_smoothing(1)).pack(side='left', padx=2)
        ttk.Button(smooth_frame, text="3", width=3,
                   command=lambda: self._set_smoothing(3)).pack(side='left', padx=1)
        ttk.Button(smooth_frame, text="5", width=3,
                   command=lambda: self._set_smoothing(5)).pack(side='left', padx=1)
        ttk.Button(smooth_frame, text="11", width=3,
                   command=lambda: self._set_smoothing(11)).pack(side='left', padx=1)
        ttk.Button(smooth_frame, text="21", width=3,
                   command=lambda: self._set_smoothing(21)).pack(side='left', padx=1)
        
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
                        title_parts.append(f"avg={averages}")
                    
                    # Add smoothing info if active
                    smooth_window = self.smoothing_window.get()
                    if smooth_window > 1:
                        title_parts.append(f"smooth={smooth_window}pt")
                    
                    title = " | ".join(title_parts) if title_parts else "Single Trace"
                    self.ax_trace.set_title(title, fontsize=9)
        
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
            y = [20 * np.log10(d['s21_mag'] + 1e-12) for d in trace_data]
        elif mode == "Phase":
            y = [d['s21_phase'] for d in trace_data]
        elif mode == "Real":
            y = [d['s21_real'] for d in trace_data]
        else:  # Imaginary
            y = [d['s21_imag'] for d in trace_data]
        
        # Apply smoothing if enabled
        window = self.smoothing_window.get()
        if window > 1 and len(y) >= window:
            y = self._smooth_data(y, window)
        
        return y
    
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
        """Set smoothing window and update plot."""
        self.smoothing_window.set(value)
        self.update_single_trace()
    
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
        # Create appropriate controller based on model selection
        model = self.magnet_model.get()
        self.magnet = create_magnet_controller(model)
        
        if self.simulation_mode.get():
            self.magnet_connected.set(True)
            self.magnet_status.config(foreground='green')
            self.status_var.set(f"{model} connected (simulation)")
        else:
            try:
                self.magnet.set_address(self.magnet_addr.get())
                if self.magnet.connect():
                    self.magnet_connected.set(True)
                    self.magnet_status.config(foreground='green')
                    self.status_var.set(f"{model} connected")
                else:
                    messagebox.showerror("Connection Error", f"Failed to connect to {model}")
            except Exception as e:
                messagebox.showerror("Connection Error", f"{model} connection failed: {e}")
    
    def on_magnet_model_change(self, event=None):
        """Handle magnet model selection change."""
        model = self.magnet_model.get()
        # Update default address based on model
        if model == "AMI 420":
            self.magnet_addr.set("22")
        elif model == "Cryomagnetics 4G":
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
                print(f"Target clamped to ±{max_v}V")
            
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
    
    def update_gate_display(self):
        """Update the current gate voltage display."""
        try:
            voltage = self.keithley.get_voltage()
            self.current_gate_display.config(text=f"{voltage:.3f} V")
        except:
            pass
    
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
        """Update the estimated scan time display based on sweep type, points, IFBW, and averages."""
        try:
            points = int(float(self.sweep_points.get()))
            ifbw = float(self.ifbw.get())
            sweep_param = self.sweep_param.get()
            averages = max(1, int(float(self.sweep_averages.get())))
            
            if points <= 0 or ifbw <= 0:
                self.scan_time_label.config(text="")
                return
            
            # Different timing for different sweep types
            if sweep_param == "Frequency (GHz)":
                # VNA batch sweep mode - fast
                sweep_time = points / ifbw
                single_sweep_time = sweep_time * 1.5 + 2.0
            elif sweep_param == "Gate Voltage (V)":
                # ~2ms Keithley + ~42ms VNA per point (with auto-range disabled)
                time_per_point = 0.050
                single_sweep_time = points * time_per_point + 2.0
            else:
                # Other point-by-point sweeps (B-field, Power, Temp)
                time_per_point = 0.10 + (1.0 / ifbw)
                single_sweep_time = points * time_per_point + 2.0
            
            # Total time with averages
            adjusted_time = single_sweep_time * averages
            
            # Format display
            if adjusted_time < 60:
                time_str = f"~{adjusted_time:.1f}s"
            else:
                minutes = int(adjusted_time // 60)
                seconds = adjusted_time % 60
                time_str = f"~{minutes}m {seconds:.0f}s"
            
            if averages > 1:
                self.scan_time_label.config(text=f"Est: {time_str} ({averages} sweeps × {single_sweep_time:.1f}s)")
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
                'averages': max(1, int(float(self.sweep_averages.get()))),
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
        
        # Update gate voltage display periodically
        self.update_gate_display()
        
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
            elif is_2d and self.current_2d_folder:
                # Save 2D plot images at end of 2D measurement
                self.save_plot_images(os.path.join(self.current_2d_folder, "plot"))
            
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
                    f.write(f"# Averages: {self.current_config.get('averages', 1)}\n")
                    
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
            "Power (dBm)": "Power",
            "Temperature (K)": "Temp"
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
            "Power (dBm)": "Power",
            "Temperature (K)": "Temp"
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
        # Check if gate voltage is non-zero
        gate_voltage = app.keithley.get_voltage()
        
        if app.measurement_engine.is_running:
            if messagebox.askokcancel("Quit", "Measurement in progress. Stop and quit?"):
                app.measurement_engine.stop()
                time.sleep(0.5)  # Give measurement time to stop
                
                # Ramp gate to zero if not already
                if abs(gate_voltage) > 0.01:
                    app.status_var.set("Ramping gate to zero before exit...")
                    root.update()
                    app.keithley.ramp_to_voltage(0.0)
                
                root.destroy()
        else:
            # Not running, but check gate voltage
            if abs(gate_voltage) > 0.01:
                if messagebox.askyesno("Gate Voltage", 
                    f"Gate voltage is at {gate_voltage:.3f}V.\n\nRamp to zero before exiting?"):
                    app.status_var.set("Ramping gate to zero...")
                    root.update()
                    app.keithley.ramp_to_voltage(0.0)
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()