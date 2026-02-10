"""Main GUI application for VNA FMR measurement system."""

import json
import os
import sys
import subprocess
import threading
import time
import queue
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import Normalize, LogNorm, SymLogNorm
import matplotlib.pyplot as plt
from matplotlib import cm
from .log_manager import log_manager
from .instruments import (VNAController, MagnetController, KeithleyController,
                          Lakeshore370Controller, create_magnet_controller)
from .measurement import MeasurementEngine

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
        
        # Trigger initial update of B-Field Sweep Settings visibility
        self.on_sweep_param_changed()
        
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
        
        # Row 1: Checkboxes and current field
        self.wait_for_field = tk.BooleanVar(value=True)
        ttk.Checkbutton(bfield_frame, text="Wait for field (2D)", variable=self.wait_for_field).grid(
            row=1, column=0, columnspan=2, padx=5, pady=2, sticky='w')
        
        self.use_continuous_mode = tk.BooleanVar(value=False)  # Default to stepped (safer)
        continuous_cb = ttk.Checkbutton(bfield_frame, text="Continuous Mode", variable=self.use_continuous_mode)
        continuous_cb.grid(row=1, column=2, padx=5, pady=2, sticky='w')
        ToolTip(continuous_cb, "Enable continuous sweep mode (faster but less accurate).\nâš ï¸ WARNING: GPIB timeouts cause field interpolation errors.\nLeave UNCHECKED for accurate field measurements at each point.")
        
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
        self.fixed_frame = ttk.LabelFrame(right_frame, text="Fixed Parameters")
        self.fixed_frame.pack(fill='x', pady=5)
        
        # Create fixed parameter entries (will be enabled/disabled based on sweep/step selection)
        self.fixed_entries = {}
        self.fixed_labels = {}  # Store labels so we can show/hide them
        
        row = 0
        self.fixed_labels['frequency'] = ttk.Label(self.fixed_frame, text="Frequency (GHz):")
        self.fixed_labels['frequency'].grid(row=row, column=0, padx=10, pady=5, sticky='w')
        self.fixed_entries['frequency'] = ttk.Entry(self.fixed_frame, textvariable=self.fixed_frequency, width=15)
        self.fixed_entries['frequency'].grid(row=row, column=1, padx=10, pady=5, sticky='w')
        ToolTip(self.fixed_entries['frequency'], "Range: 0.0001 - 18 GHz\n(auto-clamped if out of range)")
        
        row += 1
        # B-field row - will be hidden if magnet not connected
        self.fixed_labels['b_field'] = ttk.Label(self.fixed_frame, text="B-Field (T):")
        self.fixed_labels['b_field'].grid(row=row, column=0, padx=10, pady=5, sticky='w')
        self.fixed_entries['b_field'] = ttk.Entry(self.fixed_frame, textvariable=self.fixed_field, width=15)
        self.fixed_entries['b_field'].grid(row=row, column=1, padx=10, pady=5, sticky='w')
        ToolTip(self.fixed_entries['b_field'], "Range: -18 to +18 T\n(auto-clamped if out of range)")
        self.bfield_row = row  # Store row number for show/hide
        
        row += 1
        self.fixed_labels['vg'] = ttk.Label(self.fixed_frame, text="Gate Voltage (V):")
        self.fixed_labels['vg'].grid(row=row, column=0, padx=10, pady=5, sticky='w')
        self.fixed_entries['vg'] = ttk.Entry(self.fixed_frame, textvariable=self.fixed_gate, width=15)
        self.fixed_entries['vg'].grid(row=row, column=1, padx=10, pady=5, sticky='w')
        
        row += 1
        self.fixed_labels['power'] = ttk.Label(self.fixed_frame, text="VNA Power (dBm):")
        self.fixed_labels['power'].grid(row=row, column=0, padx=10, pady=5, sticky='w')
        self.fixed_entries['power'] = ttk.Entry(self.fixed_frame, textvariable=self.fixed_power, width=15)
        self.fixed_entries['power'].grid(row=row, column=1, padx=10, pady=5, sticky='w')
        ToolTip(self.fixed_entries['power'], "VNA output power setting.\nRange: -50 to +10 dBm\n\n"
                                             "Power at probe = VNA power - input attenuation\n"
                                             "(See VNA Settings for calculated probe power)")
        
        row += 1
        self.fixed_labels['temperature'] = ttk.Label(self.fixed_frame, text="Temperature:", foreground='gray')
        self.fixed_labels['temperature'].grid(row=row, column=0, padx=10, pady=5, sticky='w')
        # Temperature display DISABLED - Lakeshore causes GPIB conflicts
        self.fixed_temp_display = ttk.Label(self.fixed_frame, text="DISABLED", font=('Arial', 9), foreground='gray')
        self.fixed_temp_display.grid(row=row, column=1, padx=10, pady=5, sticky='w')
        ToolTip(self.fixed_labels['temperature'], "Temperature monitoring DISABLED\n"
                                                   "Lakeshore 370 causes GPIB conflicts with Keithley")
        
        # Update fixed parameter states
        self.update_fixed_params_state()
        
        # Initially hide B-field if magnet not connected
        self.update_bfield_visibility()
        
        # B-Field Sweep Settings (only used during experimental sweeps)
        self.bfield_sweep_frame = ttk.LabelFrame(right_frame, text="B-Field Sweep Settings")
        self.bfield_sweep_frame.pack(fill='x', pady=5)
        
        bfield_sweep_row = ttk.Frame(self.bfield_sweep_frame)
        bfield_sweep_row.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(bfield_sweep_row, text="Sweep Rate (T/min):").pack(side='left', padx=5)
        self.bfield_sweep_rate = tk.StringVar(value="0.3")
        sweep_rate_entry = ttk.Entry(bfield_sweep_row, textvariable=self.bfield_sweep_rate, width=8)
        sweep_rate_entry.pack(side='left', padx=5)
        ToolTip(sweep_rate_entry, 
                "Rate for B-field sweeps during measurements.\n"
                "Maximum limited by IFBW and hardware (0.5 T/min).\n"
                "No minimum - can be as slow as desired.\n\n"
                "Note: Tab 1 rate is used for moving between fields.")
        
        # Info label showing if rate will be limited
        self.bfield_sweep_rate_info = ttk.Label(
            self.bfield_sweep_frame,
            text="",
            font=('Arial', 8),
            foreground='gray'
        )
        self.bfield_sweep_rate_info.pack(padx=5, pady=(0, 5))
        
        # Add trace to update rate info when parameters change
        self.bfield_sweep_rate.trace_add('write', self.update_bfield_sweep_rate_info)
        self.sweep_points.trace_add('write', self.update_bfield_sweep_rate_info)
        self.sweep_start.trace_add('write', self.update_bfield_sweep_rate_info)
        self.sweep_stop.trace_add('write', self.update_bfield_sweep_rate_info)
        self.ifbw.trace_add('write', self.update_bfield_sweep_rate_info)
        self.sweep_averages.trace_add('write', self.update_bfield_sweep_rate_info)
        
        # Initially hide this frame (only show when sweep param is B-field)
        self.bfield_sweep_frame.pack_forget()
        
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
        
        # Only set defaults if this is a user-initiated change (not initial load)
        # Check if event is None (programmatic call) or if we're still initializing
        is_user_change = event is not None
        
        if is_user_change:
            # Set sensible defaults based on parameter type (only for user changes)
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
        
        # Show/hide B-Field Sweep Settings frame
        try:
            if sweep == "B-Field (T)":
                # Pack it right after Fixed Parameters frame
                if hasattr(self, 'bfield_sweep_frame') and hasattr(self, 'fixed_frame'):
                    self.bfield_sweep_frame.pack(fill='x', pady=5, after=self.fixed_frame)
                    # Update rate info
                    self.update_bfield_sweep_rate_info()
            else:
                if hasattr(self, 'bfield_sweep_frame'):
                    self.bfield_sweep_frame.pack_forget()
        except Exception as e:
            print(f"[DEBUG] Error in show/hide B-Field Sweep Settings: {e}")
            import traceback
            traceback.print_exc()
        
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
        
        # Only set defaults if this is a user-initiated change (not initial load)
        # Check if event is None (programmatic call) or if we're still initializing
        is_user_change = event is not None
        
        if is_user_change:
            # Set sensible defaults based on parameter type (only for user changes)
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
    
    def update_bfield_sweep_rate_info(self, *args):
        """Update the info label showing if B-field sweep rate will be limited."""
        try:
            # Only update if sweep parameter is B-field
            if self.sweep_param.get() != "B-Field (T)":
                return
            
            # Check if all required attributes exist (may not during initialization)
            if not hasattr(self, 'bfield_sweep_rate') or not hasattr(self, 'sweep_averages'):
                return
            
            requested_rate = float(self.bfield_sweep_rate.get())
            ifbw = float(self.ifbw.get())
            points = int(self.sweep_points.get())
            start = float(self.sweep_start.get())
            stop = float(self.sweep_stop.get())
            num_averages = max(1, int(float(self.sweep_averages.get())))
            
            # Calculate field step and max rate from VNA
            field_range = abs(stop - start)
            field_step = field_range / max(points - 1, 1)
            
            # Determine mode and timing based on averaging
            if num_averages > 1:
                # STEPPED MODE: Field stops at each point for N measurements
                # Time = field move time + settling + (measurement Ã— N averages)
                field_move_time = field_step / (requested_rate / 60.0)  # Time to move between points
                settle_time = 0.1  # Settling time at each point
                single_measurement = 3.0 / ifbw + 0.10  # VNA measurement time
                measurement_time = single_measurement * num_averages + (num_averages - 1) * 0.01
                time_per_point = field_move_time + settle_time + measurement_time
                total_time = time_per_point * points
                
                # Hardware limit
                max_rate_hardware = 0.5  # T/min
                
                # Show stepped mode timing
                info_text = f"ðŸ”µ Stepped mode ({num_averages}Ã— avg): {total_time:.0f}s ({total_time/60:.1f} min)"
                if requested_rate > max_rate_hardware:
                    info_text = f"âš ï¸ Rate limited to {max_rate_hardware} T/min by hardware"
            else:
                # CONTINUOUS MODE: Field always moving, single measurement per point
                # Rate is limited by how fast VNA can measure
                single_measurement = 3.0 / ifbw + 0.10 + 0.05  # VNA + overhead
                max_rate_vna = (field_step / single_measurement) * 60.0  # T/min
                
                # Hardware limit
                max_rate_hardware = 0.5  # T/min
                
                # Determine what will limit the rate
                if requested_rate > max_rate_vna:
                    info_text = f"âš ï¸ Limited to {min(max_rate_vna, max_rate_hardware):.3f} T/min by IFBW"
                elif requested_rate > max_rate_hardware:
                    info_text = f"âš ï¸ Limited to {max_rate_hardware:.3f} T/min by hardware"
                else:
                    sweep_time = (field_range / requested_rate) * 60  # seconds
                    info_text = f"âœ“ Continuous mode: {sweep_time:.0f}s ({sweep_time/60:.1f} min)"
            
            self.bfield_sweep_rate_info.config(text=info_text)
        except:
            # Silently ignore errors during initialization
            if hasattr(self, 'bfield_sweep_rate_info'):
                self.bfield_sweep_rate_info.config(text="")
    
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
                # NOTE: Header does NOT have # prefix so pandas can read it as column names.
                # The metadata lines above have # prefix and will be skipped by pandas comment='#'.
                if is_freq_sweep:
                    header = "Frequency_GHz,S21_Real,S21_Imag,S21_Mag_dB,S21_Phase_deg"
                elif is_power_sweep:
                    header = "Power_at_Probe_dBm,S21_Real,S21_Imag,S21_Mag_dB,S21_Phase_deg"
                else:
                    header = f"{sweep_param.replace(' ', '_')},S21_Real,S21_Imag,S21_Mag_dB,S21_Phase_deg"
                
                if has_normalized:
                    header += ",S21_Mag_dB_Normalized"
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
                    f.write(f"{sweep_val},{d['s21_real']},{d['s21_imag']},")
                    f.write(f"{mag_db},{d['s21_phase']}")
                    
                    if has_normalized:
                        norm_val = d.get('s21_mag_db_norm')
                        if norm_val is not None:
                            f.write(f",{norm_val}")
                        else:
                            f.write(",")
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
                    f.write("Frequency_GHz,S21_Mag_dB\n")
                    
                    for i, s21_db in enumerate(spectrum):
                        freq_ghz = frequencies[i] / 1e9 if i < len(frequencies) else i
                        f.write(f"{freq_ghz},{s21_db}\n")
                
                elif self.reference_data.get('per_step_db') is not None:
                    # Per-frequency reference (for gate sweep with freq step)
                    per_step = self.reference_data['per_step_db']
                    step_freqs = self.reference_data.get('step_frequencies', [])
                    
                    f.write(f"# Type: Per-frequency CW measurements\n")
                    f.write(f"# Points: {len(per_step)}\n")
                    f.write("#\n")
                    f.write("Frequency_GHz,S21_Mag_dB\n")
                    
                    for i, freq in enumerate(step_freqs):
                        if i in per_step and per_step[i] is not None:
                            f.write(f"{freq/1e9},{per_step[i]}\n")
            
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
            'use_continuous_mode': self.use_continuous_mode.get() if hasattr(self, 'use_continuous_mode') else False,
            
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
            
            # Update sweep parameter UI state (show/hide B-Field Sweep Settings frame)
            self.on_sweep_param_changed()
            
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
            # Backward compatibility: convert old force_stepped_mode to new use_continuous_mode
            if hasattr(self, 'use_continuous_mode'):
                if 'use_continuous_mode' in settings:
                    self.use_continuous_mode.set(settings['use_continuous_mode'])
                elif 'force_stepped_mode' in settings:
                    # Invert old logic: force_stepped=True means use_continuous=False
                    self.use_continuous_mode.set(not settings['force_stepped_mode'])
            
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