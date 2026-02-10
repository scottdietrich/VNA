"""Measurement engine for VNA FMR experiments."""

import json
import os
import queue
import threading
import time

import numpy as np
import pandas as pd

from .simulation import SimulatedDataGenerator
from .instruments.gate_safety import GateSafetyWrapper


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
        # Continuous sweep only works for single measurements (no averaging)
        # With averaging, we need to stop at each point to take multiple measurements
        num_averages = config.get('averages', 1)

        # Check if use_continuous_mode exists (backward compatibility)
        # Default to False (stepped mode) if attribute doesn't exist
        continuous_mode_enabled = False
        if hasattr(self, 'use_continuous_mode'):
            continuous_mode_enabled = self.use_continuous_mode.get()

        use_bfield_continuous = (
            sweep_param == "B-Field (T)" and
            not self.use_simulation and
            self.magnet.connected and
            num_averages == 1 and  # Only continuous if no averaging
            continuous_mode_enabled  # User must explicitly enable continuous mode (defaults to stepped)
        )

        # Debug output
        print(f"=== Measurement Configuration ===")
        print(f"Sweep: {sweep_param} ({sweep_start} to {sweep_stop}, {sweep_points} pts)")
        print(f"Step: {step_param} ({step_start} to {step_stop}, {step_points} pts)" if step_param else "Step: None")
        if use_vna_sweep:
            mode_str = 'VNA BATCH SWEEP'
        elif use_bfield_continuous:
            mode_str = 'B-FIELD CONTINUOUS (\u26a0\ufe0f interpolated fields, GPIB timeouts cause errors)'
        else:
            if sweep_param == "B-Field (T)":
                if num_averages > 1:
                    mode_str = f'B-FIELD STEPPED ({num_averages}\u00d7 avg)'
                else:
                    mode_str = 'B-FIELD STEPPED (measured fields at each point)'
            else:
                mode_str = 'POINT-BY-POINT'
        print(f"Mode: {mode_str}")
        print(f"Simulation: {self.use_simulation}, VNA connected: {self.vna.connected}")
        if use_bfield_continuous:
            print(f"Magnet connected: {self.magnet.connected}, Rate: {self.field_ramp_rate} T/min")
        elif sweep_param == "B-Field (T)" and num_averages > 1 and self.magnet.connected:
            print(f"Magnet connected: {self.magnet.connected}, Rate: {self.field_ramp_rate} T/min, Averaging: {num_averages}\u00d7")
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

            # Global target field grid for 2D measurements.
            # Set by the first sweep's actual recorded range, then reused
            # by all subsequent sweeps to ensure identical field axes.
            global_target_fields = None

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
                                fixed_values.get('ifbw', 100),
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
                    self.keithley.instrument.write(f':SENS:CURR:RANG {self.keithley.compliance_current}')  # Match sense range to compliance

                    # Model-specific commands
                    if self.keithley.model == '2450':
                        self.keithley.instrument.write(':SOUR:DEL 0')  # 2450: source delay
                        self.keithley.instrument.write(':SYST:AZER:STAT OFF')  # 2450: autozero
                        # 2450 doesn't have :DISP:ENAB, skip it
                    else:
                        self.keithley.instrument.write(':SOUR:DEL 0')  # 2400: source delay
                        self.keithley.instrument.write(':SYST:AZER:STAT OFF')  # 2400: autozero
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

                            # Set voltage (use model-appropriate command)
                            cmd = self.keithley.commands['set_voltage'].format(target_voltage) + '\n'
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
                # Only use this mode when averaging is disabled (num_averages == 1)
                elif use_bfield_continuous:
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
                    # Note: Continuous sweep only happens with num_averages=1 (see use_bfield_continuous logic)
                    ifbw = fixed_values.get('ifbw', 100)
                    num_freq_points = 1  # CW mode for B-field sweeps

                    # VNA measurement time: 3 time constants + overhead for CW mode
                    # Overhead includes: GPIB communication, data processing, Python overhead
                    # Empirically measured: ~100ms overhead for single-point CW measurements
                    vna_measurement_time = (num_freq_points / ifbw) * 3.0 + 0.10  # 3 time constants + 100ms overhead

                    # For CONTINUOUS sweeps, field settling doesn't make sense - the field never stops!
                    # We just need time to take the VNA measurement and process data
                    measurement_overhead = 0.05  # 50ms for data processing and GUI updates
                    min_time_per_point = vna_measurement_time + measurement_overhead

                    print(f"  VNA measurement time: {vna_measurement_time:.3f}s (IFBW={ifbw} Hz, CW mode, {num_averages} avg)")
                    print(f"  Continuous sweep mode: no field settling (field always moving)")

                    # Calculate field step size
                    field_range = abs(field_stop - field_start)
                    field_step = field_range / max(sweep_points - 1, 1)  # Step size between points

                    # Signed field range for expected field calculation (negative for down sweeps)
                    field_range_signed = field_stop - field_start

                    # Calculate maximum possible sweep rate based on VNA speed
                    # This is the fastest rate where VNA can keep up with field changes
                    max_rate_from_vna = (field_step / min_time_per_point) * 60.0  # T/min

                    # Use sweep-specific rate from Tab 2 (Measurement Control)
                    # Tab 1 rate is for moving magnet between positions, not for sweeps
                    try:
                        requested_rate = float(self.bfield_sweep_rate.get())
                    except:
                        # Fallback to Tab 1 rate if sweep rate not available
                        requested_rate = self.field_ramp_rate

                    # Apply only UPPER limits:
                    # 1. VNA measurement speed limit
                    # 2. Hardware capability limit (0.5 T/min for Cryomagnetics 4G)
                    max_hardware_rate = 0.5  # T/min
                    actual_ramp_rate = min(requested_rate, max_rate_from_vna, max_hardware_rate)

                    # Show what's limiting the rate
                    if actual_ramp_rate < requested_rate:
                        if actual_ramp_rate == max_rate_from_vna:
                            print(f"  \u26a0\ufe0f Requested rate {requested_rate:.3f} T/min too fast for IFBW={ifbw} Hz")
                            print(f"      Limited to {actual_ramp_rate:.3f} T/min (VNA can measure every {min_time_per_point:.3f}s)")
                        elif actual_ramp_rate == max_hardware_rate:
                            print(f"  \u26a0\ufe0f Requested rate {requested_rate:.3f} T/min exceeds hardware limit")
                            print(f"      Limited to {actual_ramp_rate:.3f} T/min (hardware maximum)")

                    # Calculate time per point based on actual rate
                    time_per_point = (field_step / (actual_ramp_rate / 60.0))  # seconds
                    expected_sweep_time = time_per_point * sweep_points  # seconds

                    print(f"  Requested rate: {requested_rate:.3f} T/min")
                    print(f"  Actual rate: {actual_ramp_rate:.3f} T/min")
                    print(f"  Field step: {field_step*1000:.3f} mT per point")
                    print(f"  Time per point: {time_per_point:.3f}s")
                    print(f"  Expected sweep time: {expected_sweep_time:.1f}s ({expected_sweep_time/60:.1f} min) for {sweep_points} points")

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

                                    # Check if hardware rate differs from requested (use percentage for robustness)
                                    requested_A_s = actual_ramp_rate / 60.0 / self.magnet.field_per_amp
                                    rate_difference = abs(reported_A_s - requested_A_s)
                                    rate_error_percent = (rate_difference / max(requested_A_s, 1e-6)) * 100

                                    # If hardware rate is different by more than 5%, warn and recalculate timing
                                    if rate_error_percent > 5.0:
                                        print(f"")
                                        print(f"  \u26a0\ufe0f WARNING: Hardware rate differs from requested \u26a0\ufe0f")
                                        print(f"  Software requested: {actual_ramp_rate:.4f} T/min ({requested_A_s:.6f} A/s)")
                                        print(f"  Controller reports: {actual_hardware_rate:.4f} T/min ({reported_A_s:.6f} A/s)")
                                        print(f"  Difference: {rate_error_percent:.1f}%")
                                        print(f"  Adjusting timing to match hardware rate...")
                                        print(f"")

                                        # Use hardware rate instead
                                        actual_ramp_rate = actual_hardware_rate
                                        time_per_point = (field_step / (actual_ramp_rate / 60.0))
                                        expected_sweep_time = time_per_point * sweep_points

                                        print(f"      Adjusted time per point: {time_per_point:.3f}s")
                                        print(f"      Adjusted sweep time: {expected_sweep_time:.1f}s ({expected_sweep_time/60:.1f} min)")
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
                        direction = "DOWN" if field_stop < current_field else "UP"
                        print(f"Cryomagnetics: {sweep_cmd} - sweeping {direction} to {field_stop:.4f} T at {actual_ramp_rate:.4f} T/min")

                    # Wait for initial settling after ramp starts
                    if self.field_settle_time > 0:
                        print(f"  Waiting {self.field_settle_time:.1f}s for initial field settling...")
                        time.sleep(self.field_settle_time)

                    # === FIELD ANCHOR SETUP ===
                    # Strategy: take real field readings every ~1s as anchor points,
                    # record the timestamp of each VNA measurement, then interpolate
                    # field values at all measurement times from the sparse anchors.
                    # This avoids both the drift of pure time-calculation AND the
                    # per-point GPIB blocking that caused 5s timeouts.
                    field_read_interval = 1.0  # seconds between real field reads
                    field_anchors = []  # List of (elapsed_time, real_field)
                    measurement_times = []  # Elapsed time of each VNA measurement
                    last_anchor_time = -field_read_interval  # Force a read at t=0

                    # Take initial field reading BEFORE the loop starts.
                    # Magnet has been ramping for settle_time but we record the actual field.
                    t_start = time_module.perf_counter()
                    initial_field = self.magnet.get_field(debug=False)
                    if getattr(self.magnet, '_field_read_fresh', False):
                        field_anchors.append((0.0, initial_field))
                        last_anchor_time = 0.0
                        print(f"  Starting continuous measurement...")
                        print(f"  Initial field (measured): {initial_field:.4f}T")
                        print(f"  Field anchors taken every {field_read_interval:.1f}s, interpolated between")
                    else:
                        print(f"  WARNING: Could not read initial field, falling back to requested start")
                        field_anchors.append((0.0, field_start))
                        last_anchor_time = 0.0

                    # Use time-calculated field for live GUI display during sweep.
                    # Real fields are filled in post-loop from anchors.
                    effective_rate = actual_hardware_rate if actual_hardware_rate else actual_ramp_rate

                    # Measure at regular time intervals
                    for sweep_idx in range(sweep_points):
                        if self.should_stop:
                            break

                        # Wait until it's time for this measurement
                        if sweep_idx > 0:
                            target_time = t_start + sweep_idx * time_per_point
                            current_time = time_module.perf_counter()
                            wait_time = target_time - current_time
                            if wait_time > 0:
                                time.sleep(wait_time)

                        # Periodically attempt a real field read for anchor points.
                        # IMPORTANT: get_field() can take seconds on GPIB timeout.
                        # The anchor timestamp must be taken AFTER get_field returns,
                        # not before -- otherwise the anchor says (t=10, field=0.2845)
                        # when the field was actually read at t=13. That mismatch
                        # causes np.interp to place that field value 3s too early,
                        # creating a jump in the interpolated field curve.
                        # Similarly, measurement_times must be recorded AFTER any
                        # anchor read, right before the VNA measurement, so the
                        # S21 timestamp matches when the VNA actually sampled.
                        elapsed_check = time_module.perf_counter() - t_start
                        if elapsed_check - last_anchor_time >= field_read_interval:
                            anchor_field = self.magnet.get_field(debug=False)
                            # Timestamp AFTER the read -- this is when the field was measured
                            anchor_time = time_module.perf_counter() - t_start
                            if getattr(self.magnet, '_field_read_fresh', False):
                                field_anchors.append((anchor_time, anchor_field))
                                last_anchor_time = anchor_time
                            else:
                                print(f"  [anchor] Stale read at t={anchor_time:.1f}s, skipped")

                        # Record measurement timestamp RIGHT BEFORE the VNA read.
                        # Any slow anchor read above is already done, so this
                        # timestamp accurately reflects when S21 is sampled.
                        elapsed = time_module.perf_counter() - t_start
                        measurement_times.append(elapsed)

                        # Time-calculated field for live GUI display only
                        field_change = (effective_rate / 60.0) * elapsed
                        if field_stop < field_start:
                            display_field = field_start - field_change
                        else:
                            display_field = field_start + field_change

                        # Clamp display field to target range
                        if field_stop < field_start:
                            display_field = max(display_field, field_stop)
                        else:
                            display_field = min(display_field, field_stop)

                        # Check if field has reached target based on last anchor
                        field_reached_target = False
                        if len(field_anchors) >= 2:
                            last_anchor_field = field_anchors[-1][1]
                            if field_stop > field_start and last_anchor_field >= field_stop - self.field_tolerance:
                                field_reached_target = True
                            elif field_stop < field_start and last_anchor_field <= field_stop + self.field_tolerance:
                                field_reached_target = True

                        if field_reached_target and sweep_idx > sweep_points * 0.25:
                            print(f"  Field reached target {field_stop:.4f}T at point {sweep_idx+1}/{sweep_points}")
                            print(f"  Stopping measurement")
                            # Fall through to take this final measurement, then break

                        # Take VNA measurement
                        s21 = self._get_real_data()

                        # Calculate normalized value if field reference available
                        s21_mag_db = 20 * np.log10(np.abs(s21) + 1e-12)
                        s21_mag_db_norm = None
                        if field_reference_s21_mag_db is not None:
                            s21_mag_db_norm = s21_mag_db - field_reference_s21_mag_db

                        # Store with display_field for now; real field filled in post-loop
                        sweep_data.append({
                            'sweep_value': display_field,
                            'step_value': step_val,
                            's21_real': np.real(s21),
                            's21_imag': np.imag(s21),
                            's21_mag': np.abs(s21),
                            's21_phase': np.angle(s21, deg=True),
                            's21_mag_db_norm': s21_mag_db_norm
                        })

                        # Send data to GUI (uses display_field for live plot)
                        self.data_queue.put({
                            'type': 'point',
                            'sweep_idx': sweep_idx,
                            'step_idx': step_idx,
                            'data': sweep_data[-1]
                        })

                        # Update progress
                        progress = (step_idx * sweep_points + sweep_idx + 1) / (len(step_values) * sweep_points) * 100
                        self.progress_queue.put(min(progress, 100))

                        print(f"  Point {sweep_idx+1}/{sweep_points}: Field = {display_field:.4f} T [calc] (anchors: {len(field_anchors)})")

                        # Break after recording the final measurement at target
                        if field_reached_target and sweep_idx > sweep_points * 0.25:
                            break


                    # Handle abort - stop the magnet ramp
                    if self.should_stop:
                        print(f"  B-field sweep ABORTED")
                        if hasattr(self.magnet, 'stop_ramp'):
                            self.magnet.stop_ramp()

                    t_total = time_module.perf_counter() - t_start
                    num_recorded = len(sweep_data)

                    # === FINAL FIELD ANCHOR ===
                    # Read field one last time now that the magnet has stopped (or nearly stopped).
                    # This is the most reliable reading of the sweep -- no GPIB contention.
                    try:
                        final_field = self.magnet.get_field(debug=False)
                        if final_field is not None:
                            field_anchors.append((t_total, final_field))
                            print(f"  Final field (measured): {final_field:.4f}T")
                    except Exception as e:
                        print(f"  WARNING: Final field read failed ({e})")

                    print(f"  B-field sweep complete: {num_recorded} points in {t_total:.1f}s")
                    print(f"  Field anchors collected: {len(field_anchors)}")
                    for i, (t, f) in enumerate(field_anchors):
                        print(f"    Anchor {i}: t={t:.2f}s  field={f:.4f}T")

                    # === PASS 1: INTERPOLATE REAL FIELD AT EACH MEASUREMENT TIME ===
                    # We have sparse real field readings (anchors) and a timestamp for
                    # every VNA measurement. Interpolate to get the best-estimate real
                    # field at each measurement time.
                    if len(field_anchors) >= 2 and len(measurement_times) > 0:
                        anchor_times  = np.array([t for t, f in field_anchors])
                        anchor_fields = np.array([f for t, f in field_anchors])
                        meas_times    = np.array(measurement_times)

                        # Interpolate field at each measurement timestamp
                        real_fields = np.interp(meas_times, anchor_times, anchor_fields)

                        # Update sweep_data with interpolated real fields
                        for i in range(len(sweep_data)):
                            sweep_data[i]['sweep_value'] = float(real_fields[i])

                        print(f"  Field interpolated from anchors: {real_fields[0]:.4f}T -> {real_fields[-1]:.4f}T")
                    else:
                        print(f"  WARNING: Not enough anchors ({len(field_anchors)}) to interpolate field")

                    # === PASS 2: INTERPOLATE S21 ONTO UNIFORM FIELD GRID ===
                    # The recorded points are at irregular field spacing (due to
                    # variable VNA timing). Resample onto a uniform field grid so
                    # every sweep has exactly sweep_points rows for the data loader.
                    #
                    # CRITICAL: Use the ACTUAL recorded field range from the first
                    # sweep to set the target grid, then reuse this exact grid for
                    # all subsequent sweeps in the 2D measurement. This:
                    # 1. Avoids extrapolation artifacts (flat regions at edges)
                    # 2. Ensures all sweeps share identical field axes for proper 2D stacking
                    # 3. Grounds the grid in actual measured field values, not assumptions
                    if len(sweep_data) >= 2:
                        rec_fields = np.array([d['sweep_value'] for d in sweep_data])
                        rec_real   = np.array([d['s21_real']  for d in sweep_data])
                        rec_imag   = np.array([d['s21_imag']  for d in sweep_data])
                        rec_norm   = np.array([d['s21_mag_db_norm'] if d['s21_mag_db_norm'] is not None else np.nan
                                               for d in sweep_data])

                        # On first sweep: establish the global target grid from actual recorded range
                        # On subsequent sweeps: reuse the global grid for consistency
                        if global_target_fields is None:
                            # First sweep: use actual recorded endpoints
                            actual_start = rec_fields[0]
                            actual_stop  = rec_fields[-1]

                            # Check deviation from requested range
                            start_err = abs(actual_start - field_start) * 1000  # mT
                            stop_err  = abs(actual_stop  - field_stop)  * 1000
                            if start_err > 1.0 or stop_err > 1.0:
                                print(f"  \u26a0\ufe0f Recorded field range differs from requested:")
                                print(f"      Requested: {field_start:.4f}T -> {field_stop:.4f}T")
                                print(f"      Recorded:  {actual_start:.4f}T -> {actual_stop:.4f}T  (\u0394start={start_err:.2f}mT, \u0394stop={stop_err:.2f}mT)")

                            # Establish global grid from this first sweep's actual range
                            global_target_fields = np.linspace(actual_start, actual_stop, sweep_points)
                            print(f"  Global field grid established: {actual_start:.4f}T -> {actual_stop:.4f}T ({sweep_points} points)")

                        target_fields = global_target_fields

                        # np.interp requires xp (rec_fields) increasing; also make target_fields
                        # increasing for interpolation, then reverse back if sweeping down
                        sweeping_down = (field_stop < field_start)
                        if sweeping_down:
                            # Reverse recorded data and target grid to make them increasing
                            rec_fields = rec_fields[::-1]
                            rec_real   = rec_real[::-1]
                            rec_imag   = rec_imag[::-1]
                            rec_norm   = rec_norm[::-1]
                            target_fields = target_fields[::-1]  # field_start > field_stop, so reverse to increasing

                        interp_real = np.interp(target_fields, rec_fields, rec_real)
                        interp_imag = np.interp(target_fields, rec_fields, rec_imag)

                        has_norm = not np.all(np.isnan(rec_norm))
                        if has_norm:
                            interp_norm = np.interp(target_fields, rec_fields, rec_norm)

                        # Reverse interpolated results back if we were sweeping down
                        if sweeping_down:
                            target_fields = target_fields[::-1]
                            interp_real = interp_real[::-1]
                            interp_imag = interp_imag[::-1]
                            if has_norm:
                                interp_norm = interp_norm[::-1]

                        # Rebuild sweep_data on the uniform grid
                        sweep_data = []
                        for i in range(sweep_points):
                            s21_complex = interp_real[i] + 1j * interp_imag[i]
                            sweep_data.append({
                                'sweep_value': float(target_fields[i]),
                                'step_value': step_val,
                                's21_real': float(interp_real[i]),
                                's21_imag': float(interp_imag[i]),
                                's21_mag': float(np.abs(s21_complex)),
                                's21_phase': float(np.angle(s21_complex, deg=True)),
                                's21_mag_db_norm': float(interp_norm[i]) if has_norm else None
                            })

                        print(f"  Resampled onto {sweep_points}-point uniform grid: {target_fields[0]:.4f}T -> {target_fields[-1]:.4f}T")

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

                    # Info message for B-field sweeps with averaging
                    if sweep_param == "B-Field (T)" and config.get('averages', 1) > 1:
                        num_averages = config.get('averages', 1)
                        print(f"  B-field stepped sweep: {sweep_points} points \u00d7 {num_averages} averages")
                        print(f"  Field stops at each point for {num_averages} measurements")

                    for sweep_idx, sweep_val in enumerate(sweep_values):
                        if self.should_stop:
                            break

                        current_point += 1
                        progress = current_point / total_points * 100
                        self.progress_queue.put(progress)

                        # Set sweep parameter
                        self._set_parameter(sweep_param, sweep_val, fixed_values)

                        # Get measurement(s) with averaging
                        num_averages = config.get('averages', 1)
                        if self.use_simulation:
                            # Simulation - single call (averaging not needed for sim data)
                            s21 = self._get_simulated_data(
                                sweep_param, sweep_val, step_param, step_val, fixed_values
                            )
                        else:
                            # Real measurement with averaging
                            if num_averages == 1:
                                s21 = self._get_real_data()
                            else:
                                # Average multiple measurements at this point
                                # Each _get_real_data() call triggers a new VNA measurement
                                if sweep_idx == 0:
                                    print(f"    Averaging {num_averages} measurements per point...")

                                s21_sum = 0.0 + 0.0j
                                for avg_idx in range(num_averages):
                                    s21_measurement = self._get_real_data()
                                    s21_sum += s21_measurement

                                    # Debug: Show first few measurements at first point
                                    if sweep_idx == 0 and avg_idx < 3:
                                        mag_db = 20 * np.log10(np.abs(s21_measurement) + 1e-12)
                                        print(f"      Measurement {avg_idx+1}/{num_averages}: {mag_db:.3f} dB")

                                    # Small delay between measurements (already have wait in trigger_sweep)
                                    if avg_idx < num_averages - 1:
                                        time.sleep(0.02)  # 20ms safety margin

                                s21 = s21_sum / num_averages

                                if sweep_idx == 0:
                                    avg_mag_db = 20 * np.log10(np.abs(s21) + 1e-12)
                                    print(f"      Averaged result: {avg_mag_db:.3f} dB")

                        # For B-field stepped sweeps, read the actual field after the magnet has reached target
                        # This ensures we record the true field value, not just the commanded value
                        # CRITICAL: Retry on stale reads - field accuracy is paramount
                        actual_sweep_val = sweep_val
                        if sweep_param == "B-Field (T)" and not self.use_simulation and self.magnet.connected:
                            max_retries = 10
                            retry_delay = 0.5  # seconds

                            for retry in range(max_retries):
                                measured_field = self.magnet.get_field()

                                if getattr(self.magnet, '_field_read_fresh', False):
                                    # Got a fresh read - use it
                                    actual_sweep_val = measured_field
                                    if sweep_idx < 3 or sweep_idx % 20 == 0:  # Print occasionally
                                        print(f"  Point {sweep_idx+1}/{sweep_points}: Field = {actual_sweep_val:.4f} T [measured]")
                                    break
                                else:
                                    # Stale read - retry
                                    if retry == 0:
                                        print(f"  \u26a0\ufe0f Point {sweep_idx+1}: Field read stale, retrying...")
                                    time.sleep(retry_delay)
                            else:
                                # All retries exhausted - this is serious
                                print(f"  \u274c ERROR Point {sweep_idx+1}: All {max_retries} field reads returned stale values!")
                                print(f"      Using commanded value {sweep_val:.4f} T (field accuracy unknown)")

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
                            'sweep_value': actual_sweep_val,  # Use measured field, not commanded
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

                # Verify we got the expected number of points (critical for B-field stepped sweeps)
                if sweep_param == "B-Field (T)" and not self.use_simulation:
                    expected_points = len(sweep_values)
                    actual_points = len(sweep_data)
                    if actual_points != expected_points:
                        print(f"  \u26a0\ufe0f WARNING: Expected {expected_points} points, got {actual_points}")
                        if self.should_stop:
                            print(f"      Measurement stopped early by user")
                        else:
                            print(f"      This indicates a measurement loop error!")
                    else:
                        print(f"  \u2714 Collected all {actual_points} field points")

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

                # Wait for field logic:
                # - If this is a STEP parameter: wait only if wait_for_field checkbox is enabled
                # - If this is a SWEEP parameter (not is_step): ALWAYS wait (we're in stepped mode)
                should_wait = (not is_step) or (is_step and self.wait_for_field)

                if should_wait:
                    if is_step:
                        print(f"Waiting for field to reach {value:.4f} T (tolerance: {self.field_tolerance} T)...")
                    else:
                        print(f"[Stepped mode] Waiting for field to reach {value:.4f} T...")

                    # Wait for field with stop checking
                    start_time = time.time()
                    timeout = 600  # 10 minute timeout
                    last_print_time = 0

                    while time.time() - start_time < timeout:
                        if self.should_stop:
                            print("Field wait interrupted by stop request")
                            return

                        current_field = self.magnet.get_field()

                        # Only verify field was fresh, don't rely on stale cached value
                        if not getattr(self.magnet, '_field_read_fresh', False):
                            print(f"  \u26a0\ufe0f Field read returned stale cached value, retrying...")
                            time.sleep(0.5)
                            continue

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
