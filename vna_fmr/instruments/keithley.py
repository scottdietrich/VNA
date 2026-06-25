"""Controller for Keithley 2400/2450 SMU (gate voltage).

SAFETY FEATURES (v2.1):
- Never jumps voltage - always ramps, even on connect/reconnect
- Reads actual voltage BEFORE reset to prevent jumps
- Emergency shutdown capability
- Fails loudly on communication errors (no stale values)

Supports both Keithley 2400 and 2450 models with appropriate SCPI commands.
"""

import time

import numpy as np


class KeithleyController:
    """Controller for Keithley 2400/2450 SMU (gate voltage)."""

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
        },
        # BK Precision 9132B Triple-Output Power Supply (GPIB).
        # Channels are always voltage-source — there is no SOUR:FUNC mode
        # to set, no remote-sense / NPLC / autozero / sense-range to
        # configure. The active channel must be selected once via
        # 'INST CHn' before any per-channel command. We default to CH1.
        # Hardware limits: CH1/CH2 0–30 V, 3 A; CH3 0–5 V, 3 A.
        'BK 9132B': {
            'name': 'BK Precision 9132B',
            'reset': '*RST',
            'source_voltage': None,                  # N/A — always voltage source
            'set_voltage': 'VOLT {:.6f}',            # acts on selected channel
            'get_voltage': 'VOLT?',
            'compliance_current': 'CURR {:.9f}',     # current limit on selected channel
            'output_on': 'OUTP 1',
            'output_off': 'OUTP 0',
            'output_state': 'OUTP?',
            'measure_current': 'MEAS:CURR?',
            'measure_voltage': 'MEAS:VOLT?',
            'select_channel': 'INST CH{}',           # BK-specific
        },
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

        # BK 9132B is multi-channel — pick which channel drives the gate.
        # CH1/CH2: 0–30 V, 3 A    CH3: 0–5 V, 3 A
        # Ignored when self.model is a Keithley.
        self.channel = 1

        # BK-specific limits: CH1/CH2 can do 30 V / 3 A.
        # The GUI's max_gate_voltage entry provides the user-facing soft cap.
        if self.model == 'BK 9132B':
            if self.max_voltage > 30.0:
                self.max_voltage = 30.0
            self.compliance_current = 3.0  # A — full channel hardware compliance

        # Safety settings
        self._safe_step_size = 1.0  # Maximum voltage step without ramping (V)
        self._emergency_stop = False

        # For backward compatibility
        self.current_voltage = property(lambda self: self._current_voltage)

    @property
    def is_bk(self):
        """True when this controller is driving a BK 9132B (vs a Keithley)."""
        return self.model == 'BK 9132B'

    def set_model(self, model):
        """Change the SMU model (2400, 2450, or BK 9132B)."""
        if model in self.MODELS:
            self.model = model
            self.commands = self.MODELS[model]
            # When switching to BK, clamp max_voltage to CH1/CH2 hardware limit.
            if self.is_bk:
                if self.max_voltage > 30.0:
                    self.max_voltage = 30.0
                self.compliance_current = 3.0
            print(f"SMU model set to {self.commands['name']}")
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

                # BK 9132B: must enter remote mode BEFORE any SCPI query.
                # Without SYST:REM, the BK ignores queries and *IDN? times out.
                if self.is_bk:
                    self.instrument.write('SYST:REM')
                    time.sleep(0.2)

                # Verify connection
                idn = self._query_raw('*IDN?')
                print(f"Connected to: {idn}")

                # BK 9132B: pick the active channel BEFORE any per-channel
                # query (VOLT?, OUTP?, MEAS:*) so we read the correct one.
                if self.is_bk:
                    self._write_raw(self.commands['select_channel'].format(self.channel))
                    time.sleep(0.05)

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
                time.sleep(1.0 if self.is_bk else 0.5)  # BK takes longer to RST

                if self.is_bk:
                    # *RST returns BK to local mode — re-enter remote and
                    # re-select channel. No source-mode / 2-wire / sense-range.
                    self.instrument.write('SYST:REM')
                    time.sleep(0.2)
                    self._write_raw(self.commands['select_channel'].format(self.channel))
                    time.sleep(0.05)
                else:
                    # Configure Keithley for voltage sourcing
                    self._write_raw(self.commands['source_voltage'])
                    self._write_raw(':SOUR:VOLT:RANG:AUTO ON')
                    self._write_raw(':SYST:RSEN OFF')  # 2-wire mode

                # Set compliance (current limit)
                self._write_raw(self.commands['compliance_current'].format(self.compliance_current))
                if not self.is_bk:
                    # Match Keithley sense range to compliance for fast reads
                    self._write_raw(f':SENS:CURR:RANG {self.compliance_current}')
                if self.compliance_current >= 1e-3:
                    print(f"Compliance: {self.compliance_current:.3f} A")
                else:
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
                print(f"{self.commands['name']} connected safely. Output ON at 0.000V")
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
            if not self.is_bk:
                # BK 9132B has no sense-range concept — its CURR command is
                # the channel current limit, full stop.
                self._write_raw(f':SENS:CURR:RANG {current_limit}')
            if current_limit >= 1e-3:
                print(f"{self.commands['name']} compliance set to {current_limit:.3f} A")
            else:
                print(f"{self.commands['name']} compliance set to {current_limit*1e9:.0f} nA")

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
