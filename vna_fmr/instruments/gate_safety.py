"""Gate voltage safety wrapper for Keithley SMU.

Adds safety monitoring around gate operations during measurements.
"""

import time


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
