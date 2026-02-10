"""Controller for AMI Model 420 Power Supply Programmer.

Controls superconducting magnet via GPIB interface.
Reference: AMI Model 420 manual Rev. 7
"""

import time


class MagnetController:
    """Controller for AMI Model 420 Power Supply Programmer."""

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
        self._field_read_fresh = False

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
                self._field_read_fresh = True
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
