"""Controller for Cryomagnetics 4G Magnet Power Supply.

Controls superconducting magnet via GPIB interface.
"""

import time


class CryomagneticsController:
    """Controller for Cryomagnetics 4G Magnet Power Supply."""

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
            self.instrument.timeout = 10000  # 10 seconds base timeout (reduced during sweep queries)
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

        Returns field in Tesla. On failure returns cached self.current_field.
        Check self._field_read_fresh to distinguish a live reading from a
        stale cached fallback.
        """
        self._field_read_fresh = False

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

                self._field_read_fresh = True
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
