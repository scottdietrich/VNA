"""Controller for Lakeshore 370 AC Resistance Bridge.

Read-only temperature monitoring for dilution refrigerator.
Does NOT control temperature - only reads from mixing chamber sensor.

GPIB communication requires ~1 second delays between commands.
"""

import time


class Lakeshore370Controller:
    """Controller for Lakeshore 370 AC Resistance Bridge."""

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
