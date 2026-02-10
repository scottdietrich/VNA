"""Controller for NHMFL SCM1 Magnet via TCP/LabVIEW interface.

Communicates with LabVIEW program on the data PC via TCP socket.
Based on UCSB Young's group code, modified by Andrew Woods.

The LabVIEW program must be running on the data PC before connecting.
"""

import re
import socket
import struct
import time


class NHMFLMagnetController:
    """Controller for NHMFL SCM1 Magnet via TCP/LabVIEW interface."""

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
        return struct.pack('I', len(s)) + s.encode()

    @staticmethod
    def _get_byte_size(data):
        """Unpack length prefix from LabVIEW protocol."""
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

        Returns field in Tesla. On failure returns cached value.
        Check self._field_read_fresh to distinguish a live reading from a
        stale cached fallback.
        """
        self._field_read_fresh = False

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
            self._field_read_fresh = True
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
                    display_field = self.current_field if hasattr(self, '_last_good_field') else target_field

                setpoint = status['Setpoint']
                if setpoint < 0 or setpoint > self.max_field * 1.1:
                    setpoint = target_field  # Use our target instead

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
