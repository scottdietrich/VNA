"""Controller for Copper Mountain S5180B Vector Network Analyzer.

Connection: TCP socket to S2VNA software.
To enable: In S2VNA, go to System -> Misc Setup -> Network Setup
           Enable TCP/IP Socket Server on port 5025
"""

import socket
import time

import numpy as np


class VNAController:
    """Controller for Copper Mountain S5180B VNA."""

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

    def setup_frequency_sweep(self, f_start, f_stop, num_points, ifbw, power, averages=1):
        """Configure VNA for frequency sweep measurement.

        Args:
            f_start: Start frequency in Hz
            f_stop: Stop frequency in Hz
            num_points: Number of sweep points
            ifbw: IF bandwidth in Hz
            power: Source power in dBm
            averages: Number of hardware averages (1 = no averaging)
        """
        if not self.connected:
            return False

        avg_str = f", {averages}avg" if averages > 1 else ""
        print(f"VNA setup: {f_start/1e9:.3f}-{f_stop/1e9:.3f} GHz, {num_points} pts, IFBW={ifbw}, P={power} dBm{avg_str}")

        # Abort any ongoing operation and stop continuous triggering.
        # If the VNA was in CW continuous mode, :ABOR alone does not
        # prevent it from immediately re-triggering. The DSP can hang
        # if configuration commands arrive while a sweep is active.
        self.write(":ABOR")
        self.write(":INIT:CONT OFF")
        time.sleep(0.1)

        # Set frequency range
        self.write(f":SENS:FREQ:STAR {f_start}")
        self.write(f":SENS:FREQ:STOP {f_stop}")

        # Set number of points
        self.write(f":SENS:SWE:POIN {num_points}")

        # Set IF bandwidth
        self.write(f":SENS:BAND {ifbw}")

        # Wait for VNA to process IFBW change
        time.sleep(0.05)

        # Disable smoothing
        self.write(":CALC1:SMO OFF")
        self.write(":CALC1:SMO:APER 1")

        # Configure hardware averaging
        if averages > 1:
            self.write(f":SENS:AVER:COUN {averages}")  # Set number of averages
            self.write(":SENS:AVER:TYP SWE")           # Sweep-by-sweep averaging
            self.write(":SENS:AVER ON")                 # Enable averaging
            print(f"VNA hardware averaging: {averages} sweeps")
        else:
            self.write(":SENS:AVER OFF")                # Disable averaging

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

        # Query actual settings for diagnostic
        try:
            actual_ifbw = self.query(":SENS:BAND?")
            actual_smo = self.query(":CALC1:SMO?")
            actual_avg = self.query(":SENS:AVER?")
            actual_avg_cnt = self.query(":SENS:AVER:COUN?") if averages > 1 else "1"
            print(f"VNA confirms: IFBW={actual_ifbw.strip()}, Smooth={actual_smo.strip()}, Avg={actual_avg.strip()}, AvgCnt={actual_avg_cnt.strip()}")
        except Exception as e:
            print(f"VNA diagnostic query failed: {e}")

        # Wait for VNA to process settings
        time.sleep(0.3)

        return True

    def setup_cw_mode(self, frequency, ifbw, power):
        """Configure VNA for CW (single frequency) measurement.

        RF output stays ON continuously so the sample is always excited.
        The VNA runs in continuous trigger mode; data is read on demand.
        This avoids transient effects from pulsing the RF, which is
        critical for EPR/FMR where thermal and spin equilibrium matter.
        """
        if not self.connected:
            return False

        # Store IFBW for trigger timing
        self._cw_ifbw = ifbw

        # Abort any ongoing operation and stop continuous triggering
        self.write(":ABOR")
        self.write(":INIT:CONT OFF")
        time.sleep(0.05)

        # For CW mode: set start = stop = CW frequency, 1 point
        self.write(f":SENS:FREQ:STAR {frequency}")
        self.write(f":SENS:FREQ:STOP {frequency}")
        self.write(":SENS:SWE:POIN 1")

        # Set IF bandwidth
        self.write(f":SENS:BAND {ifbw}")

        # Wait for VNA to process IFBW change
        time.sleep(0.05)

        # Disable ALL processing that could cause digitization
        self.write(":CALC1:SMO OFF")           # Smoothing off
        self.write(":SENS:AVER OFF")           # Averaging off
        self.write(":CALC1:SMO:APER 1")        # Smoothing aperture to minimum

        # Set source power
        self.write(f":SOUR:POW {power}")

        # Set S-parameter
        self.write(f":CALC1:PAR1:DEF {self.s_parameter}")

        # Set data format to real/imag pairs
        self.write(":CALC1:FORM SDAT")

        # Turn RF ON and keep it on continuously
        self.write(":OUTP ON")

        # Continuous trigger mode: VNA sweeps repeatedly, RF always on
        # Data buffer always has fresh data; we just read when ready
        self.write(":INIT:CONT ON")

        # Query actual settings for diagnostic
        try:
            actual_ifbw = self.query(":SENS:BAND?")
            actual_smo = self.query(":CALC1:SMO?")
            actual_avg = self.query(":SENS:AVER?")
            print(f"VNA CW setup: IFBW={actual_ifbw.strip()}, Smooth={actual_smo.strip()}, Avg={actual_avg.strip()}, RF=ON (continuous)")
        except Exception as e:
            print(f"VNA diagnostic query failed: {e}")

        # Allow VNA to settle and complete at least one measurement cycle
        # so the data buffer is populated before first read
        settle_time = max(0.2, 2.0 / ifbw + 0.1)
        time.sleep(settle_time)

        return True

    def trigger_sweep(self):
        """Read the latest CW measurement from the continuously-running VNA.

        In continuous mode, the VNA is always measuring and the data buffer
        always has fresh data. We just wait long enough for one complete
        measurement cycle to ensure we get a new data point, then read.
        """
        if not self.connected:
            return False

        # Wait for one full measurement cycle to complete
        # This ensures the data buffer has been updated since we last read
        ifbw = getattr(self, '_cw_ifbw', 100)
        measurement_time = 1.0 / ifbw
        wait_time = measurement_time + 0.04  # 1x measurement time + 40ms processing
        time.sleep(wait_time)

        return True

    def trigger_sweep_timed(self, num_points, ifbw, averages=1):
        """Trigger sweep and wait for completion.

        Uses time-based waiting.  The S5180B's *OPC? does not block
        over TCP, so we sleep for the estimated sweep duration.

        Args:
            num_points: Number of sweep points
            ifbw: IF bandwidth in Hz
            averages: Number of hardware averages (affects wait time)
        """
        if not self.connected:
            return False

        # Expected single-sweep time with safety margin
        single_sweep_time = float(num_points) / float(ifbw)
        adjusted_single_sweep = single_sweep_time * 1.5 + 0.5

        # Make sure RF is on
        self.write(":OUTP ON")

        # Set to single sweep mode (hold)
        self.write(":INIT:CONT OFF")
        time.sleep(0.1)

        if averages > 1:
            # Clear averaging buffer to start fresh
            self.write(":SENS:AVER:CLE")
            time.sleep(0.1)

            print(f"VNA sweep: {num_points} pts x {averages} avg, "
                  f"est {adjusted_single_sweep * averages:.0f}s...")

            # Trigger N sweeps to accumulate averaging
            for i in range(averages):
                self.write(":INIT:IMM")
                time.sleep(adjusted_single_sweep)

                if (i + 1) % 5 == 0 or i == averages - 1:
                    print(f"  Averaging: {i + 1}/{averages} sweeps complete")
        else:
            # Single sweep
            self.write(":INIT:IMM")
            time.sleep(adjusted_single_sweep + 1.5)

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

        # Note: *OPC? and *WAI do NOT work on the S5180B over TCP — they
        # return immediately without waiting.  Sweep completion is verified
        # in trigger_sweep_timed() via stale-data detection instead.

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

    def flush_and_reset(self):
        """Recover the VNA after an interrupted measurement.

        Called after abort/error to leave the VNA in a clean state:
        1. Drain any unread data from the TCP socket
        2. Send :ABOR to stop any in-progress sweep
        3. Restore continuous sweep mode so S2VNA displays live data
        4. Ensure RF output stays ON
        """
        if not self.connected or not self.socket:
            return

        print("VNA: flushing socket and resetting state...")

        # Step 1: Drain TCP receive buffer (may have partial SDAT response)
        try:
            self.socket.setblocking(False)
            drained = 0
            while True:
                try:
                    chunk = self.socket.recv(65536)
                    if not chunk:
                        break
                    drained += len(chunk)
                except (BlockingIOError, socket.error):
                    break
            self.socket.setblocking(True)
            self.socket.settimeout(self.timeout)
            if drained > 0:
                print(f"  Drained {drained} stale bytes from socket")
        except Exception as e:
            print(f"  Socket drain error: {e}")
            try:
                self.socket.setblocking(True)
                self.socket.settimeout(self.timeout)
            except Exception:
                pass

        # Step 2: Abort any in-progress sweep
        try:
            self.write(":ABOR")
            time.sleep(0.1)
        except Exception as e:
            print(f"  :ABOR error: {e}")

        # Step 3: Restore continuous sweep mode (S2VNA shows live trace)
        try:
            self.write(":INIT:CONT ON")
        except Exception as e:
            print(f"  :INIT:CONT ON error: {e}")

        # Step 4: Ensure RF output is ON
        try:
            self.write(":OUTP ON")
        except Exception as e:
            print(f"  :OUTP ON error: {e}")

        # Step 5: Drain again (the above commands may have generated responses)
        time.sleep(0.2)
        try:
            self.socket.setblocking(False)
            while True:
                try:
                    self.socket.recv(65536)
                except (BlockingIOError, socket.error):
                    break
            self.socket.setblocking(True)
            self.socket.settimeout(self.timeout)
        except Exception:
            try:
                self.socket.setblocking(True)
                self.socket.settimeout(self.timeout)
            except Exception:
                pass

        print("VNA: reset complete")
