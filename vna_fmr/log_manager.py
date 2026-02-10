"""Log manager for VNA FMR measurement system.

Captures stdout/stderr, displays in GUI, and saves to files.
"""

import io
import os
import queue
import sys
import time
import tkinter as tk
from datetime import datetime


class LogManager:
    """Manages logging to both GUI display and file.

    Captures print statements and logging output, displays them in a
    scrolling text widget, and saves to log files alongside data files.
    """

    def __init__(self):
        self.text_widget = None
        self.log_queue = queue.Queue()
        self.log_buffer = []  # Buffer for file saving
        self.max_buffer_lines = 10000  # Limit memory usage
        self.current_log_file = None
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._started = False
        self._update_pending = False
        self._last_update_time = 0
        self._min_update_interval = 0.1  # Minimum 100ms between GUI updates

    def set_text_widget(self, widget):
        """Set the tkinter Text widget for display."""
        self.text_widget = widget

    def start_capture(self):
        """Start capturing stdout/stderr."""
        if self._started:
            return
        self._started = True

        # Create custom stream that writes to both original and our queue
        self._stdout_redirector = self._StreamRedirector(
            self._original_stdout, self.log_queue, "INFO"
        )
        self._stderr_redirector = self._StreamRedirector(
            self._original_stderr, self.log_queue, "ERROR"
        )

        sys.stdout = self._stdout_redirector
        sys.stderr = self._stderr_redirector

    def stop_capture(self):
        """Restore original stdout/stderr."""
        if not self._started:
            return
        self._started = False
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

    def log(self, message, level="INFO"):
        """Add a message to the log."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted = f"[{timestamp}] {message}"

        # Add to queue for GUI update
        self.log_queue.put((level, formatted))

        # Add to buffer for file saving
        self.log_buffer.append(formatted)
        if len(self.log_buffer) > self.max_buffer_lines:
            self.log_buffer = self.log_buffer[-self.max_buffer_lines:]

        # Also print to original stdout (for console)
        self._original_stdout.write(formatted + "\n")
        self._original_stdout.flush()

    def update_display(self):
        """Process queued messages and update the text widget.

        Call this periodically from the main GUI thread.
        Throttled to prevent GUI lag.
        Returns True if any messages were processed.
        """
        if self.text_widget is None:
            # Just drain the queue if no widget
            try:
                while True:
                    self.log_queue.get_nowait()
            except queue.Empty:
                pass
            return False

        # Throttle updates to prevent GUI lag
        current_time = time.time()
        if current_time - self._last_update_time < self._min_update_interval:
            return False

        messages_processed = False
        batch_messages = []

        # Collect all pending messages (up to 50 at a time)
        try:
            for _ in range(50):
                level, message = self.log_queue.get_nowait()
                messages_processed = True
                batch_messages.append((level, message))

                # Add to buffer
                if message not in self.log_buffer:
                    self.log_buffer.append(message)
                    if len(self.log_buffer) > self.max_buffer_lines:
                        self.log_buffer = self.log_buffer[-self.max_buffer_lines:]
        except queue.Empty:
            pass

        # Update text widget in one batch
        if batch_messages:
            try:
                self.text_widget.config(state='normal')

                for level, message in batch_messages:
                    self.text_widget.insert(tk.END, message + "\n", level)

                # Auto-scroll to bottom
                self.text_widget.see(tk.END)

                # Limit display lines to prevent memory issues
                line_count = int(self.text_widget.index('end-1c').split('.')[0])
                if line_count > 500:  # Reduced from 1000
                    self.text_widget.delete('1.0', f'{line_count - 400}.0')

                self.text_widget.config(state='disabled')
            except tk.TclError:
                pass  # Widget may have been destroyed

            self._last_update_time = current_time

        return messages_processed

    def clear_display(self):
        """Clear the log display widget."""
        if self.text_widget:
            try:
                self.text_widget.config(state='normal')
                self.text_widget.delete('1.0', tk.END)
                self.text_widget.config(state='disabled')
            except tk.TclError:
                pass

    def clear_buffer(self):
        """Clear both display and buffer (for new measurement)."""
        self.clear_display()
        self.log_buffer = []

    def save_to_file(self, filepath):
        """Save log buffer to a file."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("VNA FMR Measurement Log\n")
                f.write(f"Saved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")

                for line in self.log_buffer:
                    f.write(line + "\n")

            self._original_stdout.write(f"Log saved to: {filepath}\n")
            return True
        except Exception as e:
            self._original_stdout.write(f"Error saving log: {e}\n")
            return False

    def get_log_filepath(self, data_filepath):
        """Generate log filepath from data filepath.

        For 1D: data_001.csv -> data_001_log.txt
        For 2D folder: data_001/ -> data_001/measurement_log.txt
        """
        if os.path.isdir(data_filepath):
            # 2D measurement folder
            return os.path.join(data_filepath, "measurement_log.txt")
        else:
            # 1D measurement file
            base, ext = os.path.splitext(data_filepath)
            return f"{base}_log.txt"

    class _StreamRedirector(io.StringIO):
        """Redirects a stream to both original output and a queue."""

        def __init__(self, original_stream, log_queue, level):
            super().__init__()
            self.original_stream = original_stream
            self.log_queue = log_queue
            self.level = level
            self.line_buffer = ""

        def write(self, text):
            # Write to original stream
            if self.original_stream:
                self.original_stream.write(text)
                self.original_stream.flush()

            # Buffer until we have complete lines
            self.line_buffer += text
            while '\n' in self.line_buffer:
                line, self.line_buffer = self.line_buffer.split('\n', 1)
                if line.strip():  # Skip empty lines
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    formatted = f"[{timestamp}] {line}"
                    self.log_queue.put((self.level, formatted))

        def flush(self):
            if self.original_stream:
                self.original_stream.flush()


# Global log manager instance
log_manager = LogManager()
