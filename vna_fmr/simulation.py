"""Simulated FMR data generator for GUI testing without hardware."""

import numpy as np


class SimulatedDataGenerator:
    """Generates realistic-looking FMR data for GUI testing."""

    def __init__(self):
        # FMR parameters for simulation
        self.resonance_freq = 8e9  # 8 GHz base resonance
        self.linewidth = 200e6    # 200 MHz linewidth
        self.resonance_field = 0.3  # 300 mT
        self.field_linewidth = 0.02  # 20 mT

    def lorentzian_complex(self, x, x0, gamma, amplitude=1.0):
        """Generate complex Lorentzian (absorption + dispersion)."""
        # Real part: dispersion, Imag part: absorption
        delta = x - x0
        denom = delta**2 + gamma**2
        real_part = amplitude * delta / denom
        imag_part = -amplitude * gamma / denom
        return real_part + 1j * imag_part

    def generate_s21_vs_frequency(self, frequencies, b_field=0.0, vg=0.0, power=-10, temperature=300.0):
        """Generate S21(f) data - FMR absorption spectrum."""
        # Shift resonance with field (Kittel-like)
        f_res = self.resonance_freq + b_field * 28e9  # ~28 GHz/T for electron

        # Temperature affects linewidth (broader at higher T)
        temp_linewidth = self.linewidth * (1 + (temperature - 300) / 500)

        # Gate voltage affects amplitude slightly
        amplitude = 0.1 * (1 + 0.05 * vg)

        # Power affects signal amplitude
        amplitude *= 10**((power + 10) / 20)

        # Background transmission
        s21_background = 0.8 * np.exp(-1j * frequencies / 1e10)

        # FMR signal
        s21_fmr = self.lorentzian_complex(frequencies, f_res, temp_linewidth, amplitude)

        # Add noise (more noise at higher temperature)
        noise_level = 0.005 * (1 + temperature / 1000)
        noise = noise_level * (np.random.randn(len(frequencies)) + 1j * np.random.randn(len(frequencies)))

        return s21_background + s21_fmr + noise

    def generate_s21_vs_field(self, fields, frequency=8e9, vg=0.0, power=-10, temperature=300.0):
        """Generate S21(B) data - field-swept FMR."""
        # Find resonance field for given frequency
        b_res = (frequency - self.resonance_freq) / 28e9

        # Temperature affects linewidth
        temp_linewidth = self.field_linewidth * (1 + (temperature - 300) / 500)

        amplitude = 0.1 * (1 + 0.05 * vg)
        amplitude *= 10**((power + 10) / 20)

        # Background
        s21_background = 0.8 * np.ones(len(fields), dtype=complex)

        # FMR signal
        s21_fmr = self.lorentzian_complex(fields, b_res, temp_linewidth, amplitude)

        noise_level = 0.005 * (1 + temperature / 1000)
        noise = noise_level * (np.random.randn(len(fields)) + 1j * np.random.randn(len(fields)))

        return s21_background + s21_fmr + noise

    def generate_s21_vs_gate(self, vg_values, frequency=8e9, b_field=0.3, power=-10, temperature=300.0):
        """Generate S21(Vg) data - gate-dependent transmission."""
        # Gate voltage modulates transmission (e.g., graphene Dirac point)
        # Dirac point shifts slightly with temperature
        dirac_point = 2.0 + (temperature - 300) / 200

        amplitude = 0.1 * 10**((power + 10) / 20)

        # Transmission minimum at Dirac point
        conductivity = 1 + 0.3 * np.abs(vg_values - dirac_point)
        s21_background = 0.5 * conductivity / conductivity.max()

        # Small FMR-like feature
        s21_fmr = 0.05 * self.lorentzian_complex(vg_values, dirac_point, 1.0, amplitude)

        noise_level = 0.003 * (1 + temperature / 1000)
        noise = noise_level * (np.random.randn(len(vg_values)) + 1j * np.random.randn(len(vg_values)))

        return s21_background + s21_fmr + noise

    def generate_s21_vs_power(self, powers, frequency=8e9, b_field=0.3, vg=0.0, temperature=300.0):
        """Generate S21(P) data - power dependence."""
        # Linear power dependence with some saturation at high power
        linear_response = 10**(powers / 20)
        # Saturation threshold decreases at higher temperature
        sat_threshold = 100 * (1 - (temperature - 300) / 1000)
        saturation = 1 / (1 + linear_response / sat_threshold)

        s21_mag = 0.1 * linear_response * saturation
        s21_phase = -np.pi/4 + 0.1 * powers / 10

        s21 = s21_mag * np.exp(1j * s21_phase)

        noise_level = 0.002 * (1 + temperature / 1000)
        noise = noise_level * (np.random.randn(len(powers)) + 1j * np.random.randn(len(powers)))

        return s21 + noise

    def generate_s21_vs_temperature(self, temperatures, frequency=8e9, b_field=0.3, vg=0.0, power=-10):
        """Generate S21(T) data - temperature dependence."""
        # Resonance properties change with temperature
        # Linewidth broadens, amplitude decreases, resonance may shift

        amplitude = 0.1 * 10**((power + 10) / 20)

        # Background transmission (slight temperature dependence from cables/components)
        s21_background = 0.8 * (1 - 0.0005 * (temperatures - 300))

        # FMR signal amplitude decreases with temperature (thermal fluctuations)
        # and linewidth broadens
        s21_signal = []
        for T in temperatures:
            temp_amplitude = amplitude * np.exp(-(T - 300) / 500)
            temp_linewidth = self.field_linewidth * (1 + (T - 300) / 300)
            # Small resonance shift with temperature
            b_res = (frequency - self.resonance_freq) / 28e9 + 0.001 * (T - 300) / 100
            signal = temp_amplitude * self.lorentzian_complex(np.array([b_field]), b_res, temp_linewidth, 1.0)[0]
            s21_signal.append(signal)

        s21_signal = np.array(s21_signal)

        # More noise at higher temperature
        noise_level = 0.003 * (1 + temperatures / 500)
        noise = noise_level * (np.random.randn(len(temperatures)) + 1j * np.random.randn(len(temperatures)))

        return s21_background + s21_signal + noise
