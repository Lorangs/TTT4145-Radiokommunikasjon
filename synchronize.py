import numpy as np
from scipy import signal
from numba import njit

from filter import RRCFilter
from modulation import normalize_config_modulation_name

from project_logger import get_logger
logger = get_logger(__name__)

def _interp_cubic_py(x, i) -> np.complex64:
    i1 = int(np.floor(i))
    mu = i - i1

    y0 = x[i1 - 1]
    y1 = x[i1]
    y2 = x[i1 + 1]
    y3 = x[i1 + 2]

    a0 = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3
    a1 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
    a2 = -0.5 * y0 + 0.5 * y2
    a3 = y1
    return ((a0 * mu + a1) * mu + a2) * mu + a3


@njit(cache=False, fastmath=True)
def interp_cubic(x, i) -> np.complex64:
    i1 = int(np.floor(i))
    mu = i - i1

    y0 = x[i1 - 1]
    y1 = x[i1]
    y2 = x[i1 + 1]
    y3 = x[i1 + 2]

    a0 = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3
    a1 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
    a2 = -0.5 * y0 + 0.5 * y2
    a3 = y1
    return ((a0 * mu + a1) * mu + a2) * mu + a3


def _gardner_py(
    samples: np.ndarray,
    sps: int = 8,
    Kp: float = 0.01,
    Ki: float = 0.0001,
    gate_min_energy: float = 0.0,
    gate_max_energy: float = 1e30,
    update_start_sample: int = 0,
    update_stop_sample: int = 2147483647,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = 0.0
    omega = float(sps)
    omega_min = 0.5 * float(sps)
    omega_max = 1.5 * float(sps)
    i = sps
    out = np.zeros(len(samples), dtype=np.complex64)
    errors = np.zeros(len(samples), dtype=np.float32)
    mu_trace = np.zeros(len(samples), dtype=np.float32)
    updates = np.zeros(len(samples), dtype=np.float32)
    omega_trace = np.zeros(len(samples), dtype=np.float32)

    j = 0
    while True:
        if j >= out.size:
            break

        center = i + mu
        if center - (sps // 2) - 1 < 0 or center + (sps // 2) + 2 >= len(samples):
            break

        mid = _interp_cubic_py(samples, center)
        early = _interp_cubic_py(samples, center - sps // 2)
        late = _interp_cubic_py(samples, center + sps // 2)

        raw_error = np.real((late - early) * np.conj(mid))
        energy = np.abs(early) ** 2 + np.abs(mid) ** 2 + np.abs(late) ** 2 + 1e-12
        error = raw_error / energy
        in_region = update_start_sample <= center < update_stop_sample
        update_ok = 1.0 if (in_region and gate_min_energy <= energy <= gate_max_energy) else 0.0

        if update_ok > 0.5:
            omega += Ki * error
            if omega < omega_min:
                omega = omega_min
            elif omega > omega_max:
                omega = omega_max
            mu += omega + Kp * error
        else:
            mu += omega

        step = int(np.floor(mu))
        if step < 1:
            step = 1
            mu = 0.0
        else:
            mu -= step
        i += step

        out[j] = mid
        errors[j] = error
        mu_trace[j] = mu
        updates[j] = update_ok
        omega_trace[j] = omega
        j += 1
    return out[:j], errors[:j], mu_trace[:j], updates[:j], omega_trace[:j]


@njit(cache=False, fastmath=True)
def _gardner_njit(
    samples: np.ndarray,
    sps: int = 8,
    Kp: float = 0.01,
    Ki: float = 0.0001,
    gate_min_energy: float = 0.0,
    gate_max_energy: float = 1e30,
    update_start_sample: int = 0,
    update_stop_sample: int = 2147483647,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = 0.0
    omega = float(sps)
    omega_min = 0.5 * float(sps)
    omega_max = 1.5 * float(sps)
    i = sps
    out = np.zeros(len(samples), dtype=np.complex64)
    errors = np.zeros(len(samples), dtype=np.float32)
    mu_trace = np.zeros(len(samples), dtype=np.float32)
    updates = np.zeros(len(samples), dtype=np.float32)
    omega_trace = np.zeros(len(samples), dtype=np.float32)

    j = 0
    while True:
        if j >= out.size:
            break

        center = i + mu
        if center - (sps // 2) - 1 < 0 or center + (sps // 2) + 2 >= len(samples):
            break

        mid = interp_cubic(samples, center)
        early = interp_cubic(samples, center - sps // 2)
        late = interp_cubic(samples, center + sps // 2)

        raw_error = np.real((late - early) * np.conj(mid))
        energy = (
            np.abs(early) * np.abs(early)
            + np.abs(mid) * np.abs(mid)
            + np.abs(late) * np.abs(late)
            + 1e-12
        )
        error = raw_error / energy
        in_region = update_start_sample <= center < update_stop_sample
        update_ok = 1.0 if (in_region and gate_min_energy <= energy <= gate_max_energy) else 0.0

        if update_ok > 0.5:
            omega += Ki * error
            if omega < omega_min:
                omega = omega_min
            elif omega > omega_max:
                omega = omega_max
            mu += omega + Kp * error
        else:
            mu += omega

        step = int(np.floor(mu))
        if step < 1:
            step = 1
            mu = 0.0
        else:
            mu -= step
        i += step

        out[j] = mid
        errors[j] = error
        mu_trace[j] = mu
        updates[j] = update_ok
        omega_trace[j] = omega

        j += 1
    return out[:j], errors[:j], mu_trace[:j], updates[:j], omega_trace[:j]


def _costas_loop_py(
    received_signal: np.ndarray,
    Kp: float,
    Ki: float,
    modulation_order: int,
    update_start_symbol: int = 0,
    update_stop_symbol: int = 2147483647,
    gate_min_power: float = 0.0,
    gate_max_power: float = 1e30,
) -> np.ndarray:
    loop_integral = 0.0
    vco_phase = 0.0
    out = np.zeros_like(received_signal, dtype=np.complex64)

    N = len(received_signal)
    for i in range(N):
        sample = received_signal[i] * np.exp(-1j * vco_phase)

        I = np.real(sample)
        Q = np.imag(sample)

        if modulation_order == 2:
            power = I * I + Q * Q + 1e-12
            error = (np.sign(I) * Q) / power
            error = float(np.clip(error, -0.5, 0.5))
            if (
                update_start_symbol <= i < update_stop_symbol
                and gate_min_power <= power <= gate_max_power
            ):
                loop_integral += Ki * error
                vco_phase = Kp * error + loop_integral
            out[i] = I
        else:
            error = np.sign(I) * Q - np.sign(Q) * I
            power = I * I + Q * Q + 1e-12
            if (
                update_start_symbol <= i < update_stop_symbol
                and gate_min_power <= power <= gate_max_power
            ):
                loop_integral += Ki * error
                vco_phase = Kp * error + loop_integral
            out[i] = sample

    return out

@njit(cache=False, fastmath=True)
def _costas_loop_njit(received_signal: np.ndarray,
                      Kp: float,
                      Ki:float,
                      modulation_order: int,
                      update_start_symbol: int = 0,
                      update_stop_symbol: int = 2147483647,
                      gate_min_power: float = 0.0,
                      gate_max_power: float = 1e30) -> np.ndarray:
    """Costas loop implementation optimized with Numba's JIT compilation for performance."""
    loop_integral = 0.0
    vco_phase = 0.0     # [radians] 
    out = np.zeros_like(received_signal, dtype=np.complex64)

    N = len(received_signal)
    for i in range(N):
        sample = received_signal[i] * np.exp(-1j * vco_phase)  # Mix down the signal by the current phase estimate

        I = np.real(sample)
        Q = np.imag(sample)

        # decision-directed error signal based on the modulation scheme
        if modulation_order == 2:  # BPSK
            power = I * I + Q * Q + 1e-12
            error = (np.sign(I) * Q) / power
            if error > 0.5:
                error = 0.5
            elif error < -0.5:
                error = -0.5

            # loop filter: update frequency and phase estimates
            if (
                update_start_symbol <= i < update_stop_symbol
                and gate_min_power <= power <= gate_max_power
            ):
                loop_integral += Ki * error  # Integrate the error to update frequency estimate
                vco_phase = Kp * error + loop_integral  # Update phase estimate based on proportional and integral

            # store the real part of the corrected sample for output (since BPSK only has information in the I component)
            out[i] = I

        else:  # QPSK
            error = np.sign(I) * Q - np.sign(Q) * I  # For QPSK.    
            power = I * I + Q * Q + 1e-12

            # loop filter: update frequency and phase estimates
            if (
                update_start_symbol <= i < update_stop_symbol
                and gate_min_power <= power <= gate_max_power
            ):
                loop_integral += Ki * error  # Integrate the error to update frequency estimate
                vco_phase = Kp * error + loop_integral  # Update phase estimate based on proportional and integral

            # store the corrected sample for output
            out[i] = sample

    return out



class Synchronizer:
    def __init__(self, config: dict, warmup: bool = True, use_numba: bool = True):
        """ Synchronization class that handles both coarse and fine frequency synchronization, as well as timing synchronization using a Costas loop and Gardner algorithm. Configurable via the provided config dictionary."""
        self.modulation_scheme = normalize_config_modulation_name(config)
        self.sps = int(config['modulation']['samples_per_symbol'])
        self.symbol_rate = int(float(config['modulation']['symbol_rate']))
        self.buffer_size = int(config['receiver']['buffer_size'])
        self.sample_rate = self.sps * self.symbol_rate
        self.nfft = int(config['synchronization']['nfft'])
        self.use_numba = bool(use_numba)

        self.signal_power_threshold_dB = float(config['synchronization']['signal_power_threshold_dB'])
        self.noise_floor_dB = 0.0 # linear scale, to be set after SDR connection
        
        sync_cfg = config["synchronization"]

        self.costas_Kp = float(sync_cfg.get("costas_Kp", 0.001))
        self.costas_Ki = float(sync_cfg.get("costas_Ki", 2e-6))
        self.costas_gate_power_percentile_low = float(
            sync_cfg.get("costas_gate_power_percentile_low", 15.0)
        )
        self.costas_gate_power_percentile_high = float(
            sync_cfg.get("costas_gate_power_percentile_high", 85.0)
        )
        self.gardner_Kp = float(sync_cfg.get("gardner_Kp", 0.002))
        self.gardner_Ki = float(sync_cfg.get("gardner_Ki", 1e-5))
        self.gardner_gate_energy_percentile_low = float(
            sync_cfg.get("gardner_gate_energy_percentile_low", 35.0)
        )
        self.gardner_gate_energy_percentile_high = float(
            sync_cfg.get("gardner_gate_energy_percentile_high", 80.0)
        )
        self.gardner_gate_energy_scale_low = float(
            sync_cfg.get("gardner_gate_energy_scale_low", 3.0)
        )
        self.gardner_gate_energy_scale_high = float(
            sync_cfg.get("gardner_gate_energy_scale_high", 3.0)
        )
        self.gardner_tracking_payload_symbols = int(
            sync_cfg.get("gardner_tracking_payload_symbols", 256)
        )
        self.short_equalizer_enable = bool(
            sync_cfg.get("short_equalizer_enable", True)
        )
        self.short_equalizer_tap_count = int(
            sync_cfg.get("short_equalizer_tap_count", 5)
        )
        if self.short_equalizer_tap_count < 1:
            self.short_equalizer_tap_count = 1
        if self.short_equalizer_tap_count % 2 == 0:
            self.short_equalizer_tap_count += 1
        self.short_equalizer_regularization = float(
            sync_cfg.get("short_equalizer_regularization", 1.0e-3)
        )
        self.short_equalizer_apply_if_improved = bool(
            sync_cfg.get("short_equalizer_apply_if_improved", True)
        )
        self.short_equalizer_train_on_preamble = bool(
            sync_cfg.get("short_equalizer_train_on_preamble", True)
        )
        self.short_equalizer_train_on_header = bool(
            sync_cfg.get("short_equalizer_train_on_header", True)
        )
        self.short_equalizer_min_training_symbols = int(
            sync_cfg.get("short_equalizer_min_training_symbols", 32)
        )

        print(f"Costas loop parameters: Kp={self.costas_Kp:.6f}, Ki={self.costas_Ki:.6f}")
        print(f"Gardner loop parameters: Kp={self.gardner_Kp:.6f}, Ki={self.gardner_Ki:.6f}")
        
        if self.modulation_scheme == 'BPSK':
            self.modulation_order = 2.0    
        elif self.modulation_scheme == 'QPSK':
            self.modulation_order = 4.0
        else:
            raise ValueError(f"Unsupported modulation scheme: {self.modulation_scheme}")
        
        if warmup and self.use_numba:
            # Compile the Numba-optimized functions before the first real call.
            _costas_loop_njit(
                np.zeros(self.buffer_size, dtype=np.complex64),
                self.costas_Kp,
                self.costas_Ki,
                self.modulation_order,
                gate_min_power=0.0,
                gate_max_power=1e30,
            )
            _gardner_njit(
                np.zeros(self.buffer_size, dtype=np.complex64),
                self.sps,
                self.gardner_Kp,
                self.gardner_Ki,
            )

    def set_noise_floor(self, level_dB: float):
        """Set the noise floor in dB for adaptive thresholding."""
        print(f"Setting noise floor to {level_dB:.2f} dB")
        self.noise_floor_dB = level_dB

    def coarse_frequenzy_synchronization(self, received_signal: np.ndarray) -> np.ndarray:
        """Coarse frequency synchronization using FFT-based method.
        Should be applied before timing synchronization.
        """

        raised_signal = received_signal**self.modulation_order # Remove modulation effects by raising to the power of the modulation order

        magnitude = np.fft.fftshift(np.abs(np.fft.fft(raised_signal, n=self.nfft)))  
        freqs = np.fft.fftshift(np.fft.fftfreq(self.nfft, d=1/self.sample_rate))  # Frequency bins corresponding to the FFT output

        estimated_frequenzy_offset = freqs[np.argmax(magnitude)] / self.modulation_order # Divide by modulation order to get the actual frequency offset
        
        signal_power_dB = 10 * np.log10(np.max(magnitude)**2)
        if signal_power_dB < self.noise_floor_dB + self.signal_power_threshold_dB:
            return None

        time_vector = np.arange(len(received_signal)) / self.sample_rate
        
        return received_signal * np.exp(-1j * 2 * np.pi * estimated_frequenzy_offset * time_vector)

    def normalize_matched_filter_output(
        self,
        samples: np.ndarray,
        target_rms: float = 1.0,
    ) -> np.ndarray:
        received = np.asarray(samples).astype(np.complex64, copy=False)
        if received.size == 0:
            return received

        energy = np.abs(received) ** 2
        threshold = float(np.percentile(energy, 65.0))
        selected = energy >= threshold
        if not np.any(selected):
            selected = np.ones(received.size, dtype=bool)

        energy = np.abs(received) ** 2
        threshold = float(np.percentile(energy, 65.0))
        selected = energy >= threshold
        if not np.any(selected):
            selected = np.ones(received.size, dtype=bool)

        rms_before = float(np.sqrt(np.mean(energy[selected]) + 1e-12))
        scale = 1.0 if rms_before <= 1e-12 else float(target_rms) / rms_before
        normalized = (received * scale).astype(np.complex64, copy=False)

        return normalized

    
    def fine_frequenzy_synchronization(
        self,
        received_signal: np.ndarray,
        update_start_symbol: int = 0,
        update_stop_symbol: int | None = None,
    ) -> np.ndarray:
        """Fine frequency synchronization using a costas loop."""
        stop_symbol = len(received_signal) if update_stop_symbol is None else int(update_stop_symbol)
        gate_min_power, gate_max_power, power_reference = self._costas_power_gate(
            received_signal,
            update_start_symbol=int(max(0, update_start_symbol)),
            update_stop_symbol=int(max(0, stop_symbol)),
        )
        if not self.use_numba:
            return _costas_loop_py(
                received_signal,
                self.costas_Kp,
                self.costas_Ki,
                self.modulation_order,
                update_start_symbol=int(max(0, update_start_symbol)),
                update_stop_symbol=int(max(0, stop_symbol)),
                gate_min_power=gate_min_power,
                gate_max_power=gate_max_power,
            )
        return _costas_loop_njit(
            received_signal,
            self.costas_Kp,
            self.costas_Ki,
            self.modulation_order,
            update_start_symbol=int(max(0, update_start_symbol)),
            update_stop_symbol=int(max(0, stop_symbol)),
            gate_min_power=gate_min_power,
            gate_max_power=gate_max_power,
        )

    def _costas_power_gate(
        self,
        received_signal: np.ndarray,
        update_start_symbol: int,
        update_stop_symbol: int,
    ) -> tuple[float, float, float]:
        signal_in = np.asarray(received_signal).astype(np.complex64, copy=False)
        if signal_in.size == 0:
            return 0.0, 1e30, 0.0

        start = int(max(0, update_start_symbol))
        stop = int(min(signal_in.size, max(start, update_stop_symbol)))
        active = signal_in[start:stop]
        if active.size == 0:
            active = signal_in

        power = np.abs(active) ** 2
        power_reference = float(np.median(power)) if power.size else 0.0
        if not np.isfinite(power_reference) or power_reference <= 1e-12:
            return 0.0, 1e30, max(power_reference, 0.0)

        low_pct = float(np.clip(self.costas_gate_power_percentile_low, 0.0, 100.0))
        high_pct = float(np.clip(self.costas_gate_power_percentile_high, low_pct, 100.0))
        gate_min_power = float(np.percentile(power, low_pct))
        gate_max_power = float(np.percentile(power, high_pct))
        gate_min_power = max(0.0, gate_min_power)
        gate_max_power = max(gate_min_power + 1e-12, gate_max_power)
        return float(gate_min_power), float(gate_max_power), float(power_reference)


    def _gardner_gate_parameters(
        self,
        samples: np.ndarray,
        update_start_sample: int = 0,
        update_stop_sample: int | None = None,
    ) -> dict:
        received = np.asarray(samples).astype(np.complex64, copy=False)
        sample_energy = np.abs(received) ** 2

        if sample_energy.size == 0:
            return {
                "gate_min_energy": 0.0,
                "gate_max_energy": 1e30,
                "update_start_sample": int(update_start_sample),
                "update_stop_sample": int(0 if update_stop_sample is None else update_stop_sample),
            }

        low_pct = float(np.clip(self.gardner_gate_energy_percentile_low, 0.0, 100.0))
        high_pct = float(
            np.clip(self.gardner_gate_energy_percentile_high, low_pct, 100.0)
        )
        lo = float(np.percentile(sample_energy, low_pct))
        hi = float(np.percentile(sample_energy, high_pct))
        gate_min_energy = max(1e-12, self.gardner_gate_energy_scale_low * lo)
        gate_max_energy = max(
            gate_min_energy + 1e-12,
            self.gardner_gate_energy_scale_high * hi,
        )
        stop_sample = received.size if update_stop_sample is None else int(update_stop_sample)

        return {
            "gate_min_energy": gate_min_energy,
            "gate_max_energy": gate_max_energy,
            "update_start_sample": int(max(0, update_start_sample)),
            "update_stop_sample": int(max(0, stop_sample)),
        }


    def gardner_timing_synchronization(
        self,
        samples: np.ndarray,
        update_start_sample: int = 0,
        update_stop_sample: int | None = None,
    ) -> np.ndarray:
        """Timing synchronization using the Gardner algorithm."""
        gate = self._gardner_gate_parameters(
            samples,
            update_start_sample=update_start_sample,
            update_stop_sample=update_stop_sample,
        )
        if not self.use_numba:
            return _gardner_py(
                samples,
                self.sps,
                self.gardner_Kp,
                self.gardner_Ki,
                gate["gate_min_energy"],
                gate["gate_max_energy"],
                gate["update_start_sample"],
                gate["update_stop_sample"],
            )[0]
        return _gardner_njit(
            samples,
            self.sps,
            self.gardner_Kp,
            self.gardner_Ki,
            gate["gate_min_energy"],
            gate["gate_max_energy"],
            gate["update_start_sample"],
            gate["update_stop_sample"],
        )[0]


if __name__ == "__main__":
    from yaml import safe_load
    from sdr_plots import StaticSDRPlotter
    from matplotlib.pyplot import show

    try:
        with open("setup/config.yaml", 'r') as f:
            config = safe_load(f)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        exit(1)

    synchronizer = Synchronizer(config)
    plotter = StaticSDRPlotter()
    filter = RRCFilter(config)

    ##########################################
    # Test signal Parameters
    ##########################################
    num_symbols = 256  # Number of symbols in the test signal (excluding preamble)
    frequency_offset = 1000  # [Hz]
    timing_offset = 10.4 # [fraction of symbol period]
    snr_dB = 30 # [dB]

    # Generate QPSK test signal
    test_symbols = np.random.randint(0, 4, num_symbols)  # Random QPSK symbols
    symbol_mapping = {0: 1+1j, 1: -1+1j, 2: -1-1j, 3: 1-1j}  # Gray coding for QPSK
    modulated_signal = np.array([symbol_mapping[symbol] for symbol in test_symbols], dtype=np.complex64)
    upsampled_signal = np.zeros(len(modulated_signal) * synchronizer.sps, dtype=np.complex64)
    upsampled_signal[::synchronizer.sps] = modulated_signal  # Upsample by inserting zeros between symbols
    shaped_signal = filter.apply_filter(upsampled_signal)  # Apply pulse shaping

    # Add frequency offset
    time_vector = np.arange(len(shaped_signal)) / synchronizer.sample_rate
    frequency_offset_signal = shaped_signal * np.exp(1j * 2 * np.pi * frequency_offset * time_vector)

    # Add noise
    signal_power = np.mean(np.abs(frequency_offset_signal)**2)
    noise_power = signal_power / (10**(snr_dB/10))
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(frequency_offset_signal)) + 1j * np.random.randn(len(frequency_offset_signal)))
    received_signal = frequency_offset_signal + noise 

    coarse_corrected_signal = synchronizer.coarse_frequenzy_synchronization(received_signal)

    filtered_signal = filter.apply_filter(coarse_corrected_signal)

    time_adjusted_signal = synchronizer.gardner_timing_synchronization(filtered_signal)

    fine_corrected_signal = synchronizer.fine_frequenzy_synchronization(time_adjusted_signal)


    print(f"len symbols: {len(modulated_signal)} symbols")
    print(f"len upsampled signal: {len(upsampled_signal)} samples")
    print(f"len time synchronized signal: {len(time_adjusted_signal)} samples")

    plotter.plot_constellation(received_signal, title="Constellation Before Synchronization")
    plotter.plot_constellation(coarse_corrected_signal, title="Constellation After Coarse Frequency Synchronization")
    plotter.plot_constellation(time_adjusted_signal, title="Constellation After Timing Synchronization")
    plotter.plot_constellation(fine_corrected_signal, title="Constellation After Synchronization")
    show()
