import numpy as np
from scipy import signal
from numba import njit
from frame_sequences import (
    build_sync_preamble_symbols,
    build_timing_reference_block,
)
from gold_code import get_gold_code_symbols
from filter import RRCFilter
from modulation import normalize_config_modulation_name

def _interp_linear_py(x, i) -> np.complex64:
    i0 = int(np.floor(i))
    frac = i - i0
    return x[i0] * (1 - frac) + x[i0 + 1] * frac


@njit(cache=False, fastmath=True)
def interp_linear(x, i) -> np.complex64:
    i0 = int(np.floor(i))
    frac = i - i0
    return x[i0]*(1-frac) + x[i0+1]*frac


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


def _costas_loop_trace_py(
    received_signal: np.ndarray,
    Kp: float,
    Ki: float,
    modulation_order: int,
    update_start_symbol: int = 0,
    update_stop_symbol: int = 2147483647,
    gate_min_power: float = 0.0,
    gate_max_power: float = 1e30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    loop_integral = 0.0
    vco_phase = 0.0
    projected = np.zeros_like(received_signal, dtype=np.complex64)
    corrected = np.zeros_like(received_signal, dtype=np.complex64)
    errors = np.zeros(len(received_signal), dtype=np.float32)
    phases = np.zeros(len(received_signal), dtype=np.float32)
    updates = np.zeros(len(received_signal), dtype=np.float32)

    for i in range(len(received_signal)):
        sample = received_signal[i] * np.exp(-1j * vco_phase)
        corrected[i] = sample

        I = np.real(sample)
        Q = np.imag(sample)

        if modulation_order == 2:
            power = I * I + Q * Q + 1e-12
            error = (np.sign(I) * Q) / power
            error = float(np.clip(error, -0.5, 0.5))
            update_ok = 1.0 if (
                update_start_symbol <= i < update_stop_symbol
                and gate_min_power <= power <= gate_max_power
            ) else 0.0
            if update_ok > 0.5:
                loop_integral += Ki * error
                vco_phase = Kp * error + loop_integral
            projected[i] = np.complex64(I + 0j)
        else:
            error = np.sign(I) * Q - np.sign(Q) * I
            power = I * I + Q * Q + 1e-12
            update_ok = 1.0 if (
                update_start_symbol <= i < update_stop_symbol
                and gate_min_power <= power <= gate_max_power
            ) else 0.0
            if update_ok > 0.5:
                loop_integral += Ki * error
                vco_phase = Kp * error + loop_integral
            projected[i] = sample

        errors[i] = error
        phases[i] = vco_phase
        updates[i] = update_ok

    return projected, corrected, errors, phases, updates


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
        self.header_phase_search_offset_samples = int(
            sync_cfg.get("header_phase_search_offset_samples", max(2, self.sps // 2))
        )
        self.timing_header_search_radius_symbols = int(
            sync_cfg.get("timing_header_search_radius_symbols", 8)
        )
        self.timing_header_candidate_count = int(
            sync_cfg.get("timing_header_candidate_count", 5)
        )
        self.gardner_tracking_payload_symbols = int(
            sync_cfg.get("gardner_tracking_payload_symbols", 256)
        )
        self.preamble_repeat_count = int(sync_cfg.get("preamble_repeat_count", 4))
        self.preamble_repeat_count = max(1, self.preamble_repeat_count)
        self.preamble_cfo_enable = bool(sync_cfg.get("preamble_cfo_enable", True))
        self.preamble_cfo_min_score = float(
            sync_cfg.get("preamble_cfo_min_score", 0.45)
        )
        self.preamble_symbol_min_score = float(
            sync_cfg.get("preamble_symbol_min_score", 0.5)
        )
        self.preamble_symbol_search_radius_symbols = int(
            sync_cfg.get("preamble_symbol_search_radius_symbols", 4)
        )
        self.header_fractional_timing_enable = bool(
            sync_cfg.get("header_fractional_timing_enable", True)
        )
        self.header_fractional_timing_max_offset_samples = float(
            sync_cfg.get("header_fractional_timing_max_offset_samples", 1.0)
        )
        self.header_fractional_timing_step_samples = float(
            sync_cfg.get("header_fractional_timing_step_samples", 0.25)
        )
        self.preamble_cfo_max_abs_slope_rad_per_symbol = float(
            sync_cfg.get("preamble_cfo_max_abs_slope_rad_per_symbol", 0.05)
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
        self.header_alignment_candidate_count = int(
            sync_cfg.get("header_alignment_candidate_count", 4)
        )
        self.header_alignment_soft_enable = bool(
            sync_cfg.get("header_alignment_soft_enable", True)
        )
        self.header_alignment_soft_phase_min_score = float(
            sync_cfg.get("header_alignment_soft_phase_min_score", 0.25)
        )

        print(f"Costas loop parameters: Kp={self.costas_Kp:.6f}, Ki={self.costas_Ki:.6f}")
        print(f"Gardner loop parameters: Kp={self.gardner_Kp:.6f}, Ki={self.gardner_Ki:.6f}")
        
        filter = RRCFilter(config)
        rc_filter = filter.rc_coefficients

        gold_cfg = config["gold_sequence"]
        threshold = gold_cfg.get(
            "correlation_scale_factor_threshold",
            gold_cfg.get("correlation_threshold", 0.7),
        )
        self.header_alignment_threshold = float(
            sync_cfg.get("header_alignment_threshold", threshold)
        )
        self.base_gold_symbols = get_gold_code_symbols(
            modulation_type=self.modulation_scheme,
            code_length=int(gold_cfg["code_length"]),
            code_index=int(gold_cfg.get("code_index", 0)),
        ).astype(np.complex64)
        self.base_timing_preamble_symbols = build_timing_reference_block(
            config,
            self.modulation_scheme,
        ).astype(np.complex64, copy=False)
        self.header_repeat_count = max(1, int(gold_cfg.get("header_repeat_count", 1)))
        self.gold_symbols = np.tile(
            self.base_gold_symbols,
            self.header_repeat_count,
        ).astype(np.complex64, copy=False)
        self.sync_preamble_symbols = build_sync_preamble_symbols(
            config,
            self.modulation_scheme,
        ).astype(np.complex64, copy=False)
        preamble_sequence = self.sync_preamble_symbols.copy()
        preamble_sequence_upsampled = np.zeros(len(preamble_sequence) * self.sps, dtype=np.complex64)
        preamble_sequence_upsampled[::self.sps] = preamble_sequence  
        preamble_sequence = signal.convolve(preamble_sequence_upsampled, rc_filter, mode='same', method='fft')  # Apply pulse shaping to the preamble sequence

        preamble_energy = np.sum(np.abs(preamble_sequence)**2)
        self.preamble_sequence = preamble_sequence / np.sqrt(preamble_energy)  

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
    ) -> tuple[np.ndarray, dict]:
        received = np.asarray(samples).astype(np.complex64, copy=False)
        if received.size == 0:
            return received, {
                "rms_before": 0.0,
                "scale": 1.0,
                "selected_samples": 0,
            }

        energy = np.abs(received) ** 2
        threshold = float(np.percentile(energy, 65.0))
        selected = energy >= threshold
        if not np.any(selected):
            selected = np.ones(received.size, dtype=bool)

        rms_before = float(np.sqrt(np.mean(energy[selected]) + 1e-12))
        scale = 1.0 if rms_before <= 1e-12 else float(target_rms) / rms_before
        normalized = (received * scale).astype(np.complex64, copy=False)

        return normalized, {
            "rms_before": rms_before,
            "scale": scale,
            "selected_samples": int(np.count_nonzero(selected)),
        }

    
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

    def fine_frequenzy_synchronization_with_traces(
        self,
        received_signal: np.ndarray,
        update_start_symbol: int = 0,
        update_stop_symbol: int | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Fine frequency synchronization plus loop diagnostics."""
        stop_symbol = len(received_signal) if update_stop_symbol is None else int(update_stop_symbol)
        gate_min_power, gate_max_power, power_reference = self._costas_power_gate(
            received_signal,
            update_start_symbol=int(max(0, update_start_symbol)),
            update_stop_symbol=int(max(0, stop_symbol)),
        )
        projected, corrected, errors, phases, updates = _costas_loop_trace_py(
            received_signal,
            self.costas_Kp,
            self.costas_Ki,
            self.modulation_order,
            update_start_symbol=int(max(0, update_start_symbol)),
            update_stop_symbol=int(max(0, stop_symbol)),
            gate_min_power=gate_min_power,
            gate_max_power=gate_max_power,
        )
        traces = {
            "corrected_signal": corrected,
            "error": errors,
            "phase_estimate": np.unwrap(phases.astype(np.float64)).astype(np.float32),
            "updates": updates,
            "update_start_symbol": int(max(0, update_start_symbol)),
            "update_stop_symbol": int(max(0, stop_symbol)),
            "gate_min_power": float(gate_min_power),
            "gate_max_power": float(gate_max_power),
            "gate_reference_power": float(power_reference),
        }
        return projected, traces

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

    def header_alignment_scores(self, samples: np.ndarray) -> np.ndarray:
        reference = self.preamble_sequence.astype(np.complex64, copy=False)
        received = np.asarray(samples).astype(np.complex64, copy=False)

        if received.size < reference.size:
            return np.array([], dtype=np.float32)

        raw = np.correlate(received, reference, mode="valid")
        ref_energy = float(np.vdot(reference, reference).real)
        rx_power = np.abs(received) ** 2
        window_energy = np.convolve(
            rx_power,
            np.ones(reference.size, dtype=np.float32),
            mode="valid",
        )
        denom = np.sqrt(np.maximum(ref_energy * window_energy, 1e-12))
        return (np.abs(raw) / denom).astype(np.float32, copy=False)

    def align_to_header(
        self,
        samples: np.ndarray,
        pre_roll_samples: int = 0,
    ) -> tuple[np.ndarray, dict]:
        received = np.asarray(samples).astype(np.complex64, copy=False)
        scores = self.header_alignment_scores(samples)
        if scores.size == 0:
            return samples, {
                "detected": False,
                "peak": 0.0,
                "header_start_sample": None,
                "alignment_start_sample": 0,
                "header_index_samples": 0,
            }

        top_k = max(1, min(int(self.header_alignment_candidate_count), int(scores.size)))
        top_indices = np.argsort(scores)[-top_k:][::-1]

        selected_start = int(top_indices[0])
        selected_peak = float(scores[selected_start])
        selected_phase_score = 0.0
        selected_phase = 0
        selected_offset = 0

        for candidate_start in top_indices.tolist():
            phase_info = self.acquire_header_symbol_phase(
                received,
                header_start_sample=int(candidate_start),
            )
            candidate_key = (
                float(phase_info["score"]),
                float(scores[int(candidate_start)]),
                -abs(int(phase_info["header_offset_samples"])),
            )
            selected_key = (
                float(selected_phase_score),
                float(selected_peak),
                -abs(int(selected_offset)),
            )
            if candidate_key > selected_key:
                selected_start = int(candidate_start)
                selected_peak = float(scores[selected_start])
                selected_phase_score = float(phase_info["score"])
                selected_phase = int(phase_info["phase"])
                selected_offset = int(phase_info["header_offset_samples"])

        second_peak = 0.0
        if scores.size > 1:
            second_peak = float(scores[int(np.argsort(scores)[-2])])
        peak_ratio = (
            float(selected_peak / max(second_peak, 1e-12))
            if second_peak > 0.0
            else float("inf")
        )

        detected = bool(selected_peak >= self.header_alignment_threshold)
        soft_selected = bool(
            not detected
            and self.header_alignment_soft_enable
            and selected_phase_score >= self.header_alignment_soft_phase_min_score
        )
        align_applied = bool(detected or soft_selected)
        if not align_applied:
            return samples, {
                "detected": False,
                "peak": selected_peak,
                "phase_score": selected_phase_score,
                "phase": int(selected_phase),
                "peak_ratio": peak_ratio,
                "second_peak": second_peak,
                "soft_selected": False,
                "align_applied": False,
                "header_start_sample": None,
                "alignment_start_sample": 0,
                "header_index_samples": 0,
            }

        alignment_start = max(0, selected_start - max(0, int(pre_roll_samples)))
        aligned = samples[alignment_start:]
        return aligned, {
            "detected": detected,
            "peak": selected_peak,
            "phase_score": selected_phase_score,
            "phase": int(selected_phase),
            "peak_ratio": peak_ratio,
            "second_peak": second_peak,
            "soft_selected": bool(soft_selected),
            "align_applied": True,
            "header_start_sample": int(selected_start),
            "alignment_start_sample": int(alignment_start),
            "header_index_samples": int(selected_start - alignment_start),
        }

    def header_phase_scores(
        self,
        samples: np.ndarray,
        header_start_sample: int = 0,
    ) -> np.ndarray:
        reference = self.sync_preamble_symbols.astype(np.complex64, copy=False)
        received = np.asarray(samples).astype(np.complex64, copy=False)
        start_base = max(0, int(header_start_sample))
        ref_energy = float(np.vdot(reference, reference).real)

        scores = np.zeros(self.sps, dtype=np.float32)
        for phase in range(self.sps):
            start = start_base + phase
            stop = start + reference.size * self.sps
            if stop > received.size:
                continue

            header_symbols = received[start:stop:self.sps]
            if header_symbols.size != reference.size:
                continue

            raw = np.vdot(reference, header_symbols)
            rx_energy = float(np.vdot(header_symbols, header_symbols).real)
            denom = np.sqrt(max(ref_energy * rx_energy, 1e-12))
            scores[phase] = float(np.abs(raw) / denom)
        return scores

    def acquire_header_symbol_phase(
        self,
        samples: np.ndarray,
        header_start_sample: int = 0,
    ) -> dict:
        received = np.asarray(samples).astype(np.complex64, copy=False)
        header_start = int(max(0, header_start_sample))
        search_span = max(0, int(self.header_phase_search_offset_samples))

        best_offset = 0
        best_phase = 0
        best_score = 0.0
        best_scores: np.ndarray | None = None

        for offset in range(-search_span, search_span + 1):
            candidate_start = header_start + offset
            if candidate_start < 0 or candidate_start >= received.size:
                continue
            scores = self.header_phase_scores(received, header_start_sample=candidate_start)
            if scores.size == 0:
                continue
            phase = int(np.argmax(scores))
            score = float(scores[phase])
            if score > best_score:
                best_score = score
                best_phase = phase
                best_offset = offset
                best_scores = scores

        if best_scores is None:
            return {
                "phase": 0,
                "score": 0.0,
                "scores": np.zeros(self.sps, dtype=np.float32),
                "header_start_sample": int(header_start),
                "header_offset_samples": 0,
                "refined_header_start_sample": int(header_start),
            }
        return {
            "phase": best_phase,
            "score": float(best_score),
            "scores": best_scores,
            "header_start_sample": int(header_start),
            "header_offset_samples": int(best_offset),
            "refined_header_start_sample": int(header_start + best_offset),
        }

    def acquire_header_timing_window(
        self,
        samples: np.ndarray,
        header_start_sample: int,
        payload_symbol_count: int,
        pre_symbols: int = 2,
        post_symbols: int = 16,
    ) -> tuple[np.ndarray, dict]:
        received = np.asarray(samples).astype(np.complex64, copy=False)
        header_start = max(0, int(header_start_sample))
        payload_symbols = max(0, int(payload_symbol_count))
        pre_symbols = max(0, int(pre_symbols))
        post_symbols = max(0, int(post_symbols))

        phase_info = self.acquire_header_symbol_phase(
            received,
            header_start_sample=header_start,
        )
        phase = int(phase_info["phase"])
        refined_header_start = int(phase_info["refined_header_start_sample"])

        raw_window_start = max(0, refined_header_start - pre_symbols * self.sps)
        preamble_symbols = int(self.sync_preamble_symbols.size)
        gold_header_symbols = int(self.gold_symbols.size)
        requested_symbols = (
            pre_symbols + preamble_symbols + gold_header_symbols + payload_symbols + post_symbols
        )

        aligned_window_start = min(received.size, raw_window_start + phase)
        aligned_window_stop = min(
            received.size,
            aligned_window_start + requested_symbols * self.sps + 2 * self.sps,
        )
        timing_window = received[aligned_window_start:aligned_window_stop]

        header_sample_in_window = max(
            0.0,
            float(refined_header_start - aligned_window_start),
        )
        expected_preamble_symbol_index = int(
            max(
                0.0,
                round((header_sample_in_window - float(self.sps)) / float(self.sps)),
            )
        )
        expected_header_symbol_index = int(expected_preamble_symbol_index + preamble_symbols)
        available_symbols = int(max(0, timing_window.size // self.sps))
        active_symbol_count = int(preamble_symbols) + int(gold_header_symbols) + payload_symbols
        tracking_payload_symbols = int(
            min(
                max(0, self.gardner_tracking_payload_symbols),
                payload_symbols,
            )
        )
        tracking_symbol_count = int(preamble_symbols) + int(gold_header_symbols) + tracking_payload_symbols
        update_start_sample = int(expected_preamble_symbol_index * self.sps)
        update_stop_sample = int(
            min(
                timing_window.size,
                update_start_sample + tracking_symbol_count * self.sps,
            )
        )

        metadata = {
            "phase": phase,
            "score": float(phase_info["score"]),
            "scores": phase_info["scores"],
            "header_start_sample": header_start,
            "header_offset_samples": int(phase_info["header_offset_samples"]),
            "refined_header_start_sample": int(refined_header_start),
            "raw_window_start_sample": int(raw_window_start),
            "aligned_window_start_sample": int(aligned_window_start),
            "aligned_window_stop_sample": int(aligned_window_stop),
            "window_samples": int(timing_window.size),
            "requested_symbols": int(requested_symbols),
            "available_symbols": int(available_symbols),
            "header_sample_in_window": float(header_sample_in_window),
            "expected_preamble_symbol_index": int(expected_preamble_symbol_index),
            "expected_header_symbol_index": int(expected_header_symbol_index),
            "sync_preamble_symbols": int(preamble_symbols),
            "gold_header_symbols": int(gold_header_symbols),
            "active_symbol_count": int(active_symbol_count),
            "tracking_payload_symbols": int(tracking_payload_symbols),
            "tracking_symbol_count": int(tracking_symbol_count),
            "update_start_sample": int(update_start_sample),
            "update_stop_sample": int(update_stop_sample),
        }
        return timing_window, metadata

    def estimate_header_carrier_phase(
        self,
        symbol_stream: np.ndarray,
        header_symbol_index: int = 0,
    ) -> dict:
        received = np.asarray(symbol_stream).astype(np.complex64, copy=False)
        start = max(0, int(header_symbol_index))
        stop = start + int(self.gold_symbols.size)

        if stop > received.size or start >= received.size:
            return {
                "phase_rad": 0.0,
                "score": 0.0,
                "header_symbol_index": start,
                "detected_symbols": 0,
            }

        header_symbols = received[start:stop]
        reference = self.gold_symbols[: header_symbols.size].astype(np.complex64, copy=False)
        correlation = np.vdot(reference, header_symbols)
        ref_energy = float(np.vdot(reference, reference).real)
        rx_energy = float(np.vdot(header_symbols, header_symbols).real)
        denom = np.sqrt(max(ref_energy * rx_energy, 1e-12))
        phase_rad = float(np.angle(correlation))

        return {
            "phase_rad": phase_rad,
            "score": float(np.abs(correlation) / denom),
            "header_symbol_index": start,
            "detected_symbols": int(header_symbols.size),
        }

    def estimate_header_carrier_state(
        self,
        symbol_stream: np.ndarray,
        header_symbol_index: int = 0,
    ) -> dict:
        received = np.asarray(symbol_stream).astype(np.complex64, copy=False)
        start = max(0, int(header_symbol_index))
        stop = start + int(self.gold_symbols.size)

        if stop > received.size or start >= received.size:
            return {
                "phase_rad": 0.0,
                "phase_slope_rad_per_symbol": 0.0,
                "frequency_hz": 0.0,
                "score": 0.0,
                "header_symbol_index": start,
                "detected_symbols": 0,
            }

        header_symbols = received[start:stop]
        reference = self.gold_symbols[: header_symbols.size].astype(np.complex64, copy=False)
        aligned = header_symbols * np.conj(reference)
        phase_trace = np.unwrap(np.angle(aligned.astype(np.complex128)))
        symbol_index = np.arange(header_symbols.size, dtype=np.float64)

        if phase_trace.size >= 2:
            slope, intercept = np.polyfit(symbol_index, phase_trace, 1)
        else:
            slope = 0.0
            intercept = float(phase_trace[0]) if phase_trace.size else 0.0

        residual = aligned * np.exp(-1j * (intercept + slope * symbol_index))
        correlation = np.sum(residual)
        ref_energy = float(np.vdot(reference, reference).real)
        rx_energy = float(np.vdot(header_symbols, header_symbols).real)
        denom = np.sqrt(max(ref_energy * rx_energy, 1e-12))

        return {
            "phase_rad": float(intercept),
            "phase_slope_rad_per_symbol": float(slope),
            "frequency_hz": float(slope * self.symbol_rate / (2.0 * np.pi)),
            "score": float(np.abs(correlation) / denom),
            "header_symbol_index": start,
            "detected_symbols": int(header_symbols.size),
        }

    def apply_carrier_phase_correction(
        self,
        symbol_stream: np.ndarray,
        phase_rad: float,
    ) -> np.ndarray:
        received = np.asarray(symbol_stream).astype(np.complex64, copy=False)
        return (received * np.exp(-1j * float(phase_rad))).astype(np.complex64, copy=False)

    def apply_carrier_phase_and_frequency_correction(
        self,
        symbol_stream: np.ndarray,
        phase_rad: float,
        phase_slope_rad_per_symbol: float = 0.0,
        reference_symbol_index: int = 0,
    ) -> np.ndarray:
        received = np.asarray(symbol_stream).astype(np.complex64, copy=False)
        if received.size == 0:
            return received

        symbol_index = np.arange(received.size, dtype=np.float64) - float(reference_symbol_index)
        correction_phase = float(phase_rad) + float(phase_slope_rad_per_symbol) * symbol_index
        return (received * np.exp(-1j * correction_phase)).astype(np.complex64, copy=False)

    def estimate_repeated_preamble_cfo(
        self,
        samples: np.ndarray,
        preamble_symbol_index: int = 0,
    ) -> dict:
        received = np.asarray(samples).astype(np.complex64, copy=False)
        symbol_stream = received[:: self.sps]

        repeat_length = int(self.base_timing_preamble_symbols.size)
        repeat_count = int(self.preamble_repeat_count)
        start = max(0, int(preamble_symbol_index))
        stop = start + repeat_length * repeat_count

        if (
            not self.preamble_cfo_enable
            or repeat_count < 2
            or repeat_length <= 0
            or stop > symbol_stream.size
        ):
            return {
                "phase_slope_rad_per_symbol": 0.0,
                "applied_phase_slope_rad_per_symbol": 0.0,
                "frequency_hz": 0.0,
                "applied_frequency_hz": 0.0,
                "score": 0.0,
                "repeat_count": int(repeat_count),
                "repeat_length_symbols": int(repeat_length),
                "preamble_symbol_index": int(start),
                "enabled": bool(self.preamble_cfo_enable),
                "reason": "disabled_or_insufficient",
            }

        preamble_symbols = symbol_stream[start:stop]
        repeated = preamble_symbols.reshape(repeat_count, repeat_length)
        correlations = np.array(
            [np.vdot(repeated[idx], repeated[idx + 1]) for idx in range(repeat_count - 1)],
            dtype=np.complex128,
        )

        if correlations.size == 0:
            estimated_slope = 0.0
            score = 0.0
        else:
            phase_diffs = np.unwrap(np.angle(correlations))
            estimated_slope = float(np.mean(phase_diffs) / max(1, repeat_length))
            repeated_power = np.mean(np.abs(repeated) ** 2, axis=1)
            power_scale = float(np.sqrt(np.maximum(np.mean(repeated_power[:-1] * repeated_power[1:]), 1e-12)))
            score = float(np.mean(np.abs(correlations)) / max(power_scale * repeat_length, 1e-12))

        if float(score) < float(self.preamble_cfo_min_score):
            applied_slope = 0.0
            reason = "low_score"
        else:
            max_abs = float(max(0.0, self.preamble_cfo_max_abs_slope_rad_per_symbol))
            if max_abs > 0.0:
                applied_slope = float(np.clip(estimated_slope, -max_abs, max_abs))
                reason = "clamped" if not np.isclose(applied_slope, estimated_slope) else "applied"
            else:
                applied_slope = float(estimated_slope)
                reason = "applied"

        return {
            "phase_slope_rad_per_symbol": float(estimated_slope),
            "applied_phase_slope_rad_per_symbol": float(applied_slope),
            "frequency_hz": float(estimated_slope * self.symbol_rate / (2.0 * np.pi)),
            "applied_frequency_hz": float(applied_slope * self.symbol_rate / (2.0 * np.pi)),
            "score": float(score),
            "repeat_count": int(repeat_count),
            "repeat_length_symbols": int(repeat_length),
            "preamble_symbol_index": int(start),
            "enabled": bool(self.preamble_cfo_enable),
            "reason": reason,
        }

    def detect_repeated_preamble_symbol_index(
        self,
        symbol_stream: np.ndarray,
        expected_preamble_symbol_index: int = 0,
        search_radius_symbols: int | None = None,
    ) -> dict:
        received = np.asarray(symbol_stream).astype(np.complex64, copy=False)
        reference = self.sync_preamble_symbols.astype(np.complex64, copy=False)
        reference_length = int(reference.size)
        expected_index = max(0, int(expected_preamble_symbol_index))

        if received.size < reference_length or reference_length <= 0:
            return {
                "expected_index": int(expected_index),
                "best_index": None,
                "slip_symbols": 0,
                "score": 0.0,
                "phase_rad": 0.0,
                "peak_ratio": 0.0,
                "second_score": 0.0,
                "applied": False,
                "reason": "insufficient_symbols",
            }

        radius = (
            self.preamble_symbol_search_radius_symbols
            if search_radius_symbols is None
            else int(search_radius_symbols)
        )
        radius = max(0, int(radius))
        max_start = int(received.size - reference_length)
        start = max(0, min(expected_index - radius, max_start))
        stop = max(0, min(expected_index + radius, max_start))

        ref_energy = float(np.vdot(reference, reference).real)
        best_index = None
        best_score = -1.0
        best_phase = 0.0
        second_score = -1.0

        for candidate_index in range(start, stop + 1):
            observed = received[candidate_index : candidate_index + reference_length]
            if observed.size != reference_length:
                continue

            correlation = np.vdot(reference, observed)
            rx_energy = float(np.vdot(observed, observed).real)
            denom = np.sqrt(max(ref_energy * rx_energy, 1e-12))
            score = float(np.abs(correlation) / denom)

            if score > best_score:
                second_score = best_score
                best_score = score
                best_index = int(candidate_index)
                best_phase = float(np.angle(correlation))
            elif score > second_score:
                second_score = score

        if best_index is None:
            return {
                "expected_index": int(expected_index),
                "best_index": None,
                "slip_symbols": 0,
                "score": 0.0,
                "phase_rad": 0.0,
                "peak_ratio": 0.0,
                "second_score": 0.0,
                "applied": False,
                "reason": "no_candidate",
            }

        peak_ratio = (
            float(best_score / max(second_score, 1e-12))
            if second_score > 0.0
            else float("inf")
        )
        applied = bool(best_score >= self.preamble_symbol_min_score)
        return {
            "expected_index": int(expected_index),
            "best_index": int(best_index),
            "slip_symbols": int(best_index - expected_index),
            "score": float(best_score),
            "phase_rad": float(best_phase),
            "peak_ratio": float(peak_ratio),
            "second_score": float(max(second_score, 0.0)),
            "applied": bool(applied),
            "reason": "applied" if applied else "low_score",
        }

    def apply_symbol_rate_phase_correction_to_samples(
        self,
        samples: np.ndarray,
        phase_slope_rad_per_symbol: float = 0.0,
        reference_symbol_index: int = 0,
    ) -> np.ndarray:
        received = np.asarray(samples).astype(np.complex64, copy=False)
        if received.size == 0:
            return received

        symbol_index = (
            np.arange(received.size, dtype=np.float64) / float(self.sps)
            - float(reference_symbol_index)
        )
        correction_phase = float(phase_slope_rad_per_symbol) * symbol_index
        return (received * np.exp(-1j * correction_phase)).astype(np.complex64, copy=False)

    def _short_equalizer_training_symbols(self) -> np.ndarray:
        parts: list[np.ndarray] = []
        if self.short_equalizer_train_on_preamble:
            parts.append(self.sync_preamble_symbols.astype(np.complex64, copy=False))
        if self.short_equalizer_train_on_header:
            parts.append(self.gold_symbols.astype(np.complex64, copy=False))
        if not parts:
            return np.array([], dtype=np.complex64)
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts).astype(np.complex64, copy=False)

    def apply_symbol_equalizer(
        self,
        symbol_stream: np.ndarray,
        taps: np.ndarray,
    ) -> np.ndarray:
        received = np.asarray(symbol_stream).astype(np.complex64, copy=False)
        coefficients = np.asarray(taps).astype(np.complex128, copy=False)
        if received.size == 0 or coefficients.size == 0:
            return received

        half_span = int(coefficients.size // 2)
        padded = np.pad(
            received.astype(np.complex128, copy=False),
            (half_span, half_span),
            mode="constant",
        )
        equalized = np.empty(received.size, dtype=np.complex128)
        for index in range(received.size):
            equalized[index] = np.dot(
                padded[index : index + coefficients.size],
                coefficients,
            )
        return equalized.astype(np.complex64, copy=False)

    def train_and_apply_symbol_equalizer(
        self,
        symbol_stream: np.ndarray,
        training_symbol_index: int = 0,
    ) -> tuple[np.ndarray, dict]:
        received = np.asarray(symbol_stream).astype(np.complex64, copy=False)
        identity_taps = np.array([1.0 + 0.0j], dtype=np.complex64)
        default_state = {
            "enabled": bool(self.short_equalizer_enable),
            "applied": False,
            "reason": "disabled",
            "tap_count": int(self.short_equalizer_tap_count),
            "regularization": float(self.short_equalizer_regularization),
            "training_symbol_index": int(training_symbol_index),
            "training_symbol_count": 0,
            "valid_training_symbols": 0,
            "before_error": 0.0,
            "after_error": 0.0,
            "before_mse": 0.0,
            "after_mse": 0.0,
            "rms_scale": 1.0,
            "taps": identity_taps,
        }

        if not self.short_equalizer_enable:
            return received, default_state

        training_symbols = self._short_equalizer_training_symbols()
        if training_symbols.size == 0:
            state = dict(default_state)
            state["reason"] = "no_training_symbols"
            return received, state

        start = int(training_symbol_index)
        tap_count = int(self.short_equalizer_tap_count)
        half_span = int(tap_count // 2)
        indices = []
        targets = []
        for offset, symbol in enumerate(training_symbols.tolist()):
            symbol_index = start + int(offset)
            if 0 <= symbol_index < received.size:
                indices.append(symbol_index)
                targets.append(symbol)

        valid_training_symbols = int(len(indices))
        if valid_training_symbols < max(int(self.short_equalizer_min_training_symbols), tap_count):
            state = dict(default_state)
            state.update(
                {
                    "reason": "insufficient_training",
                    "training_symbol_count": int(training_symbols.size),
                    "valid_training_symbols": valid_training_symbols,
                }
            )
            return received, state

        padded = np.pad(
            received.astype(np.complex128, copy=False),
            (half_span, half_span),
            mode="constant",
        )
        X = np.empty((valid_training_symbols, tap_count), dtype=np.complex128)
        for row, symbol_index in enumerate(indices):
            X[row, :] = padded[symbol_index : symbol_index + tap_count]

        d = np.asarray(targets, dtype=np.complex128)
        observed = received[np.asarray(indices, dtype=np.int64)].astype(np.complex128, copy=False)
        before_error = float(np.mean(np.abs(observed - d)))
        before_mse = float(np.mean(np.abs(observed - d) ** 2))

        regularization = max(0.0, float(self.short_equalizer_regularization))
        lhs = X.conj().T @ X
        if regularization > 0.0:
            lhs = lhs + regularization * np.eye(tap_count, dtype=np.complex128)
        rhs = X.conj().T @ d

        try:
            taps = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            taps = np.linalg.pinv(lhs) @ rhs

        trained = X @ taps
        input_rms = float(np.sqrt(max(np.mean(np.abs(observed) ** 2), 1e-12)))
        output_rms = float(np.sqrt(max(np.mean(np.abs(trained) ** 2), 1e-12)))
        rms_scale = 1.0 if output_rms <= 1e-12 else input_rms / output_rms
        taps = taps * rms_scale
        trained = trained * rms_scale

        after_error = float(np.mean(np.abs(trained - d)))
        after_mse = float(np.mean(np.abs(trained - d) ** 2))
        improved = bool(after_mse + 1e-12 < before_mse)

        if self.short_equalizer_apply_if_improved and not improved:
            state = dict(default_state)
            state.update(
                {
                    "reason": "not_improved",
                    "training_symbol_count": int(training_symbols.size),
                    "valid_training_symbols": valid_training_symbols,
                    "before_error": before_error,
                    "after_error": after_error,
                    "before_mse": before_mse,
                    "after_mse": after_mse,
                    "rms_scale": float(rms_scale),
                    "taps": taps.astype(np.complex64, copy=False),
                }
            )
            return received, state

        equalized = self.apply_symbol_equalizer(received, taps)
        state = dict(default_state)
        state.update(
            {
                "applied": True,
                "reason": "applied",
                "training_symbol_count": int(training_symbols.size),
                "valid_training_symbols": valid_training_symbols,
                "before_error": before_error,
                "after_error": after_error,
                "before_mse": before_mse,
                "after_mse": after_mse,
                "rms_scale": float(rms_scale),
                "taps": taps.astype(np.complex64, copy=False),
            }
        )
        return equalized, state

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

    def gardner_timing_synchronization_with_traces(
        self,
        samples: np.ndarray,
        update_start_sample: int = 0,
        update_stop_sample: int | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Timing synchronization plus Gardner loop diagnostics."""
        gate = self._gardner_gate_parameters(
            samples,
            update_start_sample=update_start_sample,
            update_stop_sample=update_stop_sample,
        )
        out, errors, mu_trace, updates, omega_trace = _gardner_py(
            samples,
            self.sps,
            self.gardner_Kp,
            self.gardner_Ki,
            gate["gate_min_energy"],
            gate["gate_max_energy"],
            gate["update_start_sample"],
            gate["update_stop_sample"],
        )
        traces = {
            "error": errors,
            "mu": mu_trace,
            "updates": updates,
            "omega": omega_trace,
            "gate_min_energy": float(gate["gate_min_energy"]),
            "gate_max_energy": float(gate["gate_max_energy"]),
            "update_start_sample": int(gate["update_start_sample"]),
            "update_stop_sample": int(gate["update_stop_sample"]),
        }
        return out, traces
    

    def time_synchronization(self, samples: np.ndarray) -> int:
        """ ML timing synchronization algorithm. Searches for the known pattern in the received signal and estimates the timing offset."""
        from matplotlib import pyplot as plt

        correlation = signal.convolve(samples, self.preamble_sequence, mode='full', method='fft')

        abs_correlation = np.abs(correlation)  # Normalize correlation to get a value between 0 and 1

        max_value = np.max(abs_correlation)

        plt.figure(figsize=(10, 4))
        plt.stem(abs_correlation)
        plt.title("Correlation with Gold Code Preamble")
        plt.xlabel("Sample Index")
        plt.ylabel("Absolute Correlation")
        plt.grid()

        return np.argmax(abs_correlation)  # Return the index of the maximum correlation as the estimated timing offset


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
