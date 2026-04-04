import numpy as np
from scipy import signal
from numba import njit
from gold_code import get_gold_code_symbols
from filter import RRCFilter

def _interp_linear_py(x, i) -> np.complex64:
    i0 = int(np.floor(i))
    frac = i - i0
    return x[i0] * (1 - frac) + x[i0 + 1] * frac


@njit(cache=False, fastmath=True)
def interp_linear(x, i) -> np.complex64:
    i0 = int(np.floor(i))
    frac = i - i0
    return x[i0]*(1-frac) + x[i0+1]*frac


def _gardner_py(samples: np.ndarray, sps: int = 8, Kp: float = 0.01, Ki: float = 0.0001) -> tuple[np.ndarray, np.ndarray]:
    mu = 0.0
    omega = 1.0
    i = sps
    out = np.zeros(len(samples) // sps + 10, dtype=np.complex64)
    errors = np.zeros(len(samples) // sps + 10, dtype=np.float32)

    j = 0
    while i < len(samples) - sps:
        mid = _interp_linear_py(samples, i + mu)
        early = _interp_linear_py(samples, i + mu - sps // 2)
        late = _interp_linear_py(samples, i + mu + sps // 2)

        error = np.real((late - early) * np.conj(mid))

        omega += Ki * error
        mu += omega + Kp * error

        step = int(np.floor(mu))
        mu -= step
        i += step

        out[j] = mid
        errors[j] = error
        j += 1
    return out, errors


@njit(cache=False, fastmath=True)
def _gardner_njit(samples: np.ndarray, sps: int = 8, Kp: float = 0.01, Ki: float = 0.0001) -> tuple[np.ndarray, np.ndarray]:
    mu = 0.0
    omega = 1.0
    i = sps
    out = np.zeros(len(samples) // sps + 10, dtype=np.complex64)
    errors = np.zeros(len(samples) // sps + 10, dtype=np.float32)

    j = 0
    while i < len(samples) - sps:

        # Linear interpolation to get the mid, early, and late samples
        i0 = int(np.floor(i))
        fraction = i - i0
        mid = samples[i0] * (1 - fraction) + samples[i0 + 1] * fraction
        early = samples[i0 - sps//2] * (1 - fraction) + samples[i0 - sps//2 + 1] * fraction
        late = samples[i0 + sps//2] * (1 - fraction) + samples[i0 + sps//2 + 1] * fraction

        error = np.real((late - early) * np.conj(mid)) # ca |mid|² 

        omega += Ki * error
        mu += omega + Kp * error

        step = int(np.floor(mu))
        mu -= step
        i += step

        out[j] = mid
        errors[j] = error

        j += 1
    return out, errors


def _costas_loop_py(received_signal: np.ndarray, Kp: float, Ki: float, modulation_order: int) -> np.ndarray:
    loop_integral = 0.0
    vco_phase = 0.0
    out = np.zeros_like(received_signal, dtype=np.complex64)

    N = len(received_signal)
    for i in range(N):
        sample = received_signal[i] * np.exp(-1j * vco_phase)

        I = np.real(sample)
        Q = np.imag(sample)

        if modulation_order == 2:
            error = np.sign(I) * Q
            loop_integral += Ki * error
            vco_phase = Kp * error + loop_integral
            out[i] = I
        else:
            error = np.sign(I) * Q - np.sign(Q) * I
            loop_integral += Ki * error
            vco_phase = Kp * error + loop_integral
            out[i] = sample

    return out


@njit(cache=False, fastmath=True)
def _costas_loop_njit(received_signal: np.ndarray,
                      Kp: float,
                      Ki:float,
                      modulation_order: int) -> np.ndarray:
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
            error = np.sign(I) * Q  # For BPSK, the error can be computed as the product of I and Q

            # loop filter: update frequency and phase estimates
            loop_integral += Ki * error  # Integrate the error to update frequency estimate
            vco_phase = Kp * error + loop_integral  # Update phase estimate based on proportional and integral

            # store the real part of the corrected sample for output (since BPSK only has information in the I component)
            out[i] = I

        else:  # QPSK
            error = np.sign(I) * Q - np.sign(Q) * I  # For QPSK.    

            # loop filter: update frequency and phase estimates
            loop_integral += Ki * error  # Integrate the error to update frequency estimate
            vco_phase = Kp * error + loop_integral  # Update phase estimate based on proportional and integral

            # store the corrected sample for output
            out[i] = sample

    return out



class Synchronizer:
    def __init__(self, config: dict, warmup: bool = True, use_numba: bool = True):
        self.modulation_scheme = config['modulation']['type'].upper().strip()
        self.sps = int(config['modulation']['samples_per_symbol'])
        self.buffer_size = int(config['receiver']['buffer_size'])
        self.sample_rate = self.sps * int(float(config['modulation']['symbol_rate']))
        self.nfft = int(config['synchronization']['nfft'])
        self.use_numba = bool(use_numba)

        self.signal_power_threshold_dB = float(config['synchronization']['signal_power_threshold_dB'])
        self.noise_floor_dB = 0.0 # linear scale, to be set after SDR connection
        
        # Calculate Costas loop parameters.
        loop_bw = float(config['synchronization']['costas_alpha']) * float(config['synchronization']['signal_bw']) / float(config['modulation']['sample_rate'])  # normalized loop bandwidth (as a fraction of the sample rate)
        zeta = float(config['synchronization']['costas_zeta'])
        wn = (4 * zeta * loop_bw) / (zeta + 1/(4*zeta))  # Natural frequency of the loop
        K0 = float(config['synchronization']['costas_K0'])

        #self.costas_Kp = (2 * zeta * wn) /  K0
        #self.costas_Ki = wn**2 / K0

        self.costas_Kp = 0.132
        self.costas_Ki = 0.00932
        self.gardner_Kp = 0.01
        self.gardner_Ki = 0.0001

        print(f"Costas loop parameters: Kp={self.costas_Kp:.6f}, Ki={self.costas_Ki:.6f}")
        
        filter = RRCFilter(config)
        rc_filter = filter.rc_coefficients

        gold_cfg = config["gold_sequence"]
        preamble_sequence = get_gold_code_symbols(
            modulation_type=self.modulation_scheme,
            code_length=int(gold_cfg["code_length"]),
            code_index=int(gold_cfg.get("code_index", 0)),
        )
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

    
    def fine_frequenzy_synchronization(self, received_signal: np.ndarray) -> np.ndarray:
        """Fine frequency synchronization using a costas loop."""
        if not self.use_numba:
            return _costas_loop_py(received_signal, self.costas_Kp, self.costas_Ki, self.modulation_order)
        return _costas_loop_njit(received_signal, self.costas_Kp, self.costas_Ki, self.modulation_order)


    def gardner_timing_synchronization(self, samples: np.ndarray) -> np.ndarray:
        if not self.use_numba:
            return _gardner_py(samples, self.sps, self.gardner_Kp, self.gardner_Ki)[0]
        return _gardner_njit(samples, self.sps, self.gardner_Kp, self.gardner_Ki)[0]
    

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
