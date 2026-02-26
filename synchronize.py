import numpy as np
from scipy import signal
from numba import njit

@njit(cache=True, fastmath=True)
def _time_sync_njit(samples: np.ndarray, 
                    sps: int, Kp: float, 
                    Ki: float) -> np.ndarray:
        """
        plain Python implementation of the Mueller and Müller timing synchronization algorithm, 
        optimized with Numba's JIT compilation for performance.

        Args:
            samples: Input signal array after coarse frequency synchronization.
            sps: Samples per symbol (oversampling factor).
            Kp: Proportional gain for the timing error correction.
            Ki: Integral gain for the timing error correction.
        Returns:
            Array of samples resampled at the symbol rate, with timing errors corrected.
        """
        mu = 0.0  # Fractional interval between samples, initialized to 0 (start at the first sample)
        omega = sps  # Initial estimate of rate of change of mu
        i = 1  # need margin for cubic interpolation
        N = len(samples) - 2

        prev_sample = np.complex64(0.0)
        prev_decision = np.complex64(0.0)
        out = [prev_sample]

        while i < N:
            xn1 = samples[i - 1]
            x0  = samples[i]
            x1  = samples[i + 1]
            x2  = samples[i + 2]

            # Cubic interpolation coefficients (Farrow structure)
            # Replaces np.dot(h, xvec) with explicit arithmetic
            v0 =  x0                                          # h0 = [0, 1, 0, 0]
            v1 = -xn1/3 - x0/2 + x1   - x2/6                # h1 = [-1/3, -0.5, 1, -1/6]
            v2 =  xn1/2 - x0   + x1/2                        # h2 = [0.5, -1, 0.5, 0]
            v3 = -xn1/6 + x0/2 - x1/2 + x2/6                # h3 = [-1/6, 0.5, -0.5, 1/6]

            current_sample = np.complex64(
                v0 + mu * (v1 + mu * (v2 + mu * v3))
            )

            current_decision = (
                  np.sign(current_sample.real)  +
                  np.sign(current_sample.imag)*1j
            )

            # Mueller & Müller timing error
            error = np.real(
                prev_decision * current_sample -
                current_decision * prev_sample
            )

            omega += Ki * error

            # Limit omega to prevent it from diverging too much
            if omega > sps + 0.5:
                omega = sps + 0.5
            elif omega < sps - 0.5:
                omega = sps - 0.5

            mu += omega + Kp * error

            prev_sample = current_sample
            prev_decision = current_decision

            out.append(current_sample)

            step = int(np.floor(mu))
            i += step
            mu -= step  # Remove the integer part from mu to keep it in the range [0, 1)

        return np.array(out)

@njit(cache=True, fastmath=True)
def _costas_loop_njit(received_signal: np.ndarray,
                      alpha: float,
                      beta:float,
                      modulation_order: int) -> np.ndarray:
    """Costas loop implementation optimized with Numba's JIT compilation for performance."""
    phase = 0.0     # [radians] 
    freq = 0.0      # [radians/sample] 
    out = np.zeros_like(received_signal, dtype=np.complex64)

    N = len(received_signal)
    for i in range(N):
        out[i] = received_signal[i] * np.exp(-1j * phase)
        sample = out[i]

        if modulation_order == 2:   # BPSK
            error = sample.real * sample.imag
        else:                       # QPSK
            error = np.sign(sample.real) * sample.imag - np.sign(sample.imag) * sample.real
        
        freq += beta * error
        phase += freq + alpha * error

        # Keep phase in the range [0, 2*pi) to prevent numerical issues
        phase = phase % (2 * np.pi)
    return out



class Synchronizer:
    def __init__(self, config: dict):
        self.modulation_scheme = config['modulation']['type'].upper().strip()
        self.sps = int(config['modulation']['samples_per_symbol'])
        self.buffer_size = int(config['receiver']['buffer_size'])
        self.sample_rate = self.sps * int(float(config['modulation']['symbol_rate']))
        self.nfft = int(config['synchronization']['nfft'])
        
        self.mm_Kp = float(config['synchronization']['mm_Kp'])
        self.mm_Ki = float(config['synchronization']['mm_Ki'])

        self.costas_alpha = float(config['synchronization']['costas_alpha'])
        self.costas_beta = float(config['synchronization']['costas_beta'])  
        
        
        if self.modulation_scheme == 'BPSK':
            self.modulation_order = 2.0    
        elif self.modulation_scheme == 'QPSK':
            self.modulation_order = 4.0
        else:
            raise ValueError(f"Unsupported modulation scheme: {self.modulation_scheme}")
        
        # compile the Numba-optimized functions with the specified parameters
        _time_sync_njit(np.zeros(self.buffer_size, dtype=np.complex64), self.sps, self.mm_Kp, self.mm_Ki)
        _costas_loop_njit(np.zeros(self.buffer_size, dtype=np.complex64), self.costas_alpha, self.costas_beta, self.modulation_order)


    def coarse_frequenzy_synchronization(self, received_signal: np.ndarray) -> np.ndarray:
        """Coarse frequency synchronization using FFT-based method.
        Should be applied before timing synchronization.
        """

        raised_signal = received_signal**self.modulation_order # Remove modulation effects by raising to the power of the modulation order

        magnitude = np.fft.fftshift(np.abs(np.fft.fft(raised_signal, n=self.nfft)))  
        freqs = np.fft.fftshift(np.fft.fftfreq(self.nfft, d=1/self.sample_rate))  # Frequency bins corresponding to the FFT output

        estimated_frequenzy_offset = freqs[np.argmax(magnitude)] / self.modulation_order # Divide by modulation order to get the actual frequency offset
        
        time_vector = np.arange(len(received_signal)) / self.sample_rate
        print(f"Estimated frequency offset: {estimated_frequenzy_offset:.2f} Hz")
        return received_signal * np.exp(-1j * 2 * np.pi * estimated_frequenzy_offset * time_vector)

    
    def fine_frequenzy_synchronization(self, received_signal: np.ndarray) -> np.ndarray:
        """Fine frequency synchronization using a costas loop."""
        return _costas_loop_njit(received_signal, self.costas_alpha, self.costas_beta, self.modulation_order)

    # Source: https://wirelesspi.com/mueller-and-muller-timing-synchronization-algorithm/ 
    def time_synchronization(self, samples: np.ndarray) -> np.ndarray:
        """Mueller and Müller timing synchronization algorithm.
        Should be applied after coarse frequency synchronization.
        Be aware that this algorithm decimates the signal by a factor of sps.
        
        Args:
            samples: Input signal array after coarse frequency synchronization.
        Returns:
            Array of samples resampled at the symbol rate, with timing errors corrected.
        """
        return _time_sync_njit(samples, self.sps, self.mm_Kp, self.mm_Ki)

            


if __name__ == "__main__":
    from yaml import safe_load
    from matplotlib.pyplot import show
    from sdr_plots import StaticSDRPlotter
    from filter import RRCFilter

    try:
        with open("setup/config.yaml", 'r') as f:
            config = safe_load(f)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        exit(1)

    synchronizer = Synchronizer(config)
    plotter = StaticSDRPlotter()
    rrc_filter = RRCFilter(config).coefficients

    ##########################################
    # Test signal Parameters
    ##########################################
    num_symbols = 1000
    frequency_offset = 100  # [Hz]
    timing_offset = 0.4  # [fraction of symbol period]

    sps = synchronizer.sps
    sample_rate = synchronizer.sample_rate
    test_signal = np.zeros(num_symbols*sps, dtype=np.complex64)
    time_vector = np.arange(len(test_signal)) / sample_rate

    modulation_scheme = config['modulation']['type'].upper().strip()
    if modulation_scheme == 'BPSK':
        print("Using BPSK modulation")
        test_signal[::sps] = 2*np.random.randint(0, 2, num_symbols) - 1
    elif modulation_scheme == 'QPSK':
        print("Using QPSK modulation")
        test_signal[::sps] = (2*np.random.randint(0, 2, num_symbols) - 1) + 1j*(2*np.random.randint(0, 2, num_symbols) - 1)
    else:
        print(f"Unsupported modulation scheme: {modulation_scheme}")
        exit(1)

    # TX processing: pulse shaping, add frequency and timing offset
    test_signal_pulse_shaped = np.convolve (test_signal, rrc_filter, mode='same') #  Apply pulse shaping
    test_signal_freq_shift = test_signal_pulse_shaped * np.exp(1j * 2*np.pi * frequency_offset * time_vector)  # Add frequency offset
    test_signal_time_shift = np.roll(test_signal_freq_shift, int(timing_offset * sps))  # Add timing offset
    test_singal_awgn = test_signal_time_shift + 0.01*(np.random.randn(len(test_signal)) + 1j*np.random.randn(len(test_signal)))  # Additive white Gaussian noise


    # RX processing: coarse frequency synchronization, matched filtering, fine frequency synchronization, timing synchronization
    test_signal_coarse_freq_adjustd = synchronizer.coarse_frequenzy_synchronization(test_singal_awgn)  
    test_signal_matched_filtered = np.convolve(test_signal_coarse_freq_adjustd, rrc_filter, mode='same')  # Matched filtering after coarse frequency synchronization    
    test_signal_time_adjusted = synchronizer.time_synchronization(test_signal_matched_filtered)
    test_signal_fine_freq_adjusted = synchronizer.fine_frequenzy_synchronization(test_signal_time_adjusted)

    plotter.plot_constellation(
        test_signal_coarse_freq_adjustd,
        title="Constellation After Coarse Frequency Synchronization"
    )

    plotter.plot_constellation(
        test_signal_matched_filtered,
        title="Constellation After Matched Filtering"
    )

    plotter.plot_constellation(
        test_signal_time_adjusted,
        title="Constellation After Mueller and Muller Timing Synchronization"
    )

    plotter.plot_constellation(
        test_signal_fine_freq_adjusted,
        title="Constellation After Costas Loop Frequency Synchronization"
    )
    show()
    
