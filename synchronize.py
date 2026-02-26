import numpy as np
from scipy import signal



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
            self.costas_error = self._costas_error_bpsk
        elif self.modulation_scheme == 'QPSK':
            self.modulation_order = 4.0
            self.costas_error = self._costas_error_qpsk
        else:
            raise ValueError(f"Unsupported modulation scheme: {self.modulation_scheme}")

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
    
    def _costas_error_bpsk(self, sample: np.complex64) -> float:
        """Pointer function for Costas loop error calculation for BPSK modulation."""
        return np.real(sample) * np.imag(sample)
    
    def _costas_error_qpsk(self, sample: np.complex64) -> float:
        """Pointer function for Costas loop error calculation for QPSK modulation."""
        return np.sign(sample.real) * sample.imag - np.sign(sample.imag) * sample.real
    
    def fine_frequenzy_synchronization(self, received_signal: np.ndarray) -> np.ndarray:
        """Fine frequency synchronization using a costas loop."""
        phase = 0.0     # [radians] 
        freq = 0.0      # [radians/sample] 
        out = np.zeros_like(received_signal, dtype=np.complex64)

        N = len(received_signal)
        for i in range(N):
            out[i] = received_signal[i] * np.exp(-1j * phase)
            error = self.costas_error(out[i])

            freq += self.costas_beta * error
            phase += freq + self.costas_alpha * error

            # Keep phase in the range [0, 2*pi) to prevent numerical issues
            phase = phase % (2 * np.pi)
        return out

        
    def mm_decision(self, sample: np.complex64) -> np.complex64:
        # QPSK decision (works for BPSK too)
        return np.sign(sample.real) + 1j*np.sign(sample.imag)

    def mm_interpolate(self, x: np.ndarray, i: int, mu: float) -> float:
        """Cubic interpolation. Dynamically computes the interpolated value at position i + mu using the four surrounding samples.
        Args:    
            x: Input signal array.
            i: Current integer sample index (must be >= 2 and <= len(x)-3 to allow for 4 samples).
            mu: Fractional offset (0 <= mu < 1) indicating how far between x[i] and x[i+1] the interpolation should occur.
        Returns:    
            Interpolated value at position i + mu.
        """
        x0 = x[i-1]
        x1 = x[i]
        x2 = x[i+1]
        x3 = x[i+2]

        a = (-0.5*x0) + (1.5*x1) - (1.5*x2) + (0.5*x3)
        b = x0 - (2.5*x1) + (2*x2) - (0.5*x3)
        c = (-0.5*x0) + (0.5*x2)
        d = x1

        return a*mu**3 + b*mu**2 + c*mu + d

    # Source: https://wirelesspi.com/mueller-and-muller-timing-synchronization-algorithm/ 
    def time_synchronization(self, samples: np.ndarray) -> np.ndarray:
        """Mueller and Müller timing synchronization algorithm.
        Should be applied after coarse frequency synchronization.
        Args:
            samples: Input signal array after coarse frequency synchronization.
        Returns:
            Array of samples resampled at the symbol rate, with timing errors corrected.
        """
        mu = 0.0  # Fractional interval between samples, initialized to 0 (start at the first sample)
        omega = self.sps  # Initial estimate of rate of change of mu
        i = 2  # need margin for cubic interp
        N = len(samples) - 3

        prev_sample = self.mm_interpolate(samples, 1, mu)
        prev_decision = self.mm_decision(prev_sample)

        out = [prev_sample]

        while i < N:
            current_sample = self.mm_interpolate(samples, i, mu)
            current_decision = self.mm_decision(current_sample)

            # Mueller & Müller timing error
            error = np.real(
                prev_decision * current_sample -
                current_decision * prev_sample
            )

            # Update omega (frequency term)
            omega += self.mm_Ki * error

            # clamp omega to prevent it from diverging
            omega = np.clip(omega, self.sps - 0.5, self.sps + 0.5)

            # Update phase
            mu += omega + self.mm_Kp * error

            prev_sample = current_sample
            prev_decision = current_decision

            out.append(current_sample)

            step = int(np.floor(mu))
            i += step
            mu -= step  # Remove the integer part from mu to keep it in the range [0, 1)

        return np.array(out)

            


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
    
