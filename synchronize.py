import numpy as np
from scipy import signal



class Synchronizer:
    def __init__(self, config: dict):
        self.modulation_scheme = config['modulation']['type'].upper().strip()
        self.sps = int(config['modulation']['samples_per_symbol'])
        self.buffer_size = int(config['receiver']['buffer_size'])
        self.sample_rate = self.sps * int(float(config['modulation']['symbol_rate']))
        self.nfft = int(config['synchronization']['nfft'])
        
        
        self.mm_mu = 0.0                                                        # Fractional interval [0, 1) for timing recovery
        self.mm_omega = self.sps                                                # RX sps estimate
        self.mm_prev_sample = 0
        self.mm_prev_decision = 0 
        self.mm_Kp = float(config['synchronization']['mm_Kp'])
        self.mm_Ki = float(config['synchronization']['mm_Ki'])
        
        
        if self.modulation_scheme == 'BPSK':
            self.modulation_order = 2.0
        elif self.modulation_scheme == 'QPSK':
            self.modulation_order = 4.0
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
    
    def fine_frequenzy_synchronization(self, received_signal: np.ndarray) -> np.ndarray:
        """Fine frequency synchronization using a costas loop."""
        
    def decision(self, sample: np.complex64) -> np.complex64:
        # QPSK decision (works for BPSK too)
        return np.sign(sample.real) + 1j*np.sign(sample.imag)

    def interpolate(self, x: np.ndarray, i: int, mu: float) -> float:
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

    def mm_timing_synchronization(self, samples: np.ndarray) -> np.ndarray:

        self.prev_sample = self.interpolate(samples, 1, self.mm_mu)
        self.prev_decision = self.decision(self.prev_sample)

        out = [self.prev_sample]

        i = 2  # need margin for cubic interp
        n = len(samples) - 3

        while i < n:
            # Interpolated symbol. 
            current_sample = self.interpolate(samples, i, self.mm_mu)

            # Make decision
            current_decision = self.decision(current_sample)

            # Mueller & Müller timing error
            error = np.real(
                self.prev_decision * current_sample -
                current_decision * self.prev_sample
            )
            # Guard against NaN or infinite error.
            if not np.isfinite(error):
                error = 0.0

            # Update omega (frequency term)
            self.mm_omega += self.mm_Ki * error

            # clamp omega to prevent it from diverging
            self.mm_omega = np.clip(self.mm_omega, self.sps - 0.5, self.sps + 0.5)

            # Update phase
            self.mm_mu += self.mm_omega + self.mm_Kp * error

            # Save previous values
            self.prev_sample = current_sample
            self.prev_decision = current_decision

            out.append(current_sample)

            # Move by floor(omega)
            step = int(np.floor(self.mm_mu))
            i += step
            self.mm_mu -= step  # Remove the integer part from mu

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

#############################################################
# Test sequenze for timing synchronization
# Uncomment to run the test sequenze. Make sure to have the config.yaml file properly set up and the required libraries installed.
#############################################################
    # generate test signal with timing offset
    timing_offset = 0.3  # Fraction of a symbol period
    num_symbols = 100
    sps = synchronizer.sps
    sample_rate = synchronizer.sample_rate
    symbols = np.zeros(num_symbols*sps, dtype=np.complex64)

    modulation_scheme = config['modulation']['type'].upper().strip()
    if modulation_scheme == 'BPSK':
        print("Using BPSK modulation")
        symbols[::sps] = 2*np.random.randint(0, 2, num_symbols) - 1
    elif modulation_scheme == 'QPSK':
        print("Using QPSK modulation")
        symbols[::sps] = (2*np.random.randint(0, 2, num_symbols) - 1) + 1j*(2*np.random.randint(0, 2, num_symbols) - 1)
    else:
        print(f"Unsupported modulation scheme: {modulation_scheme}")
        exit(1)

    test_signal = np.convolve (symbols, rrc_filter, mode='same')
    test_signal = np.roll(test_signal, int(timing_offset * sps))  # Introduce timing offset

    plotter.plot_constellation(
        test_signal,
        title="Constellation of Test Signal Before Timing Synchronization"
    )

    plotter.plot_time_domain(
        test_signal,
        sample_rate=sample_rate,
        title="Time Domain of Test Signal Before Timing Synchronization"
    )

    corrected_signal = synchronizer.mm_timing_synchronization(test_signal)

    plotter.plot_constellation(
        corrected_signal,
        title="Constellation of Test Signal After Timing Synchronization"
    )

    plotter.plot_time_domain(
        corrected_signal,
        sample_rate=sample_rate,
        title="Time Domain of Test Signal After Timing Synchronization"
    )

    show()

##############################################################
# Test sequenze for coarse frequency synchronization
# Uncomment to run the test sequenze. Make sure to have the config.yaml file properly set up and the required libraries installed.
##############################################################
"""


    # Example usage with a test signal
    frequency_offset = 20000
    num_symbols = 100
    sps = synchronizer.sps
    sample_rate = synchronizer.sample_rate
    symbols = np.zeros(num_symbols*sps, dtype=np.complex64)
    
    modulation_scheme = config['modulation']['type'].upper().strip()
    if modulation_scheme == 'BPSK':
        print("Using BPSK modulation")
        symbols[::sps] = 2*np.random.randint(0, 2, num_symbols) - 1
    elif modulation_scheme == 'QPSK':
        print("Using QPSK modulation")
        symbols[::sps] = (2*np.random.randint(0, 2, num_symbols) - 1) + 1j*(2*np.random.randint(0, 2, num_symbols) - 1)
    else:
        print(f"Unsupported modulation scheme: {modulation_scheme}")
        exit(1)
    


    test_signal = np.convolve (symbols, rrc_filter, mode='same') 

    time_vector = np.arange(len(test_signal)) / synchronizer.sample_rate
    test_signal *= np.exp(1j* 2 * np.pi * frequency_offset * time_vector)

    plotter.plot_constellation(
        test_signal,
        title="Constellation of Test Signal with Frequency Offset"
    )

    plotter.plot_psd(
        test_signal,
        center_freq=frequency_offset,
        sample_rate=synchronizer.sample_rate,
        title="PSD of Test Signal with Frequency Offset"
    )

    corrected_signal = synchronizer.coarse_frequenzy_synchronization(test_signal)

    plotter.plot_constellation(
        corrected_signal,
        title="Constellation of Corrected Signal after Coarse Frequency Synchronization"
    )

    plotter.plot_psd(
        corrected_signal,
        center_freq=0,
        sample_rate=synchronizer.sample_rate,
        title="PSD of Corrected Signal after Coarse Frequency Synchronization"
    )

    show()
"""