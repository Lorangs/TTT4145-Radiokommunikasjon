import numpy as np
from scipy import signal
from numba import njit
from barker_code import BARKER_SYMBOLS

from matplotlib import pyplot as plt
from filter import RRCFilter


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

        if modulation_order == 2.0:   # BPSK
            error = np.sign(sample.real) * sample.imag
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

        self.correlation_threshold = float(config['barker_sequence']['correlation_threshold'])
        self.noise_floor = 0.0 # linear scale, to be set after SDR connection
        
        self.costas_alpha = float(config['synchronization']['costas_alpha'])
        self.costas_beta = float(config['synchronization']['costas_beta'])  
        
        filter = RRCFilter(config)
        rc_filter = filter.rc_coefficients

        preamble_sequence = BARKER_SYMBOLS[self.modulation_scheme][int(config['barker_sequence']['code_length'])]  
        preamble_sequence_upsampled = np.zeros(len(preamble_sequence) * self.sps, dtype=np.complex64)
        preamble_sequence_upsampled[::self.sps] = preamble_sequence  
        preamble_sequence = np.convolve(preamble_sequence_upsampled, rc_filter, mode='full')  # Apply pulse shaping to the preamble sequence

        preamble_energy = np.sum(np.abs(preamble_sequence)**2)
        self.preamble_sequence = preamble_sequence / np.sqrt(preamble_energy)  

        if self.modulation_scheme == 'BPSK':
            self.modulation_order = 2.0    
        elif self.modulation_scheme == 'QPSK':
            self.modulation_order = 4.0
        else:
            raise ValueError(f"Unsupported modulation scheme: {self.modulation_scheme}")
        
        # compile the Numba-optimized functions with the specified parameters
        _costas_loop_njit(np.zeros(self.buffer_size, dtype=np.complex64), self.costas_alpha, self.costas_beta, self.modulation_order)

    def set_noise_floor(self, noise_floor_dB: float):
        """Set the noise floor in dB for adaptive thresholding."""
        self.noise_floor = 10**(noise_floor_dB/10)  # Convert dB to linear scale  

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


    def time_synchronization(self, samples: np.ndarray) -> int:
        """ ML timing synchronization algorithm. Searches for the known pattern in the received signal and estimates the timing offset."""

        correlation = signal.correlate(samples, self.preamble_sequence, mode='valid', method='fft')

        abs_correlation = np.abs(correlation)  # Normalize correlation to get a value between 0 and 1

        max_value = np.max(abs_correlation)

        if max_value > self.correlation_threshold * self.noise_floor:
            return np.argmax(abs_correlation)
        
        else: return None
        


if __name__ == "__main__":
    from yaml import safe_load
    from sdr_plots import StaticSDRPlotter
    

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
    num_symbols = 500
    frequency_offset = 100  # [Hz]
    timing_offset = -0.5 # [fraction of symbol period]
    snr_dB = 10 # [dB]

    sps = synchronizer.sps
    sample_rate = synchronizer.sample_rate

    test_signal = np.zeros(num_symbols, dtype=np.complex64)
    test_signal = (2*np.random.randint(0, 2, num_symbols) - 1) + 1j*(2*np.random.randint(0, 2, num_symbols) - 1)
    test_signal = np.concatenate([BARKER_SYMBOLS[synchronizer.modulation_scheme][13], test_signal])

    test_signal_upsampled = np.zeros(len(test_signal) * sps, dtype=np.complex64)
    test_signal_upsampled[::sps] = test_signal  # Upsample by inserting zeros between symbols
    time_vector = np.arange(len(test_signal_upsampled)) / sample_rate

    # TX processing: pulse shaping, add frequency and timing offset
    test_signal = signal.convolve(test_signal_upsampled, rrc_filter, mode='same', method='fft') #  Apply pulse shapin
    
    test_signal = test_signal * np.exp(1j * 2*np.pi * frequency_offset * time_vector)  # Add frequency offset
    test_signal = np.roll(test_signal, int(timing_offset * sps))  # Add timing offset

    test_signal += 10**(-snr_dB/10) * (np.random.randn(len(test_signal)) + 1j*np.random.randn(len(test_signal)))  # Add AWGN noise
    #test_signal += 0.5 + 1j*0.2 # add DC offset
    awgn_signal = test_signal.copy()

    # RX processing: coarse frequency synchronization, matched filtering, fine frequency synchronization, timing synchronization

    test_signal = synchronizer.coarse_frequenzy_synchronization(test_signal)  
    coarse_freq_corrected_signal = test_signal.copy()

    test_signal = signal.convolve(test_signal, rrc_filter, mode='same', method='fft')  # Matched filtering after coarse frequency synchronization   

    plotter.plot_eye_diagram(
        test_signal,
        samples_per_symbol=sps,
        title="Eye Diagram after Coarse Frequency Synchronization and Matched Filtering",
        num_traces=100,
        symbols_per_trace=2
    )

    delay = synchronizer.time_synchronization(test_signal) # Timing synchronization without preamble
    print(f"Estimated timing offset (samples): {delay}")
    test_signal = test_signal[delay::sps]  # Compensate for timing offset

    downsampled_signal = test_signal.copy()

    test_signal = synchronizer.fine_frequenzy_synchronization(test_signal)  # Fine frequency synchronization using Costas loop
    fine_freq_corrected_signal = test_signal.copy()

    plotter.plot_constellation(
        awgn_signal, 
        title="Received Signal Constellation with AWGN"
    )

    plotter.plot_constellation(
        coarse_freq_corrected_signal, 
        title="After Coarse Frequency Synchronization"
    )
    plotter.plot_constellation(
        downsampled_signal, 
        title="After Timing Synchronization (Downsampled)"
    )
    plotter.plot_constellation(
        fine_freq_corrected_signal, 
        title="After Fine Frequency Synchronization (Costas Loop)"
    )
    plt.show()