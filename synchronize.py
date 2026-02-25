import numpy as np
from scipy import signal



class Synchronizer:
    def __init__(self, config: dict):
        self.modulation_scheme = config['modulation']['type'].upper().strip()
        self.sps = int(config['modulation']['samples_per_symbol'])
        self.buffer_size = int(config['receiver']['buffer_size'])
        self.interpolation_factor = int(config['synchronization']['interpolation_factor'])
        self.sample_rate = self.sps * int(float(config['modulation']['symbol_rate']))
        self.coarse_freq_search_factor = int(config['synchronization']['coarse_freq_search_factor'])

        if self.modulation_scheme == 'BPSK':
            self.modulation_order = 2.0
        elif self.modulation_scheme == 'QPSK':
            self.modulation_order = 4.0
        else:
            raise ValueError(f"Unsupported modulation scheme: {self.modulation_scheme}")

    def coarse_frequenzy_synchronization(self, received_signal: np.ndarray) -> np.ndarray:
        """Coarse frequency synchronization using FFT-based method."""

        signal_raised = received_signal**self.modulation_order # Remove modulation effects by raising to the power of the modulation order

        magnitude = np.fft.fftshift(np.abs(np.fft.fft(signal_raised)))  

        freqs = np.fft.fftshift(np.fft.fftfreq(len(signal_raised), d=1/self.sample_rate))

        estimated_frequenzy_offset = freqs[np.argmax(magnitude)] / self.modulation_order # Divide by modulation order to get the actual frequency offset
        
        time_vector = np.arange(len(received_signal)) / self.sample_rate
        
        return received_signal * np.exp(-1j * 2 * np.pi * estimated_frequenzy_offset * time_vector)
    
    def fine_frequenzy_synchronization(self, received_signal: np.ndarray) -> np.ndarray:
        """Fine frequency synchronization using a costas loop."""
        
        
    
    def delay_synchronization(self, received_signal: np.ndarray) -> np.ndarray:
        """Delay synchronization using a matched filter for the known barker preamble."""


        


if __name__ == "__main__":
    from yaml import safe_load
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


    # Example usage with a test signal
    frequency_offset = 500
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

    from matplotlib.pyplot import show
    show()
  