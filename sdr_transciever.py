"""
Test and measure the PSD of transmitter of Adalm Pluto SDR
"""

import adi
from yaml import safe_load
from sdr_plots import LiveSDRPlotter, StaticSDRPlotter
import queue
import time
import threading
from scipy import signal
import numpy as np
import barker_code



class SDRTransciever:  
    def __init__(self, config_file: str ="config.yaml"):
        """Initialize SDR with given configuration."""
        print("\n------------------------------------------------")
        print("Initilizing Adalm Pluto SDR Transciever...")
        print("------------------------------------------------\n")

        try: 
            with open(config_file, 'r') as f:
                config = safe_load(f)
        except Exception as e:
            print(f"Error loading config file: {e}")
            raise e

        self.config = config
        self.sdr = None

        self.rx_thread = None
        self.rx_running = False

        self.tx_queue = queue.Queue()
        self.rx_queue = queue.Queue()

        # Generate pulse shaping filter
        self.rrc_filter = self._generate_rrc_filter()
        self.barker_seq = barker_code.generate_barker_code(int(self.config['preamble']['barker_code_length']))

    
    def __del__(self):
        """Destructor to ensure SDR is disconnected."""
        self.stop_receiving()
        if self.sdr:
            del self.sdr
            self.sdr = None
            print("SDR disconnected and resources released.")

    def connect(self):
        """Connect to Adalm Pluto SDR and configure."""
        print("Connecting to Adalm Pluto at ip:", self.config['radio']['ip_address'])
        
        try:
            self.sdr = adi.Pluto(self.config['radio']['ip_address'])

            # Configure TX
            self.sdr.tx_rf_bandwidth = int(float(self.config['transmitter']['tx_bandwidth']))
            self.sdr.tx_lo = int(float(self.config['transmitter']['tx_carrier']))
            self.sdr.tx_hardwaregain_chan0 = int(float(self.config['transmitter']['tx_gain_dB']))
            self.sdr.sample_rate = int(float(self.config['modulation']['sample_rate']))

            # Configure RX
            self.sdr.rx_rf_bandwidth = int(float(self.config['receiver']['rx_bandwidth']))
            self.sdr.rx_lo = int(float(self.config['receiver']['rx_carrier']))
            self.sdr.rx_buffer_size = int(self.config['receiver']['buffer_size'])
            self.sdr.gain_control_mode_chan0 = self.config['receiver']['gain_mode']
            if self.config['receiver']['gain_mode'] == 'manual':
                self.sdr.rx_hardwaregain_chan0 = float(self.config['receiver']['rx_gain_dB'])
            
            
            print(f"Connected to ADALM Pluto at {int(float(self.config['transmitter']['tx_carrier']))/1e6:.3f} MHz")
            return True
        
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
        

    def _generate_rrc_filter(self):
        """Generate Root Raised Cosine (RRC) filter for pulse shaping."""
        from scipy.signal import fir_filter_design as design
        from scipy.signal import lfilter

        roll_off = float(self.config['modulation']['roll_off'])
        sps = int(self.config['modulation']['samples_per_symbol'])
        num_taps = int(self.config['modulation']['rrc_filter_span']) * sps + 1  # Ensure odd number of taps

        t = np.arange(-num_taps//2, num_taps//2) / sps
        h = np.zeros_like(t)

        symbols_per_second = int(float(self.config['modulation']['symbol_rate']))

        for i, time in enumerate(t):
            if time == 0:
                h[i] = (1 + roll_off * (4/np.pi - 1)) * symbols_per_second

            elif np.abs(time) == 1 / (4 * roll_off * symbols_per_second):
                h[i] = (
                    roll_off * symbols_per_second / (np.sqrt(2))) * (
                        (1 + 2/np.pi) * np.sin(np.pi / (4 * roll_off)) +
                        (1 - 2/np.pi) * np.cos(np.pi / (4 * roll_off))
                    )

            else:
                h[i] = (
                        np.sin(np.pi * time * symbols_per_second * (1 - roll_off)) +
                        4 * roll_off * time * symbols_per_second * np.cos(np.pi * time * symbols_per_second * (1 + roll_off))
                    ) * (
                        symbols_per_second  / (
                            np.pi * time * symbols_per_second * (1 - (4 * roll_off * time * symbols_per_second) ** 2)
                        )   
                    )
    
        # Normalize filter energy
        h /= np.sqrt(np.sum(h**2))
        return h


    def modulate_bpsk(self, symbols):
        """Modulate bipolar symbols using BPSK. Adds barker code preamble.
        Args:
            message (np.array): symbols to modulate (1 and -1).
        
        Returns:
            np.array: Complex IQ baseband samples.
        """    

        # add preamble
        symbols_with_preamble = np.concatenate((self.barker_seq, symbols))

        sps = int(self.config['modulation']['samples_per_symbol'])

        # Upsample
        upsampled = np.zeros(len(symbols_with_preamble) * sps)
        upsampled[::sps] = symbols_with_preamble
        
        # Apply RRC pulse shaping filter
        shaped = np.convolve(upsampled, self.rrc_filter, mode='same')
        
        # Scale for int16 range for Adalm Pluto
        complex_signal = 2**14 * shaped.astype(np.complex64)

        return complex_signal.astype(np.complex64)

        

    def demodulate_bpsk(self, samples):
        """Demodulate BPSK signal from samples.
        
        Args:
            samples (np.array): Complex IQ baseband samples.
        
        Returns:
            Message object if demodulation is successful, else None.
        """


    

    def _detect_barker_code(self, signal):
        """Detect Barker code preamble in the signal.
        
        Args:
            signal (np.array): Filtered baseband signal.
        
        Returns:
            Index after preamble if found, else None.
        """

        # Cross-correlate to find preamble
        correlation = signal.correlate(signal, self.barker_preamble_upsampled, mode='valid')
        
        peak_index = np.argmax(np.abs(correlation))
        peak_value = np.abs(correlation[peak_index])

        threshold = 0.8 * np.max(np.abs(correlation))

        if peak_value > threshold:
            return peak_index + len(self.barker_preamble_upsampled)
        else:
            return None

    def stop_receiving(self):
        """Stop receiving data from SDR"""
        self.rx_running = False
        if self.rx_thread:
            self.rx_thread.join(timeout=2)
            self.rx_thread = None
        print("Stopped receiving data")

    def generate_test_signal(self, duration=1.0, freq=100e3):
        """Generate a test signal: a simple sine wave.
        
            Args:
                freq (float): Frequency of the sine wave in Hz.
                sample_rate (float): Sample rate in samples per second.
                duration (float): Duration of the signal in seconds.
        """
        import numpy as np

        t = np.arange(0, duration, 1/int(float(self.config['radio']['sample_rate'])))
        signal = 0.5 * np.exp(2.0j * np.pi * freq * t)  
        signal *= 2**14 # Scale to int16 range for Adalm Pluto.
        return signal.astype(np.complex64)

if __name__ == "__main__":
    from matplotlib.pyplot import show
    transciever = SDRTransciever("config.yaml")


    if transciever.connect() == False:
        print("Failed to connect to SDR. Exiting.")
        exit(1)
    
    plotter = StaticSDRPlotter()

    signal = transciever.generate_test_signal(duration=1.0, freq=100e3)

    plotter.plot_time_domain(signal, sample_rate=transciever.config['radio']['sample_rate'], title="Transmitted Signal - Time Domain")


    modulated_signal = transciever.modulate_bpsk(signal)
    
    plotter.plot_constellation(modulated_signal, title="Modulated BPSK Signal - Constellation")

    show()
    del transciever
    del plotter