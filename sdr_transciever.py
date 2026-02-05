"""
Test and measure the PSD of transmitter of Adalm Pluto SDR
"""

import adi
from yaml import safe_load
from sdr_plots import LiveSDRPlotter, StaticSDRPlotter
import queue
from scipy import signal
import numpy as np



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
  
    def __del__(self):
        """Destructor to ensure SDR is disconnected."""
        self.sdr.tx_destroy_buffer() # Ensure TX buffer is destroyed
        self.stop_receiving()
        if self.sdr:
            del self.sdr
            self.sdr = None
            print("SDR disconnected and resources released.")

    def connect(self):
        """Connect to Adalm Pluto SDR and configure."""
        try:
            self.sdr = adi.Pluto(self.config['radio']['ip_address'])
            print(f"Pluto {self.sdr.uri}")

            # Configure TX
            self.sdr.tx_rf_bandwidth = int(float(self.config['transmitter']['tx_bandwidth']))
            self.sdr.tx_lo = int(float(self.config['transmitter']['tx_carrier']))
            self.sdr.tx_hardwaregain_chan0 = int(float(self.config['transmitter']['tx_gain_dB']))
            self.sdr.sample_rate = int(float(self.config['modulation']['sample_rate']))
            self.sdr.tx_cyclic_buffer = bool(self.config['transmitter']['tx_cyclic_buffer'])
            
            print(f"Sample Rate\t: {self.sdr.sample_rate/1e6:.3f} MHz")
            print(f"TX Bandwidth\t: {self.sdr.tx_rf_bandwidth/1e6:.3f} MHz")
            print(f"TX Carrier\t: {self.sdr.tx_lo/1e6:.3f} MHz")
            print(f"TX Gain\t\t: {self.sdr.tx_hardwaregain_chan0} dB")
            print(f"TX Cyclic Buffer\t: {self.sdr.tx_cyclic_buffer}")

            # Configure RX

            self.sdr.rx_rf_bandwidth = int(float(self.config['receiver']['rx_bandwidth']))
            self.sdr.rx_lo = int(float(self.config['receiver']['rx_carrier']))
            self.sdr.rx_buffer_size = int(self.config['receiver']['buffer_size'])
            self.sdr.gain_control_mode_chan0 = self.config['receiver']['gain_mode']
            if self.config['receiver']['gain_mode'] == 'manual':
                self.sdr.rx_hardwaregain_chan0 = float(self.config['receiver']['rx_gain_dB'])
                print(f"RX Gain\t\t: {self.sdr.rx_hardwaregain_chan0} dB")
            else:
                print(f"RX Gain\t\t: {self.sdr.gain_control_mode_chan0} mode")

            print(f"RX Bandwidth\t: {self.sdr.rx_rf_bandwidth/1e6:.3f} MHz")
            print(f"RX Carrier\t: {self.sdr.rx_lo/1e6:.3f} MHz")
            print(f"RX Gain Mode\t: {self.sdr.gain_control_mode_chan0}")
            print("=================================================================\n")
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

        t = np.arange(-num_taps//2, num_taps//2, step=1) / sps
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

        t = np.arange(0, duration, 1/int(float(self.config['modulation']['sample_rate'])))
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
#
#    signal = transciever.generate_test_signal(duration=1.0, freq=100e3)
#    transciever.sdr.tx(signal)
#
#    plotter.plot_time_domain(signal, sample_rate=float(transciever.config['modulation']['sample_rate']), title="Transmitted Signal - Time Domain")
#    
#
#    print(f"Signal transmitted:\n\
#            Length:\t{len(signal)} samples\n\
#            Duration:\t{len(signal)/float(transciever.config['modulation']['sample_rate']):.3f} seconds\n\
#            Frequency:\t{500e3/1e6:.3f} MHz\n")
#
    show()

    #plotter = LiveSDRPlotter(
    #    data_callback=transciever.sdr.rx, 
    #    sample_rate=float(transciever.config['modulation']['sample_rate']),
    #    center_freq=float(transciever.config['receiver']['rx_carrier'])
    #    )

    #plotter.start_spectrum_analyzer()
    samples = transciever.sdr.rx()
    plotter.plot_spectrogram(
        samples, 
        sample_rate=float(transciever.config['modulation']['sample_rate']),
        title="Received Signal - Spectrogram"
        )

    show()

    del transciever
    del plotter