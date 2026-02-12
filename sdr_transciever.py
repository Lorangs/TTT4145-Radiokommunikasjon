"""
Test and measure the PSD of transmitter of Adalm Pluto SDR
"""


import adi
from sdr_plots import LiveSDRPlotter, StaticSDRPlotter
import queue
from scipy import signal
import numpy as np
import logging


class SDRTransciever:  
    def __init__(self, config: dict):
        """Initialize SDR with given configuration."""
        logging.info("Initializing SDR Transciever...")
        self.config = config
        self.sdr = None
  
    def __del__(self):
        """Destructor to ensure SDR is disconnected."""
        if self.sdr:
            self.sdr.tx_destroy_buffer() # Ensure TX buffer is destroyed
            del self.sdr
            self.sdr = None
            logging.info("SDR Transciever resources cleaned up.")

    def connect(self):
        """Connect to Adalm Pluto SDR and configure."""
        try:
            self.sdr = adi.Pluto(self.config['radio']['ip_address'])
            logging.info(f"Connected to SDR at {self.config['radio']['ip_address']}.")

            # Configure TX
            self.sdr.tx_rf_bandwidth = int(float(self.config['transmitter']['tx_bandwidth']))
            self.sdr.tx_lo = int(float(self.config['transmitter']['tx_carrier']))
            self.sdr.tx_hardwaregain_chan0 = int(float(self.config['transmitter']['tx_gain_dB']))
            self.sdr.sample_rate = int(float(self.config['modulation']['sample_rate']))
            self.sdr.tx_cyclic_buffer = bool(self.config['transmitter']['tx_cyclic_buffer'])
            
            logging.info(f"TX Bandwidth\t: {self.sdr.tx_rf_bandwidth/1e6:.3f} MHz")
            logging.info(f"TX Carrier\t: {self.sdr.tx_lo/1e6:.3f} MHz")
            logging.info(f"TX Gain\t\t: {self.sdr.tx_hardwaregain_chan0} dB")
            logging.info(f"Sample Rate\t: {self.sdr.sample_rate/1e6:.3f} MS/s")
            logging.info(f"TX Cyclic Buffer: {self.sdr.tx_cyclic_buffer}")

            # Configure RX

            self.sdr.rx_rf_bandwidth = int(float(self.config['receiver']['rx_bandwidth']))
            self.sdr.rx_lo = int(float(self.config['receiver']['rx_carrier']))
            self.sdr.rx_buffer_size = int(self.config['receiver']['buffer_size'])
            self.sdr.gain_control_mode_chan0 = self.config['receiver']['gain_mode']
            if self.config['receiver']['gain_mode'] == 'manual':
                self.sdr.rx_hardwaregain_chan0 = float(self.config['receiver']['rx_gain_dB'])
                logging.info(f"RX Gain\t\t: {self.sdr.rx_hardwaregain_chan0} dB (manual mode)")
            else:
                logging.info(f"RX Gain Mode\t: {self.sdr.gain_control_mode_chan0} (automatic mode)")

            logging.info(f"RX Bandwidth\t: {self.sdr.rx_rf_bandwidth/1e6:.3f} MHz")
            logging.info(f"RX Carrier\t: {self.sdr.rx_lo/1e6:.3f} MHz")
            logging.info(f"RX Buffer Size\t: {self.sdr.rx_buffer_size} samples")

            return True
        
        except Exception as e:
            logging.error(f"Error connecting to SDR: {e}")
            return False


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
    
    noise_signal = 2**14 * 0.01 * (np.random.randn(1024) + 1j*np.random.randn(1024))

    transciever.sdr.tx(noise_signal.astype(np.complex64))

    recieved_signal = transciever.sdr.rx()

    plotter = StaticSDRPlotter()

    plotter.plot_psd(
        recieved_signal, 
        title="Received Signal PSD", 
        center_freq=transciever.sdr.rx_lo, 
        sample_rate=100e7
    )
    show()

    del transciever
    del plotter