"""
Test and measure the PSD of transmitter of Adalm Pluto SDR
"""

import adi
from yaml import safe_load
from sdr_plots import StaticSDRPlotter




class SDR:  
    def __init__(self, config_file: str ="config.yaml"):
        """Initialize SDR with given configuration."""
    
        with open(config_file, 'r') as f:
            config = safe_load(f)

        self.config = config
        self.sample_rate = int(float(config['radio']['sample_rate']))
        self.center_frequency = int(float(config['radio']['center_frequency']))

        self.sdr                        = adi.Pluto(config['radio']['ip_address'])
        self.sdr.sample_rate            = int(float(config['radio']['sample_rate']))

        self.sdr.tx_rf_bandwidth        = int(float(config['transmitter']['tx_rf_bandwidth']))
        self.sdr.tx_lo                  = int(float(config['transmitter']['tx_lo']))
        self.sdr.tx_hardwaregain_chan0  = config['transmitter']['tx_gain_dB']
        self.sdr.tx_cyclic_buffer       = config['transmitter']['tx_cyclic_buffer_enable']

        self.sdr.rx_rf_bandwidth        = int(float(config['receiver']['rx_rf_bandwidth']))
        self.sdr.rx_hardwaregain_chan0  = config['receiver']['rx_gain_dB']
        self.sdr.rx_lo                  = int(float(config['receiver']['rx_lo']))
        self.sdr.rx_buffer_size         = config['receiver']['buffer_size']
        self.sdr.gain_control_mode_chan0 = config['radio']['gain_mode']

    def transmit(self, tx_data):
        """Transmit data using the SDR."""
        self.sdr.tx(tx_data)

    def generate_test_signal(self, duration=1.0, freq=100e3):
        """Generate a test signal: a simple sine wave.
        
            Args:
                freq (float): Frequency of the sine wave in Hz.
                sample_rate (float): Sample rate in samples per second.
                duration (float): Duration of the signal in seconds.
        """
        import numpy as np

        t = np.arange(0, duration, 1/self.sample_rate)
        signal = 0.5 * np.exp(2.0j * np.pi * freq * t)  
        signal *= 2**14 # Scale to int16 range for Adalm Pluto.
        return signal.astype(np.complex64)

def main():
    sdr = SDR("config.yaml")

    test_signal = sdr.generate_test_signal()

    sdr.transmit(test_signal)

    plotter = StaticSDRPlotter()

    plotter.plot_psd(
        test_signal, 
        sample_rate= sdr.sample_rate,
        center_freq= sdr.center_frequency,
        title="Transmitted Signal PSD",

    )

    plotter.plot_time_domain(
        samples=test_signal,
        sample_rate=sdr.sample_rate,
        title="Transmitted Signal Time Domain",
    )

    from matplotlib.pyplot import show 
    show()


if __name__ == "__main__":
    main()
