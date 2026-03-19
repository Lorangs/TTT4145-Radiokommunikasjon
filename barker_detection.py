"""
This module defines the BarkerDetector class, 
which is responsible for detecting Barker codes in a received signal 
and removing them to extract the underlying message signal. 
The Barker code is used as a preamble for synchronization and detection of incoming messages in the RadioGram system. 
"""

from scipy import signal
import numpy as np
from barker_code import BARKER_SYMBOLS
import logging

class BarkerDetector:
    def __init__(self, config: dict):
        modulation_type = str(config['modulation']['type']).upper().strip()
        barker_config = config['barker_sequence']
        code_length = int(barker_config['code_length'])

        try:
            self.barker_symbols = BARKER_SYMBOLS[modulation_type][code_length]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported Barker configuration: modulation_type={modulation_type}, "
                f"code_length={code_length}"
            ) from exc

        threshold = barker_config.get(
            'correlation_scale_factor_threshold',
            barker_config.get('correlation_threshold'),
        )
        if threshold is None:
            raise ValueError(
                "Missing Barker correlation threshold. Expected "
                "'correlation_scale_factor_threshold' or 'correlation_threshold'."
            )
        self.correlation_scale_factor_threshold = float(threshold)
        self.noise_floor_dB = None  # To be set after SDR connection

    def set_noise_floor_dB(self, noise_floor_dB: float):
        """Set the noise floor in dB for adaptive thresholding."""
        self.noise_floor_dB = noise_floor_dB

    def add_barker_symbols(self, signal: np.ndarray) -> np.ndarray:
        """Add Barker code to the beginning of the signal."""
        return np.concatenate((self.barker_symbols, signal))
    
    def remove_barker_symbols(self, signal: np.ndarray, start_index: int) -> np.ndarray:
        """Remove Barker code from the signal starting at the specified index."""
        if start_index < 0 or start_index + len(self.barker_symbols) > len(signal):
            logging.warning("Invalid start index for removing Barker code.")
            return signal
        return signal[start_index + len(self.barker_symbols):]
    
if __name__ == "__main__":

    from yaml import safe_load
    try:
        with open("setup/config.yaml", "r") as f:
            config = safe_load(f)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        exit(1)

    detector = BarkerDetector(config)
    detector.set_noise_floor_dB(-90)  # Example noise floor, adjust as needed
    
    # Example usage with a test signal containing a Barker code
     # Random noise, noise floor around -90 dB
    noise = 10**(-90/20) * (np.random.randn(1000) + 1j*np.random.randn(1000))
    
    # insert Barker code at a random position in the noise
    # SNR of around 1 dB for the Barker code
    modulation_type = str(config['modulation']['type']).upper().strip()
    code_length = int(config['barker_sequence']['code_length'])
    barker_symbols = 10**(-89/20) * BARKER_SYMBOLS[modulation_type][code_length]

    insert_position = 500
    test_signal = noise.copy()
    test_signal[insert_position:insert_position+len(barker_symbols)] += barker_symbols
    detected_index = detector.detect(test_signal)
    if detected_index is not None:
        print(f"Barker code detected at index: {detected_index}")
    else:
        print("Barker code not detected.")

    from sdr_plots import StaticSDRPlotter
    plotter = StaticSDRPlotter()
    plotter.plot_time_domain(test_signal, title="Test Signal with Barker Code", sample_rate=1e6)
    from matplotlib.pyplot import show
    show()
