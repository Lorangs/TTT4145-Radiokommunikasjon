from scipy import signal
import numpy as np
from barker_code import BARKER_SYMBOLS

class BarkerDetector:
    def __init__(self, config: dict):
        self.barker_code = BARKER_SYMBOLS[int(config['barker_sequence']['code_length'])]
        self.correlation_scale_factor_threshold = float(config['barker_sequence']['correlation_scale_factor_threshold'])
        self.noise_floor_dB = None  # To be set after SDR connection

    def set_noise_floor_dB(self, noise_floor_dB: float):
        """Set the noise floor in dB for adaptive thresholding."""
        self.noise_floor_dB = noise_floor_dB

    def detect(self, received_signal: np.array) -> int:
        """Detect Barker code in the received signal and return the index of the start of the code."""
        correlation = signal.correlate(received_signal, self.barker_code, mode='valid')
        correlation_abs = np.abs(correlation)
        peak_index = np.argmax(correlation_abs)
        peak_value = correlation_abs[peak_index]
        threshold = self.correlation_scale_factor_threshold * self.noise_floor_dB if self.noise_floor_dB is not None else 0
        if peak_value < threshold:
            return None
        return int(peak_index)
    
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
    barker_code = 10**(-89/20) * BARKER_SYMBOLS[int(config['barker_sequence']['code_length'])]   

    insert_position = 500
    test_signal = noise.copy()
    test_signal[insert_position:insert_position+len(barker_code)] += barker_code
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