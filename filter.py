import commpy
import numpy as np
from sdr_plots import StaticSDRPlotter

from yaml import safe_load

try:
    with open("setup/config.yaml", 'r') as f:
        config = safe_load(f)
except Exception as e:
    print(f"Error loading config file: {e}")
    raise e

class RRCFilter:
    """Class to generate and manage Root Raised Cosine (RRC) filters."""
    
    def __init__(self, rolloff: float, span: int, sps: int):
        self.rolloff = rolloff
        self.span = span
        self.sps = sps
        self.time_vector, self.coefficients = self._generate_rrc_filter()

    def _generate_rrc_filter(self) -> np.ndarray:
        """Generate RRC filter coefficients using commpy."""
        
        time_idx, h = commpy.filters.rrcosfilter(
            N=self.sps * self.span, 
            alpha=self.rolloff, 
            Ts=1.0, 
            Fs=self.sps)

        return time_idx, h
      

    def apply_filter(self, signal: np.ndarray) -> np.ndarray:
        """Apply RRC filter to the input signal."""
        return np.convolve(signal, self.coefficients, mode='same')
    
    def plot_filter_response(self):
        """Plot the impulse and frequency response of the RRC filter."""
        plotter = StaticSDRPlotter()
        plotter.plot_filter_response(self.coefficients, 
                                     self.time_vector, 
                                     sample_rate=self.sps * self.span, 
                                     title="RRC Filter Response")
        
        from matplotlib.pyplot import show
        show()

if __name__ == "__main__":
    # Example usage of RRCFilter
    rolloff = float(config['modulation']['rrc_roll_off'])
    span = int(config['modulation']['rrc_filter_span'])
    sps = int(config['modulation']['samples_per_symbol'])
    rrc_filter = RRCFilter(rolloff, span, sps)
    rrc_filter.plot_filter_response()
