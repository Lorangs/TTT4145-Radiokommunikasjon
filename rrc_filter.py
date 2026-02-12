
import numpy as np
from sdr_plots import StaticSDRPlotter



class RRCFilter:
    """Class to generate and manage Root Raised Cosine (RRC) filters."""
    
    def __init__(self, config: dict):
        """Initialize RRC filter parameters and generate coefficients.
            Args:
                rolloff (float): Roll-off factor of the RRC filter (0 < rolloff <= 1).
                span (int): Filter span in symbols (number of symbol durations the filter covers).
                sps (int): Samples per symbol (oversampling factor).
        """
        self.rolloff = float(config['modulation']['rrc_roll_off'])
        self.symbol_periode = 1 / float(config['modulation']['symbol_rate'])
        self.sps = int(config['modulation']['samples_per_symbol'])
        self.filter_span = int(config['modulation']['rrc_filter_span'])
        self.time_vector, self.coefficients = self._generate_rrc_filter()

    def _generate_rrc_filter(self) -> np.ndarray:
        """Generate RRC filter coefficients using commpy."""
        

        num_taps = self.filter_span * self.sps + 1
        sample_periode = self.symbol_periode / self.sps
        time_vector = np.arange(-num_taps//2, num_taps//2 + 1) * sample_periode

        h = np.zeros_like(time_vector)

        for i, t in enumerate(time_vector):
            if t == 0.0:
                h[i] = 1 + self.rolloff * (4 / np.pi - 1)
            
            elif np.abs(t) == self.symbol_periode / (4 * self.rolloff):
                h[i] = (
                        self.rolloff / np.sqrt(2)
                    ) * (
                        (1 + 2/np.pi) * np.sin(np.pi/(4*self.rolloff)) + 
                        (1 - 2/np.pi) * np.cos(np.pi/(4*self.rolloff))
                        )
            else:
                h[i] = (
                    np.sin(np.pi * (t / self.symbol_periode) * (1 - self.rolloff))  + 
                    4 * self.rolloff * t / self.symbol_periode * 
                    np.cos(np.pi * (t / self.symbol_periode) * (1 + self.rolloff))
                ) / (
                    np.pi * (t / self.symbol_periode) * (1 - (4 * self.rolloff * t / self.symbol_periode)**2)
                )
            
        h = h / self.symbol_periode  # Normalize filter coefficients by symbol period to ensure unit energy

        return time_vector, h

    def apply_filter(self, signal: np.ndarray) -> np.ndarray:
        """Apply RRC filter to the input signal."""
        return np.convolve(signal, self.coefficients, mode='same')
    
    def plot_filter_response(self):
        """Plot the impulse and frequency response of the RRC filter."""
        plotter = StaticSDRPlotter()
        plotter.plot_filter_response(self.coefficients, 
                                     self.time_vector, 
                                     sample_rate= self.sps / self.symbol_periode, 
                                     title="RRC Filter Response")
        
        from matplotlib.pyplot import show
        show()

if __name__ == "__main__":
    from yaml import safe_load

    try:
        with open("setup/config.yaml", 'r') as f:
            config = safe_load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        raise e

    # Example usage of RRCFilter
    rrc_filter = RRCFilter(config)
    rrc_filter.plot_filter_response()
