
import numpy as np
import logging

class RRCFilter:
    """Class to generate and manage Root Raised Cosine (RRC) filters."""
    
    def __init__(self, config: dict):
        """Initialize RRC filter parameters and generate coefficients.
            Args:
                rolloff (float): Roll-off factor of the RRC filter (0 < rolloff <= 1).
                span (int): Filter span in symbols (number of symbol durations the filter covers).
                sps (int): Samples per symbol (oversampling factor).
        """
        self.hardware_filter_enabled = bool(config['radio']['rrc_filter_enable'])
        self.rx_bandwidth = int(float(config['receiver']['rx_bandwidth']))
        self.tx_bandwidth = int(float(config['transmitter']['tx_bandwidth']))
        self.rolloff = float(config['modulation']['rrc_roll_off'])
        self.symbol_periode = 1 / float(config['modulation']['symbol_rate'])
        self.sps = int(config['modulation']['samples_per_symbol'])
        self.filter_span = int(config['modulation']['rrc_filter_span'])

        self.time_vector, self.coefficients = self._generate_rrc_filter()
    
        if self.hardware_filter_enabled:
            self.write_filter_to_file(config['radio']['rrc_filter'])


    def _generate_rrc_filter(self) -> np.ndarray:
        """Generate RRC filter coefficients using commpy."""
        

        num_taps = self.filter_span * self.sps 

        sample_periode = self.symbol_periode / self.sps

        # If hardware filtering is enabled, the total number of filer
        # coefficients must be divisible by 16
        if self.hardware_filter_enabled:
            time_vector = np.arange(-num_taps//2 + 0.5, num_taps//2 + 0.5) * sample_periode
        else:
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
            
        h = h / np.sqrt(np.sum(h**2))  # Normalize filter coefficients to unit energy

        return time_vector, h

    def apply_filter(self, signal: np.ndarray) -> np.ndarray:
        """Apply RRC filter to the input signal."""
        return np.convolve(signal, self.coefficients, mode='same')
        

    def write_filter_to_file(self, filename: str = "rrc_filter.ftr"):
        """Write RRC filter coefficients to a file."""
        try:
            with open(filename, 'w') as f:
                # write header
                f.write(f"TX 3 GAIN 0 INT 2\n")
                f.write(f"RX 3 GAIN 0 DEC 2\n")
                f.write(f"BWTX {self.tx_bandwidth}\n")
                f.write(f"BWRX {self.rx_bandwidth}\n")

                # write coefficients
                rrc_taps = self.coefficients * (2**15 - 1)  # Scale to 16-bit integer range
                for tap in rrc_taps:
                    t = int(round(tap))
                    f.write(f"{t},{t}\n") 
        
            logging.info(f"Filter coefficients successfully written to {filename}.")

        except Exception as e:
            logging.error(f"Error writing filter coefficients to file: {e}")
            raise e
        
    

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

    from scipy import signal
    rc = signal.convolve(rrc_filter.coefficients, rrc_filter.coefficients, mode='same')
    
    from sdr_plots import StaticSDRPlotter
    plotter = StaticSDRPlotter()

    plotter.plot_filter_response(rrc_filter.coefficients, 
                                 rrc_filter.time_vector, 
                                 sample_rate= rrc_filter.sps / rrc_filter.symbol_periode, 
                                 title="Root Raised Cosine Filter Response")

    plotter.plot_filter_response(rc, 
                                 rrc_filter.time_vector, 
                                 sample_rate= rrc_filter.sps / rrc_filter.symbol_periode, 
                                 title="Raised Cosine Filter Response")
    
    print(f"RRC length:\t{len(rrc_filter.coefficients)}\nRRC energy:\t{np.sum(rrc_filter.coefficients**2):.4f}")
    print(f"RC length:\t{len(rc)}\nRC energy:\t{np.sum(rc**2):.4f}")
    print(f"Time vector:\n{rrc_filter.time_vector}")

    if rrc_filter.hardware_filter_enabled:
        rx_generated_filter = []
        tx_generated_filter = []
        try:
            with open ("rrc_filter_coefficients.ftr", 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("BWTX"):
                        tx_bandwidth = int(line.split()[1])
                    elif line.startswith("BWRX"):
                        rx_bandwidth = int(line.split()[1])
                    elif line[0].isdigit() or line[0] == '-':
                        tx_coeff, rx_coeff = line.strip().split(',')
                        tx_generated_filter.append(int(tx_coeff))
                        rx_generated_filter.append(int(rx_coeff))
        
        except Exception as e:
            print(f"Error reading generated filter coefficients: {e}")
            raise e
        
        rc_generated_filter = signal.convolve(tx_generated_filter, rx_generated_filter, mode='same')
        
        plotter.plot_filter_response(np.array(tx_generated_filter), 
                                    rrc_filter.time_vector, 
                                    sample_rate= rrc_filter.sps / rrc_filter.symbol_periode, 
                                    title="Generated RRC Filter Response (TX)")
        
        plotter.plot_filter_response(np.array(rx_generated_filter),
                                        rrc_filter.time_vector, 
                                        sample_rate= rrc_filter.sps / rrc_filter.symbol_periode, 
                                        title="Generated RRC Filter Response (RX)")
        
        plotter.plot_filter_response(rc_generated_filter,
                                        rrc_filter.time_vector, 
                                        sample_rate= rrc_filter.sps / rrc_filter.symbol_periode, 
                                        title="Generated Raised Cosine Filter Response (TX*RX)")
        

    from matplotlib.pyplot import show
    show()


