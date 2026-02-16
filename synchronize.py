import numpy as np
from scipy import signal


class Synchronizer:
    def __init__(self, config: dict):
        self.modulation_scheme = config['modulation']['type'].upper().strip()
        self.sps = int(config['modulation']['samples_per_symbol'])
        self.buffer_size = int(config['receiver']['buffer_size'])
        self.mm_reactance_factor = float(config['synchronization']['mm_reactance_factor'])
        self.interpolation_factor = int(config['synchronization']['interpolation_factor'])
    
    def coarse_time_synchronization(self, received_signal: np.ndarray) -> np.ndarray:
        """Mueller and Muller (M&M) timing error detector for coarse time synchronization."""
        if self.modulation_scheme == 'QPSK':
            return self._bpsk_coarse_time_sync(received_signal)
        elif self.modulation_scheme == 'BPSK':
            return self._qpsk_coarse_time_sync(received_signal)
        else:
            raise ValueError(f"Unsupported modulation scheme: {self.modulation_scheme}")
    
    def _bpsk_coarse_time_sync(self, received_signal: np.ndarray) -> np.ndarray:
        """M&M (Mueller & Muller) timing error detector for BPSK."""
        mu = 0 # initial estimate of phase of sample
        out = np.zeros(self.buffer_size + 10, dtype=np.complex64)
        out_rail = np.zeros(self.buffer_size + 10, dtype=np.complex64) # to hold the "railed" version of the output (i.e., only the sign of the real and imaginary parts)
        i_in = 0 # input samples index
        i_out = 2 # output index (let first two outputs be 0)

        interpolated_signal = signal.resample_poly(received_signal, self.interpolation_factor, 1) # interpolate the input signal to increase timing resolution

        while i_out < self.buffer_size and i_in+16 < self.buffer_size:
            out[i_out] = interpolated_signal[i_in * self.interpolation_factor + int(mu * self.interpolation_factor)] # grab what we think is the "best" sample
            out_rail[i_out] = int(np.real(out[i_out]) > 0) + 1j*int(np.imag(out[i_out]) > 0)
            x = (out_rail[i_out] - out_rail[i_out-2]) * np.conj(out[i_out-1])
            y = (out[i_out] - out[i_out-2]) * np.conj(out_rail[i_out-1])
            mm_val = np.real(y - x)
            mu += self.sps + self.mm_reactance_factor*mm_val
            i_in += int(np.floor(mu)) # round down to nearest int since we are using it as an index
            mu = mu - np.floor(mu) # remove the integer part of mu
            i_out += 1 # increment output index

        return out[2:i_out] # remove the first two, and anything after i_out (that was never filled out)
        
    
    def _qpsk_coarse_time_sync(self, received_signal: np.ndarray) -> np.ndarray:
        """M&M (Mueller & Muller) timing error detector for QPSK."""
        # For QPSK, the timing error can be estimated using the product of the real and imaginary parts
        mu = 0 # initial estimate of phase of sample
        out = np.zeros(self.buffer_size + 10, dtype=np.complex64)
        out_rail = np.zeros(self.buffer_size + 10, dtype=np.complex64) # to hold the "railed" version of the output (i.e., only the sign of the real and imaginary parts)
        i_in = 0 # input samples index
        i_out = 2 # output index (let first two outputs be 0)

        interpolated_signal = signal.resample_poly(received_signal, self.interpolation_factor, 1) # interpolate the input signal to increase timing resolution
        while i_out < self.buffer_size and i_in+16 < self.buffer_size:
            out[i_out] = interpolated_signal[i_in * self.interpolation_factor + int(mu * self.interpolation_factor)] # grab what we think is the "best" sample
            out_rail[i_out] = int(np.real(out[i_out]) > 0) + 1j*int(np.imag(out[i_out]) > 0)
            x = (out_rail[i_out] - out_rail[i_out-2]) * np.conj(out[i_out-1])
            y = (out[i_out] - out[i_out-2]) * np.conj(out_rail[i_out-1])
            mm_val = np.real(y - x)
            mu += self.sps + self.mm_reactance_factor*mm_val
            i_in += int(np.floor(mu)) # round down to nearest int since we are using it as an index
            mu = mu - np.floor(mu) # remove the integer part of mu
            i_out += 1 # increment output index        

        return out[2:i_out] # remove the first two, and anything after i_out (that was never filled out)
        


if __name__ == "__main__":
    from yaml import safe_load

    try:
        with open("setup/config.yaml", 'r') as f:
            config = safe_load(f)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        exit(1)

    synchronizer = Synchronizer(config)
    # Example usage with a test signal
  