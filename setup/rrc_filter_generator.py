"""
This module generates a Root Raised Cosine (RRC) filter based on the configuration specified in the config.yaml file. 
The generated filter coefficients are then written to a file in a format compatible with the SDR transceiver.
"""


import commpy
from yaml import safe_load
from matplotlib.pyplot import plot, show, stem
import numpy as np

import adi
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from sdr_plots import StaticSDRPlotter


try: 
    with open("setup/config.yaml", 'r') as f:
        config = safe_load(f)
except Exception as e:
    print(f"Error loading config file: {e}")
    raise e


def try_filter_in_sdr():
    """Test if the generated filter can be loaded into the SDR."""
    try:
        sdr = adi.Pluto(config['radio']['ip_address'])

    # Configure TX
        sdr.tx_rf_bandwidth = int(float(config['transmitter']['tx_bandwidth']))
        sdr.tx_lo = int(float(config['transmitter']['tx_carrier']))
        sdr.tx_hardwaregain_chan0 = int(float(config['transmitter']['tx_gain_dB']))
        sdr.sample_rate = int(float(config['modulation']['sample_rate']))
        sdr.tx_cyclic_buffer = bool(config['transmitter']['tx_cyclic_buffer'])

        # Configure RX

        sdr.rx_rf_bandwidth = int(float(config['receiver']['rx_bandwidth']))
        sdr.rx_lo = int(float(config['receiver']['rx_carrier']))
        sdr.rx_buffer_size = int(config['receiver']['buffer_size'])
        sdr.gain_control_mode_chan0 = config['receiver']['gain_mode']
        if config['receiver']['gain_mode'] == 'manual':
            sdr.rx_hardwaregain_chan0 = float(config['receiver']['rx_gain_dB'])

        # set TX and RX filter
        sdr.filter = config['radio']['rrc_filter']

        print(len(sdr.rx()))

    except Exception as e:
        print(f"Error connecting to SDR: {e}")
        exit(1)

def rrc_filter(config_file: str ="config.yaml") -> tuple[np.ndarray, np.ndarray]:
    """Generate Root Raised Cosine (RRC) filter based on configuration."""

    sdr = config['radio']['sdr']

    rolloff = float(config['modulation']['rrc_roll_off'])
    span = int(config['modulation']['rrc_filter_span'])
    sample_rate = float(config['modulation']['sample_rate'])
    sps = int(config['modulation']['samples_per_symbol'])
    symbol_rate = float(config['modulation']['symbol_rate'])

    t, rrc_taps = commpy.filters.rrcosfilter(N=span * sps, alpha=rolloff, Ts=1/symbol_rate, Fs=sample_rate)
    return t, rrc_taps 




def write_filter_to_file(rrc_taps, filename: str = "rrc_filter.ftr"):
    """Write RRC filter coefficients to a file."""
    try:
        with open(filename, 'w') as f:
            # write header
            f.write(f"TX 3 GAIN 0 INT 2\n")
            f.write(f"RX 3 GAIN 0 DEC 2\n")
            f.write(f"BWTX {int(float(config['transmitter']['tx_bandwidth']))}\n")
            f.write(f"BWRX {int(float(config['receiver']['rx_bandwidth']))}\n")


            # write coefficients
            rrc_taps *= 32767  # Scale to 16-bit integer range
            for tap in rrc_taps:
                t = int(round(tap))
                f.write(f"{t},{t}\n") 
       
        print(f"Filter coefficients successfully written to {filename}.")

    except Exception as e:
        print(f"Error writing filter coefficients to file: {e}")
        raise e
    
if __name__ == "__main__":
    try_filter_in_sdr()