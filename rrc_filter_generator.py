import commpy
from yaml import safe_load
from matplotlib.pyplot import plot, show

try: 
    with open("config.yaml", 'r') as f:
        config = safe_load(f)
except Exception as e:
    print(f"Error loading config file: {e}")
    raise e

def rrc_filter(config_file: str ="config.yaml"):
    """Generate Root Raised Cosine (RRC) filter based on configuration."""


    rolloff = float(config['modulation']['rrc_roll_off'])
    span = int(config['modulation']['rrc_filter_span'])
    sample_rate = float(config['modulation']['sample_rate'])
    sps = int(config['modulation']['samples_per_symbol'])

    t, rrc_taps = commpy.filters.rrcosfilter(N=span * sps, alpha=rolloff, Ts=sps/sample_rate, Fs=sample_rate)
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
    rrc_taps = rrc_filter()[1]
    write_filter_to_file(rrc_taps)