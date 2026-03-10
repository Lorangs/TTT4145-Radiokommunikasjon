"""
Simple tester for the SDR transceiver system. This script utilizes the 
SDR Adalm Pluto Cyclic TX buffer feature to transmit a known test message repeatedly, 
while the receiver captures the signal and plots its characteristics.

Perfect for testing end to end functionality of each module.
"""


from yaml import safe_load
import numpy as np
from sdr_transciever import SDRTransciever
from filter import RRCFilter
from barker_code import BARKER_SYMBOLS
from modulation import ModulationProtocol
from datagram import Datagram, msgType
from synchronize import Synchronizer

from scipy import signal
from time import sleep
from sdr_plots import StaticSDRPlotter
import matplotlib.pyplot as plt



if __name__ == "__main__":


    try:
        with open("setup/config.yaml", 'r') as f:
            config = safe_load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        exit(1)

    plotter = StaticSDRPlotter()

    sdr  = SDRTransciever(config)
    rrc_filter = RRCFilter(config)
    modulator = ModulationProtocol(config)
    synchronizer = Synchronizer(config)
    sps = int(config['modulation']['samples_per_symbol'])
    sample_rate = int(float(config['modulation']['sample_rate']))


    if sdr.connect() == False:
        print("Failed to connect to SDR. Exiting.")
        exit(1)


    test_1 = np.ones(64, dtype=np.int8)  + 1j*np.ones(64, dtype=np.int8)  # Example test message (128 bytes of value 1)
    test_2 = -np.ones(64, dtype=np.int8) - 1j*np.ones(64, dtype=np.int8)  # Example test message (128 bytes of value 1)
    modulated_message = np.concatenate([test_1, test_2])  # Combine test messages to create a longer message

    print(modulated_message)
    #modulated_message = np.concatenate([BARKER_SYMBOLS[config['modulation']['type'].upper()][config['barker_sequence']['code_length']], modulated_message])  # Prepend Barker code for synchronization

    upsampled_message = np.zeros(len(modulated_message) * sps, dtype=np.complex64)
    upsampled_message[::sps] = modulated_message  # Upsample by inserting zeros between symbols

    # zero pad to ensure we have enough samples for the filter to settle
    pad = np.zeros(len(rrc_filter.coefficients), dtype=np.complex64)  
    modulated_message = np.concatenate([pad, upsampled_message, pad])
    
    modulated_message = rrc_filter.apply_filter(modulated_message) 
    modulated_message *= 2**14 # Scale to int16 range for Adalm Pluto.

    print(f"Upsampled and filtered message length: {len(modulated_message)} samples")

    sdr.sdr.tx(modulated_message)
    
    for _ in range(10):
        sdr.sdr.rx()


    received_signal = sdr.sdr.rx()
    received_signal /= np.sqrt(np.mean(np.abs(received_signal)**2))  # Normalize received signal power

    print(f"len received signal: {len(received_signal)} samples")

    coarse_corrected_signal = synchronizer.coarse_frequenzy_synchronization(received_signal)

    filtered_signal = rrc_filter.apply_filter(coarse_corrected_signal) 

    plotter.plot_time_domain(filtered_signal, sample_rate, title="Time Domain of Filtered Signal Before Synchronization")

    time_synchronized_signal = synchronizer.gardner_timing_synchronization(filtered_signal)

    fine_corrected_signal = synchronizer.fine_frequenzy_synchronization(time_synchronized_signal)

    print(f"len time synchronized signal: {len(time_synchronized_signal)}")

    plotter.plot_constellation(fine_corrected_signal, title="Constellation after Costas Loop")
    plotter.plot_constellation(time_synchronized_signal, title="Constellation after Gardner Timing Synchronization")
    plotter.plot_constellation(coarse_corrected_signal, title="Constellation after Coarse Frequency Synchronization")
    plotter.plot_eye_diagram(time_synchronized_signal, sps, title="Eye Diagram after Gardner Timing Synchronization")
    plotter.plot_constellation(received_signal, title="Constellation of Received Signal (No Synchronization)")
    plotter.plot_psd(filtered_signal, sample_rate, title="PSD of Received Signal")

   
    plt.show()

    sdr.sdr.tx_destroy_buffer() # Destroy the TX buffer to stop transmission after one message
                           



    