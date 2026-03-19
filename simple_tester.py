"""
Simple tester for the SDR transceiver system. This script utilizes the 
SDR Adalm Pluto Cyclic TX buffer feature to transmit a known test message repeatedly, 
while the receiver captures the signal and plots its characteristics.

Perfect for testing end to end functionality of each module.
"""


import numpy as np
import matplotlib.pyplot as plt
from yaml import safe_load

from datagram import Datagram, msgType
from filter import RRCFilter, BWLPFilter
from modulation import ModulationProtocol
from sdr_transciever import SDRTransciever
from synchronize import Synchronizer
from sdr_plots import StaticSDRPlotter



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
    frontend_filter = BWLPFilter(config)
    synchronizer = Synchronizer(config)
    sps = int(config['modulation']['samples_per_symbol'])
    sample_rate = int(float(config['modulation']['sample_rate']))



    if sdr.connect() == False:
        print("Failed to connect to SDR. Exiting.")
        exit(1)

    synchronizer.set_noise_floor(sdr.measure_noise_floor_dB())  # Measure noise floor and set it in the synchronizer for adaptive thresholding

    payload_text = "Scrambler test payload " * 4
    datagram = Datagram.as_string(payload_text, msg_type=msgType.DATA)
    tx_bits = modulator.pack_message_bits(datagram)
    ones = int(np.count_nonzero(tx_bits))
    zeros = int(tx_bits.size - ones)
    print(f"Scrambled bit balance: zeros={zeros}, ones={ones}")

    modulated_message = modulator.modulate_message(datagram).astype(np.complex64)
    upsampled_message = np.zeros(len(modulated_message) * sps, dtype=np.complex64)
    upsampled_message[::sps] = modulated_message

    # zero pad to ensure we have enough samples for the filter to settle
    print(f"len filter taps: {len(rrc_filter.coefficients)}")
    pad = np.zeros(len(rrc_filter.coefficients) - 1, dtype=np.complex64)  
    modulated_message = np.concatenate([pad, upsampled_message, pad])
    
    modulated_message = rrc_filter.apply_filter(modulated_message) 
    modulated_message *= 2**14 # Scale to int16 range for Adalm Pluto.

    print(f"Upsampled and filtered message length: {len(modulated_message)} samples")

    sdr.sdr.tx(modulated_message)
    
    for _ in range(10):
        sdr.sdr.rx()


    received_signal = sdr.sdr.rx()

    print(f"len received signal: {len(received_signal)} samples")

    coarse_corrected_signal = synchronizer.coarse_frequenzy_synchronization(received_signal)
    frontend_filtered_signal = frontend_filter.apply_filter(coarse_corrected_signal)
    filtered_signal = rrc_filter.apply_filter(frontend_filtered_signal)

    plotter.plot_time_domain(filtered_signal, sample_rate, title="Time Domain of Filtered Signal Before Synchronization")

    time_synchronized_signal = synchronizer.gardner_timing_synchronization(filtered_signal)

    fine_corrected_signal = synchronizer.fine_frequenzy_synchronization(time_synchronized_signal)

    print(f"len time synchronized signal: {len(time_synchronized_signal)}")

    plotter.plot_constellation(fine_corrected_signal, title="Constellation after Costas Loop")
    plotter.plot_constellation(time_synchronized_signal, title="Constellation after Gardner Timing Synchronization")
    plotter.plot_constellation(coarse_corrected_signal, title="Constellation after Coarse Frequency Synchronization")
    plotter.plot_eye_diagram(time_synchronized_signal, sps, title="Eye Diagram after Gardner Timing Synchronization")
    plotter.plot_constellation(filtered_signal, title="Constellation of Received Signal (No Synchronization)")
    plotter.plot_psd(received_signal, sample_rate, title="PSD of Received Signal")

    plt.show()

    sdr.sdr.tx_destroy_buffer() # Destroy the TX buffer to stop transmission after one message
                           
