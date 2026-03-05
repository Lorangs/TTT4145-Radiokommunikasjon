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

from sdr_plots import StaticSDRPlotter
from matplotlib.pyplot import show



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


    if sdr.connect() == False:
        print("Failed to connect to SDR. Exiting.")
        exit(1)

    noise_floor = sdr.sdr.rx()
    fft_noise = np.fft.fftshift(np.fft.fft(noise_floor, n = len(noise_floor))) / len(noise_floor)
    mag_noise = np.abs(fft_noise)**2
    sorted_noise = np.sort(mag_noise)
    noise_bins = sorted_noise[:int(0.8*len(sorted_noise))]  # Take the lower 80% of the noise bins to avoid outliers
    noise_floor_dB = 10 * np.log10(np.mean(noise_bins))

    print(f"Measured noise floor: {noise_floor_dB:.2f} dB")
    

    test_message = "Hello, SDR!"
    msg_id = 101
    datagram = Datagram(msg_id=msg_id, payload=np.frombuffer(test_message.encode('utf-8'), dtype=np.uint8), msg_type=msgType.DATA)
    print(F"Original message:\t{test_message}")
    print(f"Datagram:\t{datagram}")

    modulated_message = modulator.modulate_message(datagram)
    modulated_message = np.concatenate([BARKER_SYMBOLS[config['modulation']['type'].upper()][config['barker_sequence']['code_length']], modulated_message])  # Prepend Barker code for synchronization
    upsampled_message = np.zeros(len(modulated_message) * sps, dtype=np.complex64)
    upsampled_message[::sps] = modulated_message  # Upsample by inserting zeros between symbols

    modulated_message = rrc_filter.apply_filter(upsampled_message)
    
    upsampled_message *= 32768 # Scale to int16 range for Adalm Pluto.

    sdr.sdr.tx(upsampled_message)

    for i in range(10):
        sdr.sdr.rx()    # Flush the RX buffer to ensure we get the most recent transmission

    received_signal = sdr.sdr.rx()

    received_signal -= np.mean(received_signal)  # Remove DC offset

    plotter.plot_psd(
        received_signal, 
        title="Received Signal PSD", 
        center_freq=0, 
        sample_rate=int(float(config['modulation']['sample_rate']))
    )

    received_fft = np.fft.fftshift(np.fft.fft(received_signal, n = len(received_signal))) / len(received_signal)
    received_mag = np.abs(received_fft)**2
    received_signal_strength_dB = 10 * np.log10(np.max(received_mag))

    plotter.plot_constellation(received_signal, title="Received Signal Constellation")
 
    print(f"Received signal strength: {received_signal_strength_dB:.2f} dB")

    coarse_freq_adjusted = synchronizer.coarse_frequenzy_synchronization(received_signal)
 
    filtered_signal = rrc_filter.apply_filter(coarse_freq_adjusted)

    plotter.plot_psd(
        filtered_signal, 
        title="Filtered Signal PSD", 
        center_freq=0, 
        sample_rate=int(float(config['modulation']['sample_rate']))
    
    )
    plotter.plot_constellation(filtered_signal, title="Constellation of Filtered Signal")
    plotter.plot_eye_diagram(filtered_signal, title="Eye Diagram of Downsampled Signal", samples_per_symbol=sps, num_traces=100, symbols_per_trace=2)

    delay = synchronizer.time_synchronization(filtered_signal)

    plotter.plot_time_domain(filtered_signal, title="Time Domain of Filtered Signal", sample_rate=int(float(config['modulation']['sample_rate'])), max_samples=32678)

    if delay is None:
        print("Failed to detect Barker code. Exiting.")
        
    downsampled_signal = filtered_signal[delay::sps]

    print(f"Timing offset (samples): {delay}")
    plotter.plot_constellation(downsampled_signal, title="Constellation of Downsampled Signal")

    fine_freq_adjusted = synchronizer.fine_frequenzy_synchronization(downsampled_signal)
    plotter.plot_constellation(fine_freq_adjusted, title="Constellation of Fine Frequency Adjusted Signal")

    #demodulated_message = modulator.demodulate_message(fine_freq_adjusted)
    #print(f"Demodulated datagram:\t{demodulated_message}")

    show()


    