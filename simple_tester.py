from yaml import safe_load
import numpy as np
from sdr_transciever import SDRTransciever
from filter import RRCFilter
from barker_code import BARKER_SYMBOLS
from barker_detection import BarkerDetector
from modulation import ModulationProtocol
from datagram import Datagram, msgType

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
    barker_detector = BarkerDetector(config)
    modulator = ModulationProtocol(config)


    if sdr.connect() == False:
        print("Failed to connect to SDR. Exiting.")
        exit(1)

    test_message = "Hello, SDR!"
    msg_id = 101
    datagram = Datagram(msg_id=msg_id, payload=np.frombuffer(test_message.encode('utf-8'), dtype=np.uint8), msg_type=msgType.DATA)
    print(F"Original message:\t{test_message}")
    print(f"Datagram:\t{datagram}")

    modulated_message = modulator.modulate_message(datagram)
    modulated_message = barker_detector.add_barker_code(modulated_message)
    modulated_message = rrc_filter.apply_filter(modulated_message)

    modulated_message *= 2**14 # Scale to int16 range for Adalm Pluto.

    sdr.sdr.tx(modulated_message)

    for i in range(10):
        sdr.sdr.rx()

    received_signal = sdr.sdr.rx()

    filtered_signal = rrc_filter.apply_filter(received_signal)

    plotter.plot_time_domain(filtered_signal, sample_rate=float(config['modulation']['sample_rate']), title="Received Signal (Time Domain)")
    plotter.plot_constellation(filtered_signal, title="Received Signal (Constellation)")
    plotter.plot_psd(filtered_signal, title="Received Signal (PSD)", center_freq=sdr.sdr.rx_lo, sample_rate=float(config['modulation']['sample_rate']))
    plotter.plot_eye_diagram(filtered_signal, title="Received Signal (Eye Diagram)", samples_per_symbol=rrc_filter.sps)
    show()


    