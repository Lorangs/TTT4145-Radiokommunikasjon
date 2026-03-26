from commpy import modulation
import numpy as np

class ModulationProtocol:
    def __init__(self, config: dict):
        modulation_config = config['modulation']
        modulation_type = str(modulation_config['type']).upper().strip()
        modulation_order = int(modulation_config['order'])
        
        
        if modulation_type == "PSK":
            self.modulator = modulation.PSKModem(modulation_order)

        elif modulation_type == "QAM":
            self.modulator = modulation.QAMModem(modulation_order)
        else:
            raise ValueError(f"Unsupported modulation type: {modulation_type}")

    def modulate_message(self, bit_stream: np.ndarray) -> np.ndarray:
        """Modulate a bit stream into a complex baseband signal."""
        return self.modulator.modulate(bit_stream)
    
    def demodulate_signal(self, signal: np.ndarray) -> np.ndarray:
        """Demodulate a complex baseband signal back into a bit stream."""
        return self.modulator.demodulate(signal, demod_type='hard')
    
if __name__ == "__main__":
    from sdr_plots import StaticSDRPlotter

    plotter = StaticSDRPlotter()

    modulation_config = {
        'modulation': {
            'type': 'PSK',
            'order': 8      # QPSK modulation
        }
    }
    protocol = ModulationProtocol(modulation_config)

    test_bits = np.random.randint(0, 2, 99)  # Generate a random bit stream of length 100
    modulated_signal = protocol.modulate_message(test_bits)
    print("Modulated signal:")
    print(modulated_signal)

    plotter.plot_constellation(modulated_signal, title="Modulated Signal Constellation")

    demodulated_bits = protocol.demodulate_signal(modulated_signal)

    if np.all(test_bits == demodulated_bits):
        print("Demodulation successful, bits match!")
    else:        
        print("Demodulation failed, bits do not match.")

    from matplotlib import pyplot as plt
    plt.show()