"""
Demo script for testing SDR plotting library
Generates synthetic signals to demonstrate all visualization capabilities
"""

import numpy as np
import matplotlib.pyplot as plt
from sdr_plots import SDRPlotter, quick_plot


def generate_test_signal(sample_rate=2e6, symbol_rate=250e3, duration=0.01, noise_std=0.1, signal_type='qpsk'):
    """
    Generate test signals for demonstration.
    
    Args:
        sample_rate: Sample rate in Hz
        duration: Signal duration in seconds
        noise_std: Standard deviation of noise
        signal_type: 'qpsk', 'bpsk', 'noise', 'tone'
        
    Returns:
        Complex IQ samples
    """
    num_samples = int(sample_rate * duration)
    t = np.arange(num_samples) / sample_rate
    
    if signal_type == 'qpsk':
        # Generate QPSK signal
        samples_per_symbol = int(sample_rate / symbol_rate)
        num_symbols = num_samples // samples_per_symbol
        
        # Random QPSK symbols
        symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], size=num_symbols)
        symbols /= np.sqrt(2)  # Normalize
        
        # Upsample and pulse shape (simple rectangular)
        signal = np.repeat(symbols, samples_per_symbol)[:num_samples]
        
        # Add some noise
        noise = (np.random.normal(0, noise_std,num_samples) + 1j*np.random.normal(0, noise_std, num_samples)) / np.sqrt(2) # Normalize noise power
        signal += noise
        
    elif signal_type == 'bpsk':
        # Generate BPSK signal
        samples_per_symbol = int(sample_rate / symbol_rate)
        num_symbols = num_samples // samples_per_symbol
        
        # Random BPSK symbols
        symbols = np.random.choice([1, -1], size=num_symbols)
        signal = np.repeat(symbols, samples_per_symbol)[:num_samples]
        
        # Add carrier and noise
        carrier_freq = 100e3
        signal = signal * np.exp(2j * np.pi * carrier_freq * t)
        noise = (np.random.normal(0, noise_std,num_samples) + 1j*np.random.normal(0, noise_std, num_samples)) / np.sqrt(2) # Normalize noise power
        signal += noise
        
    elif signal_type == 'tone':
        # Generate single tone
        freq = 100e3  # 100 kHz offset
        signal = np.exp(2j * np.pi * freq * t)
        
    elif signal_type == 'noise':
        # White noise
        signal = (np.random.normal(0, noise_std,num_samples) + 1j*np.random.normal(0, noise_std, num_samples)) / np.sqrt(2) # Normalize noise power
    
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")
    
    return signal.astype(np.complex64)


def demo_basic_plots(noise_std=0.1, signal_type='qpsk'):
    """Demonstrate basic plotting functions."""
    print('\n'+"=" * 60)
    print("Demo 1: Basic Time and Frequency Domain Plots")
    print("=" * 60)
    
    # Generate test signal
    sample_rate = 2e6
    signal = generate_test_signal(sample_rate=sample_rate, signal_type=signal_type, noise_std=noise_std)
    
    # Create plotter
    plotter = SDRPlotter(style='dark_background')
    
    # Time domain
    print("Plotting time domain...")
    plotter.plot_time_domain(signal, sample_rate)
    
    # Frequency spectrum
    print("Plotting frequency spectrum...")
    plotter.plot_frequency_spectrum(signal, sample_rate)
    
    plt.show()


def demo_advanced_plots(noise_std=0.1, signal_type='qpsk'):
    """Demonstrate advanced plotting functions."""
    print('\n'+"=" * 60)
    print("Demo 2: Advanced Plots (Constellation, PSD, Spectrogram)")
    print("=" * 60)
    
    sample_rate = 2e6
    signal = generate_test_signal(sample_rate=sample_rate, signal_type=signal_type, duration=0.02, noise_std=noise_std)
    
    plotter = SDRPlotter(style='dark_background')
    
    # Constellation diagram
    print("Plotting constellation...")
    plotter.plot_constellation(signal[::8])  # Decimate for better visualization
    
    # Power Spectral Density
    print("Plotting PSD...")
    plotter.plot_psd(signal, sample_rate)
    
    # Spectrogram
    print("Plotting spectrogram...")
    plotter.plot_spectrogram(signal, sample_rate)
    
    plt.show()


def demo_eye_diagram(signal_type='bpsk', noise_std=0.1):
    """Demonstrate eye diagram."""
    print('\n'+"=" * 60)
    print("Demo 3: Eye Diagram")
    print("=" * 60)
    
    sample_rate = 2e6
    signal = generate_test_signal(sample_rate=sample_rate, signal_type=signal_type, duration=0.01, noise_std=noise_std)
    
    plotter = SDRPlotter(style='dark_background')
    
    # Eye diagram
    symbol_rate = 250e3
    samples_per_symbol = int(sample_rate / symbol_rate)
    print(f"Samples per symbol: {samples_per_symbol}")
    
    plotter.plot_eye_diagram(signal, samples_per_symbol)
    
    plt.show()


def demo_quick_plot():
    """Demonstrate quick plot convenience function."""
    print("=" * 60)
    print("Demo 4: Quick Plot Function")
    print("=" * 60)
    
    sample_rate = 2e6
    signal = generate_test_signal(sample_rate=sample_rate, signal_type='tone')
    
    # Quick FFT plot
    print("Quick FFT plot...")
    quick_plot(signal, sample_rate, plot_type='fft', title="Single Tone Signal")
    

def demo_all_plots(signal_type='qpsk', noise_std=0.1):
    """Show all plots together."""
    print('\n'+"=" * 60)
    print("Demo 5: All Plots Dashboard")
    print("=" * 60)
    
    sample_rate = 2e6
    signal = generate_test_signal(sample_rate=sample_rate, signal_type=signal_type, duration=0.02, noise_std=noise_std)
    
    plotter = SDRPlotter(style='dark_background')
    
    symbol_rate = 250e3
    samples_per_symbol = int(sample_rate / symbol_rate)
    
    plotter.plot_all(signal, sample_rate, samples_per_symbol=samples_per_symbol)


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("SDR Plotting Library Demo")
    print("="*60 + "\n")
    
    signal_type = {
        1:  'qpsk',
        2:  'bpsk',
        3:  'noise',
        4:  'tone'
    }

    demos = {
        1: ('Basic Plots (Time & Frequency)', demo_basic_plots),
        2: ('Advanced Plots (Constellation, PSD, Spectrogram)', demo_advanced_plots),
        3: ('Eye Diagram', demo_eye_diagram),
        4: ('Quick Plot', demo_quick_plot),
        5: ('All Plots Dashboard', demo_all_plots),
        6: ('Run All Demos', None)
    }

    print("Select signal type for demos:")
    for key, val in signal_type.items():
        print(f"  {key}: {val}")

    signal_choice = input("Select signal type for demos [default: 1]: ").strip()
    try:
        signal_choice = int(signal_choice)
    except ValueError:
        signal_choice = 1

    if signal_choice not in [1, 2, 3, 4]:
        print("Invalid choice. Using default: qpsk.")
        signal_choice = 1

    signal_type = signal_type.get(signal_choice, 'qpsk')

    noise_std = input("Enter noise standard deviation [default: 0.1]: ").strip()
    if not noise_std:
        print("Invalid choice. Using default noise std: 0.1")
        noise_std = 0.1
    else:
        try:
            noise_std = float(noise_std)
        except ValueError:
            print("Invalid input for noise standard deviation. Using default 0.1.")
            noise_std = 0.1


    print("Available demos:")
    for key, (desc, _) in demos.items():
        print(f"  {key}: {desc}")
    
    choice = input("\nEnter demo number: ").strip()
    try:
        choice = int(choice)
    except ValueError:
        choice = None
    
    if choice == 6:
        for key in [1, 2, 3, 4, 5]:
            demos[key][1](noise_std=noise_std, signal_type=signal_type)
            input("\nPress Enter for next demo...")
    elif choice in demos:
        demos[choice][1](noise_std=noise_std, signal_type=signal_type)
    else:
        print("Invalid choice. Running demo 1...")
        demo_basic_plots(noise_std=noise_std, signal_type=signal_type)


if __name__ == "__main__":
    main()
