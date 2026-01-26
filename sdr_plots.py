"""
SDR Visualization Library
Plotting utilities for radio waveform analysis and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, Tuple


class SDRPlotter:
    """
    A comprehensive plotting library for SDR signal visualization.
    Supports time domain, frequency domain, constellation, and other common RF plots.
    """
    
    def __init__(self, style: str = 'dark_background'):
        """
        Initialize the SDR plotter with a matplotlib style.
        
        Args:
            style: Matplotlib style ('dark_background', 'seaborn', 'default', etc.)
        """
        self.style = style
        plt.style.use(style)
        
    def plot_time_domain(self, 
                        samples: np.ndarray, 
                        sample_rate: float,
                        title: str = "Time Domain Signal",
                        max_samples: int = 2048,
                        figsize: Tuple[int, int] = (12, 6)) -> Figure:
        """
        Plot I/Q samples in time domain.
        
        Args:
            samples: Complex IQ samples
            sample_rate: Sample rate in Hz
            title: Plot title
            max_samples: Maximum number of samples to display
            figsize: Figure size (width, height)
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Limit samples for performance
        samples = samples[:max_samples]
        time = np.arange(len(samples)) / sample_rate * 1e6  # Convert to microseconds
        
        # Plot I (Real) component
        ax1.plot(time, samples.real, linewidth=0.8, alpha=0.8)
        ax1.set_ylabel('I (In-phase)', fontsize=10)
        ax1.set_title(title, fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Plot Q (Imaginary) component
        ax2.plot(time, samples.imag, linewidth=0.8, alpha=0.8, color='orange')
        ax2.set_xlabel('Time (μs)', fontsize=10)
        ax2.set_ylabel('Q (Quadrature)', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_frequency_spectrum(self,
                               samples: np.ndarray,
                               sample_rate: float,
                               center_freq: float = 0,
                               title: str = "Frequency Spectrum",
                               nfft: int = 1024,
                               figsize: Tuple[int, int] = (12, 6)) -> Figure:
        """
        Plot frequency spectrum (FFT) of the signal.
        
        Args:
            samples: Complex IQ samples
            sample_rate: Sample rate in Hz
            center_freq: Center frequency in Hz (for display)
            title: Plot title
            nfft: FFT size
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Compute FFT
        fft_data = np.fft.fftshift(np.fft.fft(samples, n=nfft))
        fft_db = 20 * np.log10(np.abs(fft_data) + 1e-12)  # Convert to dB
        
        # Frequency axis
        freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1/sample_rate))
        if center_freq > 0:
            freqs = (freqs + center_freq) / 1e6  # Convert to MHz
            xlabel = 'Frequency (MHz)'
        else:
            freqs = freqs / 1e3  # Convert to kHz
            xlabel = 'Frequency (kHz)'
        
        ax.plot(freqs, fft_db, linewidth=1)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('Magnitude (dB)', fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_psd(self,
                samples: np.ndarray,
                sample_rate: float,
                center_freq: float = 0,
                title: str = "Power Spectral Density",
                nperseg: int = 1024,
                figsize: Tuple[int, int] = (12, 6)) -> Figure:
        """
        Plot Power Spectral Density using Welch's method.
        
        Args:
            samples: Complex IQ samples
            sample_rate: Sample rate in Hz
            center_freq: Center frequency in Hz
            title: Plot title
            nperseg: Length of each segment for Welch's method
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        from scipy import signal
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Compute PSD using Welch's method
        freqs, psd = signal.welch(samples, fs=sample_rate, nperseg=nperseg,
                                  return_onesided=False, scaling='density')
        freqs = np.fft.fftshift(freqs)
        psd = np.fft.fftshift(psd)
        psd_db = 10 * np.log10(psd + 1e-12)
        
        if center_freq > 0:
            freqs = (freqs + center_freq) / 1e6  # MHz
            xlabel = 'Frequency (MHz)'
        else:
            freqs = freqs / 1e3  # kHz
            xlabel = 'Frequency (kHz)'
        
        ax.plot(freqs, psd_db, linewidth=1)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('PSD (dB/Hz)', fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_constellation(self,
                          symbols: np.ndarray,
                          title: str = "Constellation Diagram",
                          figsize: Tuple[int, int] = (8, 8),
                          alpha: float = 0.5) -> Figure:
        """
        Plot constellation diagram of symbols.
        
        Args:
            symbols: Complex symbols to plot
            title: Plot title
            figsize: Figure size
            alpha: Point transparency
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.scatter(symbols.real, symbols.imag, alpha=alpha, s=2)
        ax.set_xlabel('In-phase', fontsize=10)
        ax.set_ylabel('Quadrature', fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Add reference circles
        max_val = max(np.abs(symbols.real).max(), np.abs(symbols.imag).max())
        ax.set_xlim(-max_val*1.2, max_val*1.2)
        ax.set_ylim(-max_val*1.2, max_val*1.2)
        
        plt.tight_layout()
        return fig
    
    def plot_spectrogram(self,
                        samples: np.ndarray,
                        sample_rate: float,
                        title: str = "Spectrogram",
                        nperseg: int = 256,
                        figsize: Tuple[int, int] = (12, 6)) -> Figure:
        """
        Plot spectrogram (time-frequency representation).
        
        Args:
            samples: Complex IQ samples
            sample_rate: Sample rate in Hz
            title: Plot title
            nperseg: Length of each segment
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        from scipy import signal
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Compute spectrogram
        freqs, times, Sxx = signal.spectrogram(samples, fs=sample_rate,
                                               nperseg=nperseg,
                                               return_onesided=False)
        
        # Convert to dB and shift
        Sxx_db = 10 * np.log10(np.fft.fftshift(Sxx, axes=0) + 1e-12)
        freqs = np.fft.fftshift(freqs) / 1e3  # Convert to kHz
        
        im = ax.pcolormesh(times * 1e3, freqs, Sxx_db, shading='auto', cmap='viridis')
        ax.set_xlabel('Time (ms)', fontsize=10)
        ax.set_ylabel('Frequency (kHz)', fontsize=10)
        ax.set_title(title, fontsize=12)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power (dB)', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_eye_diagram(self,
                        samples: np.ndarray,
                        samples_per_symbol: int,
                        title: str = "Eye Diagram",
                        num_traces: int = 100,
                        figsize: Tuple[int, int] = (10, 6)) -> Figure:
        """
        Plot eye diagram for symbol timing analysis.
        
        Args:
            samples: Real or complex samples
            samples_per_symbol: Number of samples per symbol
            title: Plot title
            num_traces: Number of symbol traces to overlay
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # If complex, plot both I and Q
        if np.iscomplexobj(samples):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
            self._plot_eye_single(samples.real, samples_per_symbol, ax1, 
                                 num_traces, "Eye Diagram - I")
            self._plot_eye_single(samples.imag, samples_per_symbol, ax2, 
                                 num_traces, "Eye Diagram - Q")
        else:
            self._plot_eye_single(samples, samples_per_symbol, ax, 
                                 num_traces, title)
        
        plt.tight_layout()
        return fig
    
    def _plot_eye_single(self, samples, samples_per_symbol, ax, num_traces, title):
        """Helper function to plot single eye diagram."""
        # Reshape into symbol periods
        num_symbols = len(samples) // samples_per_symbol
        num_traces = min(num_traces, num_symbols - 2)
        
        time = np.arange(2 * samples_per_symbol) / samples_per_symbol
        
        for i in range(num_traces):
            start = i * samples_per_symbol
            end = start + 2 * samples_per_symbol
            if end <= len(samples):
                ax.plot(time, samples[start:end], 'b-', alpha=0.1, linewidth=0.5)
        
        ax.set_xlabel('Symbol Period', fontsize=10)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 2])
    
    def plot_magnitude_phase(self,
                           samples: np.ndarray,
                           sample_rate: float,
                           title: str = "Magnitude and Phase",
                           max_samples: int = 2048,
                           figsize: Tuple[int, int] = (12, 6)) -> Figure:
        """
        Plot magnitude and phase of complex samples.
        
        Args:
            samples: Complex IQ samples
            sample_rate: Sample rate in Hz
            title: Plot title
            max_samples: Maximum samples to display
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        samples = samples[:max_samples]
        time = np.arange(len(samples)) / sample_rate * 1e6  # μs
        
        # Magnitude
        magnitude = np.abs(samples)
        ax1.plot(time, magnitude, linewidth=0.8)
        ax1.set_ylabel('Magnitude', fontsize=10)
        ax1.set_title(title, fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Phase
        phase = np.angle(samples)
        ax2.plot(time, phase, linewidth=0.8, color='orange')
        ax2.set_xlabel('Time (μs)', fontsize=10)
        ax2.set_ylabel('Phase (rad)', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_all(self,
                samples: np.ndarray,
                sample_rate: float,
                center_freq: float = 0,
                samples_per_symbol: Optional[int] = None) -> None:
        """
        Create a comprehensive dashboard with multiple plots.
        
        Args:
            samples: Complex IQ samples
            sample_rate: Sample rate in Hz
            center_freq: Center frequency in Hz
            samples_per_symbol: Samples per symbol (for eye diagram)
        """
        self.plot_time_domain(samples, sample_rate)
        self.plot_frequency_spectrum(samples, sample_rate, center_freq)
        self.plot_constellation(samples)
        self.plot_spectrogram(samples, sample_rate)
        
        if samples_per_symbol:
            self.plot_eye_diagram(samples, samples_per_symbol)
        
        plt.show()


# Convenience function for quick plotting
def quick_plot(samples: np.ndarray, 
              sample_rate: float,
              plot_type: str = 'time',
              **kwargs):
    """
    Quick plotting function for common visualizations.
    
    Args:
        samples: Complex IQ samples
        sample_rate: Sample rate in Hz
        plot_type: Type of plot ('time', 'fft', 'psd', 'constellation', 'spectrogram')
        **kwargs: Additional arguments passed to the plotting function
    """
    plotter = SDRPlotter()
    
    plot_map = {
        'time': plotter.plot_time_domain,
        'fft': plotter.plot_frequency_spectrum,
        'psd': plotter.plot_psd,
        'constellation': plotter.plot_constellation,
        'spectrogram': plotter.plot_spectrogram,
    }
    
    if plot_type in plot_map:
        fig = plot_map[plot_type](samples, sample_rate, **kwargs)
        plt.show()
        return fig
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")
