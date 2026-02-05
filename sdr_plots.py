"""
SDR Visualization Library
Plotting utilities for radio waveform analysis and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, Tuple
from scipy import signal
import sys
import warnings


class StaticSDRPlotter:
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
        try:
            plt.style.use(style)
        except Exception as e:
            warnings.warn(f"Could not load style '{style}', using default. Error: {e}")
            plt.style.use('default')

    def __del__(self):
        """Destructor to clean up resources."""
        plt.close('all')
    

    def plot_time_domain(self, 
                        samples: np.ndarray, 
                        sample_rate: float,
                        title: str = "Time Domain Signal",
                        max_samples: int = 2048,
                        figsize: Tuple[int, int] = (12, 6)) -> Optional[Figure]:
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
        try:
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
        except Exception as e:
            print(f"Unexpected error in plot_time_domain: {e}")
            return None
    
    def plot_frequency_spectrum(self,
                               samples: np.ndarray,
                               sample_rate: float,
                               center_freq: float = 0,
                               title: str = "Frequency Spectrum",
                               nfft: int = 1024,
                               figsize: Tuple[int, int] = (12, 6)) -> Optional[Figure]:
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
        try:
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
 
        except Exception as e:
            print(f"Unexpected error in plot_frequency_spectrum: {e}")
            return None
    
    def plot_psd(self,
                samples: np.ndarray,
                sample_rate: float,
                center_freq: float = 0,
                title: str = "Power Spectral Density",
                nperseg: int = 1024,
                figsize: Tuple[int, int] = (12, 6)) -> Optional[Figure]:
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
        try:            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Compute PSD using Welch's method. Window defaults to 'hann'
            freqs, psd = signal.welch(samples, fs=sample_rate, nperseg=nperseg,
                                      return_onesided=False, scaling='density')
            freqs = np.fft.fftshift(freqs)
            psd = np.fft.fftshift(psd)
            psd_db = 10 * np.log10(psd + 1e-12) # add small value to avoid log(0)
            
            # Adjust frequency axis if in baseband or passband
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

        except Exception as e:
            print(f"Unexpected error in plot_psd: {e}")
            return None
    
    def plot_constellation(self,
                          symbols: np.ndarray,
                          title: str = "Constellation Diagram",
                          figsize: Tuple[int, int] = (8, 8),
                          alpha: float = 0.5) -> Optional[Figure]:
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
        try:    
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
        except Exception as e:
            print(f"Unexpected error in plot_constellation: {e}")
            return None
    
    def plot_spectrogram(self,
                        samples: np.ndarray,
                        sample_rate: float,
                        title: str = "Spectrogram",
                        nperseg: int = 256,
                        figsize: Tuple[int, int] = (12, 6)) -> Optional[Figure]:
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
        try:        
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
        
        except Exception as e:
            print(f"Unexpected error in plot_spectrogram: {e}")
            return None
    
    def plot_eye_diagram(self,
                        samples: np.ndarray,
                        samples_per_symbol: int,
                        title: str = "Eye Diagram",
                        num_traces: int = 100,
                        figsize: Tuple[int, int] = (10, 6)) -> Optional[Figure]:
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
        try:
            if samples_per_symbol <= 0:
                raise ValueError(f"samples_per_symbol must be positive, got {samples_per_symbol}")
                return None
            
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
        except Exception as e:
            print(f"Error in plot_eye_diagram: {e}")
            return None
    
    def _plot_eye_single(self, samples, samples_per_symbol, ax, num_traces, title):
        """Helper function to plot single eye diagram."""
        # Reshape into symbol periods
        num_symbols = len(samples) // samples_per_symbol
        num_traces = min(num_traces, np.abs(num_symbols - 2))
        
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
                           figsize: Tuple[int, int] = (12, 6)) -> Optional[Figure]:
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
        try:
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
        except Exception as e:
            print(f"Unexpected error in plot_magnitude_phase: {e}")
            return None
    
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
        try:

            self.plot_time_domain(samples, sample_rate)
            self.plot_magnitude_phase(samples, sample_rate)
            self.plot_frequency_spectrum(samples, sample_rate, center_freq)
            self.plot_psd(samples, sample_rate, center_freq)
            self.plot_constellation(samples)
            self.plot_spectrogram(samples, sample_rate)
            
            if samples_per_symbol:
                self.plot_eye_diagram(samples, samples_per_symbol)
        except Exception as e:
            print(f"Error in plot_all: {e}")
        


class LiveSDRPlotter(StaticSDRPlotter):
    """
    Live plotting class for real-time SDR visualization.
    Uses matplotlib FuncAnimation for smooth, non-blocking updates.
    """
    
    def __init__(self, 
                 data_callback,
                 sample_rate: float,
                 center_freq: float = 0,
                 update_interval: int = 50,
                 max_samples: int = 2048,
                 waterfall_history: int = 100,
                 style: str = 'dark_background'):
        """
        Initialize live plotter.
        
        Args:
            data_callback: Function that returns new IQ samples when called (e.g., sdr.rx)
            sample_rate: Sample rate in Hz
            center_freq: Center frequency in Hz
            update_interval: Update period in milliseconds (e.g., 50ms = 20 fps)
            max_samples: Maximum samples to display in time domain
            waterfall_history: Number of FFT lines to keep in waterfall
            style: Matplotlib style
        """
        super().__init__(style)
        self.data_callback = data_callback
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.update_interval = update_interval
        self.max_samples = max_samples
        self.waterfall_history = waterfall_history
        
        # Animation objects (created when starting)
        self.animation = None
        self.fig = None
        self.running = False
        
        # Waterfall data buffer
        self.waterfall_data = None
        
        # Setup signal handler for graceful shutdown
        self._setup_signal_handler()
    
    def _setup_signal_handler(self):
        """Setup signal handler for Ctrl+C."""
        def signal_handler(sig, frame):
            print("\nInterrupt received, stopping live plot...")
            self.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
    
    def start_spectrum_analyzer(self, 
                               nfft: int = 1024,
                               figsize: Tuple[int, int] = (12, 6)):
        """
        Start live spectrum analyzer (FFT plot that updates continuously).
        
        Args:
            nfft: FFT size
            figsize: Figure size
        """
        try:
            from matplotlib.animation import FuncAnimation
        except ImportError:
            print("Error: matplotlib animation support required")
            return
        
        try:
            # Create figure
            self.fig, self.ax = plt.subplots(figsize=figsize)
            self.ax.set_xlabel('Frequency (MHz)' if self.center_freq > 0 else 'Frequency (kHz)', 
                              fontsize=10)
            self.ax.set_ylabel('Magnitude (dB)', fontsize=10)
            self.ax.set_title('Live Spectrum Analyzer', fontsize=12)
            self.ax.grid(True, alpha=0.3)
            
            # Initialize line
            self.line, = self.ax.plot([], [], linewidth=1)
            
            # Store parameters
            self.nfft = nfft
            
            # Animation update function
            def update(frame):
                if not self.running:
                    return self.line,
                
                try:
                    # Get new samples
                    samples = self.data_callback()
                    
                    if samples is None or len(samples) == 0:
                        return self.line,
                    
                    # Decimate if needed
                    if len(samples) > self.nfft:
                        samples = samples[:self.nfft]
                    
                    # Compute FFT
                    fft_data = np.fft.fftshift(np.fft.fft(samples, n=self.nfft))
                    fft_db = 20 * np.log10(np.abs(fft_data) + 1e-12)
                    
                    # Frequency axis
                    freqs = np.fft.fftshift(np.fft.fftfreq(self.nfft, 1/self.sample_rate))
                    if self.center_freq > 0:
                        freqs = (freqs + self.center_freq) / 1e6  # MHz
                    else:
                        freqs = freqs / 1e3  # kHz
                    
                    # Update line
                    self.line.set_data(freqs, fft_db)
                    
                    # Auto-scale y-axis
                    if len(fft_db) > 0:
                        self.ax.set_ylim([np.min(fft_db) - 5, np.max(fft_db) + 5])
                        self.ax.set_xlim([freqs[0], freqs[-1]])
                    
                except KeyboardInterrupt:
                    self.stop()
                except Exception as e:
                    print(f"Error in spectrum update: {e}")
                
                return self.line,
            
            # Create animation
            self.animation = FuncAnimation(self.fig, update, interval=self.update_interval,
                                          blit=True, cache_frame_data=False)
            self.running = True
            
            plt.tight_layout()
            print("Live spectrum analyzer started. Press Ctrl+C to stop.")
            plt.show()
        except Exception as e:
            print(f"Error starting spectrum analyzer: {e}")
            self.stop()
    
    def start_waterfall(self,
                       nfft: int = 1024,
                       figsize: Tuple[int, int] = (12, 8)):
        """
        Start live waterfall display (spectrogram-style).
        
        Args:
            nfft: FFT size
            figsize: Figure size
        """
        try:
            from matplotlib.animation import FuncAnimation
        except ImportError:
            print("Error: matplotlib animation support required")
            return
        
        try:
            # Create figure
            self.fig, self.ax = plt.subplots(figsize=figsize)
            self.ax.set_xlabel('Frequency (MHz)' if self.center_freq > 0 else 'Frequency (kHz)',
                              fontsize=10)
            self.ax.set_ylabel('Time (updates)', fontsize=10)
            self.ax.set_title('Live Waterfall', fontsize=12)
            
            # Initialize waterfall buffer
            self.waterfall_data = np.zeros((self.waterfall_history, nfft))
            self.nfft = nfft
            
            # Frequency axis
            freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1/self.sample_rate))
            if self.center_freq > 0:
                freqs = (freqs + self.center_freq) / 1e6  # MHz
            else:
                freqs = freqs / 1e3  # kHz
            
            self.freqs = freqs
            
            # Initialize image
            self.waterfall_img = self.ax.imshow(self.waterfall_data,
                                               aspect='auto',
                                               extent=[freqs[0], freqs[-1], 0, self.waterfall_history],
                                               cmap='viridis',
                                               interpolation='nearest',
                                               vmin=-80, vmax=0)
            
            cbar = plt.colorbar(self.waterfall_img, ax=self.ax)
            cbar.set_label('Power (dB)', fontsize=10)
            
            # Animation update function
            def update(frame):
                if not self.running:
                    return self.waterfall_img,
                
                try:
                    # Get new samples
                    samples = self.data_callback()
                    
                    if samples is None or len(samples) == 0:
                        return self.waterfall_img,
                    
                    # Decimate if needed
                    if len(samples) > self.nfft:
                        samples = samples[:self.nfft]
                    
                    # Compute FFT
                    fft_data = np.fft.fftshift(np.fft.fft(samples, n=self.nfft))
                    fft_db = 20 * np.log10(np.abs(fft_data) + 1e-12)
                    
                    # Scroll waterfall: move old data up, add new line at bottom
                    self.waterfall_data = np.roll(self.waterfall_data, 1, axis=0)
                    self.waterfall_data[0, :] = fft_db
                    
                    # Update image
                    self.waterfall_img.set_data(self.waterfall_data)
                    
                except KeyboardInterrupt:
                    self.stop()
                except Exception as e:
                    print(f"Error in waterfall update: {e}")
                
                return self.waterfall_img,
            
            # Create animation
            self.animation = FuncAnimation(self.fig, update, interval=self.update_interval,
                                          blit=True, cache_frame_data=False)
            self.running = True
            
            plt.tight_layout()
            print("Live waterfall started. Press Ctrl+C to stop.")
            plt.show()
        except Exception as e:
            print(f"Error starting waterfall: {e}")
            self.stop()
    
    def start_time_domain(self, figsize: Tuple[int, int] = (12, 6)):
        """
        Start live time domain plot (I/Q signals).
        
        Args:
            figsize: Figure size
        """
        try:
            from matplotlib.animation import FuncAnimation
        except ImportError:
            print("Error: matplotlib animation support required")
            return
        
        try:
            # Create figure
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=figsize)
            self.ax1.set_ylabel('I (In-phase)', fontsize=10)
            self.ax1.set_title('Live Time Domain Signal', fontsize=12)
            self.ax1.grid(True, alpha=0.3)
            
            self.ax2.set_xlabel('Time (μs)', fontsize=10)
            self.ax2.set_ylabel('Q (Quadrature)', fontsize=10)
            self.ax2.grid(True, alpha=0.3)
            
            # Initialize lines
            self.line_i, = self.ax1.plot([], [], linewidth=0.8)
            self.line_q, = self.ax2.plot([], [], linewidth=0.8, color='orange')
            
            # Animation update function
            def update(frame):
                if not self.running:
                    return self.line_i, self.line_q
                
                try:
                    # Get new samples
                    samples = self.data_callback()
                    
                    if samples is None or len(samples) == 0:
                        return self.line_i, self.line_q
                    
                    # Decimate for display
                    samples = samples[:self.max_samples]
                    time = np.arange(len(samples)) / self.sample_rate * 1e6  # μs
                    
                    # Update lines
                    self.line_i.set_data(time, samples.real)
                    self.line_q.set_data(time, samples.imag)
                    
                    # Auto-scale
                    self.ax1.set_xlim([0, time[-1]])
                    self.ax2.set_xlim([0, time[-1]])
                    
                    i_max = np.max(np.abs(samples.real))
                    q_max = np.max(np.abs(samples.imag))
                    self.ax1.set_ylim([-i_max*1.1, i_max*1.1])
                    self.ax2.set_ylim([-q_max*1.1, q_max*1.1])
                    
                except KeyboardInterrupt:
                    self.stop()
                except Exception as e:
                    print(f"Error in time domain update: {e}")
                
                return self.line_i, self.line_q
            
            # Create animation
            self.animation = FuncAnimation(self.fig, update, interval=self.update_interval,
                                          blit=True, cache_frame_data=False)
            self.running = True
            
            plt.tight_layout()
            print("Live time domain plot started. Press Ctrl+C to stop.")
            plt.show()
        except Exception as e:
            print(f"Error starting time domain plot: {e}")
            self.stop()
    
    def start_constellation(self, 
                          decimation: int = 8,
                          figsize: Tuple[int, int] = (8, 8)):
        """
        Start live constellation diagram.
        
        Args:
            decimation: Show every Nth sample
            figsize: Figure size
        """
        try:
            from matplotlib.animation import FuncAnimation
        except ImportError:
            print("Error: matplotlib animation support required")
            return
        
        try:
            # Create figure
            self.fig, self.ax = plt.subplots(figsize=figsize)
            self.ax.set_xlabel('In-phase', fontsize=10)
            self.ax.set_ylabel('Quadrature', fontsize=10)
            self.ax.set_title('Live Constellation Diagram', fontsize=12)
            self.ax.grid(True, alpha=0.3)
            self.ax.axis('equal')
            
            # Initialize scatter
            self.scatter = self.ax.scatter([], [], alpha=0.5, s=2)
            
            # Store decimation
            self.decimation = decimation
            
            # Animation update function
            def update(frame):
                if not self.running:
                    return self.scatter,
                
                try:
                    # Get new samples
                    samples = self.data_callback()
                    
                    if samples is None or len(samples) == 0:
                        return self.scatter,
                    
                    # Decimate
                    symbols = samples[::self.decimation]
                    
                    # Update scatter
                    self.scatter.set_offsets(np.c_[symbols.real, symbols.imag])
                    
                    # Auto-scale
                    if len(symbols) > 0:
                        max_val = max(np.abs(symbols.real).max(), np.abs(symbols.imag).max())
                        self.ax.set_xlim(-max_val*1.2, max_val*1.2)
                        self.ax.set_ylim(-max_val*1.2, max_val*1.2)
                    
                except KeyboardInterrupt:
                    self.stop()
                except Exception as e:
                    print(f"Error in constellation update: {e}")
                
                return self.scatter,
            
            # Create animation
            self.animation = FuncAnimation(self.fig, update, interval=self.update_interval,
                                          blit=True, cache_frame_data=False)
            self.running = True
            
            plt.tight_layout()
            print("Live constellation diagram started. Press Ctrl+C to stop.")
            plt.show()
        except Exception as e:
            print(f"Error starting constellation diagram: {e}")
            self.stop()
    
    def stop(self):
        """Stop the live animation."""
        if self.animation and self.running:
            try:
                self.animation.event_source.stop()
                self.running = False
                if self.fig is not None:
                    plt.close(self.fig)
                print("Live plotting stopped.")
            except Exception as e:
                print(f"Error stopping animation: {e}")
        elif not self.running:
            print("No active animation to stop.")
    
    def __del__(self):
        """Cleanup on object deletion."""
        try:
            self.stop()
        except:
            pass
