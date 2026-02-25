"""
SDR Visualization Library
Plotting utilities for radio waveform analysis and visualization
"""

# Standard library imports
import logging
from typing import Optional, Tuple, Dict
import time

# Third party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy import signal

from PyQt6.QtCore import QSize, Qt, QThread, pyqtSignal, QObject, QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
import pyqtgraph as pg
from queue import Queue, Empty, Full



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
            plt.style.use('default')

    def __del__(self):
        """Destructor to clean up resources."""
        plt.close('all')
    
    def plot_filter_response(self,
                                coefficients: np.ndarray,
                                time_vector: np.ndarray,
                                sample_rate: float,
                                title: str = "Filter Impulse and Frequency Response",
                                figsize: Tuple[int, int] = (12, 6)) -> Optional[Figure]:
        """Plot the impulse response and frequency response of a filter.
        
        Args:
            coefficients: Filter coefficients
            time_vector: Time vector corresponding to coefficients
            sample_rate: Sample rate in Hz
            title: Plot title
            figsize: Figure size
        """
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
            
            # Impulse response
            ax1.stem(time_vector, coefficients)

            ax1.set_xlabel('Time (s)', fontsize=10)
            ax1.set_ylabel('Amplitude', fontsize=10)
            ax1.set_title(f"{title} - Impulse Response", fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # Frequency response
            nfft = 2048
            H = np.fft.fft(coefficients, n=nfft)
            H_shifted = np.fft.fftshift(H)
            H_dB = 20 * np.log10(np.abs(H_shifted) + 1e-12)  # dB, add small value to avoid log(0)

            freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1/sample_rate)) / 1e3  # kHz

            ax2.plot(freqs, H_dB, linewidth=1)
            ax2.set_xlabel('Frequency (kHz)', fontsize=10)
            ax2.set_ylabel('Magnitude (dB)', fontsize=10)
            ax2.set_title(f"{title} - Frequency Response", fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Unexpected error in plot_filter_response: {e}")
            return None


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
            if np.max(freqs) > 1e6:
                freqs = (freqs + center_freq) / 1e6  # MHz
                xlabel = 'Frequency (MHz)'
            elif np.max(freqs) > 1e3:
                freqs = (freqs + center_freq) / 1e3  # kHz
                xlabel = 'Frequency (kHz)'
            else:
                freqs = freqs + center_freq  # Hz
                xlabel = 'Frequency (Hz)'
            
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
            
            ax.set_aspect('equal', adjustable='box')

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
            
            # If complex, plot both I and Q
            if np.iscomplexobj(samples):
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
                self._plot_eye_single(samples.real, samples_per_symbol, ax1, 
                                     num_traces, "Eye Diagram - I")
                self._plot_eye_single(samples.imag, samples_per_symbol, ax2, 
                                     num_traces, "Eye Diagram - Q")
            else:
                fig, ax = plt.subplots(figsize=figsize)
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
                ax.plot(time, samples[start:end], color='orange', alpha=0.5, linewidth=0.5)
        
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
        


class StaticPlotSignaler(QObject):
    """Helper class to signal main thread for static plot updates."""
    plot_requested = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()



class LivePlotWorker(QObject):
    """
    Worker object that processes SDR data and emits signals for plot updates.
    Runs in a separate thread to avoid blocking the GUI.
    """
    
    # PyQt Signals for updating plots
    time_plot_update = pyqtSignal(np.ndarray)           # Raw IQ samples
    freq_plot_update = pyqtSignal(np.ndarray, np.ndarray)  # (frequencies, PSD values)
    waterfall_plot_update = pyqtSignal(np.ndarray)      # Spectrogram data
    constellation_plot_update = pyqtSignal(np.ndarray)  # Constellation symbols
    end_of_run = pyqtSignal()                           # Signals completion of one update cycle
    
    def __init__(self, config: Dict, data_queue: Queue):
        """
        Initialize the worker with configuration and data queue.
        
        Args:
            config: Configuration dictionary from config.yaml
            data_queue: Thread-safe queue for receiving samples from RX loop
        """
        super().__init__()
        
        # Extract configuration parameters
        plotter_config = config.get('plotter', {})
        self.sample_rate = float(config['modulation']['sample_rate'])
        self.center_freq = float(plotter_config.get('center_freq', 866.5e6))
        self.num_rows = int(plotter_config.get('num_rows', 200))
        self.time_plot_samples = int(plotter_config.get('time_plot_samples', 500))
        self.psd_nperseg = int(plotter_config.get('psd_nperseg', 1024))
        self.max_constellation_points = int(plotter_config.get('max_constellation_points', 500))
        self.psd_window = str(plotter_config.get('psd_window', 'hann'))
        self.psd_scaling = str(plotter_config.get('psd_scaling', 'density'))
        self.psd_max_points = int(plotter_config.get('psd_max_points', 2048))
        
        # Data queue for receiving samples
        self.data_queue = data_queue
        
        # Initialize spectrogram buffer (frequency bins x num_rows)
        self.fft_size = self.psd_nperseg
        self.spectrogram = -50 * np.ones((self.fft_size, self.num_rows))
        
        # Running average for PSD smoothing
        self.PSD_avg = -50 * np.ones(self.fft_size)
        self.psd_alpha = 0.1  # Smoothing factor (0.1 = slow, 0.9 = fast)
        
        # Constellation point buffer (circular buffer)
        self.constellation_buffer = np.zeros(self.max_constellation_points, dtype=np.complex64)
        self.constellation_index = 0
        
        # Control flag
        self.running = True
        
    def run(self):
        """
        Main processing loop. Retrieves data from queue and emits plot updates.
        Called repeatedly via QTimer in the main window.
        """
        if not self.running:
            return
            
        try:
            # Try to get data from queue (non-blocking)
            samples = self.data_queue.get_nowait()
            
            # === Time Domain Update ===
            # Emit only the first N samples for time plot
            time_samples = samples[:min(len(samples), self.time_plot_samples)]
            self.time_plot_update.emit(time_samples)
            
            # === Frequency Domain (PSD) Update ===
            # Compute PSD using FFT
            fft_data = np.fft.fftshift(np.fft.fft(samples, n=self.fft_size))
            PSD = 10.0 * np.log10(np.abs(fft_data)**2 / self.fft_size + 1e-12)
            
            # Apply exponential moving average for smoothing
            self.PSD_avg = self.PSD_avg * (1 - self.psd_alpha) + PSD * self.psd_alpha
            
            # Generate frequency axis
            freqs = np.fft.fftshift(np.fft.fftfreq(self.fft_size, 1/self.sample_rate))
            freqs = (freqs + self.center_freq) / 1e6  # Convert to MHz
            
            self.freq_plot_update.emit(freqs, self.PSD_avg)
            
            # === Waterfall (Spectrogram) Update ===
            # Roll spectrogram and insert new row
            self.spectrogram = np.roll(self.spectrogram, 1, axis=1)
            self.spectrogram[:, 0] = PSD
            self.waterfall_plot_update.emit(self.spectrogram)
            
            # === Constellation Update ===
            # Add new samples to circular buffer
            num_new = min(len(samples), self.max_constellation_points)
            end_idx = min(self.constellation_index + num_new, self.max_constellation_points)
            samples_to_add = num_new if end_idx - self.constellation_index >= num_new else end_idx - self.constellation_index
            
            self.constellation_buffer[self.constellation_index:self.constellation_index + samples_to_add] = samples[:samples_to_add]
            self.constellation_index = (self.constellation_index + samples_to_add) % self.max_constellation_points
            
            self.constellation_plot_update.emit(self.constellation_buffer.copy())
            
        except Empty:
            # No data available, skip this cycle
            pass
        except Exception as e:
            logging.error(f"Error in LivePlotWorker.run(): {e}")
        
        # Signal end of run to trigger next cycle
        self.end_of_run.emit()
    
    def stop(self):
        """Stop the worker."""
        self.running = False



class LiveSDRPlotter(QMainWindow):
    """
    Real-time SDR signal visualization window using PyQt6 and pyqtgraph.
    
    Displays:
    - Time domain (I/Q components)
    - Frequency spectrum (PSD)
    - Waterfall/Spectrogram
    - Constellation diagram
    
    Each plot has auto-range buttons for easy scaling.
    """
    
    def __init__(self, config: Dict, data_queue: Queue):
        """
        Initialize the live plotter window.
        
        Args:
            config: Configuration dictionary from config.yaml
            data_queue: Thread-safe queue for receiving samples
        """
        super().__init__()
        
        self.config = config
        self.data_queue = data_queue
        
        # Extract configuration
        plotter_config = config.get('plotter', {})
        self.sample_rate = float(config['modulation']['sample_rate'])
        self.center_freq = float(plotter_config.get('center_freq', 866.5e6))
        self.update_interval = int(plotter_config.get('update_interval', 100))
        self.time_plot_samples = int(plotter_config.get('time_plot_samples', 500))
        self.fft_size = int(plotter_config.get('psd_nperseg', 1024))
        self.num_rows = int(plotter_config.get('num_rows', 200))
        
        # For auto-range calculations
        self.spectrogram_min = -50
        self.spectrogram_max = 0
        
        # Setup window
        self.setWindowTitle("SDR Live Signal Analyzer")
        self.setFixedSize(QSize(1400, 900))
        
        # Initialize UI
        self._setup_ui()
        
        # Initialize worker thread
        self._setup_worker()
        
        logging.info("LiveSDRPlotter initialized successfully")
    
    def _setup_ui(self):
        """Setup the user interface with all plots and controls."""
        
        # Main layout
        layout = QGridLayout()
        
        # ==================== Time Domain Plot ====================
        self.time_plot = pg.PlotWidget(
            labels={'left': 'Amplitude', 'bottom': 'Sample Index'},
            title='Time Domain (I/Q)'
        )
        self.time_plot.setMouseEnabled(x=False, y=True)
        self.time_plot.setYRange(-1.1, 1.1)
        self.time_plot.addLegend()
        
        # Create plot curves for I and Q
        self.time_curve_i = self.time_plot.plot([], pen='c', name='I (In-phase)')
        self.time_curve_q = self.time_plot.plot([], pen='y', name='Q (Quadrature)')
        
        layout.addWidget(self.time_plot, 0, 0)
        
        # Time plot controls
        time_controls = QVBoxLayout()
        
        btn_time_auto = QPushButton('Auto Range')
        btn_time_auto.clicked.connect(lambda: self.time_plot.autoRange())
        time_controls.addWidget(btn_time_auto)
        
        btn_time_adc = QPushButton('ADC Limits\n(-1 to +1)')
        btn_time_adc.clicked.connect(lambda: self.time_plot.setYRange(-1.1, 1.1))
        time_controls.addWidget(btn_time_adc)
        
        time_controls.addStretch()
        layout.addLayout(time_controls, 0, 1)
        
        # ==================== Frequency Domain Plot ====================
        self.freq_plot = pg.PlotWidget(
            labels={'left': 'PSD (dB)', 'bottom': 'Frequency (MHz)'},
            title='Power Spectral Density'
        )
        self.freq_plot.setMouseEnabled(x=False, y=True)
        self.freq_plot.setYRange(-60, 10)
        
        self.freq_curve = self.freq_plot.plot([], pen='g')
        
        layout.addWidget(self.freq_plot, 1, 0)
        
        # Frequency plot controls
        freq_controls = QVBoxLayout()
        
        btn_freq_auto = QPushButton('Auto Range')
        btn_freq_auto.clicked.connect(lambda: self.freq_plot.autoRange())
        freq_controls.addWidget(btn_freq_auto)
        
        freq_controls.addStretch()
        layout.addLayout(freq_controls, 1, 1)
        
        # ==================== Waterfall Plot ====================
        waterfall_layout = QHBoxLayout()
        
        self.waterfall_plot = pg.PlotWidget(
            labels={'left': 'Time (rows)', 'bottom': 'Frequency Bin'},
            title='Waterfall (Spectrogram)'
        )
        self.waterfall_image = pg.ImageItem(axisOrder='col-major')
        self.waterfall_plot.addItem(self.waterfall_image)
        self.waterfall_plot.setMouseEnabled(x=False, y=False)
        
        waterfall_layout.addWidget(self.waterfall_plot)
        
        # Colorbar for waterfall
        self.colorbar = pg.HistogramLUTWidget()
        self.colorbar.setImageItem(self.waterfall_image)
        self.colorbar.item.gradient.loadPreset('viridis')
        self.waterfall_image.setLevels((-50, 0))
        
        waterfall_layout.addWidget(self.colorbar)
        
        layout.addLayout(waterfall_layout, 2, 0)
        
        # Waterfall controls
        waterfall_controls = QVBoxLayout()
        
        btn_waterfall_auto = QPushButton('Auto Range\n(±2σ)')
        btn_waterfall_auto.clicked.connect(self._auto_range_waterfall)
        waterfall_controls.addWidget(btn_waterfall_auto)
        
        waterfall_controls.addStretch()
        layout.addLayout(waterfall_controls, 2, 1)
        
        # ==================== Constellation Plot ====================
        self.constellation_plot = pg.PlotWidget(
            labels={'left': 'Quadrature', 'bottom': 'In-phase'},
            title='Constellation Diagram'
        )
        self.constellation_plot.setAspectLocked(True)
        self.constellation_plot.setXRange(-1.5, 1.5)
        self.constellation_plot.setYRange(-1.5, 1.5)
        
        self.constellation_scatter = pg.ScatterPlotItem(
            size=3, pen=None, brush=pg.mkBrush(100, 200, 255, 120)
        )
        self.constellation_plot.addItem(self.constellation_scatter)
        
        layout.addWidget(self.constellation_plot, 3, 0)
        
        # Constellation controls
        constellation_controls = QVBoxLayout()
        
        btn_const_auto = QPushButton('Auto Range')
        btn_const_auto.clicked.connect(lambda: self.constellation_plot.autoRange())
        constellation_controls.addWidget(btn_const_auto)
        
        btn_const_unit = QPushButton('Unit Circle\n(-1.5 to +1.5)')
        btn_const_unit.clicked.connect(self._reset_constellation_range)
        constellation_controls.addWidget(btn_const_unit)
        
        btn_const_clear = QPushButton('Clear Points')
        btn_const_clear.clicked.connect(self._clear_constellation)
        constellation_controls.addWidget(btn_const_clear)
        
        constellation_controls.addStretch()
        layout.addLayout(constellation_controls, 3, 1)
        
        # ==================== Status Bar ====================
        self.status_label = QLabel('Status: Waiting for data...')
        self.status_label.setStyleSheet('color: #888; font-size: 10px;')
        layout.addWidget(self.status_label, 4, 0)
        
        # FPS counter
        self.fps_label = QLabel('FPS: --')
        self.fps_label.setStyleSheet('color: #888; font-size: 10px;')
        layout.addWidget(self.fps_label, 4, 1)
        
        # Set central widget
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        
        # For FPS calculation
        self.last_update_time = time.time()
        self.frame_count = 0
    
    def _setup_worker(self):
        """Setup the worker thread for data processing."""
        
        # Create worker and thread
        self.worker_thread = QThread()
        self.worker_thread.setObjectName('PlotWorker_Thread')
        
        self.worker = LivePlotWorker(self.config, self.data_queue)
        self.worker.moveToThread(self.worker_thread)
        
        # Connect signals to slots
        self.worker.time_plot_update.connect(self._update_time_plot)
        self.worker.freq_plot_update.connect(self._update_freq_plot)
        self.worker.waterfall_plot_update.connect(self._update_waterfall_plot)
        self.worker.constellation_plot_update.connect(self._update_constellation_plot)
        self.worker.end_of_run.connect(self._on_worker_cycle_complete)
        
        # Start worker when thread starts
        self.worker_thread.started.connect(self.worker.run)
        self.worker_thread.start()
    
    # ==================== Plot Update Callbacks ====================
    
    def _update_time_plot(self, samples: np.ndarray):
        """Update the time domain plot with new samples."""
        self.time_curve_i.setData(samples.real)
        self.time_curve_q.setData(samples.imag)
    
    def _update_freq_plot(self, freqs: np.ndarray, psd: np.ndarray):
        """Update the frequency domain plot."""
        self.freq_curve.setData(freqs, psd)
    
    def _update_waterfall_plot(self, spectrogram: np.ndarray):
        """Update the waterfall/spectrogram plot."""
        self.waterfall_image.setImage(spectrogram, autoLevels=False)
        
        # Calculate statistics for auto-range
        sigma = np.std(spectrogram)
        mean = np.mean(spectrogram)
        self.spectrogram_min = mean - 2 * sigma
        self.spectrogram_max = mean + 2 * sigma
    
    def _update_constellation_plot(self, symbols: np.ndarray):
        """Update the constellation diagram."""
        # Filter out zero values (unfilled buffer positions)
        valid_symbols = symbols[symbols != 0]
        if len(valid_symbols) > 0:
            self.constellation_scatter.setData(
                valid_symbols.real, 
                valid_symbols.imag
            )
    
    def _on_worker_cycle_complete(self):
        """Called when worker completes one processing cycle."""
        # Update FPS counter
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_update_time
        
        if elapsed >= 1.0:
            fps = self.frame_count / elapsed
            self.fps_label.setText(f'FPS: {fps:.1f}')
            self.frame_count = 0
            self.last_update_time = current_time
        
        self.status_label.setText('Status: Receiving data')
        
        # Schedule next worker run
        QTimer.singleShot(self.update_interval, self.worker.run)
    
    # ==================== Control Button Callbacks ====================
    
    def _auto_range_waterfall(self):
        """Auto-range the waterfall colormap based on signal statistics."""
        self.waterfall_image.setLevels((self.spectrogram_min, self.spectrogram_max))
        self.colorbar.setLevels(self.spectrogram_min, self.spectrogram_max)
    
    def _reset_constellation_range(self):
        """Reset constellation plot to unit circle range."""
        self.constellation_plot.setXRange(-1.5, 1.5)
        self.constellation_plot.setYRange(-1.5, 1.5)
    
    def _clear_constellation(self):
        """Clear the constellation diagram."""
        self.constellation_scatter.setData([], [])
        if hasattr(self, 'worker'):
            self.worker.constellation_buffer.fill(0)
            self.worker.constellation_index = 0
    
    # ==================== Lifecycle Methods ====================
    
    def close_all(self):
        """Clean up resources and close the window."""
        logging.info("Closing LiveSDRPlotter...")
        
        # Stop worker
        if hasattr(self, 'worker'):
            self.worker.stop()
        
        # Stop thread
        if hasattr(self, 'worker_thread') and self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait(2000)  # Wait up to 2 seconds
        
        # Close window
        self.close()
        
        logging.info("LiveSDRPlotter closed successfully")
    
    def closeEvent(self, event):
        """Handle window close event."""
        self.close_all()
        event.accept()




class LivePlotWindow(QMainWindow):
    """Base class for individual plot windows."""
    
    def __init__(self, title: str, size: Tuple[int, int] = (600, 400)):
        super().__init__()
        self.setWindowTitle(title)
        self.setFixedSize(QSize(size[0], size[1]))
        
        # Main layout
        layout = QHBoxLayout()
        self.plot_layout = QVBoxLayout()
        self.controls_layout = QVBoxLayout()
        
        layout.addLayout(self.plot_layout, stretch=4)
        layout.addLayout(self.controls_layout, stretch=1)
        
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)


class TimePlotWindow(LivePlotWindow):
    """Separate window for time domain plot."""
    
    def __init__(self):
        super().__init__("Time Domain (I/Q)", (700, 400))
        
        # Create plot
        self.plot = pg.PlotWidget(
            labels={'left': 'Amplitude', 'bottom': 'Sample Index'},
            title='Time Domain (I/Q)'
        )
        self.plot.setMouseEnabled(x=False, y=True)
        self.plot.setYRange(-1.1, 1.1)
        self.plot.addLegend()
        
        self.curve_i = self.plot.plot([], pen='c', name='I (In-phase)')
        self.curve_q = self.plot.plot([], pen='y', name='Q (Quadrature)')
        
        self.plot_layout.addWidget(self.plot)
        
        # Controls
        btn_auto = QPushButton('Auto Range')
        btn_auto.clicked.connect(lambda: self.plot.autoRange())
        self.controls_layout.addWidget(btn_auto)
        
        btn_adc = QPushButton('ADC Limits\n(-1 to +1)')
        btn_adc.clicked.connect(lambda: self.plot.setYRange(-1.1, 1.1))
        self.controls_layout.addWidget(btn_adc)
        
        self.controls_layout.addStretch()
    
    def update_plot(self, samples: np.ndarray):
        """Update the plot with new samples."""
        self.curve_i.setData(samples.real)
        self.curve_q.setData(samples.imag)


class FreqPlotWindow(LivePlotWindow):
    """Separate window for frequency domain plot."""
    
    def __init__(self):
        super().__init__("Power Spectral Density", (700, 400))
        
        self.plot = pg.PlotWidget(
            labels={'left': 'PSD (dB)', 'bottom': 'Frequency (MHz)'},
            title='Power Spectral Density'
        )
        self.plot.setMouseEnabled(x=False, y=True)
        self.plot.setYRange(-60, 10)
        
        self.curve = self.plot.plot([], pen='g')
        
        self.plot_layout.addWidget(self.plot)
        
        # Controls
        btn_auto = QPushButton('Auto Range')
        btn_auto.clicked.connect(lambda: self.plot.autoRange())
        self.controls_layout.addWidget(btn_auto)
        
        self.controls_layout.addStretch()
    
    def update_plot(self, freqs: np.ndarray, psd: np.ndarray):
        """Update the plot with new data."""
        self.curve.setData(freqs, psd)


class WaterfallPlotWindow(LivePlotWindow):
    """Separate window for waterfall/spectrogram plot."""
    
    def __init__(self):
        super().__init__("Waterfall (Spectrogram)", (800, 500))
        
        # Waterfall plot with colorbar
        waterfall_layout = QHBoxLayout()
        
        self.plot = pg.PlotWidget(
            labels={'left': 'Time (rows)', 'bottom': 'Frequency Bin'},
            title='Waterfall (Spectrogram)'
        )
        self.image = pg.ImageItem(axisOrder='col-major')
        self.plot.addItem(self.image)
        self.plot.setMouseEnabled(x=False, y=False)
        
        waterfall_layout.addWidget(self.plot)
        
        # Colorbar
        self.colorbar = pg.HistogramLUTWidget()
        self.colorbar.setImageItem(self.image)
        self.colorbar.item.gradient.loadPreset('viridis')
        self.image.setLevels((-50, 0))
        
        waterfall_layout.addWidget(self.colorbar)
        
        self.plot_layout.addLayout(waterfall_layout)
        
        # For auto-range
        self.spectrogram_min = -50
        self.spectrogram_max = 0
        
        # Controls
        btn_auto = QPushButton('Auto Range\n(±2σ)')
        btn_auto.clicked.connect(self._auto_range)
        self.controls_layout.addWidget(btn_auto)
        
        self.controls_layout.addStretch()
    
    def _auto_range(self):
        """Auto-range colormap based on statistics."""
        self.image.setLevels((self.spectrogram_min, self.spectrogram_max))
        self.colorbar.setLevels(self.spectrogram_min, self.spectrogram_max)
    
    def update_plot(self, spectrogram: np.ndarray):
        """Update the waterfall with new data."""
        self.image.setImage(spectrogram, autoLevels=False)
        
        sigma = np.std(spectrogram)
        mean = np.mean(spectrogram)
        self.spectrogram_min = mean - 2 * sigma
        self.spectrogram_max = mean + 2 * sigma


class ConstellationPlotWindow(LivePlotWindow):
    """Separate window for constellation diagram."""
    
    def __init__(self):
        super().__init__("Constellation Diagram", (500, 500))
        
        self.plot = pg.PlotWidget(
            labels={'left': 'Quadrature', 'bottom': 'In-phase'},
            title='Constellation Diagram'
        )
        self.plot.setAspectLocked(True)
        self.plot.setXRange(-1.5, 1.5)
        self.plot.setYRange(-1.5, 1.5)
        
        self.scatter = pg.ScatterPlotItem(
            size=3, pen=None, brush=pg.mkBrush(100, 200, 255, 120)
        )
        self.plot.addItem(self.scatter)
        
        self.plot_layout.addWidget(self.plot)
        
        # Controls
        btn_auto = QPushButton('Auto Range')
        btn_auto.clicked.connect(lambda: self.plot.autoRange())
        self.controls_layout.addWidget(btn_auto)
        
        btn_unit = QPushButton('Unit Circle\n(-1.5 to +1.5)')
        btn_unit.clicked.connect(self._reset_range)
        self.controls_layout.addWidget(btn_unit)
        
        btn_clear = QPushButton('Clear Points')
        btn_clear.clicked.connect(self._clear)
        self.controls_layout.addWidget(btn_clear)
        
        self.controls_layout.addStretch()
        
        # Buffer for clearing
        self.clear_callback = None
    
    def _reset_range(self):
        self.plot.setXRange(-1.5, 1.5)
        self.plot.setYRange(-1.5, 1.5)
    
    def _clear(self):
        self.scatter.setData([], [])
        if self.clear_callback:
            self.clear_callback()
    
    def update_plot(self, symbols: np.ndarray):
        """Update the constellation with new symbols."""
        valid_symbols = symbols[symbols != 0]
        if len(valid_symbols) > 0:
            self.scatter.setData(valid_symbols.real, valid_symbols.imag)


class LiveSDRPlotterMultiWindow:
    """
    Multi-window version of LiveSDRPlotter.
    Each plot type gets its own window that can be moved independently.
    """
    
    def __init__(self, config: Dict, data_queue: Queue):
        self.config = config
        self.data_queue = data_queue
        
        # Extract configuration
        plotter_config = config.get('plotter', {})
        self.update_interval = int(plotter_config.get('update_interval', 100))
        
        # Create individual windows
        self.time_window = TimePlotWindow()
        self.freq_window = FreqPlotWindow()
        self.waterfall_window = WaterfallPlotWindow()
        self.constellation_window = ConstellationPlotWindow()
        
        # Store all windows for easy management
        self.windows = [
            self.time_window,
            self.freq_window,
            self.waterfall_window,
            self.constellation_window
        ]
        
        # Position windows in a grid
        self._arrange_windows()
        
        # Setup worker thread
        self._setup_worker()
        
        # For FPS tracking
        self.last_update_time = time.time()
        self.frame_count = 0
        
        logging.info("LiveSDRPlotterMultiWindow initialized with 4 separate windows")
    
    def _arrange_windows(self):
        """Arrange windows in a 2x2 grid on screen."""
        # Get screen geometry (approximate positioning)
        x_offset = 50
        y_offset = 50
        
        # Top-left: Time domain
        self.time_window.move(x_offset, y_offset)
        
        # Top-right: Frequency domain
        self.freq_window.move(x_offset + 720, y_offset)
        
        # Bottom-left: Waterfall
        self.waterfall_window.move(x_offset, y_offset + 450)
        
        # Bottom-right: Constellation
        self.constellation_window.move(x_offset + 820, y_offset + 450)
    
    def _setup_worker(self):
        """Setup the worker thread for data processing."""
        self.worker_thread = QThread()
        self.worker_thread.setObjectName('PlotWorker_Thread')
        
        self.worker = LivePlotWorker(self.config, self.data_queue)
        self.worker.moveToThread(self.worker_thread)
        
        # Connect signals to individual window update methods
        self.worker.time_plot_update.connect(self.time_window.update_plot)
        self.worker.freq_plot_update.connect(self.freq_window.update_plot)
        self.worker.waterfall_plot_update.connect(self.waterfall_window.update_plot)
        self.worker.constellation_plot_update.connect(self.constellation_window.update_plot)
        self.worker.end_of_run.connect(self._on_worker_cycle_complete)
        
        # Set up constellation clear callback
        self.constellation_window.clear_callback = self._clear_constellation_buffer
        
        self.worker_thread.started.connect(self.worker.run)
        self.worker_thread.start()
    
    def _clear_constellation_buffer(self):
        """Clear the worker's constellation buffer."""
        if hasattr(self, 'worker'):
            self.worker.constellation_buffer.fill(0)
            self.worker.constellation_index = 0
    
    def _on_worker_cycle_complete(self):
        """Called when worker completes one cycle."""
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_update_time
        
        if elapsed >= 1.0:
            fps = self.frame_count / elapsed
            # Update title of time window with FPS
            self.time_window.setWindowTitle(f"Time Domain (I/Q) - {fps:.1f} FPS")
            self.frame_count = 0
            self.last_update_time = current_time
        
        QTimer.singleShot(self.update_interval, self.worker.run)
    
    def show(self):
        """Show all windows."""
        for window in self.windows:
            window.show()
    
    def close_all(self):
        """Close all windows and cleanup."""
        logging.info("Closing LiveSDRPlotterMultiWindow...")
        
        if hasattr(self, 'worker'):
            self.worker.stop()
        
        if hasattr(self, 'worker_thread') and self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait(2000)
        
        for window in self.windows:
            window.close()
        
        logging.info("LiveSDRPlotterMultiWindow closed")
