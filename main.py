"""
To display logging file in a terminal with color coding, use the following command:
tail -f log/$(date +%Y-%m-%d)-debug.log

optionally pipe with grep:
| grep --color=always "ERROR\|WARNING\|INFO\|DEBUG"

to filter for specific log levels while keeping color coding.

"""

# import system modules
import os
import sys
import time
import select
import logging
import threading
from queue import Queue, Empty, Full
from datetime import datetime
from typing import Dict
import signal
import atexit

# import third party moduels
import numpy as np
from yaml import safe_load

# import modules
from chat_tui import ChatTUI
from modulation import ModulationProtocol
from datagram import Datagram, msgType
from sdr_transciever import SDRTransciever
from filter import RRCFilter
from barker_detection import BarkerDetector
from synchronize import Synchronizer


# Optional imports for debug mode
try:
    from sdr_plots import LiveSDRPlotter, LiveSDRPlotterMultiWindow, StaticSDRPlotter, StaticPlotSignaler
    from matplotlib.pyplot import show
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QTimer
    HAS_PLOTTER = True
except ImportError:
    HAS_PLOTTER = False
    LiveSDRPlotter = None
    LiveSDRPlotterMultiWindow = None
    StaticPlotSignaler = None
    QApplication = None


class SDRChatApp:
    def __init__(self, config_file: str ="setup/config.yaml"):
        """Initialize the SDR Chat Application."""

        # ================= read configuration file =================
        try:
            with open(config_file, 'r') as f:
                config = safe_load(f)
        except Exception as e:
            print(f"Error loading config file: {e}")
            raise e

        self.config: Dict = config

        # ================= Initialize Modules with configuration =================
        self.modulation_protocol = ModulationProtocol(config)
        self.matched_filter = RRCFilter(config)
        self.tui = ChatTUI(config)
        self.barker_detector = BarkerDetector(config)
        self.synchronizer = Synchronizer(config)
        self.sdr = SDRTransciever(config) # must be initilized after Matched Filter module.

        # ================== Threading and synchronization primitives ==================
        self.running: bool = False
        self.rx_thread: threading.Thread = None
        self.tx_thread: threading.Thread = None
        self.tui_thread: threading.Thread = None
        self.tui_refresh_event: threading.Event = threading.Event()

        # ================== Message queues for inter-thread communication ==================
        self.tx_queue: Queue[Datagram] = Queue(maxsize=int(config['radio']['queue_size']))
        self.rx_queue: Queue[Datagram] = Queue(maxsize=int(config['radio']['queue_size']))
    
        # ================== Logging setup ==================
        log_dir = "log"
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"{datetime.now().date()}-chat-history.txt")
        self.debug_file = os.path.join(log_dir, f"{datetime.now().date()}-debug.log")
        self._setup_logging(str(config['radio']['log_level']).upper().strip())

        try:
            with open(self.log_file, 'a') as f:
                f.write(f"\n\n--- New Chat Session Started at {datetime.now().time()} ---\n")
        except Exception as e:
            logging.error(f"Error initializing chat history log: {e}")
            raise e

        # ================== Debug mode setup ==================
        self.debug_mode = bool(config['radio']['debug_mode'])
        self.qapp = None
        self.plotter = None
        self.plot_data_queue: Queue[np.ndarray] = Queue(maxsize=32)
        self.static_plotter= StaticSDRPlotter()  # For static plots of filter response, constellation, etc.
        self.static_plot_queue: Queue[Dict[str, np.ndarray]] = Queue(maxsize=8)  # Queue for static plot data (e.g., filter response, constellation points)

        # Initialize plotter if debug mode is enabled
        if self.debug_mode and HAS_PLOTTER:
            logging.info("Debug mode enabled - initializing live plotter")
            self._init_plotter()
        elif self.debug_mode and not HAS_PLOTTER:
            logging.warning("Debug mode enabled but PyQt6/pyqtgraph not available")
            self.debug_mode = False
        
                
        # ====================== Setup signal handlers for graceful shutdown =====================
        atexit.register(self._cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logging.info("SDR Chat Application initialized successfully.")

    def __del__(self):
        """Fallback cleanup if explicit cleanup wasn't called."""
        if not hasattr(self, '_cleaned_up'):
            try:
                self._cleanup()
            except:
                pass  # Silence errors in __del__

    def _signal_handler(self, signum, frame):
        """Handle termination signals for graceful shutdown."""
        logging.info(f"Signal {signum} received. Initiating graceful shutdown...")
        self.running = False

    def _cleanup(self):
        """Clean up resources, stop threads, and disconnect SDR. This function is designed to be idempotent and can be safely called multiple times."""
        try:
            if not hasattr(self, '_cleaned_up'):
                self._cleaned_up = True  # Ensure cleanup only runs once
                logging.info("Starting cleanup...")

                # 1. Stop all threads and signal them to shutdown
                if hasattr(self, 'running'):
                    self.running = False  # Signal threads to stop

                # 2. Close debug plots before joining threads
                if hasattr(self, 'plotter') and self.plotter is not None:
                    try:
                        self.plotter.close_all()
                        logging.info("Closed debug plot windows.")
                    except Exception as e:
                        logging.error(f"Error closing debug plot windows: {e}")

                # 3. Clean up queues
                if hasattr(self, 'rx_queue'):
                    while not self.rx_queue.empty():
                        try:
                            self.rx_queue.get_nowait()
                        except Empty:
                            break
                if hasattr(self, 'tx_queue'):
                    while not self.tx_queue.empty():
                        try:
                            self.tx_queue.get_nowait()
                        except Empty:
                            break

                # 4. Join threads with timeout
                threads = [
                    ('RX', self.rx_thread),
                    ('TX', self.tx_thread),
                    ('TUI', self.tui_thread)
                ]

                for thread_name, thread in threads:
                    if thread is not None and thread.is_alive():
                        logging.info(f"Waiting for {thread_name} thread to finish...")
                        thread.join(timeout=5.0)
                        if thread.is_alive():
                            logging.warning(f"{thread_name} thread did not stop gracefully")
                        else:
                            logging.info(f"{thread_name} thread stopped")
                
                # 5. Disconnect SDR
                if hasattr(self, 'sdr') and self.sdr is not None:
                    try:
                        self.sdr.disconnect()
                        logging.info("SDR disconnected successfully.")
                    except Exception as e:
                        logging.error(f"Error disconnecting SDR: {e}")

                # 6. delete filter file if it exists
                if hasattr(self.matched_filter, 'hardware_filter_enable') and self.matched_filter.hardware_filter_enable:
                    filter_file = self.config['radio']['hardware_filter_file']
                    if os.path.exists(filter_file):
                        os.remove(filter_file)
                        logging.info(f"Deleted temporary filter file: {filter_file}")

                # 7. Close log files
                if hasattr(self, 'log_file'):
                    try:
                        with open(self.log_file, 'a') as f:
                            f.write(f"\n--- Chat Session Ended at {datetime.now().strftime('%H:%M:%S')} ---\n")
                    except Exception as e:
                        logging.error(f"Error closing chat log: {e}")

                # Force flush logging handlers
                for handler in logging.root.handlers[:]:
                    try:
                        handler.flush()
                        handler.close()
                    except:
                        pass

                logging.info("Cleanup completed successfully.")
        except Exception as e:
            logging.error(f"Error during application cleanup: {e}")


    # ================= Start and Stop of sub threads =================
    def start(self):
        """Start the SDR Chat Application."""
        if self.sdr.connect():  
            self.barker_detector.set_noise_floor_dB(self.sdr.measure_noise_floor_dB())
        else:
            logging.debug("Failed to connect to SDR.")
            return False
        
        self.running = True
        self.rx_thread = threading.Thread(target=self._rx_loop, daemon=True, name="RX_Thread")
        self.tx_thread = threading.Thread(target=self._tx_loop, daemon=True, name="TX_Thread")
        self.tui_thread = threading.Thread(target=self._tui_loop, daemon=True, name="TUI_Thread")

        self.rx_thread.start()
        self.tx_thread.start()
        self.tui_thread.start()

        logging.info("Chat Application started successfully.")

        return True

    def stop(self):
        """Stop the SDR Chat Application."""
        logging.info("Stopping SDR Chat Application...")
        self.running = False

        # Wait for threads to finish
        if self.rx_thread and self.rx_thread.is_alive():
            self.rx_thread.join(timeout=2)
            if self.rx_thread.is_alive():
                logging.warning("RX thread did not stop in time")
                
        if self.tx_thread and self.tx_thread.is_alive():
            self.tx_thread.join(timeout=2)
            if self.tx_thread.is_alive():
                logging.warning("TX thread did not stop in time")
                
        if self.tui_thread and self.tui_thread.is_alive():
            self.tui_thread.join(timeout=2)
            if self.tui_thread.is_alive():
                logging.warning("TUI thread did not stop in time")
        
        logging.info("All threads stopped.")

    # ================== Debug plotter setup ==================
    def _init_plotter(self):
        """Initialize the live plotter for debug mode."""
        try:
            if QApplication.instance() is None:
                self.qapp = QApplication(sys.argv)
            else:
                self.qapp = QApplication.instance()
            
            # Choose between single-window or multi-window mode
            use_multi_window = self.config.get('plotter', {}).get('multi_window', True)
            
            if use_multi_window:
                self.plotter = LiveSDRPlotterMultiWindow(self.config, self.plot_data_queue)
            else:
                self.plotter = LiveSDRPlotter(self.config, self.plot_data_queue)
            
            self.plotter.show()

            # Setup static plot signaler for thread-safe plot requests
            self.static_plot_signaler = StaticPlotSignaler()
            self.static_plot_signaler.plot_requested.connect(self._handle_static_plot)
           
            logging.info(f"Live plotter initialized ({'multi-window' if use_multi_window else 'single-window'} mode)")
        except Exception as e:
            logging.error(f"Failed to initialize live plotter: {e}")
            self.debug_mode = False
            self.plotter = None


    def _handle_static_plot(self, plot_data: dict):
        """Handle static plot request (runs in main thread)."""
        try:
            plot_type = plot_data.get('type')
            data = plot_data.get('data')
            title = plot_data.get('title', '')
            
            if plot_type == 'time_domain':
                self.static_plotter.plot_time_domain(
                    data, 
                    float(self.config['modulation']['sample_rate']),
                    title=title
                )
            elif plot_type == 'constellation':
                self.static_plotter.plot_constellation(data, title=title)
            elif plot_type == 'psd':
                sample_rate = float(plot_data.get('sample_rate', self.config['modulation']['sample_rate']))
                center_freq = float(plot_data.get('center_freq', self.config['plotter']['center_freq']))
                self.static_plotter.plot_psd(data, sample_rate, center_freq=center_freq, title=title)
            
            show(block=False)
            
        except Exception as e:
            logging.error(f"Error handling static plot: {e}")

    def request_static_plot(self, plot_data: dict):
        """Thread-safe method to request a static plot from any thread."""
        if self.debug_mode and hasattr(self, 'static_plot_signaler'):
            self.static_plot_signaler.plot_requested.emit(plot_data)



    # ================= Message Handling =================
    def queue_ack(self, msg_id: np.uint8):
        """Send an ACK for the recieved datagram carrying its msg_id."""
        try:
            ack_datagram = Datagram.as_ack(msg_id=msg_id)
            self.tx_queue.put(ack_datagram)
            logging.info(f"Enqueue ACK for transmission.\tmsg_ID: {msg_id}")
            return True
        
        except Exception as e:
            logging.error(f"Failed to enqueue ACK datagram ID {msg_id}: {e}")
            return False

    def queue_datagram(self, datagram: Datagram) -> bool:
        """Enqueue a datagram for transmission."""
        try:
            self.tx_queue.put(datagram, timeout=1)
            logging.info(f"Enqueued datagram for transmission.\tmsg_ID: {datagram.get_msg_id}")
            return True 
        
        except Exception as e:
            logging.error(f"Failed to enqueue datagram ID {datagram.get_msg_id}: {e}")
            return False
        

        
    # ================= Callback loops for threads =================
    def _rx_loop(self):
        """Receive loop - continuously receive data from SDR and process it."""
        logging.info("RX loop started.")

        while self.running:
            try:
                received_signal = self.sdr.sdr.rx()

                if self.matched_filter.hardware_filter_enable:
                    filtered_signal = received_signal  # Assume hardware filtering is applied by the SDR
                else:
                    filtered_signal = self.matched_filter.apply_filter(received_signal)

                 # === Send data to plotter if debug mode is enabled ===
                if self.debug_mode and self.plotter is not None:
                    try:
                        # Non-blocking put - drop if queue is full
                        self.plot_data_queue.put_nowait(filtered_signal.copy())
                    except Full:
                        pass  # Drop frame if plotter can't keep up

                synchronized_signal = self.synchronizer.synchronize(filtered_signal)
                decimated_signal = self.modulation_protocol.downsample_symbols(synchronized_signal)
                barker_index = self.barker_detector.detect(decimated_signal)

                if barker_index is not None:
                    try:
                        recieved_signal = self.barker_detector.remove_barker_code(received_signal, barker_index)
                        received_message = self.modulation_protocol.demodulate_message(recieved_signal)
                    except ValueError as e:
                        logging.warning(f"Message demodulation failed: {e}")
                        continue
                    except Exception as e:
                        logging.error(f"Unexpected error during demodulation: {e}")
                        continue

                    self.rx_queue.put(received_message)
                    self.tui_refresh_event.set()  # Signal TUI to refresh display

                    if received_message.msg_type == msgType.DATA:
                        logging.info(f"Received datagram: {received_message}")
                        self.queue_ack(received_message.get_msg_id)
                        self.chat_history_log(f"Received: [ID:{received_message.get_msg_id}]\t{received_message.get_payload.tobytes().decode('utf-8', errors='replace')}")
                    else:
                        logging.info(f"Received ACK for msg_ID: {received_message.get_msg_id}")
                        self.chat_history_log(f"Received: [ID:{received_message.get_msg_id}]\tACK")
                        
            except Exception as e:
                #logging.error(f"Error in receive loop: {e}")
                time.sleep(0.1)  # Sleep briefly to avoid tight error loop
                continue

        logging.info("RX loop stopped.")

    def _tx_loop(self):
        """Transmit loop - continuously check for outgoing messages and transmit them."""
        logging.info("TX loop started.")

        while self.running:
            try:
                datagram: Datagram = self.tx_queue.get(timeout=0.1)  # Wait for message to send
                modulated_signal = self.modulation_protocol.modulate_message(datagram)
                signal_with_barker = self.barker_detector.add_barker_code(modulated_signal)
                upsampled_signal = self.modulation_protocol.upsample_symbols(signal_with_barker)

                if self.matched_filter.hardware_filter_enable:
                    filtered_signal = upsampled_signal  # Assume hardware filtering is applied by the SDR TODO: Not working as inteded
                else:
                    filtered_signal = self.matched_filter.apply_filter(upsampled_signal)

                if self.debug_mode:
                    logging.debug(f"TX loop got datagram from queue: {datagram}")

                    #self.request_static_plot({
                    #    'type': 'time_domain',
                    #    'data': filtered_signal.copy(),
                    #    'title': f"Transmitted Signal"})
                    
                    #self.request_static_plot({
                    #    'type': 'constellation',
                    #    'data': filtered_signal.copy(),
                    #    'title': f"Transmitted Constellation"})
                    #
                    #self.request_static_plot({
                    #    'type': 'psd',
                    #    'data': filtered_signal.copy(),
                    #    'title': f"Transmitted Signal PSD",
                    #    'sample_rate': self.config['modulation']['sample_rate'],
                    #    'center_freq': self.config['plotter']['center_freq']
                    #})

                self.sdr.send_signal(filtered_signal)

                logging.info(f"Transmitted datagram: {datagram.get_msg_id}")
            except Empty:
                time.sleep(0.1)  # No message to send, sleep briefly
                continue  # No message to send, continue loop
            except Exception as e:
                logging.error(f"Error: {e}")
                time.sleep(0.1)  # Sleep briefly to avoid tight error loop
                continue

        logging.info("TX loop stopped.")

    def _tui_loop(self):
        """TUI loop - continuously check for user input and enqueue messages to send."""
        logging.info("TUI loop started.")

        self.tui.render_screen()  # Initial render of TUI

        while self.running:
            try:

                # Wait for either: screen refresh event OR stdin input (max 0.5s timeout)
                ready_to_read, _, _ = select.select([sys.stdin], [], [], 0.1)
                
                if self.tui_refresh_event.is_set():
                    while not self.rx_queue.empty():
                        try:
                            received_datagram: Datagram = self.rx_queue.get_nowait()
                            self.tui.add_message(received_datagram)
                            logging.debug(f"TUI processed received datagram ID: {received_datagram.get_msg_id}")
                        except Empty:
                            break  # No more messages to process
                    self.tui.render_screen()  # Update TUI display
                    self.tui_refresh_event.clear()  # Reset event

                if ready_to_read:
                    user_input = sys.stdin.readline().strip()
                    if user_input.lower() == "/quit":
                        logging.info("User requested to quit. Stopping application...")
                        self.running = False
                        break
                    elif user_input.startswith("/"):
                        logging.warning(f"Unknown command: {user_input}")
                        continue  # Ignore unknown commands

                    # send message as datagram
                    datagram = Datagram.as_string(user_input, msg_type=msgType.DATA)
                    self.queue_datagram(datagram)
                    self.tui.add_message(datagram)  # Add sent message to TUI display
                    self.tui.render_screen()  # Update TUI display after sending message
                    self.chat_history_log(f"Sent: [ID:{datagram.get_msg_id}]\t{user_input}")

      
            except Exception as e:
                logging.error(f"Error in TUI loop: {e}")
                continue

            time.sleep(0.1)  # Sleep briefly to avoid tight error loop

        logging.info("TUI loop stopped.")

        
    # ==================================Logging ======================================
    def chat_history_log(self, message: str):
        """Append a message to the chat history log file."""
        try:
            with open(self.log_file, 'a') as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
        except Exception as e:
            logging.error(f"Error writing to chat history log: {e}")

    def _setup_logging(self, debug_mode: str='INFO'):
        """Setup Python's built-in logging system"""
        
        log_level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'WARN': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        log_level = log_level_map.get(debug_mode, 'INFO')
        

        # Add color formatting
        class ColoredFormatter(logging.Formatter):
            COLORS = {
                'DEBUG': '\033[36m',    # Cyan
                'INFO': '\033[32m',     # Green
                'WARNING': '\033[33m',  # Yellow
                'ERROR': '\033[31m',    # Red
                'CRITICAL': '\033[35m'  # Magenta
            }
            RESET = '\033[0m'
            
            def format(self, record):
                log_message = super().format(record)
                return f"{self.COLORS.get(record.levelname, '')}{log_message}{self.RESET}"
        
        # File handler - writes all logs to file
        file_handler = logging.FileHandler(self.debug_file, mode='a')
        file_handler.setLevel(log_level)  
        file_handler.setFormatter(ColoredFormatter(
            fmt='[%(asctime)s] [%(levelname)-8s] [%(threadName)-12s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))

        # Configure root logger
        logging.basicConfig(
            level=logging.DEBUG,  # Capture all levels
            handlers=[file_handler],
            force=True  # Ensure that logging configuration is applied even if logging was previously configured
        )
        
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('pyqtgraph').setLevel(logging.WARNING)


        logging.info("\n\n--- New Application Session Started ---")
        logging.info(f"Logging configured: level={debug_mode}, file={self.debug_file}")



if __name__ == "__main__":
    app = None
    try:
        app = SDRChatApp()
        if not app.start():
            logging.critical("Failed to start SDR Chat Application. Exiting.")
            sys.exit(1)

        logging.info("Entering main loop. Press Ctrl+C to exit.")
        
        # Main loop - process Qt events if debug mode is enabled
        while app.running:
            if app.debug_mode and app.qapp is not None:
                # Process Qt events to keep plotter responsive
                app.qapp.processEvents()
                
            time.sleep(0.01)  # Small sleep to prevent CPU spinning

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Stopping application...")

    except Exception as e:
        logging.error(f"Unexpected error in main loop: {e}")

    finally:
        if app is not None:
            app.stop()
        logging.info("SDR Chat Application has been stopped.")
        sys.exit(0)