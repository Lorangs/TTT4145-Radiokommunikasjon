"""
To display logging file in a terminal with color coding, use the following command:
tail -f log/$(date +%Y-%m-%d)-debug.log

optionally pipe with grep:
| grep --color=always "ERROR\\|WARNING\\|INFO\\|DEBUG"

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
from modulation import normalize_config_modulation_name
from datagram import Datagram, msgType
from sdr_transciever import SDRTransciever
from filter import RRCFilter
from gold_detection import GoldCodeDetector, rank_gold_candidates
from synchronize import Synchronizer
from forward_error_correction import FCCodec
from convolutional_coder import ConvolutionalCoder
from interleaver import Interleaver
from project_logger import configure_project_logging, get_configured_log_level
from ARQ import StopAndWaitARQ
from RX_pipeline import run_rx_pipeline
from TX_pipeline import build_scrambler, build_tx_burst, decode_payload_symbols


# ================= read configuration file =================
try:
    with open("setup/config.yaml", 'r') as f:
        config = safe_load(f)
except Exception as e:
    print(f"Error loading config file: {e}")
    raise e

# Optional imports for debug mode
if config.get('radio', {}).get('debug_mode', False):
    from sdr_plots import LiveSDRPlotter, LiveSDRPlotterMultiWindow, StaticSDRPlotter, StaticPlotSignaler
    from matplotlib.pyplot import show
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QTimer
else:
    LiveSDRPlotter = None
    LiveSDRPlotterMultiWindow = None
    StaticPlotSignaler = None
    QApplication = None


class SDRChatApp:
    def __init__(self, config_file: Dict):
        """Initialize the SDR Chat Application."""
        self.config: Dict = config

        # ================= Initialize Modules with configuration =================
        self.modulation_name = normalize_config_modulation_name(config)
        self.interleaver = Interleaver(config)
        self.scrambler = build_scrambler(config)
        self.fec_codec = FCCodec(config)
        self.conv_coder = ConvolutionalCoder(config)
        self.matched_filter = RRCFilter(config)
        self.tui = ChatTUI(config)
        self.gold_detector = GoldCodeDetector(config)
        self.synchronizer = Synchronizer(config)
        self.sdr = SDRTransciever(config) # must be initilized after Matched Filter module.
        self.guard_symbols = int(config.get('gold_sequence', {}).get('guard_symbols', 32))
        reference_burst = build_tx_burst(
            config=self.config,
            datagram=Datagram.as_ack(np.uint8(0)),
            modulation_name=self.modulation_name,
            samples_per_symbol=int(self.config['modulation']['samples_per_symbol']),
            gold_detector=self.gold_detector,
            rrc_filter=self.matched_filter,
            fec=self.fec_codec,
            interleaver=self.interleaver,
            conv_coder=self.conv_coder,
            scrambler=self.scrambler,
            guard_symbols=self.guard_symbols,
        )
        self.payload_symbol_count = int(reference_burst.channel_payload_symbols.size)
        self.expected_tx_samples = int(reference_burst.tx_signal.size)

        # ================== Threading and synchronization primitives ==================
        self.stop_event: threading.Event = threading.Event()
        self.rx_thread: threading.Thread = None
        self.tx_thread: threading.Thread = None
        self.tui_thread: threading.Thread = None
        self.tui_refresh_event: threading.Event = threading.Event()

        # ================== Message queues for inter-thread communication ==================
        self.tx_queue: Queue[Datagram] = Queue(maxsize=int(config['radio']['queue_size']))
        self.control_tx_queue: Queue[Datagram] = Queue(maxsize=int(config['radio']['queue_size']))
        self.rx_queue: Queue[Datagram] = Queue(maxsize=int(config['radio']['queue_size']))
        self.ui_status_queue: Queue[tuple[str, np.uint8]] = Queue(maxsize=int(config['radio']['queue_size']))
        reliability_cfg = config.get('reliability', {})
        self.reliability = StopAndWaitARQ(
            ack_timeout_s=float(reliability_cfg.get('ack_timeout_seconds', 1.5)),
            max_retries=int(reliability_cfg.get('max_retries', 2)),
            duplicate_cache_size=int(reliability_cfg.get('duplicate_cache_size', 128)),
        )
    
        # ================== Logging setup ==================
        log_dir = "log"
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"{datetime.now().date()}-chat-history.txt")
        self.debug_file = os.path.join(log_dir, f"{datetime.now().date()}-debug.log")
        configure_project_logging(
            level_name=get_configured_log_level(config),
            session_name="debug",
            log_file=self.debug_file,
            console=True,
            file_output=True,
        )

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
        self.static_plotter = StaticSDRPlotter() if self.debug_mode else None
        self.static_plot_queue: Queue[Dict[str, np.ndarray]] = Queue(maxsize=8)  # Queue for static plot data (e.g., filter response, constellation points)

        if self.debug_mode:
            logging.info("Debug mode enabled - initializing live plotter")
            self._init_plotter()
        
                
        # ====================== Setup signal handlers for graceful shutdown =====================
        atexit.register(self._cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logging.info("SDR Chat Application initialized successfully.")

   
    # ================= Start and Stop of sub threads =================
    def start(self):
        """Start the SDR Chat Application."""
        if self.sdr.connect():  
            self.synchronizer.set_noise_floor(self.sdr.measure_noise_floor_dB())
        else:
            logging.debug("Failed to connect to SDR.")
            return False
        
        self.stop_event.clear()
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
        self.stop_event.set()

        # Wait for threads to finish
        if self.rx_thread and self.rx_thread.is_alive():
            self.rx_thread.join(timeout = 2)
            if self.rx_thread.is_alive():
                logging.warning("RX thread did not stop in time")
                
        if self.tx_thread and self.tx_thread.is_alive():
            self.tx_thread.join(timeout = 2)
            if self.tx_thread.is_alive():
                logging.warning("TX thread did not stop in time")
                
        if self.tui_thread and self.tui_thread.is_alive():
            self.tui_thread.join(timeout = 2)
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
        """Enqueue ACK for the recieved datagram carrying its msg_id."""
        try:
            ack_datagram = Datagram.as_ack(msg_id=msg_id)
            self.control_tx_queue.put(ack_datagram, timeout=1)
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

    def _queue_ui_status(self, event_type: str, msg_id: np.uint8) -> None:
        try:
            self.ui_status_queue.put_nowait((event_type, np.uint8(msg_id)))
            self.tui_refresh_event.set()
        except Full:
            logging.debug("UI status queue full; dropping %s for msg_ID=%s", event_type, int(msg_id))

    def _transmit_datagram_once(self, datagram: Datagram):
        tx_burst = build_tx_burst(
            config=self.config,
            datagram=datagram,
            modulation_name=self.modulation_name,
            samples_per_symbol=int(self.config['modulation']['samples_per_symbol']),
            gold_detector=self.gold_detector,
            rrc_filter=self.matched_filter,
            fec=self.fec_codec,
            interleaver=self.interleaver,
            conv_coder=self.conv_coder,
            scrambler=self.scrambler,
            guard_symbols=self.guard_symbols,
        )
        filtered_signal = (tx_burst.tx_signal / (2**14)).astype(np.complex64, copy=False)

        if self.debug_mode:
            logging.debug(f"TX loop got datagram from queue: {datagram}")

        self.sdr.send_signal(filtered_signal)
        logging.info(
            "Transmitted %s datagram: %s",
            datagram.get_msg_type.name,
            int(datagram.get_msg_id),
        )

    def _drain_control_tx_queue(self):
        """Transmit queued ACKs before continuing with lower-priority DATA traffic."""
        while not self.stop_event.is_set():
            try:
                datagram = self.control_tx_queue.get_nowait()
            except Empty:
                return
            self._transmit_datagram_once(datagram)

    def _transmit_with_ack_retry(self, datagram: Datagram):
        result = self.reliability.transmit_with_retry(
            datagram,
            send_once=self._transmit_datagram_once,
            stop_event=self.stop_event,
            drain_control_queue=self._drain_control_tx_queue,
            on_retry_timeout=self._log_retry_timeout,
        )

        if result.ack_received:
            logging.info(
                "ACK received for msg_ID=%s after %d attempt(s).",
                int(datagram.get_msg_id),
                result.attempts,
            )
            return

        if not self.stop_event.is_set():
            logging.warning(
                "No ACK received for msg_ID=%s after %d attempt(s).",
                int(datagram.get_msg_id),
                result.attempts,
            )
            self.chat_history_log(
                f"Send failed: [ID:{int(datagram.get_msg_id)}]\t"
                f"{datagram.payload_text(trim_padding=True)}"
            )
            self._queue_ui_status('failed', datagram.get_msg_id)

    def _log_retry_timeout(self, datagram: Datagram, next_attempt: int, max_attempts: int):
        logging.warning(
            "ACK timeout for msg_ID=%s, retrying (%d/%d).",
            int(datagram.get_msg_id),
            next_attempt,
            max_attempts,
        )

        
    # ================= Callback loops for threads =================
    def _decode_received_datagram(self, rx_state) -> Datagram | None:
        """Decode the first valid datagram candidate from a synchronized RX burst."""
        phase_candidates = rank_gold_candidates(
            rx_state.fine_signal,
            self.gold_detector,
            self.modulation_name,
            expected_index=rx_state.expected_header_index,
            search_radius=int(max(0, self.synchronizer.timing_header_search_radius_symbols)),
            top_k=int(max(1, self.synchronizer.timing_header_candidate_count)),
        )

        gold_cfg = self.config.get('gold_sequence', {})
        decode_candidate_count = int(max(1, gold_cfg.get('decode_candidate_count', 2)))
        decode_candidate_min_peak = float(
            gold_cfg.get(
                'decode_candidate_min_peak',
                self.gold_detector.correlation_scale_factor_threshold,
            )
        )
        expected_header_index = (
            int(rx_state.expected_header_index)
            if rx_state.expected_header_index is not None
            else None
        )
        exact_index_candidates = []
        fallback_candidates = []
        seen_candidates: set[tuple[int, complex]] = set()

        for candidate in phase_candidates:
            candidate_index = candidate['index']
            if candidate_index is None:
                continue
            if float(candidate['peak']) < decode_candidate_min_peak:
                continue

            candidate_key = (int(candidate_index), complex(candidate['rotation']))
            if candidate_key in seen_candidates:
                continue
            seen_candidates.add(candidate_key)

            if expected_header_index is not None and int(candidate_index) == expected_header_index:
                exact_index_candidates.append(candidate)
            else:
                fallback_candidates.append(candidate)

        exact_index_candidates.sort(key=lambda candidate: float(candidate['peak']), reverse=True)
        fallback_candidates.sort(key=lambda candidate: float(candidate['peak']), reverse=True)
        eligible_candidates = (
            exact_index_candidates[:decode_candidate_count]
            if exact_index_candidates
            else fallback_candidates[:decode_candidate_count]
        )

        for candidate in eligible_candidates:
            candidate_index = candidate['index']
            if candidate_index is None:
                continue

            required_symbols = (
                int(candidate_index)
                + int(self.gold_detector.gold_symbols.size)
                + int(self.payload_symbol_count)
            )
            if required_symbols > int(candidate['decisions'].size):
                continue

            payload_symbols = self.gold_detector.remove_gold_symbols(
                candidate['decisions'],
                candidate_index,
            )
            payload_symbols = np.asarray(payload_symbols).astype(
                np.complex64,
                copy=False,
            ).reshape(-1)[: self.payload_symbol_count]

            try:
                return decode_payload_symbols(
                    payload_symbols=payload_symbols,
                    modulation_name=self.modulation_name,
                    fec=self.fec_codec,
                    interleaver=self.interleaver,
                    conv_coder=self.conv_coder,
                    scrambler=self.scrambler,
                )
            except Exception as e:
                logging.debug(
                    "Candidate decode failed at index %s with peak %.3f: %s",
                    str(candidate_index),
                    float(candidate['peak']),
                    e,
                )

        return None

    def _rx_loop(self):
        """Receive loop - continuously receive data from SDR and process it."""
        logging.info("RX loop started.")

        while not self.stop_event.is_set():
            try:
                received_signal = self.sdr.sdr.rx()
                rx_state = run_rx_pipeline(
                    config=self.config,
                    modulation_name=self.modulation_name,
                    received_signal=received_signal,
                    expected_tx_samples=self.expected_tx_samples,
                    payload_symbol_count=self.payload_symbol_count,
                    guard_symbols=self.guard_symbols,
                    gold_detector=self.gold_detector,
                    rrc_filter=self.matched_filter,
                    synchronizer=self.synchronizer,
                )


                # === Send data to plotter if debug mode is enabled ===
                if self.debug_mode and self.plotter is not None:
                    try:
                        # Non-blocking put - drop if queue is full
                        self.plot_data_queue.put_nowait(rx_state.fine_signal.copy())
                    except Full:
                        pass  # Drop frame if plotter can't keep up
                # ================================================================

                received_message = self._decode_received_datagram(rx_state)
                if received_message is not None:
                    if received_message.get_msg_type == msgType.DATA:
                        self.queue_ack(received_message.get_msg_id)
                        if self.reliability.mark_inbound_data(received_message):
                            self.rx_queue.put(received_message)
                            self.tui_refresh_event.set()  # Signal TUI to refresh display
                            logging.info(f"Received datagram: {received_message}")
                            self.chat_history_log(
                                f"Received: [ID:{received_message.get_msg_id}]\t"
                                f"{received_message.payload_text(trim_padding=True)}"
                            )
                        else:
                            logging.info(
                                "Received duplicate DATA datagram for msg_ID=%s; resent ACK only.",
                                int(received_message.get_msg_id),
                            )
                    else:
                        self.reliability.acknowledge(received_message.get_msg_id)
                        self.rx_queue.put(received_message)
                        self.tui_refresh_event.set()  # Signal TUI to refresh display
                        logging.info(f"Received ACK for msg_ID: {received_message.get_msg_id}")
                        self.chat_history_log(f"Received: [ID:{received_message.get_msg_id}]\tACK")
                        
            except Exception as e:
                logging.debug(f"RX loop skipped buffer: {e}")
                time.sleep(0.1)  # Sleep briefly to avoid tight error loop
                continue

        logging.info("RX loop stopped.")

    def _tx_loop(self):
        """Transmit loop - continuously check for outgoing messages and transmit them."""
        logging.info("TX loop started.")

        while not self.stop_event.is_set():
            try:
                self._drain_control_tx_queue()
                datagram: Datagram = self.tx_queue.get(timeout=0.1)  # Wait for message to send
                if datagram.get_msg_type == msgType.DATA:
                    self._transmit_with_ack_retry(datagram)
                else:
                    self._transmit_datagram_once(datagram)
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

        while not self.stop_event.is_set():
            try:

                # Wait for either: screen refresh event OR stdin input (max 0.5s timeout)
                ready_to_read, _, _ = select.select([sys.stdin], [], [], 0.1)
                
                if self.tui_refresh_event.is_set():
                    while not self.rx_queue.empty():
                        try:
                            received_datagram: Datagram = self.rx_queue.get_nowait()
                            self.tui.add_message(received_datagram, is_local=False)
                            logging.debug(f"TUI processed received datagram ID: {received_datagram.get_msg_id}")
                        except Empty:
                            break  # No more messages to process
                    while not self.ui_status_queue.empty():
                        try:
                            event_type, msg_id = self.ui_status_queue.get_nowait()
                        except Empty:
                            break

                        if event_type == 'failed':
                            self.tui.mark_failed(msg_id)
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
                    while len(user_input.encode('utf-8')) > Datagram.PAYLOAD_SIZE:
                        logging.warning("Input message is too long and will be truncated to fit payload size.")
                        sliced_user_input = user_input[: Datagram.PAYLOAD_SIZE]
                        datagram = Datagram.as_string(sliced_user_input, msg_type=msgType.DATA)
                        self.queue_datagram(datagram)
                        user_input = user_input[Datagram.PAYLOAD_SIZE :]  # Remove the part that was sent
                   
                    # Final slice (or if input was already short enough)
                    sliced_user_input = user_input
                    datagram = Datagram.as_string(user_input, msg_type=msgType.DATA)
                    self.queue_datagram(datagram)
                    self.tui.add_message(datagram, is_local=True)  # Add sent message to TUI display
                    self.tui.render_screen()  # Update TUI display after sending message
                    self.chat_history_log(f"Sent: [ID:{datagram.get_msg_id}]\t{sliced_user_input}")

        
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

    # ================== Cleanup and Signal Handling ==================
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
        self.stop_event.set()

    def _cleanup(self):
        """Clean up resources, stop threads, and disconnect SDR. This function is designed to be idempotent and can be safely called multiple times."""
        try:
            if not hasattr(self, '_cleaned_up'):
                self._cleaned_up = True  # Ensure cleanup only runs once
                logging.info("Starting cleanup...")

                # 1. Stop all threads and signal them to shutdown
                if hasattr(self, 'stop_event'):
                    self.stop_event.set()

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
                    filter_file = self.config['filter']['hardware_filter_file']
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




if __name__ == "__main__":
    app = None
    try:
        app = SDRChatApp(config)
        if not app.start():
            logging.critical("Failed to start SDR Chat Application. Exiting.")
            sys.exit(1)

        logging.info("Entering main loop. Press Ctrl+C to exit.")
        
        # Main loop - process Qt events if debug mode is enabled
        while not app.stop_event.is_set():
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
