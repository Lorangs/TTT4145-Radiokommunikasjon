"""
Single sided transmission SDR Chat Application Receiver
"""

# import system modules
import os
import sys
import time
import logging
import threading
from queue import Queue, Empty, Full
from datetime import datetime
import signal
import atexit

if os.name == "nt":
    import msvcrt
else:
    import select

# import third party moduels
import numpy as np
from yaml import safe_load


# import modules
from chat_tui import ChatTUI
from modulation import ModulationProtocol
from datagram import Datagram, msgType
from sdr_transciever import SDRTransciever
from filter import RRCFilter
from gold_detection import GoldCodeDetector
from equalizer import equalize_from_known_header
from synchronize import Synchronizer
from forward_error_correction import FCCodec
from convolutional_coder import ConvolutionalCoder
from interleaver import Interleaver
from scrambler import LFSRScrambler
from project_logger import configure_project_logging, get_configured_log_level



##############################################################################################
# ================= Callback loops for threads =================
##############################################################################################
def _rx_loop():
    """Receive loop - continuously receive data from SDR and process it."""
    global rx_queue, plot_data_queue, debug_mode, plotter
    logging.debug("RX loop started.")

    while not stop_event.is_set():
        try:
            received_signal = sdr.sdr.rx()

            coarse_freq_adjusted = synchronizer.coarse_frequenzy_synchronization(received_signal)
            if coarse_freq_adjusted is None:
                continue    # skip if signal is too weak to process

            padded_signal = matched_filter.pad_signal_front_and_back(coarse_freq_adjusted)  
            filtered_signal = matched_filter.apply_filter(padded_signal)
            time_adjusted = synchronizer.gardner_timing_synchronization(filtered_signal)
            fine_freq_adjusted = synchronizer.fine_frequenzy_synchronization(time_adjusted)
            gold_index, _ = gold_detector.detect_with_rotation(
                fine_freq_adjusted,
                EXPECTED_PAYLOAD_SYMBOLS,
            )

            # === Send data to plotter if debug mode is enabled ===
            if debug_mode and plotter is not None:
                try:
                    # Non-blocking put - drop if queue is full
                    plot_data_queue.put_nowait(fine_freq_adjusted.copy())
                except Full:
                    pass  # Drop frame if plotter can't keep up
                except Exception as e:
                    logging.error(f"Error sending data to plotter: {e}")
                    pass


            if gold_index is None:
                # logging.debug("Gold code not detected in received signal. Skipping processing of this signal.")
                continue   # skip if gold code is not detected, likely not a valid signal to process
            if not gold_detector.candidate_fits_frame(
                len(fine_freq_adjusted), 
                gold_index,
                EXPECTED_PAYLOAD_SYMBOLS
            ):
                continue
            best_rotation = gold_detector.estimate_rotation_from_gold(
                fine_freq_adjusted,
                gold_index,
            )
          
            # Try the Gold-based rotation first, then fall back to the other
            # allowed constellation rotations if decode fails.
            selected_rotated_signal = None
            selected_frame_synched_signal = None
            received_datagram = None

            def apply_equalizer(rotated_signal: np.ndarray) -> np.ndarray:
                if not EQUALIZER_ENABLED:
                    return rotated_signal
                return equalize_from_known_header(
                    rotated_signal,
                    gold_index,
                    gold_detector.gold_symbols[0],
                )

            for rotation in _decode_rotation_fallback_order(
                best_rotation,
                modulation_protocol.modulation_type,
            ):
                try:
                    rotated_signal = gold_detector.rotate_signal(fine_freq_adjusted, rotation)
                    equalized_signal = apply_equalizer(rotated_signal)
                    frame_synched_signal = gold_detector.remove_gold_symbols(
                        equalized_signal,
                        gold_index,
                        EXPECTED_PAYLOAD_SYMBOLS,
                    )
                    received_bits = modulation_protocol.demodulate_signal(frame_synched_signal)

                    logging.debug(
                        "Frame sync: gold_index=%s tried_rotation=%s frame_symbols=%d received_bytes=%d",
                        gold_index,
                        rotation,
                        len(frame_synched_signal),
                        len(received_bits),
                    )

                    conv_decoded_bytes = conv_coder.decode(received_bits)
                    descrambled_bytes = scrambler.apply(conv_decoded_bytes)
                    interleaved_bytes = interleaver.deinterleave(descrambled_bytes)
                    fec_decoded_bits = fec_codec.decode(interleaved_bytes)
                    received_datagram = Datagram.unpack(fec_decoded_bits)

                    selected_rotated_signal = equalized_signal
                    selected_frame_synched_signal = frame_synched_signal
                    break
                except (ValueError, RuntimeError) as e:
                    logging.debug(
                        "Rotation decode attempt failed: gold_index=%s rotation=%s error=%s",
                        gold_index,
                        rotation,
                        e,
                    )
                    continue

            if received_datagram is None:
                continue

            if debug_mode:
                # === Constellation, PSD, and Eye Diagram plots when debug is enabled and gold code is detected ===   
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                eye_window = _extract_eye_window(
                    filtered_signal,
                    gold_index,
                    EXPECTED_PAYLOAD_SYMBOLS,
                    SAMPLES_PER_SYMBOL,
                )
                eye_offset = _best_eye_offset(eye_window, SAMPLES_PER_SYMBOL)
                aligned_eye_window = eye_window[eye_offset:]

                capture_plot_if_enabled(
                    "rx_gold_detect",
                    "psd",
                    coarse_freq_adjusted,
                    title=f"RX PSD After Coarse Frequency Correction {timestamp}",
                    stem=f"rx_psd_after_coarse_frequency_correction_{timestamp}",
                    sample_rate=float(config["modulation"]["sample_rate"]),
                    center_freq=float(config["plotter"]["center_freq"]),
                )
                capture_plot_if_enabled(
                    "rx_gold_detect",
                    "eye",
                    aligned_eye_window,
                    title=f"RX Eye Diagram After Matched Filter {timestamp}",
                    stem=f"rx_eye_after_matched_filter_{timestamp}",
                    samples_per_symbol=SAMPLES_PER_SYMBOL,
                )
                capture_plot_if_enabled(
                    "rx_gold_detect",
                    "constellation",
                    time_adjusted,
                    title=f"RX Constellation After Gardner Timing Recovery {timestamp}",
                    stem=f"rx_constellation_after_gardner_timing_{timestamp}",

                )
                capture_plot_if_enabled(
                    "rx_gold_detect",
                    "constellation",
                    selected_rotated_signal,
                    title=f"RX Stream Constellation After Selected Rotation {timestamp}",
                    stem=f"rx_stream_constellation_after_selected_rotation_{timestamp}",
                )
                capture_plot_if_enabled(
                    "rx_gold_detect",
                    "constellation",
                    selected_frame_synched_signal,
                    title=f"RX Payload Constellation After Selected Rotation {timestamp}",
                    stem=f"rx_payload_constellation_after_selected_rotation_{timestamp}",
                )
                capture_plot_if_enabled(
                    "rx_gold_detect",
                    "symbol_eye",
                    selected_frame_synched_signal.real,
                    title=f"RX Payload Symbol Eye I After Selected Rotation {timestamp}",
                    stem=f"rx_payload_symbol_eye_i_after_selected_rotation_{timestamp}",
                )
                capture_plot_if_enabled(
                    "rx_gold_detect",
                    "symbol_eye",
                    selected_frame_synched_signal.imag,
                    title=f"RX Payload Symbol Eye Q After Selected Rotation {timestamp}",
                    stem=f"rx_payload_symbol_eye_q_after_selected_rotation_{timestamp}",
                )

            if received_datagram.get_msg_type == msgType.DATA:
                logging.info(f"Received datagram: {received_datagram}")
                try:
                    rx_queue.put(received_datagram)
                    tui_refresh_event.set()  # Signal TUI to refresh display
                except Full:
                    logging.error(f"RX queue is full. Dropping received datagram ID {received_datagram.get_msg_id}.")
                    continue


        except ValueError as e:
            logging.warning(f"Did not receive valid signal: {e}")
            continue
        except RuntimeError as e:
            logging.error(f"Runtime error in RX loop: {e}")
            stop_event.set()  # Trigger shutdown on critical errors
            break
        except Exception as e:
            logging.error(f"Unexpected error in RX loop: {e}")
            continue

    logging.debug("RX loop stopped.")

def _tui_loop():
    """
        TUI loop - continuously check for user input and enqueue messages to send.
        Only render if there are new messages or user input to process, to avoid unnecessary CPU usage and flickering.
    """
    global rx_queue
    logging.debug("TUI loop started.")

    tui.render_screen()  # Initial render of TUI

    while not stop_event.is_set():
        try:

            if tui_refresh_event.is_set():
                while not rx_queue.empty():
                    try:
                        received_datagram: Datagram = rx_queue.get_nowait()
                        tui.add_message(received_datagram, sent_by_self=False)
                        logging.debug(f"TUI processed received datagram ID: {received_datagram.get_msg_id}")
                    except Empty:
                        break  # No more messages to process
                tui_refresh_event.clear()  # Reset event
                tui.render_screen()  # Update TUI display

        except Exception as e:
            logging.error(f"Error in TUI loop: {e}")
            continue

        time.sleep(0.1)  # Sleep briefly to reduce CPU usage.
    logging.debug("TUI loop stopped.")

        

# ================= Start and Stop of sub threads =================
def start():
    """Start the SDR Chat Application."""
    global rx_thread, tui_thread
    
    if sdr.connect():  
        synchronizer.set_noise_floor(sdr.measure_noise_floor_dB())
    else:
        logging.debug("Failed to connect to SDR.")
        return False
    
    try:
        stop_event.clear()
        rx_thread = threading.Thread(target=_rx_loop, daemon=True, name="RX_Thread")
        tui_thread = threading.Thread(target=_tui_loop, daemon=True, name="TUI_Thread")

        rx_thread.start()
        tui_thread.start()
        return True
    
    except Exception as e:
        logging.error(f"Error starting threads: {e}")
        stop_event.set()
        return False


def stop():
    """Stop the SDR Chat Application."""
    global rx_thread, tui_thread 
    logging.info("Stopping SDR Chat Application...")
    stop_event.set()

    for name, thread in (("RX_Thread", rx_thread), ("TUI_Thread", tui_thread)):
        if thread and thread.is_alive():
            try:
                thread.join(timeout=2.0)
                if thread.is_alive():
                    logging.warning(f"{name} thread did not stop within timeout")
            except Exception as e:
                logging.error(f"Error waiting for {name} thread: {e}")

    # clear references
    rx_thread = None
    tui_thread = None    

def _signal_handler(signum, frame):
    """Handle termination signals for graceful shutdown."""
    logging.info(f"Signal {signum} received. Initiating graceful shutdown...")
    stop_event.set()

def _cleanup():
    """Clean up resources safely. Idempotent."""
    global _cleaned_up

    lock = globals().get("_cleanup_lock", None)
    if lock is None:
        return

    with lock:
        if _cleaned_up:
            return
        _cleaned_up = True

    logging.info("Starting cleanup...")

    stop()
    
    # Close debug plot windows
    if plotter is not None:
        try:
            plotter.close_all()
            logging.info("Closed debug plot windows.")
        except Exception as e:
            logging.error(f"Error closing debug plot windows: {e}")

    # Drain queues
    for q in (rx_queue, plot_data_queue):
        while not q.empty():
            try:
                q.get_nowait()
            except Empty:
                break

    # Disconnect SDR
    try:
        if sdr is not None:
            sdr.disconnect()
            logging.info("SDR disconnected successfully.")
    except Exception as e:
        logging.error(f"Error disconnecting SDR: {e}")

    # Remove temporary filter file
    try:
        if hasattr(matched_filter, "hardware_filter_enable") and matched_filter.hardware_filter_enable:
            filter_file = config["filter"]["hardware_filter_file"]
            if os.path.exists(filter_file):
                os.remove(filter_file)
                logging.info(f"Deleted temporary filter file: {filter_file}")
    except Exception as e:
        logging.error(f"Error deleting temporary filter file: {e}")

    # Session end marker
    try:
        with open(log_file, "a") as f:
            f.write(f"\n--- Chat Session Ended at {datetime.now().strftime('%H:%M:%S')} ---\n")
    except Exception as e:
        logging.error(f"Error closing chat log: {e}")

    logging.info("Cleanup completed successfully.")




##################################################################################
# ================== Helper functions for plotting and logging ==================
##################################################################################
def request_static_plot(plot_data: dict):
    """Thread-safe method to request a static plot from any thread."""
    global debug_mode, static_plot_signaler
    if debug_mode and hasattr(static_plot_signaler, 'plot_requested'):
        static_plot_signaler.plot_requested.emit(plot_data)

def capture_plot_if_enabled(
    event_name: str,
    plot_type: str,
    data: np.ndarray,
    title: str,
    stem: str,
    **extra,
):
    """Capture and save a plot if enabled in configuration."""
    global static_plotter
    capture_cfg = config.get("plot_capture", {})

    if static_plotter is None:
        return
    if not capture_cfg.get("enabled", False):
        return

    if event_name == "tx_burst" and not capture_cfg.get("save_tx_burst", False):
        return
    if event_name == "rx_gold_detect" and not capture_cfg.get("save_rx_gold_detect", False):
        return

    plot_request = {
        "type": plot_type,
        "data": np.asarray(data).copy(),
        "title": title,
        "save": True,
        "stem": stem,
    }
    plot_request.update(extra)
    request_static_plot(plot_request)

def _handle_static_plot(plot_data: dict):
    """Handle static plot request (runs in main thread)."""
    try:
        plot_type = plot_data.get('type')
        data = plot_data.get('data')
        title = plot_data.get('title', '')
        fig = None

        if plot_type == 'time_domain':
            fig = static_plotter.plot_time_domain(
                data, 
                float(plot_data.get('sample_rate', config['modulation']['sample_rate'])),
                title=title
            )
        elif plot_type == 'constellation':
            fig = static_plotter.plot_constellation(data, title=title)
        elif plot_type == 'psd':
            sample_rate = float(plot_data.get('sample_rate', config['modulation']['sample_rate']))
            center_freq = float(plot_data.get('center_freq', config['plotter']['center_freq']))
            fig = static_plotter.plot_psd(data, sample_rate, center_freq=center_freq, title=title)
        elif plot_type == 'eye':
            fig = static_plotter.plot_eye_diagram(
                data,
                int(plot_data.get('samples_per_symbol', config['modulation']['samples_per_symbol'])),
                title=title,
            )
        elif plot_type == 'symbol_eye':
            fig = static_plotter.plot_symbol_eye(
                data,
                title=title,
            )

        if fig is None:
            return

        if plot_data.get("save", False):
            output_dir = config.get("plot_capture", {}).get("output_dir", "artifacts/rf_plots")
            static_plotter.save_named_figure(
                fig,
                output_dir=output_dir,
                stem=plot_data.get("stem", plot_type),
                title=title,
                close=True,
            )
        else:
            show(block=False)
        
    except Exception as e:
        logging.error(f"Error handling static plot: {e}")


    ##################################################################################
    # ================== Helper functions for runtime ==================
    ###################################################################################

def _best_eye_offset(samples: np.ndarray, samples_per_symbol: int) -> int:
    """
    Pick the sample offset that gives the strongest symbol-energy separation for an eye plot.
    """
    sample_array = np.asarray(samples)
    if samples_per_symbol <= 0 or sample_array.size < samples_per_symbol * 4:
        return 0

    best_offset = 0
    best_score = -np.inf

    for offset in range(samples_per_symbol):
        decimated = sample_array[offset::samples_per_symbol]
        if decimated.size < 8:
            continue

        score = float(np.mean(np.abs(decimated) ** 2))
        if score > best_score:
            best_score = score
            best_offset = offset

    return best_offset

def _extract_eye_window(
    filtered_signal: np.ndarray,
    gold_index: int,
    expected_payload_symbols: int,
    samples_per_symbol: int,
    eye_symbols_margin: int = 16,
) -> np.ndarray:
    """
    Crop a matched-filtered sample window around the detected frame for eye plotting.

    The frame start is detected after timing recovery in symbol units, so this crop is an
    approximate sample-domain window around the same region of the matched-filter output.
    """
    gold_len = len(gold_detector.gold_symbols[0])
    total_symbols = gold_len + expected_payload_symbols + gold_len

    start_symbol = max(0, int(gold_index) - int(eye_symbols_margin))
    stop_symbol = int(gold_index) + total_symbols + int(eye_symbols_margin)

    start_sample = start_symbol * int(samples_per_symbol)
    stop_sample = min(len(filtered_signal), stop_symbol * int(samples_per_symbol))

    return np.asarray(filtered_signal)[start_sample:stop_sample]

def _decode_rotation_fallback_order(
    best_rotation: int,
    modulation_name: str,
) -> tuple[int, ...]:
    """
    Return the ordered list of rotations to try during payload decode.

    The Gold-based phase estimate is tried first.
    For QPSK, the remaining quadrants are tried after that.
    """
    mod = modulation_name.upper().strip()

    if mod == "QPSK":
        ordered = [best_rotation, 0, 90, 180, 270]
    elif mod == "BPSK":
        ordered = [best_rotation, 0, 180]
    else:
        ordered = [best_rotation]

    unique: list[int] = []
    for rotation in ordered:
        if rotation not in unique:
            unique.append(rotation)

    return tuple(unique)

def calculate_expected_payload_symbols(
    config: dict,
) -> int:
    """
    Calculate the expected number of payload symbols based on configuration and modulation type.
        args:   config: Configuration dictionary loaded from YAML file.
            conv_coder: Convolutional coder instance.
            modulation_name: Name of the modulation type.
        returns: Expected number of symbols in the payload after modulation and coding. 
    """
    mod_type = config["modulation"]["type"].upper().strip()
    if mod_type == "BPSK":
        bps = 1
    elif mod_type == "QPSK":
        bps = 2
    else:
        raise ValueError(f"Unsupported modulation type for payload sizing: {mod_type}")

    datagram_bytes = int(config["datagram"]["total_size"])
    reed_solomon_bytes = int(config["coding"]["rs_added_bytes"])
    tail_byte = 1 # Tail bits added by convolutional coder (assumes 1 byte of tail bits, adjust if different)

    conv_n = int(config["coding"]["conv_n"])
    conv_input_bits = (datagram_bytes + reed_solomon_bytes) * 8
    conv_output_bits = (conv_input_bits + tail_byte * 8) * conv_n

    logging.debug(f"Calculated expected payload symbols: {conv_output_bits // bps}")
    return conv_output_bits // bps

    

if __name__ == "__main__":
    # ================= read configuration file =================
    try:
        with open("setup/config.yaml", 'r') as f:
            config = safe_load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        raise e
    
    # Constants derived from configuration
    SAMPLES_PER_SYMBOL = int(config['modulation']['samples_per_symbol'])
    MAX_RETRIES = int(config['datagram']['max_retries'])  # Maximum number of retransmission attempts for unacknowledged messages
    ACK_TIMEOUT_ms = float(config['datagram']['ack_timeout_ms'])  # Timeout for waiting for ACKs (converted to milliseconds
    GUARD_SYMBOLS = np.zeros(
        int(config['transmitter']['tx_guard_symbols']) * SAMPLES_PER_SYMBOL,
        dtype=np.complex64,
    )  # Sample-rate guard interval inserted after upsampling/filtering.
    TX_PEAK_SCALE = float(config['transmitter']['tx_peak_scale']) # Normalization factor for TX Bursts
    link_control_config = config.get("link_control", {})
    EQUALIZER_ENABLED = bool(config["synchronization"].get("short_equalizer_enable", True))
    PAYLOAD_SIZE = int(config['datagram']['payload_size'])
    # ================== Logging setup ==================
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{datetime.now().date()}-chat-history.txt")
    debug_file = os.path.join(log_dir, f"{datetime.now().date()}-debug.log")
    configure_project_logging(
        level_name=get_configured_log_level(config),
        session_name="debug",
        log_file=debug_file,
        console=bool(config['logging']['console']),
        file_output=bool(config['logging']['file']),
    )

    try:
        with open(log_file, 'a') as f:
            f.write(f"\n\n--- New Chat Session Started at {datetime.now().time()} ---\n")
    except Exception as e:
        logging.error(f"Error initializing chat history log: {e}")
        raise e
    
    # ================== Signal handlers for graceful shutdown ==================
    atexit.register(_cleanup)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    logging.info("SDR Chat Application initialized successfully.")


    # ====================== Optional imports for debug mode ========================
    if config.get('radio', {}).get('debug_mode', False):
        from sdr_plots import LiveSDRPlotter, LiveSDRPlotterMultiWindow, StaticSDRPlotter, StaticPlotSignaler
        from matplotlib.pyplot import show
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import QTimer
    else:
        LiveSDRPlotter = None
        LiveSDRPlotterMultiWindow = None
        StaticSDRPlotter = None
        QApplication = None


    # ================= Initialize Modules with configuration =================
    modulation_protocol = ModulationProtocol(config)
    interleaver = Interleaver(config)
    scrambler = LFSRScrambler(config)
    fec_codec = FCCodec(config)
    conv_coder = ConvolutionalCoder(config)
    matched_filter = RRCFilter(config)
    tui = ChatTUI(config)
    gold_detector = GoldCodeDetector(config)
    synchronizer = Synchronizer(config)
    sdr = SDRTransciever(config)  # Must be initialized after matched_filter.

    # ================= Initialize additional constants =================
    EXPECTED_PAYLOAD_SYMBOLS = calculate_expected_payload_symbols(config)

    # ================== Threading and synchronization primitives ==================
    stop_event: threading.Event = threading.Event()
    tui_refresh_event: threading.Event = threading.Event()
    rx_thread: threading.Thread = None
    tui_thread: threading.Thread = None

    _cleaned_up = False
    _cleanup_lock = threading.Lock()

    # ================== Message queues for inter-thread communication ==================
    rx_queue: Queue[Datagram] = Queue(maxsize=int(config['radio']['queue_size']))       # Queue for incoming messages received by the RX thread to be processed by the TUI thread
    

    # ================== Debug mode setup ==================
    debug_mode = bool(config['radio']['debug_mode'])
    qapp = None
    plotter = None
    plot_data_queue: Queue[np.ndarray] = Queue(maxsize=32)
    static_plotter = StaticSDRPlotter() if debug_mode else None
    static_plot_signaler = None

    if debug_mode:
        logging.info("Debug mode enabled - initializing live plotter")

        try:
            if QApplication.instance() is None:
                qapp = QApplication(sys.argv)
            else:
                qapp = QApplication.instance()
            
            # Choose between single-window or multi-window mode
            use_multi_window = config.get('plotter', {}).get('multi_window', True)
            
            if use_multi_window:
                plotter = LiveSDRPlotterMultiWindow(config, plot_data_queue)
            else:
                plotter = LiveSDRPlotter(config, plot_data_queue)
            
            plotter.show()

            # Setup static plot signaler for thread-safe plot requests
            static_plot_signaler = StaticPlotSignaler()
            static_plot_signaler.plot_requested.connect(_handle_static_plot)
            
            logging.info(f"Live plotter initialized ({'multi-window' if use_multi_window else 'single-window'} mode)")
        except Exception as e:
            logging.error(f"Failed to initialize live plotter: {e}")
            debug_mode = False
            plotter = None

    # ======================= start application =========================
    if start():
        logging.info("SDR Chat Application is running. Press Ctrl+C to stop.")

        try:
            if debug_mode and qapp is not None:
                # Keep Qt alive and allow graceful shutdown from stop_event
                shutdown_timer = QTimer()
                shutdown_timer.timeout.connect(lambda: qapp.quit() if stop_event.is_set() else None)
                shutdown_timer.start(100)
                qapp.exec()
            else:
                # Headless mode main loop
                while not stop_event.is_set():
                    time.sleep(1)

        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt received. Stopping application...")
            stop_event.set()

        finally:
            _cleanup()

    else:
        logging.error("Failed to start SDR Chat Application.")
        stop()
        sys.exit(1)

    sys.exit(0)
