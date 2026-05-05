"""
Main single sided transmitter implementation for SDR Chat Application.
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

##################################################################################
# ============================== Message Handling ================================
##################################################################################
def queue_datagram(datagram: Datagram) -> bool:
    """Enqueue a datagram for transmission."""
    global tx_queue
    try: 
        tx_queue.put_nowait(datagram)
        logging.info(f"Queued datagram ID {datagram.get_msg_id} for transmission.")
        return True
    except Full:
        logging.error(f"Failed to queue datagram ID {datagram.get_msg_id}. TX queue is full.")
        return False




##############################################################################################
# ================= Callback loops for threads =================
##############################################################################################
def _tx_loop():
    """Transmit loop - continuously check for outgoing messages and transmit them."""
    global tx_queue, debug_mode
    logging.debug("TX loop started.")

    while not stop_event.is_set():
        try:
            tx_datagram: Datagram = tx_queue.get_nowait() # Wait for message to send

            fec_coded_data = fec_codec.encode(tx_datagram.pack())
            interleaved_data = interleaver.interleave(fec_coded_data)
            scrambled_data = scrambler.apply(interleaved_data)
            conv_coded_data = conv_coder.encode(scrambled_data)
            modulated_signal = modulation_protocol.modulate_message(conv_coded_data)
            signal_with_gold = gold_detector.add_gold_symbols(modulated_signal)
            upsampled_signal = modulation_protocol.upsample_symbols(signal_with_gold)

            if matched_filter.hardware_filter_enable:
                filtered_signal = upsampled_signal  # Assume hardware filtering is applied by the SDR TODO: Not working as inteded
            else:
                padded_signal = matched_filter.pad_signal_front_and_back(upsampled_signal)  # Pad signal to avoid edge effects from filtering
                filtered_signal = matched_filter.apply_filter(padded_signal)


            if debug_mode:
                logging.debug(f"TX loop got datagram from queue: {tx_datagram}")
                
                # Constellation and PSD plots when debug is enabled
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                capture_plot_if_enabled(
                    "tx_burst",
                    "constellation",
                    modulated_signal,
                    title=f"TX Payload Constellation Before Gold Framing {timestamp}",
                    stem=f"tx_payload_constellation_before_gold_framing_{timestamp}",
                )
                capture_plot_if_enabled(
                    "tx_burst",
                    "psd",
                    filtered_signal,
                    title=f"TX Framed Burst PSD Before Guard Insertion {timestamp}",
                    stem=f"tx_framed_burst_psd_before_guard_insertion_{timestamp}",
                    sample_rate=float(config["modulation"]["sample_rate"]),
                    center_freq=float(config["plotter"]["center_freq"]),
                )
                
            # add guard symbols before and after the signal.
            signal_for_transmission = np.concatenate([GUARD_SYMBOLS, filtered_signal, GUARD_SYMBOLS])
            signal_for_transmission = _normalize_tx_burst(signal_for_transmission, TX_PEAK_SCALE)
      
            sdr.send_signal(signal_for_transmission)

            time.sleep(0.005)  # Sleep briefly to allow SDR to process transmission
            logging.info(f"Transmitted datagram: {tx_datagram.get_msg_id}")
        except Empty:
            continue  # No message to send, loop again
        except RuntimeError as e:
            logging.error(f"Runtime error in TX loop: {e}")
            stop_event.set()  # Trigger shutdown on critical errors
            break
        except Exception as e:
            logging.error(f"Error: {e}")
            continue

    logging.debug("TX loop stopped.")

def _tui_loop():
    """
        TUI loop - continuously check for user input and enqueue messages to send.
        Only render if there are new messages or user input to process, to avoid unnecessary CPU usage and flickering.
    """
    logging.debug("TUI loop started.")

    tui.render_screen()  # Initial render of TUI

    while not stop_event.is_set():
        try:
            user_input = _poll_user_input()
            if user_input is not None:
                if user_input.lower() == "/quit":
                    logging.info("User requested to quit. Stopping application...")
                    stop_event.set()
                    break
                elif user_input.startswith("/"):
                    logging.warning(f"Unknown command: {user_input}")
                    continue  # Ignore unknown commands

                sliced_user_input = _slice_text_to_payload_chunks(user_input, PAYLOAD_SIZE-1)
                sent_any = False

                for chunk in sliced_user_input:
                    datagram = Datagram.as_string(chunk, msg_type=msgType.DATA)
                    if queue_datagram(datagram):
                        tui.add_message(datagram, sent_by_self=True)  # Add sent message to TUI display
                        sent_any = True
                    else:
                        logging.error("Failed to queue message for transmission. Stopping further chunks.")
                        break
                    sent_any = True
                if sent_any:
                    tui.render_screen() 

        except Exception as e:
            logging.error(f"Error in TUI loop: {e}")
            continue

        time.sleep(0.1)  # Sleep briefly to reduce CPU usage.
    logging.debug("TUI loop stopped.")

def _slice_text_to_payload_chunks(text: str, max_payload_bytes: int) -> list[str]:
    """Split text into UTF-8-safe chunks, each <= max_payload_bytes."""
    if max_payload_bytes <= 0:
        raise ValueError("max_payload_bytes must be > 0")

    chunks: list[str] = []
    current_chars: list[str] = []
    current_bytes = 0

    for ch in text:
        ch_bytes = len(ch.encode("utf-8"))

        # Defensive: skip impossible-to-fit single chars
        if ch_bytes > max_payload_bytes:
            logging.warning("Skipping character that exceeds payload byte limit.")
            continue

        if current_chars and (current_bytes + ch_bytes > max_payload_bytes):
            chunks.append("".join(current_chars))
            current_chars = [ch]
            current_bytes = ch_bytes
        else:
            current_chars.append(ch)
            current_bytes += ch_bytes

    if current_chars:
        chunks.append("".join(current_chars))

    return chunks

_windows_input_buffer = ""  # Buffer for accumulating user input on Windows, since msvcrt.getwch() reads one character at a time
def _poll_user_input() -> str | None:
    """Read one terminal line without blocking the TUI loop."""
    global _windows_input_buffer

    if os.name != "nt":
        ready_to_read, _, _ = select.select([sys.stdin], [], [], 0.1)
        if ready_to_read:
            return sys.stdin.readline().strip()
        return None

    if not msvcrt.kbhit():
        return None

    while msvcrt.kbhit():
        char = msvcrt.getwch()

        if char in ("\r", "\n"):
            completed = _windows_input_buffer
            _windows_input_buffer = ""
            print()
            return completed.strip()

        if char == "\003":
            raise KeyboardInterrupt

        if char == "\b":
            if _windows_input_buffer:
                _windows_input_buffer = _windows_input_buffer[:-1]
                print("\b \b", end="", flush=True)
            continue

        if char in ("\x00", "\xe0"):
            if msvcrt.kbhit():
                msvcrt.getwch()
            continue

        _windows_input_buffer += char
        print(char, end="", flush=True)

    return None



# ================= Start and Stop of sub threads =================
def start():
    """Start the SDR Chat Application."""
    global tx_thread, tui_thread
    
    if sdr.connect():  
        synchronizer.set_noise_floor(sdr.measure_noise_floor_dB())
    else:
        logging.debug("Failed to connect to SDR.")
        return False
    
    try:
        stop_event.clear()
        tx_thread = threading.Thread(target=_tx_loop, daemon=True, name="TX_Thread")
        tui_thread = threading.Thread(target=_tui_loop, daemon=True, name="TUI_Thread")
        tx_thread.start()
        tui_thread.start()
        return True
    
    except Exception as e:
        logging.error(f"Error starting threads: {e}")
        stop_event.set()
        return False


def stop():
    """Stop the SDR Chat Application."""
    global tx_thread, tui_thread
    logging.info("Stopping SDR Chat Application...")
    stop_event.set()

    for name, thread in (("TX_Thread", tx_thread), ("TUI_Thread", tui_thread)):
        if thread and thread.is_alive():
            try:
                thread.join(timeout=2.0)
                if thread.is_alive():
                    logging.warning(f"{name} thread did not stop within timeout")
            except Exception as e:
                logging.error(f"Error waiting for {name} thread: {e}")

    # clear references
    tx_thread = None
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
    for q in (tx_queue, plot_data_queue):
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

def _normalize_tx_burst(signal: np.ndarray, target_peak: float) -> np.ndarray:
    """
    Scale one TX burst to a fixed peak amplitude before sending it to Pluto.
    args:    
        signal. The complex baseband signal representing the TX burst to be transmitted.
        target_peak. The desired peak amplitude to which the signal should be normalized before transmission.
    returns: 
        A new complex numpy array representing the normalized TX burst, scaled to the specified target peak amplitude.
    """

    tx_signal = np.asarray(signal).astype(np.complex64, copy=False)
    peak = float(np.max(np.abs(tx_signal))) if tx_signal.size else 0.0
    if peak <= 0.0:
        return tx_signal
    return (float(target_peak) * tx_signal / peak).astype(np.complex64)



    

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

    # ================== Threading and synchronization primitives ==================
    stop_event: threading.Event = threading.Event()
    tui_refresh_event: threading.Event = threading.Event()
    rx_thread: threading.Thread = None
    tx_thread: threading.Thread = None
    tui_thread: threading.Thread = None
    ack_timeout_thread: threading.Thread = None 

    _cleaned_up = False
    _cleanup_lock = threading.Lock()

    # ================== Message queues for inter-thread communication ==================
    tx_queue: Queue[Datagram] = Queue(maxsize=int(config['radio']['queue_size']))       # Queue for outgoing messages to be transmitted by the TX thread
    
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
