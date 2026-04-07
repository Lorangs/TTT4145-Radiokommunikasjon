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
from modulation import ModulationProtocol
from datagram import Datagram, msgType
from sdr_transciever import SDRTransciever
from filter import RRCFilter
from gold_detection import GoldCodeDetector
from synchronize import Synchronizer
from forward_error_correction import FCCodec
from convolutional_coder import ConvolutionalCoder
from interleaver import Interleaver
from scrambler import LFSRScrambler
from project_logger import configure_project_logging, get_configured_log_level


# ================= Message Handling =================
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

def _track_sent_data(datagram: Datagram) -> None:
    msg_id = int(datagram.get_msg_id)
    now_ms = time.time() * 1000.0
    with pending_lock:
        # insert once; keeps original insertion order
        if msg_id not in pending_ack:
            pending_ack[msg_id] = {
                "datagram": datagram,
                "retries": 0,
                "last_sent_ms": now_ms,
            }
        else:
            # do not reinsert key (keeps FIFO order)
            pending_ack[msg_id]["last_sent_ms"] = now_ms

def _ack_received(msg_id: int) -> None:
    with pending_lock:
        pending_ack.pop(msg_id, None)

def _retransmit_oldest_pending() -> None:
    with pending_lock:
        if not pending_ack:
            return
        oldest_msg_id = next(iter(pending_ack))  # first inserted key
        entry = pending_ack[oldest_msg_id]
        dgram = entry["datagram"]
        if entry["retries"] >= MAX_RETRIES:
            logging.warning(f"Max retries reached for datagram ID {oldest_msg_id}. Giving up.")
            pending_ack.pop(oldest_msg_id, None)
            return
        entry["retries"] += 1

    queue_datagram(dgram)  # Re-enqueue for retransmission

##############################################################################################
# ================= Callback loops for threads =================
##############################################################################################
def _rx_loop():
    """Receive loop - continuously receive data from SDR and process it."""
    logging.info("RX loop started.")

    while not stop_event.is_set():
        try:
            received_signal = sdr.sdr.rx()

            coarse_freq_adjusted = synchronizer.coarse_frequenzy_synchronization(received_signal)
            if coarse_freq_adjusted is None:
                continue    # skip if signal is too weak to process

            filtered_signal = matched_filter.apply_filter(coarse_freq_adjusted)
            time_adjusted = synchronizer.gardner_timing_synchronization(filtered_signal)
            fine_freq_adjusted = synchronizer.fine_frequenzy_synchronization(time_adjusted)

            gold_index = gold_detector.detect(fine_freq_adjusted)
            ## Do the downsampling

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
                continue    # skip if gold code is not detected, likely not a valid signal to process
          
            received_symbols = gold_detector.remove_gold_symbols(
                fine_freq_adjusted,
                gold_index,
            )

            received_bits = modulation_protocol.demodulate_signal(received_symbols)
            conv_decoded_bits = conv_coder.decode(received_bits)
            deinterleaved_bits = interleaver.deinterleave(conv_decoded_bits)
            descrambled_bits = scrambler.apply(deinterleaved_bits)
            fec_decoded_bits = fec_codec.rs_decode(descrambled_bits)
            received_datagram = Datagram.unpack(fec_decoded_bits)

            try:
                rx_queue.put(received_datagram)
            except Full:
                logging.error(f"RX queue is full. Dropping received datagram ID {received_datagram.get_msg_id}.")
                continue

            tui_refresh_event.set()  # Signal TUI to refresh display

            if received_datagram.get_msg_type == msgType.DATA:
                logging.info(f"Received datagram: {received_datagram}")
                ack_datagram = Datagram.as_ack(msg_id=received_datagram.get_msg_id)
                queue_datagram(ack_datagram)
      
            # mark message as acknowledged if ACK received, so it won't be retransmitted.
            elif received_datagram.get_msg_type == msgType.ACK:
                logging.info(f"Received ACK for msg_ID: {received_datagram.get_msg_id}")
                _ack_received(int(received_datagram.get_msg_id))
            

            # retransmit the previous sent message.
            elif received_datagram.get_msg_type == msgType.NACK:
                logging.info(f"Received NACK for msg_ID: {received_datagram.get_msg_id}")
                _retransmit_oldest_pending()
                
            else:
                logging.warning(f"Received message with unknown type: {received_datagram.get_msg_type}")
                raise ValueError("Unknown message type received.")
                
        except ValueError as e:
            logging.warning(f"Did not receive valid signal: {e}")
            nack_datagram = Datagram.as_nack()
            queue_datagram(nack_datagram)
            time.sleep(0.1)  # Sleep briefly to avoid tight error loop
            continue
        except RuntimeError as e:
            logging.error(f"Runtime error in RX loop: {e}")
            stop_event.set()  # Trigger shutdown on critical errors
            break
        except Exception as e:
            logging.error(f"Unexpected error in RX loop: {e}")
            time.sleep(0.1)  # Sleep briefly to avoid tight error loop
            continue

    logging.info("RX loop stopped.")

def _tx_loop():
    """Transmit loop - continuously check for outgoing messages and transmit them."""
    logging.info("TX loop started.")

    while not stop_event.is_set():
        try:
            tx_datagram: Datagram = tx_queue.get(timeout=0.1) # Wait for message to send

            fec_coded_data = fec_codec.encode(tx_datagram.pack())
            scrambled_data = scrambler.apply(fec_coded_data)
            interleaved_data = interleaver.interleave(scrambled_data)
            conv_coded_data = conv_coder.encode(interleaved_data)
            modulated_signal = modulation_protocol.modulate_message(conv_coded_data)
            signal_with_gold = gold_detector.add_gold_symbols(modulated_signal)
            upsampled_signal = modulation_protocol.upsample_symbols(signal_with_gold)

            if matched_filter.hardware_filter_enable:
                filtered_signal = upsampled_signal  # Assume hardware filtering is applied by the SDR TODO: Not working as inteded
            else:
                filtered_signal = matched_filter.apply_filter(upsampled_signal)

            if debug_mode:
                logging.debug(f"TX loop got datagram from queue: {tx_datagram}")

            sdr.send_signal(filtered_signal)
            if tx_datagram.get_msg_type == msgType.DATA:
                _track_sent_data(tx_datagram)  # Track sent data for ACK handling
        
            time.sleep(0.1)  # Sleep briefly to allow SDR to process transmission

            logging.info(f"Transmitted datagram: {tx_datagram.get_msg_id}")
        except Empty:
            continue  # No message to send, loop again
        except RuntimeError as e:
            logging.error(f"Runtime error in TX loop: {e}")
            stop_event.set()  # Trigger shutdown on critical errors
            break
        except Exception as e:
            logging.error(f"Error: {e}")
            time.sleep(0.1)  # Sleep briefly to avoid tight error loop
            continue

    logging.info("TX loop stopped.")

def _tui_loop():
    """TUI loop - continuously check for user input and enqueue messages to send."""
    logging.info("TUI loop started.")

    tui.render_screen()  # Initial render of TUI

    while not stop_event.is_set():
        try:

            # Wait for either: screen refresh event OR stdin input (max 0.5s timeout)
            ready_to_read, _, _ = select.select([sys.stdin], [], [], 0.1)
            
            if tui_refresh_event.is_set():
                while not rx_queue.empty():
                    try:
                        received_datagram: Datagram = rx_queue.get_nowait()
                        tui.add_message(received_datagram)
                        logging.debug(f"TUI processed received datagram ID: {received_datagram.get_msg_id}")
                    except Empty:
                        break  # No more messages to process
                tui.render_screen()  # Update TUI display
                tui_refresh_event.clear()  # Reset event

            if ready_to_read:
                user_input = sys.stdin.readline().strip()
                if user_input.lower() == "/quit":
                    logging.info("User requested to quit. Stopping application...")
                    stop_event.set()
                    break
                elif user_input.startswith("/"):
                    logging.warning(f"Unknown command: {user_input}")
                    continue  # Ignore unknown commands

                # send message as datagram
                while len(user_input.encode('utf-8')) > int(config['datagram']['payload_size']):
                    logging.warning("Input message is too long and will be truncated to fit payload size.")
                    sliced_user_input = user_input[: int(config['datagram']['payload_size'])]
                    datagram = Datagram.as_string(sliced_user_input, msg_type=msgType.DATA)
                    queue_datagram(datagram)
                    user_input = user_input[int(config['datagram']['payload_size']) :]  # Remove the part that was sent
                
                # Final slice (or if input was already short enough)
                sliced_user_input = user_input
                datagram = Datagram.as_string(sliced_user_input, msg_type=msgType.DATA)
                queue_datagram(datagram)
                tui.add_message(datagram)  # Add sent message to TUI display
                tui.render_screen()  # Update TUI display after sending message
                chat_history_log(f"Sent: [ID:{datagram.get_msg_id}]\t{sliced_user_input}")
    
        except Exception as e:
            logging.error(f"Error in TUI loop: {e}")
            continue

        time.sleep(0.1)  # Sleep briefly to avoid tight error loop
    logging.info("TUI loop stopped.")
        

def _ack_timeout_loop():
    logging.info("ACK timeout loop started.")

    while not stop_event.is_set():
        now_ms = time.time() * 1000.0
        timed_out_msg_ids = []

        with pending_lock:
            for msg_id, entry in pending_ack.items():
                if (now_ms - float(entry["last_sent_ms"])) > ACK_TIMEOUT_ms:
                    timed_out_msg_ids.append(msg_id)

        for msg_id in timed_out_msg_ids:
            with pending_lock:
                entry = pending_ack.get(msg_id)
                if entry is None:
                    continue

                if entry["retries"] >= MAX_RETRIES:
                    logging.warning(f"Max retries reached for datagram ID {msg_id}. Giving up.")
                    pending_ack.pop(msg_id, None)
                    continue

                entry["retries"] += 1
                entry["last_sent_ms"] = now_ms
                dgram = entry["datagram"]

            try:
                tx_queue.put_nowait(dgram)
                logging.info(f"Timeout retransmit for datagram ID {msg_id} (retry {entry['retries']}).")
            except Full:
                logging.warning(f"TX queue full. Could not retransmit datagram ID {msg_id}.")

        time.sleep(max(0.05, ACK_TIMEOUT_ms / 1000.0 / 2.0))

    logging.info("ACK timeout loop stopped.")
# ...existing code...


# ================= Start and Stop of sub threads =================
def start():
    """Start the SDR Chat Application."""
    global rx_thread, tx_thread, tui_thread, ack_timeout_thread
    
    if sdr.connect():  
        synchronizer.set_noise_floor(sdr.measure_noise_floor_dB())
    else:
        logging.debug("Failed to connect to SDR.")
        return False
    
    try:
        stop_event.clear()
        rx_thread = threading.Thread(target=_rx_loop, daemon=True, name="RX_Thread")
        tx_thread = threading.Thread(target=_tx_loop, daemon=True, name="TX_Thread")
        tui_thread = threading.Thread(target=_tui_loop, daemon=True, name="TUI_Thread")
        ack_timeout_thread = threading.Thread(target=_ack_timeout_loop, daemon=True, name="ACK_Timeout_Thread")
        rx_thread.start()
        tx_thread.start()
        tui_thread.start()
        ack_timeout_thread.start()
        return True
    
    except Exception as e:
        logging.error(f"Error starting threads: {e}")
        stop_event.set()
        return False


def stop():
    """Stop the SDR Chat Application."""
    global rx_thread, tx_thread, tui_thread, ack_timeout_thread
    logging.info("Stopping SDR Chat Application...")
    stop_event.set()

    for name, thread in (("RX", rx_thread), ("TX", tx_thread), ("TUI", tui_thread), ("ACK Timeout", ack_timeout_thread)):
        if thread and thread.is_alive():
            try:
                thread.join(timeout=2.0)
                if thread.is_alive():
                    logging.warning(f"{name} thread did not stop within timeout")
            except Exception as e:
                logging.error(f"Error waiting for {name} thread: {e}")

    # clear references
    rx_thread = None
    tx_thread = None
    tui_thread = None    
    ack_timeout_thread = None

def _signal_handler(signum, frame):
    """Handle termination signals for graceful shutdown."""
    logging.info(f"Signal {signum} received. Initiating graceful shutdown...")
    stop_event.set()

def _cleanup():
    """Clean up resources safely. Idempotent."""
    global _cleaned_up

    with _cleanup_lock:
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
    for q in (rx_queue, tx_queue, plot_data_queue):
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
            filter_file = config["radio"]["hardware_filter_file"]
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


# ==================================Logging ======================================
#def chat_history_log(message: str):
#    """Append a message to the chat history log file."""
#    try:
#        with open(log_file, 'a') as f:
#            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
#    except Exception as e:
#        logging.error(f"Error writing to chat history log: {e}")
#


##################################################################################
# ================== Helper functions for plotting and logging ==================
##################################################################################
def request_static_plot(plot_data: dict):
    """Thread-safe method to request a static plot from any thread."""
    if debug_mode and hasattr(static_plot_signaler, 'plot_requested'):
        static_plot_signaler.plot_requested.emit(plot_data)

def _handle_static_plot(plot_data: dict):
    """Handle static plot request (runs in main thread)."""
    try:
        plot_type = plot_data.get('type')
        data = plot_data.get('data')
        title = plot_data.get('title', '')
        
        if plot_type == 'time_domain':
            static_plotter.plot_time_domain(
                data, 
                float(config['modulation']['sample_rate']),
                title=title
            )
        elif plot_type == 'constellation':
            static_plotter.plot_constellation(data, title=title)
        elif plot_type == 'psd':
            sample_rate = float(plot_data.get('sample_rate', config['modulation']['sample_rate']))
            center_freq = float(plot_data.get('center_freq', config['plotter']['center_freq']))
            static_plotter.plot_psd(data, sample_rate, center_freq=center_freq, title=title)
        
        show(block=False)
        
    except Exception as e:
        logging.error(f"Error handling static plot: {e}")




if __name__ == "__main__":
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
        StaticSDRPlotter = None
        #StaticPlotSignaler = None
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
    sdr = SDRTransciever(config) # must be initilized after Matched Filter module.

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
    rx_queue: Queue[Datagram] = Queue(maxsize=int(config['radio']['queue_size']))       # Queue for incoming messages received by the RX thread to be processed by the TUI thread
    pending_ack: Dict[int, Dict] = {}  # Dictionary to track pending ACKs with retry counts and datagram info
    pending_lock = threading.Lock()  # Lock to synchronize access to pending_ack
    MAX_RETRIES = int(config['coding']['max_retries'])  # Maximum number of retransmission attempts for unacknowledged messages
    ACK_TIMEOUT_ms = float(config['radio']['ack_timeout_ms'])  # Timeout for waiting for ACKs (converted to milliseconds

    # ================== Logging setup ==================
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{datetime.now().date()}-chat-history.txt")
    debug_file = os.path.join(log_dir, f"{datetime.now().date()}-debug.log")
    configure_project_logging(
        level_name=get_configured_log_level(config),
        session_name="debug",
        log_file=debug_file,
        console=True,
        file_output=True,
    )

    try:
        with open(log_file, 'a') as f:
            f.write(f"\n\n--- New Chat Session Started at {datetime.now().time()} ---\n")
    except Exception as e:
        logging.error(f"Error initializing chat history log: {e}")
        raise e

    # ================== Debug mode setup ==================
    debug_mode = bool(config['radio']['debug_mode'])
    qapp = None
    plotter = None
    plot_data_queue: Queue[np.ndarray] = Queue(maxsize=32)
    static_plotter = StaticSDRPlotter() if debug_mode else None
    static_plot_queue: Queue[Dict[str, np.ndarray]] = Queue(maxsize=8)  # Queue for static plot data (e.g., filter response, constellation points)
    static_plotter_signaler = None

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

    atexit.register(_cleanup)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    logging.info("SDR Chat Application initialized successfully.")


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
                    time.sleep(0.1)

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