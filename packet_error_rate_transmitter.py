
# import system modules
import os
import sys
import time
import logging
import select
import threading
from queue import Queue, Empty, Full
from datetime import datetime
from typing import Dict
import signal
import atexit

# import third party moduels
import numpy as np
from numpy import typing as npt
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


NUMBER_OF_DATAGRAMS = 100

STAGES = (
    "Generating Datagrams",
    "Transmitting Datagrams",
)
SPINNER_FRAMES = ("|", "/", "-", "\\")

def _reset_tx_progress(total: int) -> None:
    global tx_total_messages, tx_sent_messages
    with progress_lock:
        tx_total_messages = int(total)
        tx_sent_messages = 0


def _inc_tx_sent() -> None:
    global tx_sent_messages
    with progress_lock:
        tx_sent_messages += 1


def _tx_progress_snapshot() -> tuple[int, int]:
    with progress_lock:
        return tx_sent_messages, tx_total_messages


def _format_progress_bar(done: int, total: int, width: int = 40) -> str:
    if total <= 0:
        return f"[{'-' * width}]   0.00% (0/0)"
    done = max(0, min(done, total))
    filled = int((done / total) * width)
    bar = "#" * filled + "-" * (width - filled)
    pct = (done / total) * 100.0
    return f"[{bar}] {pct:6.2f}% ({done}/{total})"


def generate_test_datagrams(num_datagrams: int) -> list[Datagram]:
    datagrams = []
    for i in range(num_datagrams):
        msg_id = i % 256  # Wrap around at 255
        timestamp_ms = int(time.time() * 1000) % (1<<32)  # Current time in ms, wrapped to fit in uint32
        
        _payload = np.array([], dtype=np.uint8) # Simple payload: string representation of the index
        copy_i = i  # Make a copy of i to manipulate for payload generation
        while copy_i > 0:
            _payload = np.append(arr=_payload, values=np.uint8(copy_i % 256))
            copy_i //= 256

        dgram = Datagram(
            msg_id=msg_id, 
            timestamp_ms=timestamp_ms,
            msg_type=msgType.DATA,
            payload=_payload
        )
        datagrams.append(dgram)
    return datagrams

def num_bit_errors(a: npt.NDArray[np.uint8], b: npt.NDArray[np.uint8]) -> int:
    """Count the number of bit errors in a binary array."""
    x = np.bitwise_xor(a, b)
    return int(np.unpackbits(x).sum())


def num_datagram_errors(original: list[Datagram], received: list[Datagram]) -> int:
    """Count the number of datagrams with any bit errors."""
    error_count = 0
    received_copy = received.copy()
    for dgram in original:
        match = None
        for r in received_copy:
            if dgram.get_payload.all() == r.get_payload.all():
                match = r
                break

        if match is None:
            error_count += 1  # No matching datagram found, count as error
        else:
            received_copy.remove(match)  # Remove matched datagram to prevent duplicate matches
    return error_count


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

            # add guard symbols before and after the signal.
            signal_for_transmission = np.concatenate([GUARD_SYMBOLS, filtered_signal, GUARD_SYMBOLS])
            signal_for_transmission = _normalize_tx_burst(signal_for_transmission, TX_PEAK_SCALE)

            sdr.send_signal(signal_for_transmission)
            _inc_tx_sent()
            logging.info(f"Transmitted datagram: {tx_datagram.get_msg_id}")
            time.sleep(0.005)
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


def _set_stage(stage_name: str):
    global current_stage_idx
    with stage_lock:
        idx = STAGES.index(stage_name)
        current_stage_idx = idx
        for i in range(idx):
            stage_done[i] = True

def _mark_all_done():
    global current_stage_idx
    with stage_lock:
        for i in range(len(STAGES)):
            stage_done[i] = True
        current_stage_idx = len(STAGES) - 1
    
def _tui_loop():
    """Render PER test progress in terminal with rotating spinner."""
    frame_idx = 0
    try:
        while not stop_event.is_set():
            with stage_lock:
                idx = current_stage_idx
                done_snapshot = stage_done.copy()

            sent, total = _tx_progress_snapshot()

            sys.stdout.write("\033[2J\033[H")
            sys.stdout.write("PER Test Progress\n")
            sys.stdout.write("=================\n")

            spinner = SPINNER_FRAMES[frame_idx % len(SPINNER_FRAMES)]
            frame_idx += 1

            for i, label in enumerate(STAGES):
                if done_snapshot[i]:
                    suffix = "(done)"
                elif i == idx:
                    suffix = f"({spinner})"
                else:
                    suffix = "( )"
                sys.stdout.write(f"- {label} {suffix}\n")

            sys.stdout.write("\n")
            sys.stdout.write(f"Transmitted: {_format_progress_bar(sent, total)}\n")

            sys.stdout.flush()
            time.sleep(0.12)
    finally:
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()


# ================= Start and Stop of sub threads =================
def start():
    """Start the SDR Chat Application."""
    global  tx_thread, tui_thread, ack_timeout_thread
    


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

    for name, thread in (("TX", tx_thread), ("TUI", tui_thread)):
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

    # Drain queues

    while not tx_queue.empty():
        try:
            tx_queue.get_nowait()
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
    ACK_ENABLED = bool(link_control_config.get("enable_ack", True))
    NACK_ENABLED = bool(link_control_config.get("enable_nack", False))
    RETRANSMIT_ENABLED = bool(link_control_config.get("enable_retransmit", True))
    PENDING_TRACKING_ENABLED = bool(link_control_config.get("track_pending_data", True))
    EQUALIZER_ENABLED = bool(config["synchronization"].get("short_equalizer_enable", True))
    EQUALIZER_REGULARIZATION = float(
    config["synchronization"].get("short_equalizer_regularization", 1.0e-3)
)
    # ================== Logging setup ==================
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{datetime.now().date()}-chat-history.txt")
    debug_file = os.path.join(log_dir, f"{datetime.now().date()}-debug.log")
    configure_project_logging(
        level_name=get_configured_log_level(config),
        session_name="debug",
        log_file=debug_file,
        console=bool(config["logging"].get("console", False)),
        file_output=bool(config["logging"].get("file", True)),
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

    # ================= Initialize additional constants =================
    EXPECTED_PAYLOAD_SYMBOLS = calculate_expected_payload_symbols(config)

    # ================== Threading and synchronization primitives ==================
    stop_event: threading.Event = threading.Event()
    tui_refresh_event: threading.Event = threading.Event()
    tx_thread: threading.Thread = None
    tui_thread: threading.Thread = None

    progress_lock = threading.Lock()
    tx_total_messages = 0
    tx_sent_messages = 0
    stage_lock = threading.Lock()
    current_stage_idx = 0
    stage_done = [False] * len(STAGES)

    _cleaned_up = False
    _cleanup_lock = threading.Lock()

    # ================== Message queues for inter-thread communication ==================
    tx_queue: Queue[Datagram] = Queue(maxsize=NUMBER_OF_DATAGRAMS)       # Queue for outgoing messages to be transmitted by the TX thread

    # ======================= start application =========================
    if start():
        logging.info("SDR Chat Application is running. Press Ctrl+C to stop.")

        try:
            _set_stage("Generating Datagrams")
            test_arr = generate_test_datagrams(NUMBER_OF_DATAGRAMS)

            _reset_tx_progress(len(test_arr))

            _set_stage("Transmitting Datagrams")
            for dgram in test_arr:
                queue_datagram(dgram)

            while (not tx_queue.empty()) and (not stop_event.is_set()):
                time.sleep(1)  # Wait for all messages to be transmitted
  
            time.sleep(2)

            _mark_all_done()
            
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
