
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

NUMBER_OF_DATAGRAMS = 100      # Number of datagrams to measure start time and end time for calculating packet error rate. Adjust as needed for testing.

SPINNER = ['|', '/', '-', '\\']

num_received_datagrams = 0      # counter for datagrams received during datarate test.
test_start_time = None         # Timestamp when the first datagram is received during the test.
test_end_time = None           # Timestamp when the last datagram is received during the test.

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
def _rx_loop():
    """Receive loop - continuously receive data from SDR and process it."""
    global num_received_datagrams, test_start_time, test_end_time
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

    
            ### The following section attempts to decode the payload using the best rotation estimate from the gold code.
            ### Falls back to trying other rotations if decoding fails. 
            ### This is to handle cases where the gold-based rotation estimate is not perfect, which can happen at low SNR.

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

                    conv_decoded_bytes = conv_coder.decode(received_bits)
                    descrambled_bytes = scrambler.apply(conv_decoded_bytes)
                    interleaved_bytes = interleaver.deinterleave(descrambled_bytes)
                    fec_decoded_bits = fec_codec.decode(interleaved_bytes)
                    received_datagram = Datagram.unpack(fec_decoded_bits)

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

            num_received_datagrams += 1
            if num_received_datagrams == 1:
                logging.info(f"Received first datagram ID {received_datagram.get_msg_id}. Starting timer for datagram error rate test.")
                test_start_time = time.time()
            elif num_received_datagrams == NUMBER_OF_DATAGRAMS:
                test_end_time = time.time()
                stop_event.set()  # Stop after receiving the specified number of datagrams for testing

        except ValueError as e:
            logging.warning(f"Did not receive valid signal: {e}")
            #time.sleep(0.05)
            continue
        except RuntimeError as e:
            logging.error(f"Runtime error in RX loop: {e}")
            stop_event.set()  # Trigger shutdown on critical errors
            break
        except Exception as e:
            logging.error(f"Unexpected error in RX loop: {e}")
            ##time.sleep(0.05)
            continue

    logging.debug("RX loop stopped.")

def _tui_loop():
    """TUI loop - continuously refresh the terminal user interface."""
    global num_received_datagrams
    logging.debug("TUI loop started.")
    i = 0
    num_spinner_states = len(SPINNER)
    while not stop_event.is_set():
        try:
            print("\033c", end="")  # Clear terminal
            print(f"Measuring packet error rate... Received {num_received_datagrams} datagrams so far.")
            print(f"({SPINNER[i % num_spinner_states]})")
            i += 1
            time.sleep(2)  # Adjust refresh rate as needed
            
        except Exception as e:
            logging.error(f"Error in TUI loop: {e}")
            continue
    logging.debug("TUI loop stopped.")



# ================= Start and Stop of sub threads =================
def start():
    """Start the SDR Chat Application."""
    global rx_thread, tui_thread
    logging.info("Starting SDR Chat Application...")

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


    try:
        if rx_thread is not None:
            rx_thread.join(timeout=3.0)
            if rx_thread.is_alive():
                logging.warning(f"RX thread did not stop within timeout")
        if tui_thread is not None:
            tui_thread.join(timeout=3.0)
            if tui_thread.is_alive():
                logging.warning(f"TUI thread did not stop within timeout")
        
    except Exception as e:
        logging.error(f"Error waiting for threads: {e}")

    # clear references
    rx_thread = None 
    tui_thread = None



def _signal_handler(signum, frame):
    """Handle termination signals for graceful shutdown."""
    # Terminate application after calculating results
    logging.info(f"Received signal {signum}. Initiating shutdown...")
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
    while not rx_queue.empty():
        try:
            rx_queue.get_nowait()
        except Empty:
            break

    # Disconnect SDR
    try:
        if sdr is not None:
            sdr.disconnect()
            logging.info("SDR disconnected successfully.")
    except Exception as e:
        logging.error(f"Error disconnecting SDR: {e}")

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

def print_test_results():
    elapsed_time = (
        test_end_time - test_start_time 
        if (test_start_time is not None and test_end_time is not None) 
        else None
    )
    datarate = (num_received_datagrams / elapsed_time) if elapsed_time else None

    print(f"Received {num_received_datagrams} datagrams during test.")
    print(f"Elapsed time: {elapsed_time:.2f} seconds" if elapsed_time else "Elapsed time: N/A")
    print(f"Datarate: {datarate:.2f} datagrams/second" if datarate else "Datarate: N/A")



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
    gold_detector = GoldCodeDetector(config)
    synchronizer = Synchronizer(config)
    sdr = SDRTransciever(config) # must be initilized after Matched Filter module.

    # ================= Initialize additional constants =================
    EXPECTED_PAYLOAD_SYMBOLS = calculate_expected_payload_symbols(config)

    # ================== Threading and synchronization primitives ==================
    stop_event: threading.Event = threading.Event()
    tui_refresh_event: threading.Event = threading.Event()
    rx_thread: threading.Thread = None
    tui_thread: threading.Thread = None
    _cleaned_up = False
    _cleanup_lock = threading.Lock()

    finalized = False
    finalize_lock = threading.Lock()

    # ================== Message queues for inter-thread communication ==================
    rx_queue: Queue[Datagram] = Queue(maxsize=NUMBER_OF_DATAGRAMS)       # Queue for incoming messages received by the RX thread to be processed by the TUI thread
    
    # ======================= start application =========================
    if start():
        logging.info("SDR Chat Application is running. Press Ctrl+C to stop.")
        try:
            while not stop_event.is_set():
                time.sleep(5)  # Main thread can perform periodic tasks here if needed

        except KeyboardInterrupt:
            stop_event.set()

        finally:
            stop()
            print_test_results()
            _cleanup()

    else:
        logging.error("Failed to start SDR Chat Application.")
        stop()
        sys.exit(1)

    sys.exit(0)
