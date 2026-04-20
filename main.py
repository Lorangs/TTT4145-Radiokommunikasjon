"""Main runtime for the Pluto SDR chat application."""

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

chat_history_lock = threading.Lock()

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

def _append_chat_history(datagram: Datagram, received: bool) -> None:
    """Append user-visible DATA messages to the session chat history."""
    if datagram.get_msg_type != msgType.DATA:
        return

    direction = "RECV" if received else "SENT"
    payload_text = datagram.payload_text(trim_padding=True)
    payload_text = payload_text.replace("\r", "\\r").replace("\n", "\\n")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        with chat_history_lock:
            with open(log_file, "a", encoding="utf-8") as history_file:
                history_file.write(
                    f"[{timestamp}] [{direction}] "
                    f"ID:{int(datagram.get_msg_id)} {payload_text}\n"
                )
    except Exception as e:
        logging.error(f"Error writing chat history: {e}")

def _find_pending_index(msg_id: int) -> int | None:
    for i, (pending_msg_id, _, _, _) in enumerate(pending_ack):
        if pending_msg_id == msg_id:
            return i
    return None

def _track_sent_data(datagram: Datagram) -> None:
    """Track sent DATA datagrams for potential retransmission if ACK is not received."""
    msg_id = int(datagram.get_msg_id)
    now_ms = time.time() * 1000.0
    with pending_lock:
        idx = _find_pending_index(msg_id)
        if idx is None:
            pending_ack.append((msg_id, 0, datagram, now_ms))
        else:
            _, retries, _, _ = pending_ack[idx]
            pending_ack[idx] = (msg_id, retries, datagram, now_ms)

def _ack_received(msg_id: int) -> None:
    """Handle received ACK by removing the corresponding datagram from pending_ack."""
    with pending_lock:
        idx = _find_pending_index(msg_id)
        if idx is not None:
            pending_ack.pop(idx)


def _retransmit_oldest_pending() -> None:
    """Retransmit the oldest pending datagram if any exist and have not exceeded max retries."""
    with pending_lock:
        if not pending_ack:
            return

        # oldest = smallest last_sent_ms
        oldest_idx = min(range(len(pending_ack)), key=lambda i: pending_ack[i][3])
        msg_id, retries, dgram, _ = pending_ack[oldest_idx]

        if retries >= MAX_RETRIES:
            logging.warning(f"Max retries reached for datagram ID {msg_id}. Giving up.")
            pending_ack.pop(oldest_idx)
            return

        now_ms = time.time() * 1000.0
        pending_ack[oldest_idx] = (msg_id, retries + 1, dgram, now_ms)

    queue_datagram(dgram)

##############################################################################################
# ================= Callback loops for threads =================
##############################################################################################
def _rx_loop():
    """Receive loop - continuously receive data from SDR and process it."""
    logging.debug("RX loop started.")

    while not stop_event.is_set():
        try:
            received_signal = sdr.sdr.rx()

            coarse_freq_adjusted = synchronizer.coarse_frequenzy_synchronization(received_signal)
            if coarse_freq_adjusted is None:
                continue    # skip if signal is too weak to process
            #logging.debug("Signal detected above noise floor. Proceeding with synchronization and decoding.")

            padded_signal = matched_filter.pad_signal_front_and_back(coarse_freq_adjusted)  
            filtered_signal = matched_filter.apply_filter(padded_signal)
            #logging.debug("Applied matched filter to received signal.")

            #normalized_matched_filtered = synchronizer.normalize_matched_filter_output(filtered_signal)
            #logging.debug("Normalized matched filter output for synchronization.")

            time_adjusted = synchronizer.gardner_timing_synchronization(filtered_signal)
            #logging.debug("Performed Gardner timing synchronization on received signal.")

            fine_freq_adjusted = synchronizer.fine_frequenzy_synchronization(time_adjusted)
            #logging.debug("Performed fine frequency synchronization on received signal.")

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
                center_freq=float(config["receiver"]["rx_carrier"]),
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

            tui_refresh_event.set()  # Signal TUI to refresh display

            if received_datagram.get_msg_type == msgType.DATA:
                logging.info(f"Received datagram: {received_datagram}")
                try:
                    rx_queue.put(received_datagram)
                except Full:
                    logging.error(f"RX queue is full. Dropping received datagram ID {received_datagram.get_msg_id}.")
                    continue

                if ACK_ENABLED:
                    ack_datagram = Datagram.as_ack(msg_id=received_datagram.get_msg_id)
                    queue_datagram(ack_datagram)
      
            # mark message as acknowledged if ACK received, so it won't be retransmitted.
            elif received_datagram.get_msg_type == msgType.ACK:
                logging.info(f"Received ACK for msg_ID: {received_datagram.get_msg_id}")
                if PENDING_TRACKING_ENABLED:
                    _ack_received(int(received_datagram.get_msg_id))
            

            # retransmit the previous sent message.
            elif received_datagram.get_msg_type == msgType.NACK:
                logging.info(f"Received NACK. Retransmitting oldest pending message if any.")
                if RETRANSMIT_ENABLED:
                    _retransmit_oldest_pending()
                
            else:
                logging.warning(f"Received message with unknown type: {received_datagram.get_msg_type}")
                raise ValueError("Unknown message type received.")
                
        except ValueError as e:
            logging.warning(f"Did not receive valid signal: {e}")
            if NACK_ENABLED:
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

    logging.debug("RX loop stopped.")

def _tx_loop():
    """Transmit loop - continuously check for outgoing messages and transmit them."""
    logging.debug("TX loop started.")

    while not stop_event.is_set():
        try:
            tx_datagram: Datagram = tx_queue.get(timeout=0.1) # Wait for message to send

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
                    center_freq=float(config["transmitter"]["tx_carrier"]),
                )
                


            # add guard symbols before and after the signal.
            signal_for_transmission = np.concatenate([GUARD_SYMBOLS, filtered_signal, GUARD_SYMBOLS])
            signal_for_transmission = _normalize_tx_burst(signal_for_transmission, TX_PEAK_SCALE)

            sdr.send_signal(signal_for_transmission)
            #logging.debug("Datagram length:\t %d bytes.", len(tx_datagram.pack()))
            #logging.debug("FEC coded data:\t %s.", fec_coded_data)
            #logging.debug("FEC coded data length:\t %d bytes.", len(fec_coded_data))
            #logging.debug("Scrambled data length:\t %d bytes.", len(scrambled_data))
            #logging.debug("Scrambled data:\t %s", scrambled_data)  
            #logging.debug("Conv coded data length:\t %d bits.", len(conv_coded_data))
            #logging.debug("Conv coded data:\t %s", conv_coded_data)  # Print first 192 bits of conv coded data for debugging
            #logging.debug("Modulated signal length:\t %d symbols.", len(modulated_signal))
            #logging.debug("Modulated signal:\t %s", modulated_signal[:24])  # Print first 24 symbols of modulated signal for debugging
            #logging.debug("Signal with Gold length:\t %d symbols.", len(signal_with_gold))
            #logging.debug("Upsampled signal length:\t %d symbols.", len(upsampled_signal))
            #logging.debug("Filtered signal length:\t %d symbols.", len(filtered_signal))
            #logging.debug("Signal for transmission length:\t %d symbols.", len(signal_for_transmission))

            if (
                tx_datagram.get_msg_type == msgType.DATA
                and PENDING_TRACKING_ENABLED
            ):
                _track_sent_data(tx_datagram) 
            
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

    logging.debug("TX loop stopped.")

def _tui_loop():
    """TUI loop - continuously check for user input and enqueue messages to send."""
    logging.debug("TUI loop started.")

    tui.render_screen()  # Initial render of TUI

    while not stop_event.is_set():
        try:

            if tui_refresh_event.is_set():
                while not rx_queue.empty():
                    try:
                        received_datagram: Datagram = rx_queue.get_nowait()
                        tui.add_message(received_datagram, received=True)
                        logging.debug(f"TUI processed received datagram ID: {received_datagram.get_msg_id}")
                    except Empty:
                        break  # No more messages to process
                tui.render_screen()  # Update TUI display
                tui_refresh_event.clear()  # Reset event

            user_input = _poll_user_input()
            if user_input is not None:
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
                    _append_chat_history(datagram, received=False)
                    user_input = user_input[int(config['datagram']['payload_size']) :]  # Remove the part that was sent
                
                # Final slice (or if input was already short enough)
                sliced_user_input = user_input
                datagram = Datagram.as_string(sliced_user_input, msg_type=msgType.DATA)
                queue_datagram(datagram)
                _append_chat_history(datagram, received=False)
                tui.add_message(datagram)  # Add sent message to TUI display
                tui.render_screen()  # Update TUI display after sending message
    
        except Exception as e:
            logging.error(f"Error in TUI loop: {e}")
            continue

        time.sleep(0.1)  # Sleep briefly to avoid tight error loop
    logging.debug("TUI loop stopped.")


_windows_input_buffer = ""


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
        
def _ack_timeout_loop():
    """ACK timeout loop - periodically check for pending ACKs and retransmit if necessary."""
    logging.debug("ACK timeout loop started.")

    while not stop_event.is_set():
        if not RETRANSMIT_ENABLED:
            time.sleep(0.1)
            continue

        now_ms = time.time() * 1000.0
        to_retransmit: list[tuple[int, Datagram, int]] = []
        to_remove: list[int] = []

        with pending_lock:
            for i, (msg_id, retries, dgram, last_sent_ms) in enumerate(pending_ack):
                if (now_ms - last_sent_ms) <= ACK_TIMEOUT_ms:
                    continue

                if retries >= MAX_RETRIES:
                    logging.warning(f"Max retries reached for datagram ID {msg_id}. Giving up.")
                    to_remove.append(i)
                    continue

                pending_ack[i] = (msg_id, retries + 1, dgram, now_ms)
                to_retransmit.append((msg_id, dgram, retries + 1))

            for i in reversed(to_remove):
                pending_ack.pop(i)

        for msg_id, dgram, retry_count in to_retransmit:
            try:
                tx_queue.put_nowait(dgram)
                logging.info(f"Timeout retransmit for datagram ID {msg_id} (retry {retry_count}).")
            except Full:
                logging.warning(f"TX queue full. Could not retransmit datagram ID {msg_id}.")

        time.sleep(max(0.05, ACK_TIMEOUT_ms / 1000.0 / 2.0))

    logging.debug("ACK timeout loop stopped.")


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
    capture_cfg = config.get("plot_capture", {})

    if not debug_mode or static_plotter is None:
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
            center_freq = float(
                plot_data.get(
                    'center_freq',
                    config.get('receiver', {}).get(
                        'rx_carrier',
                        config.get('plotter', {}).get('center_freq', 0.0),
                    ),
                )
            )
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
    tx_thread: threading.Thread = None
    tui_thread: threading.Thread = None
    ack_timeout_thread: threading.Thread = None 

    _cleaned_up = False
    _cleanup_lock = threading.Lock()

    # ================== Message queues for inter-thread communication ==================
    tx_queue: Queue[Datagram] = Queue(maxsize=int(config['radio']['queue_size']))       # Queue for outgoing messages to be transmitted by the TX thread
    rx_queue: Queue[Datagram] = Queue(maxsize=int(config['radio']['queue_size']))       # Queue for incoming messages received by the RX thread to be processed by the TUI thread
    
    # List to track pending ACKs with retry counts and datagram info. 
    # (msg_id, retry_count, datagram, last_sent_ms)
    pending_ack: list[tuple[int, int, Datagram, float]] = []  
    pending_lock = threading.Lock()  # Lock to synchronize access to pending_ack


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
                    time.sleep(0.5)

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
