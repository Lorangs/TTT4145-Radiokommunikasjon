

import threading
import time
from datetime import datetime
from chat_tui import ChatTUI
from modulate_datagram import ModulationProtocol
from datagram import Datagram, msgType
from sdr_transciever import SDRTransciever
from queue import Queue, Empty
from typing import Optional

import os
import sys
import logging

from yaml import safe_load
import numpy as np

class SDRChatApp:
    def __init__(self, config_file: str ="setup/config.yaml"):
        """Initialize the SDR Chat Application."""

        try:
            with open(config_file, 'r') as f:
                config = safe_load(f)
        except Exception as e:
            print(f"Error loading config file: {e}")
            raise e
        
        # Create logs directory if it doesn't exist
        log_dir = "log"
        os.makedirs(log_dir, exist_ok=True)

        self.log_file = os.path.join(log_dir, f"{datetime.now().date()}-chat-history.txt")
        self.debug_file = os.path.join(log_dir, f"{datetime.now().date()}-debug.log")
        self._setup_logging(str(config['radio']['log_level']).upper().strip())

        try:
            with open(self.log_file, 'a') as f:
                f.write(f"\n\n--- New Chat Session Started at {datetime.now().time()} ---\n")
        except Exception as e:
            print(f"Error initializing chat history log: {e}")
            raise e

        self.modulation_protocol = ModulationProtocol(config)
        self.tui = ChatTUI()
        self.sdr = SDRTransciever(config)

        self.running: Optional[threading.Thread] = False
        self.rx_thread: Optional[threading.Thread] = None
        self.tx_thread: Optional[threading.Thread] = None
        self.tui_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()

        self.tx_queue: Queue[Datagram] = Queue(maxsize=32)  # Queue for outgoing messages
        self.rx_queue: Queue[Datagram] = Queue(maxsize=32)  # Queue for incoming messages

    def __del__(self):
        try:
            if hasattr(self, 'running') and self.running:
                self.stop()
            
            # Clean up queues
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

            logging.info("SDR Chat Application resources cleaned up.")
        except Exception as e:
            logging.error(f"Error during application cleanup: {e}")


    
    def stop(self):
        """Stop the SDR Chat Application."""
        logging.info("Stopping SDR Chat Application...")
        self.running = False
        self.shutdown_event.set()  # Signal threads to shutdown


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
  


    def start(self):
        """Start the SDR Chat Application."""
        if not self.sdr.connect():
            logging.debug("Failed to connect to SDR. Cannot start application.")
            #return False
        noiser_flor_dB = 10*np.log10(np.mean(self.sdr.sdr.rx()))
        logging.info(f"SDR connected successfully. Noise floor: {noiser_flor_dB:.2f} dB")

        self.modulation_protocol.set_noise_floor_dB(noiser_flor_dB)
        self.running = True

        self.rx_thread = threading.Thread(target=self._rx_loop, daemon=True, name="RX_Thread")
        self.tx_thread = threading.Thread(target=self._tx_loop, daemon=True, name="TX_Thread")
        self.tui_thread = threading.Thread(target=self._tui_loop, daemon=True, name="TUI_Thread")

        self.rx_thread.start()
        self.tx_thread.start()
        self.tui_thread.start()

        logging.info("SDR Chat Application started successfully.")

        return True
    
    
    def send_ack(self):
        """Send an ACK datagram."""
        self.tx_queue.put(Datagram(msg_type=msgType.ACK))
        logging.info("Sent ACK datagram.")

    def send_datagram(self, payload: str = "", msg_type: msgType = msgType.DATA) -> bool:
        """Enqueue a datagram for transmission."""
        try:
        
            datagram = Datagram.from_string(payload, msg_type=msg_type)

            self.tx_queue.put(datagram, timeout=1)
            self.tui.add_message(datagram)  # Add sent message to TUI display
            logging.info(f"Enqueued datagram for transmission: {datagram}")
            return True
        
        except Exception as e:
            logging.error(f"Failed to enqueue datagram: {e}")
            return False
        

    def _rx_loop(self):
        """Receive loop - continuously receive data from SDR and process it."""
        logging.info("RX loop started.")

        while self.running:
            try:
                received_signal = self.sdr.sdr.rx()

                barker_index = self.modulation_protocol.detect_barker_sequence(received_signal)

                if barker_index is not None:
                    try:
                        received_message = self.modulation_protocol.demodulate_message(received_signal[barker_index:])
                    except ValueError as e:
                        logging.warning(f"Message demodulation failed: {e}")
                        continue
                    except Exception as e:
                        logging.error(f"Unexpected error during demodulation: {e}")
                        continue

                    self.rx_queue.put(received_message)
                    logging.info(f"Received datagram: {received_message}")
                    if received_message.msg_type == msgType.DATA:
                        self.send_ack()
                        
           
                    
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
                #self.sdr.sdr.tx(modulated_signal)
                logging.info(f"Transmitted datagram: {datagram}")
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
        while self.running:
            try:
                self.tui._render_screen()  # Update TUI display

                user_input = input("> ")

                if user_input is not None:
                    user_input = user_input.strip()
                    if user_input.lower() == "/quit":
                        logging.info("User requested to quit. Stopping application...")
                        self.running = False
                        break
                    elif user_input.startswith("/"):
                        logging.warning(f"Unknown command: {user_input}")
                        continue  # Ignore unknown commands

                    self.send_datagram(user_input, msg_type=msgType.DATA)
                

            except Exception as e:
                logging.error(f"Error in TUI loop: {e}")
                continue

            time.sleep(0.1)  # Sleep briefly to avoid tight error loop

        logging.info("TUI loop stopped.")


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
            handlers=[file_handler]
        )

        logging.info("\n\n--- New Application Session Started ---")
        logging.info(f"Logging configured: level={debug_mode}, file={self.debug_file}")



if __name__ == "__main__":
    app = SDRChatApp()
    
    if not app.start():
        logging.critical("Failed to start SDR Chat Application. Exiting.")
        sys.exit(1)

    # main input loop
    try:
        while app.running:
            time.sleep(1)  # Main thread can perform other tasks or just sleep
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Stopping application...")
        app.running = False
    
    except Exception as e:
        logging.error(f"Unexpected error in main loop: {e}")
        app.running = False
       
    app.stop()
