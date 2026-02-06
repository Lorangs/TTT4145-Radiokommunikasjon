

import threading
import time
from chat_tui import ChatTUI
from modulate_datagram import ModulationProtocol
from datagram import Datagram, msgType
from sdr_transciever import SDRTransciever
from queue import Queue, Empty

import sys
from yaml import safe_load
import numpy as np

class SDRChatApp:
    def __init__(self, config_file: str ="config.yaml"):
        """Initialize the SDR Chat Application."""

        try:
            with open(config_file, 'r') as f:
                config = safe_load(f)
        except Exception as e:
            print(f"Error loading config file: {e}")
            raise e

        self.modulation_protocol = ModulationProtocol(config)
        self.tui = ChatTUI()
        self.sdr = SDRTransciever(config)

        self.running = False
        self.rx_thread = None
        self.tx_thread = None
        self.tui_thread = None

        self.tx_queue = Queue(maxsize=32)  # Queue for outgoing messages
        self.rx_queue = Queue(maxsize=32)  # Queue for incoming messages
    
    def start(self):
        """Start the SDR Chat Application."""
        if not self.sdr_transceiver.connect():
            print("Failed to connect to SDR. Exiting.")
            return False

        self.tui.start()
        self.running = True

        self.rx_thread = threading.Thread(target=self._rx_loop, daemon=True, name="RX_Thread")
        self.tx_thread = threading.Thread(target=self._tx_loop, daemon=True, name="TX_Thread")
        self.tui_thread = threading.Thread(target=self._tui_loop, daemon=True, name="TUI_Thread")

        self.rx_thread.start()
        self.tx_thread.start()
        self.tui_thread.start()

        print("SRD Chat Application started")

        return True
    
    def stop(self):
        """Stop the SDR Chat Application."""
        print("Stopping SDR Chat Application...")
        self.running = False
        self.sdr_transceiver.stop_receiving()
        self.tui.stop()

        if self.rx_thread and self.rx_thread.is_alive():
            self.rx_thread.join(timeout=2)
        if self.tx_thread and self.tx_thread.is_alive():
            self.tx_thread.join(timeout=2)
        if self.tui_thread and self.tui_thread.is_alive():
            self.tui_thread.join(timeout=2)

        del self.sdr

        print("SDR Chat Application stopped.")

    
    def send_ack(self):
        """Send an ACK datagram."""
        self.send_datagram(Datagram(msg_type=msgType.ACK))

    def send_datagram(self, payload: str = "", msg_type: msgType = msgType.DATA) -> bool:
        """Enqueue a datagram for transmission."""
        try:
            if msg_type == msgType.ACK:
                datagram = Datagram(msg_type=msgType.ACK)
            else:
                datagram = Datagram(msg_type=msg_type, payload=np.frombuffer(payload.encode(), dtype=np.uint8))

            self.tx_queue.put(datagram, timeout=1)
            print(f"Enqueued datagram for transmission: {datagram}")
            return True
        
        except Exception as e:
            print(f"Error enqueuing datagram: {e}")
            return False
        

    def _rx_loop(self):
        """Receive loop - continuously receive data from SDR and process it."""
        print("RX loop started.")
        while self.running:
            try:
                received_signal = self.sdr.sdr.rx()

                barker_index = self.modulation_protocol.detect_barker_sequence(received_signal)

                if barker_index is not None:
                    try:
                        received_message = self.modulation_protocol.demodulate_message(received_signal[barker_index:])
                    except ValueError as e:
                        print(f"Error demodulating message: {e}")
                        continue
                    except Exception as e:
                        print(f"Error in receive loop: {e}")
                        continue

                    if received_message.msg_type == msgType.DATA:
                        self.send_datagram(Datagram(msg_type=msgType.ACK))
                        
                    # TODO - Add message processing and display in TUI
                    


            except Exception as e:
                print(f"Error in receive loop: {e}")
                time.sleep(0.1)  # Sleep briefly to avoid tight error loop
                continue

        print("RX loop stopped.")

    def _tx_loop(self):
        """Transmit loop - continuously check for outgoing messages and transmit them."""
        print("TX loop started.")
        while self.running:
            try:
                datagram = self.tx_queue.get(timeout=0.1)  # Wait for message to send
                modulated_signal = self.modulation_protocol.modulate_message(datagram)
                self.sdr.sdr.tx(modulated_signal)
            except Empty:
                time.sleep(0.1)  # No message to send, sleep briefly
                continue  # No message to send, continue loop
            except Exception as e:
                print(f"Error in transmit loop: {e}")
                time.sleep(0.1)  # Sleep briefly to avoid tight error loop
                continue

        print("TX loop stopped.")

    def _tui_loop(self):
        """TUI loop - continuously check for user input and enqueue messages to send."""
        print("TUI loop started.")
        while self.running:
            try:
                self.tui.update()  # Update TUI display

                user_input = self.tui.get_user_input(timeout=0.5)  # Wait for user input

                if user_input is not None:
                    datagram = Datagram(msg_type=msgType.DATA, payload=np.frombuffer(user_input.encode(), dtype=np.uint8))
                    self.send_datagram(datagram)

            except Exception as e:
                print(f"Error in TUI loop: {e}")
                time.sleep(0.1)  # Sleep briefly to avoid tight error loop
                continue

        print("TUI loop stopped.")


if __name__ == "__main__":
    app = SDRChatApp()
    
    if not app.start():
        print("Failed to start SDR Chat Application. Exiting.")
        sys.exit(1)

    # main input loop
    while app.running:
        try:
            user_input = input(">\t")

            if user_input.strip().lower() == "/quit":
                print("Exiting chat...")
                break
            else:
                datagram = Datagram(msg_type=msgType.DATA, payload=np.frombuffer(user_input.encode(), dtype=np.uint8))
                app.send_datagram(datagram)

        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Exiting chat...")
            break
        except Exception as e:
            print(f"Error in main loop: {e}")
            break

    app.stop()
