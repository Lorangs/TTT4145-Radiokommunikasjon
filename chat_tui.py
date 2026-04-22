"""
Terminal UI for SDR Chat Application
Simple terminal-based chat interface with message status display
"""

import sys
import threading
from datetime import datetime
from collections import deque
from project_logger import get_logger
from datagram import Datagram, msgType


logger = get_logger(__name__)

class msgStatus:
    SENT = "S"
    ACKED = "A"
    RECEIVED = "R"

class ChatTUI:
    """Simple terminal-based chat UI"""
    
    def __init__(self, config: dict):
        """
        Initialize chat UI
        Args:
            max_display_messages: Maximum messages to display on screen
        """
        self.lock: threading.RLock = threading.RLock()  # Lock to synchronize access to messages list
        self.num_display_messages: int = config['radio']['num_tui_msg']
        self.messages: list[tuple[msgStatus, Datagram]] = [] # (msgStatus, Datagram)
        self.current_input_line: str = ""
        logger.info("Chat TUI initialized.")
      
    def close(self):
        """Clean up resources if needed"""
        logger.info("Closing Chat TUI...")
        with self.lock:
            self.messages.clear()

    def __del__(self):
        """Destructor to ensure resources are cleaned up"""
        try:
            self.close()
        except Exception as e:
            pass

    def _clear_screen(self):
        """Clear terminal screen"""
        print("\033[2J\033[H", end="")

    def _draw_input_line(self):
        """Draw the current input line at the bottom of the screen"""
        print("-" * 80)
        print("> " + self.current_input_line, end="", flush=True)
    
    def _print_header(self):
        """Print chat header"""
        print("=" * 80)
        print(" " * 25 + "RadioGram Chat Application")
        print("=" * 80)
        print("Commands: /quit to exit.")

    def print_messages(self):
        """Print all messages in the chat display"""
        
        sorted_messages = self.sort_messages()  # Ensure messages are sorted by timestamp before displaying
        display_string = ""
        for status, dgram in sorted_messages:
            msg_payload = dgram.payload_text(trim_padding=True)
            timestamp_ms = dgram.get_timestamp_ms
            time_str = datetime.fromtimestamp(timestamp_ms / 1000).strftime('%H:%M:%S')
            display_string += f"[{time_str}][{status}]\t{msg_payload}\n"
        print(display_string)

    def sort_messages(self) -> list[tuple[msgStatus, Datagram]]:
        """Sort messages by timestamp (oldest first)"""
        with self.lock:
            sorted_messages = sorted(self.messages, key=lambda x: x[1].get_timestamp_ms)
        return sorted_messages

    def add_message(self, datagram: Datagram, sent_by_us: bool = False):
        """Add a message to the chat display. Must be used with Lock when modifying messages list.
        Args:
            datagram: Datagram object containing message and metadata
            sent_by_us: Flag indicating if the message was sent by the user
        """
        if sent_by_us:
            if datagram.get_msg_type != msgType.DATA:
                return
            
            msg_id = datagram.get_msg_id
            with self.lock:
                if any(dgram.get_msg_id == msg_id for _, dgram in self.messages):
                    return
                self.messages.append((msgStatus.SENT, datagram))
                self.delete_old_messages()  # Ensure we don't exceed max display messages
     
        else:
            # Incoming message, add to display as received
            if datagram.get_msg_type == msgType.DATA:
                with self.lock:
                    self.messages.append((msgStatus.RECEIVED, datagram))
                    self.delete_old_messages()  # Ensure we don't exceed max display messages

            elif datagram.get_msg_type == msgType.ACK:
                acked_msg_id = datagram.get_msg_id
                with self.lock:
                    for i, (ack_status, dgram) in enumerate(self.messages):
                        if dgram.get_msg_id == acked_msg_id:
                            self.messages[i] = (msgStatus.ACKED, dgram) 
                            break
            else: return
        
    def delete_old_messages(self):
        """Delete old messages to prevent overflow. 
        Args:
            max_messages: Maximum number of messages to keep in display
        """
        with self.lock:
            if len(self.messages) > self.num_display_messages:
                sorted_messages = self.sort_messages()  # Ensure messages are sorted by timestamp before deleting
                self.messages = sorted_messages[-self.num_display_messages:]

    def set_current_input(self, text: str):
        """update the current typedd input line."""
        with self.lock:
            self.current_input_line = text

    def render_screen(self):
        """Render the chat screen with current messages"""
        self._clear_screen()
        self._print_header()
        self.print_messages()
        self._draw_input_line()

if __name__ == "__main__":
    # Example usage of ChatTUI
    demo_config = {
        'radio': {
            'num_tui_msg': 100
        }
    }
    chat_ui = ChatTUI(demo_config)

    import numpy as np

    # Fixed base time for reproducible output
    base_dt = datetime(2026, 4, 22, 0, 0, 0)
    base_ms = int(base_dt.timestamp() * 1000)

    # Intentionally out-of-order by timestamp offset (seconds)
    # (msg_id, offset_sec, text)
    data_messages = []
    for i in range(130):
        msg_id = i + 1  # Unique message ID
        offset_s = i    # second the message was created
        payload = f"Message {i}\t+{offset_s}s"
        data_messages.append((msg_id, offset_s, payload))


    # Add ACKs out of order too (optional)
    ack_order = np.random.permutation([msg_id for msg_id, _, _ in data_messages])  # Random ACK order
    ack_order = ack_order[:100]  # ACK only the first 100 messages for demonstration
    
    # Add DATA datagrams in this out-of-order sequence
    while data_messages:
        random_idx = np.random.randint(0, len(data_messages))   
        msg_id, offset_s, text = data_messages.pop(random_idx) 
        ts_ms = base_ms + offset_s * 1000
        dgram = Datagram.as_string(
            text=text,
            timestamp_ms=ts_ms,
            msg_id=msg_id,
            msg_type=msgType.DATA
        )
        chat_ui.add_message(dgram)

    chat_ui.render_screen()  # Initial render with out-of-order messages


    for msg_id in ack_order:
        ack_ts_ms = base_ms + 60 + msg_id  # later than all data
        ack = Datagram.as_ack(msg_id=msg_id, timestamp_ms=ack_ts_ms)
        chat_ui.add_message(ack)

    chat_ui.render_screen()




