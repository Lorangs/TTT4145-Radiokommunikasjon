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
    RECEIVED = "R"
    ACKED = "A"

class ChatTUI:
    """Simple terminal-based chat UI"""
    
    def __init__(self, config: dict):
        """
        Initialize chat UI
        Args:
            max_display_messages: Maximum messages to display on screen
        """
        self.num_display_messages = config['radio']['num_tui_msg']
        self.messages: list[tuple[msgStatus, Datagram]] = [] # (ACK status, Datagram)
        logger.info("Chat TUI initialized.")
      

    def __del__(self):
        """Cleanup resources if needed"""
        for msg in self.messages:
            del msg  # Explicitly delete messages if needed (not usually necessary in Python)
        del self.messages
        logger.info("Chat TUI destroyed.")

    def _clear_screen(self):
        """Clear terminal screen"""
        print("\033[2J\033[H", end="")
    
    def _print_header(self):
        """Print chat header"""
        print("=" * 80)
        print(" " * 25 + "RadioGram Chat Application")
        print("=" * 80)
        print("Commands: /quit to exit.")

    def print_messages(self):
        """Print all messages in the chat display"""
        
        self.sort_messages()  # Ensure messages are sorted by timestamp before displaying
        display_string = ""
        for ack_status, dgram in self.messages:
            msg_payload = dgram.payload_text(trim_padding=True)
            time_str = datetime.fromtimestamp(dgram.get_timestamp_ms / 1000).strftime('%H:%M:%S')
            display_string += f"\n[{time_str}][{ack_status}]\t{msg_payload}"
        print(display_string)

    def sort_messages(self):
        """Sort messages by timestamp (oldest first)"""
        self.messages.sort(key=lambda x: x[1].get_timestamp_ms)

    def add_message(self, datagram: Datagram, sent_by_self: bool = False):
        """Add a message to the chat display
        Args:
            datagram: Datagram object containing message and metadata
            sent_by_self: Whether the message was sent by the user
        """
        if sent_by_self:
            if datagram.get_msg_type == msgType.DATA:
                self.messages.append((msgStatus.SENT, datagram))
                self.delete_old_messages()  # Ensure we don't exceed max display messages

        else:
            if datagram.get_msg_type == msgType.ACK:
                acked_msg_id = datagram.get_msg_id
                for i, (ack_status, dgram) in enumerate(self.messages):
                    if dgram.get_msg_id == acked_msg_id:
                        self.messages[i] = (msgStatus.ACKED, dgram)  # Update ACK status to True
                        break
            elif datagram.get_msg_type == msgType.DATA:
                self.messages.append((msgStatus.RECEIVED, datagram))
                self.delete_old_messages()  # Ensure we don't exceed max display messages
            else:
                return # Ignore NACK messages for display
        
    def delete_old_messages(self):
        """Delete old messages to prevent overflow
        Args:
            max_messages: Maximum number of messages to keep in display
        """
        if len(self.messages) > self.num_display_messages:
            self.sort_messages()  # Ensure messages are sorted by timestamp before deleting
            self.messages = self.messages[-self.num_display_messages:]

    def render_screen(self):
        """Render the chat screen with current messages"""
        self._clear_screen()
        self._print_header()
        self.print_messages()
        print('-' * 80)
        print("> ", end="", flush=True)  # Prompt for user input

if __name__ == "__main__":
    # Example usage of ChatTUI
   
    chat_ui = ChatTUI()

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




