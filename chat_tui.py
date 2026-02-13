"""
Terminal UI for SDR Chat Application
Simple terminal-based chat interface with message status display
"""

import sys
import threading
from datetime import datetime
from collections import deque
import logging
from datagram import Datagram, msgType


class ChatTUI:
    """Simple terminal-based chat UI"""
    
    def __init__(self, config: dict):
        """
        Initialize chat UI
        Args:
            max_display_messages: Maximum messages to display on screen
        """
        self.max_display_messages = int(config['radio']['display_number_of_messages'])
        self.messages: deque[str] = deque(maxlen=self.max_display_messages)  # Store recent messages for display

        logging.info("Chat TUI initialized.")

    def __del__(self):
        """Cleanup resources if needed"""
        for msg in self.messages:
            del msg  # Explicitly delete messages if needed (not usually necessary in Python)
        del self.messages
        logging.info("Chat TUI destroyed.")

    def _clear_screen(self):
        """Clear terminal screen"""
        print("\033[2J\033[H", end="")
    
    def _print_header(self):
        """Print chat header"""
        print("=" * 80)
        print(" " * 25 + "RadioGram Chat Application")
        print("=" * 80)
        print("Commands: /quit to exit.")
        
    def add_message(self, datagram: Datagram):
        """Add a message to the chat display
        Args:
            datagram: Datagram object containing message and metadata
        """
        if datagram.get_msg_type == msgType.DATA:
            if len(datagram.get_payload) > 60:
                message_text = datagram.get_payload[:60].tobytes().decode('utf-8', errors='replace') + "..."
            else:
                message_text = datagram.get_payload.tobytes().decode('utf-8', errors='replace')

            timestamp = datetime.now().strftime("%H:%M:%S")
            display_message = f"[{timestamp}][S]\t{message_text}"
            self.messages.append(display_message)

        else:
            for i, msg in enumerate(self.messages):
                if f"ID:{datagram.get_msg_id}" in msg:
                    self.messages[i]= msg.replace("[S]", "[R]") # Mark message as received
                    break

    def render_screen(self):
        """Render the chat screen with current messages"""
        self._clear_screen()
        self._print_header()
        for msg in self.messages:
            print(msg)
        print('-' * 80)
        print("> ", end="", flush=True)  # Prompt for user input

if __name__ == "__main__":
    # Example usage of ChatTUI
    chat_ui = ChatTUI()

    import numpy as np

    msgID = []
    
    # Simulate sending messages
    for i in range(5):
        payload = f"Hello, this is message {i}"
        datagram = Datagram.from_string(payload)
        msgID.append(datagram.get_msg_id)  # Store message ID for later reference
        chat_ui.add_message(datagram)
        chat_ui.render_screen()


    ack_datagram = Datagram.from_ack(msg_id=msgID[2])  # Create ACK for the third message
    chat_ui.add_message(ack_datagram)
    chat_ui.render_screen()