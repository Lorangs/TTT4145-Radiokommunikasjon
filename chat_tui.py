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
    
    def __init__(self, max_display_messages=20):
        """
        Initialize chat UI
        Args:
            max_display_messages: Maximum messages to display on screen
        """
        self.max_display_messages = max_display_messages
        self.messages = deque(maxlen=max_display_messages)  # Store recent messages for display
        self.lock = threading.Lock()  # Lock for thread-safe message updates

        logging.info("Chat TUI initialized.")

    
    def _clear_screen(self):
        """Clear terminal screen"""
        print("\033[2J\033[H", end="")
    
    def _print_header(self):
        """Print chat header"""
        print("=" * 80)
        print(" " * 25 + "RadioGram Chat Application")
        print("=" * 80)
        print("Commands: /quit to exit, /export to save chat history")
        
    def add_message(self, datagram: Datagram):
        """Add a message to the chat display
        Args:
            datagram: Datagram object containing message and metadata
        """
        with self.lock:
            if datagram.msg_type == msgType.DATA:
                if len(datagram.payload) > 60:
                    message_text = datagram.payload[:60].tobytes().decode('utf-8', errors='replace') + "..."
                else:
                    message_text = datagram.payload.tobytes().decode('utf-8', errors='replace')
            else:
                message_text = f"<ACK for {datagram.payload_size} bytes>"
                # TODO - Add more detailed ACK info if needed (e.g. timestamp, original message preview, etc.)
            
            self.messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message_text}")
            

    def _render_screen(self):
        """Render the chat screen with current messages"""
        self._clear_screen()
        self._print_header()
        for msg in self.messages:
            print(msg)
        print("-" * 80)
        