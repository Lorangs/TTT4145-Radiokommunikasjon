"""
Terminal UI for SDR Chat Application
Simple terminal-based chat interface with message status display
"""

import sys
import threading
from datetime import datetime
from collections import deque


class ChatTUI:
    """Simple terminal-based chat UI"""
    
    def __init__(self, max_display_messages=20):
        """
        Initialize chat UI
        Args:
            max_display_messages: Maximum messages to display on screen
        """
        self.messages = deque(maxlen=max_display_messages)
        self.input_buffer = ""
        self.running = False
        self.lock = threading.Lock()
        
    def start(self):
        """Start the UI"""
        self.running = True
        self._clear_screen()
        self._print_header()
        self._refresh_display()
    
    def stop(self):
        """Stop the UI"""
        self.running = False
        print("\n\nChat session ended.")
    
    def _clear_screen(self):
        """Clear terminal screen"""
        print("\033[2J\033[H", end="")
    
    def _print_header(self):
        """Print chat header"""
        print("=" * 80)
        print(" " * 25 + "RADOGRAM DATA CHAT")
        print("=" * 80)
        print("Commands: /quit to exit, /export to save chat history")
        print("-" * 80)
    
    