"""
Terminal UI for SDR Chat Application
Simple terminal-based chat interface with message status display
"""

import sys
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
        self.messages: deque[str] = deque(maxlen=int(config['radio']['queue_size']) ) # Store recent messages for display
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

    def _update_local_message_status(self, msg_id, new_status: str) -> bool:
        message_id_token = f"[ID:{int(msg_id)}]"
        for index, message in enumerate(self.messages):
            if message_id_token not in message or "[IN]" in message:
                continue

            updated_message = message
            for old_status in ("[S]", "[R]", "[F]"):
                if old_status in updated_message:
                    updated_message = updated_message.replace(old_status, new_status, 1)
                    self.messages[index] = updated_message
                    return True
        return False

    def mark_acknowledged(self, msg_id) -> bool:
        return self._update_local_message_status(msg_id, "[R]")

    def mark_failed(self, msg_id) -> bool:
        return self._update_local_message_status(msg_id, "[F]")

    def add_message(self, datagram: Datagram, is_local: bool = True):
        """Add a message to the chat display
        Args:
            datagram: Datagram object containing message and metadata
        """
        if datagram.get_msg_type == msgType.DATA:
            message_text = datagram.payload_text(trim_padding=True)
            if len(message_text) > 60:
                message_text = message_text[:60] + "..."
            else:
                message_text = message_text

            timestamp = datetime.now().strftime("%H:%M:%S")
            direction_status = "[S]" if is_local else "[IN]"
            display_message = (
                f"[{timestamp}][ID:{int(datagram.get_msg_id)}]{direction_status}\t{message_text}"
            )
            self.messages.append(display_message)

        else:
            self.mark_acknowledged(datagram.get_msg_id)

    def render_screen(self, current_input: str = ""):
        """Render the chat screen with current messages"""
        self._clear_screen()
        self._print_header()
        for msg in self.messages:
            print(msg)
        print('-' * 80)
        print(f"> {current_input}", end="", flush=True)  # Prompt for user input

if __name__ == "__main__":
    # Example usage of ChatTUI
    config = {"radio": {"queue_size": 32}}
    chat_ui = ChatTUI(config)

    import numpy as np

    msgID = []
    
    # Simulate sending messages
    for i in range(5):
        payload = f"Hello, this is message {i}"
        datagram = Datagram.as_string(payload)
        msgID.append(datagram.get_msg_id)  # Store message ID for later reference
        chat_ui.add_message(datagram, is_local=True)
        chat_ui.render_screen()


    ack_datagram = Datagram.as_ack(msg_id=msgID[2])  # Create ACK for the third message
    chat_ui.add_message(ack_datagram)
    chat_ui.render_screen()
