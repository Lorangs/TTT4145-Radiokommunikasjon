"""
This module defines the Datagram class, 
which represents a structured message format for communication.
"""


import numpy as np
from dataclasses import dataclass
from enum import Enum

class msgType(Enum):
    DATA = 0
    ACK = 1

@dataclass(init=False, repr=False)
class Datagram():
    """ 
    Messaage format:
        - msg_type: 1 byte string (DATA or ACK)
        - msg_id: 1 byte np.uint8 (message ID for tracking)
        - payload: up to 254 bytes (actual message data). The payload is padded with whitespace if it is shorter than 254 bytes.
    """
    _msg_id: np.uint8
    _msg_type: msgType
    _payload: np.ndarray

    def __init__(self, 
                 msg_id: np.uint8 | None = None,
                 msg_type: msgType = msgType.DATA, 
                 payload: np.ndarray = np.array([], dtype=np.uint8), 
            ):
        """Initialize datagram with payload and message type. Automatically computes CRC16 checksum.
            Args:
                msg_type (msgType): Type of message (DATA or ACK).
                payload (np.ndarray): Payload data as a numpy array of uint8. If None, it will be treated as an empty payload.
        """

        if len(payload) > 254:
            raise ValueError("Payload size exceeds maximum of 254 bytes.")
        
        if payload.dtype != np.uint8:
            raise ValueError("Payload must be a numpy array of uint8.")
        
        if len(payload) < 254:
            # Pad payload with whitespace (ASCII 0x20) to ensure fixed size of 254 bytes
            padding_length = 254 - len(payload)
            payload = np.concatenate((payload, np.full(padding_length, 0x20, dtype=np.uint8)))
        
        self._msg_id = msg_id if msg_id is not None else np.random.randint(0, 256, dtype=np.uint8)
        self._msg_type = msg_type        
        self._payload = payload


    @classmethod
    def as_ack(cls, msg_id: np.uint8) -> 'Datagram':
        """Create an ACK datagram for a given message ID."""
        return cls(msg_id=msg_id, msg_type=msgType.ACK, payload=np.array([], dtype=np.uint8))
    
    @classmethod
    def as_string(cls, 
                    text: str, 
                    msg_id: np.uint8 | None = None,
                    msg_type: msgType = msgType.DATA, 
                    encoding: str = 'utf-8'
                    ) -> 'Datagram':
        """
        Create a datagram from a text string.
        
        Args:
            text: Text message to send
            msg_type: Message type (default: DATA)
            encoding: Text encoding (default: utf-8)
            
        Returns:
            Datagram object
        """
        payload = np.frombuffer(text.strip().encode(encoding), dtype=np.uint8)
        return cls(msg_type=msg_type, payload=payload)
    
    @classmethod
    def as_bytes(cls, 
                   data: bytes, 
                   msg_id: np.uint8 | None = None,
                   msg_type: msgType = msgType.DATA
                   ) -> 'Datagram':
        """
        Create a datagram from raw bytes.
        
        Args:
            data: Raw bytes to send
            msg_type: Message type (default: DATA)
            
        Returns:
            Datagram object
        """
        payload = np.frombuffer(data, dtype=np.uint8)
        return cls(msg_type=msg_type, payload=payload)

    def pack(self) -> bytes:
        """Pack datagram into a single numpy array of uint8."""
        return (
            bytes([self._msg_id]) +
            bytes([self._msg_type.value]) +
            bytes(self._payload)
        )

    @classmethod
    def unpack(cls, data: bytes) -> 'Datagram':
        """Unpack datagram from a byte array.
        Args:
            data (bytes): Byte array containing the packed datagram.
        Returns:
            Datagram: Unpacked datagram object.
        Raises:
            ValueError: If the data is corrupted.
        """

        if len(data) != 256:
            raise ValueError("Data length must be exactly 256 bytes (1+1+254).")

        msg_id = np.uint8(data[0])
        msg_type = msgType(data[1])
        payload = np.frombuffer(data[2:], dtype=np.uint8).copy()

        # Route through __init__ so validation/padding rules stay centralized
        return cls(msg_id=msg_id, msg_type=msg_type, payload=payload)

    @property
    def get_payload(self) ->  np.ndarray:
        """Get payload data."""
        return self._payload

    @property
    def get_msg_id(self) -> np.uint8:
        """Get the message ID used in the datagram."""
        return self._msg_id
    
    @property
    def get_msg_type(self) -> msgType:
        """Get message type."""
        return self._msg_type
    
    def __repr__(self) -> str:
        return (f"\n\tmsg_ID:\t\t{self._msg_id},\n"
                f"\tmsg_type:\t{self._msg_type.name},\n"
                f"\tpayload:\n{self._payload}\n")


if __name__ == "__main__":
    # Example usage
    #payload = np.array([1, 2, 3, 4], dtype=np.uint8)
    test_msg = "Hello, World!"
    payload = np.frombuffer(test_msg.encode('utf-8'), dtype=np.uint8)
    datagram = Datagram(msg_type=msgType.DATA, payload=payload)
    print(f"Original datagram: {datagram}")

    packed_data = datagram.pack()
    print(f"Packed data: {packed_data.hex()}\n")

    unpacked_datagram = Datagram.unpack(packed_data)
    print(f"Unpacked datagram: {unpacked_datagram}")
