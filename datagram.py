"""
This module defines the Datagram class, 
which represents a structured message format for communication.
"""
import numpy as np
from dataclasses import dataclass
from enum import Enum
import time
from binascii import crc_hqx


##########################################################
# Constants defining datagram structure and sizes
# These are duplicates of config.yaml values.
##########################################################
MSG_ID_SIZE = 1
MSG_TYPE_SIZE = 1
TIMESTAMP_SIZE = 4
PAYLOAD_LENGTH_SIZE = 1
CRC16_SIZE = 2
HEADER_SIZE = MSG_ID_SIZE + MSG_TYPE_SIZE + TIMESTAMP_SIZE + PAYLOAD_LENGTH_SIZE + CRC16_SIZE
PAYLOAD_SIZE = 247
TOTAL_SIZE = HEADER_SIZE + PAYLOAD_SIZE
PAD_BYTE = np.uint8(0x00)


class msgType(Enum):
    DATA = 0
    ACK = 1
    NACK = 2

@dataclass(init=False, repr=False)
class Datagram():
    """ 
    Messaage format:
        - msg_type: 1 byte string (DATA or ACK)
        - msg_id: 1 byte np.uint8 (message ID for tracking)
        - timestamp: 4 bytes np.uint32 (UNIX timestamp in seconds when the datagram was created)
        - payload_length: 1 byte np.uint8 logical payload length
        - payload_crc16: 2 bytes CRC-16-CCITT over the logical datagram contents
        - payload: fixed 247-byte field. Unused bytes are zero-padded on the wire.
    """

    _msg_id: np.uint8
    _msg_type: msgType
    _timestamp_ms: np.uint32
    _payload_length: np.uint8
    _payload_crc16: np.uint16
    _payload: np.ndarray

    @staticmethod
    def _compute_crc16(
        msg_id: np.uint8,
        msg_type: msgType,
        timestamp_ms: np.uint32,
        payload_length: np.uint8,
        payload: np.ndarray,
    ) -> np.uint16:
        logical_payload = np.asarray(payload, dtype=np.uint8)[: int(payload_length)]
        crc_input = (
            bytes([np.uint8(msg_id)])
            + bytes([np.uint8(msg_type.value)])
            + np.uint32(timestamp_ms).tobytes()
            + bytes([np.uint8(payload_length)])
            + logical_payload.tobytes()
        )
        return np.uint16(crc_hqx(crc_input, 0xFFFF))

    def __init__(self, 
                 msg_id: np.uint8 | None = None,
                 msg_type: msgType = msgType.DATA, 
                 timestamp_ms: np.uint32 | None = None,
                 payload: np.ndarray = np.array([], dtype=np.uint8),
            ):
        """Initialize datagram with payload and message type.
            Args:
                msg_type (msgType): Type of message (DATA or ACK).
                payload (np.ndarray): Payload data as a numpy array of uint8. If None, it will be treated as an empty payload.
        """

        payload = np.asarray(payload, dtype=np.uint8)

        if len(payload) > PAYLOAD_SIZE:
            raise ValueError(f"Payload size exceeds maximum of {PAYLOAD_SIZE} bytes.")

        self._msg_id = msg_id if msg_id is not None else np.random.randint(0, 256, dtype=np.uint8)
        self._msg_type = msg_type        

        if timestamp_ms is not None:
            self._timestamp_ms = np.uint32(int(timestamp_ms) & 0xFFFFFFFF)
        else:
            self._timestamp_ms = np.uint32(int(time.time() * 1000) & 0xFFFFFFFF)  # Keep only the lowest 32 bits

        self._payload = payload.copy()
        self._payload_length = np.uint8(payload.size)
        self._payload_crc16 = self._compute_crc16(
            msg_id=self._msg_id,
            msg_type=self._msg_type,
            timestamp_ms=self._timestamp_ms,
            payload_length=self._payload_length,
            payload=self._payload,
        )


    @classmethod
    def as_ack(cls, msg_id: np.uint8) -> 'Datagram':
        """Create an ACK datagram for a given message ID."""
        return cls(msg_id=msg_id, msg_type=msgType.ACK, payload=np.array([], dtype=np.uint8))
    
    @classmethod
    def as_nack(cls, msg_id: np.uint8) -> 'Datagram':
        """Create a NACK datagram for a given message ID."""
        return cls(msg_id=msg_id, msg_type=msgType.NACK, payload=np.array([], dtype=np.uint8))
    
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
        payload = np.frombuffer(text.encode(encoding), dtype=np.uint8)
        return cls(msg_id=msg_id, msg_type=msg_type, payload=payload)
    
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
        return cls(msg_id=msg_id, msg_type=msg_type, payload=payload)

    def pack(self) -> bytes:
        """Pack datagram into a single numpy array of uint8."""
        padding_length = PAYLOAD_SIZE - int(self._payload_length)
        padded_payload = np.concatenate(
            (
                self._payload,
                np.full(padding_length, PAD_BYTE, dtype=np.uint8),
            )
        )
        return (
            bytes([self._msg_id]) +
            bytes([self._msg_type.value]) +
            self._timestamp_ms.tobytes() +
            bytes([self._payload_length]) +
            self._payload_crc16.tobytes() +
            bytes(padded_payload)
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

        if len(data) != TOTAL_SIZE:
            raise ValueError(
                f"Data length must be exactly {TOTAL_SIZE} bytes "
                f"({HEADER_SIZE}+{PAYLOAD_SIZE})."
            )

        msg_id = np.uint8(data[0])
        msg_type = msgType(data[1])

        timestamp_start = MSG_ID_SIZE + MSG_TYPE_SIZE       
        timestamp_end = timestamp_start + TIMESTAMP_SIZE
        timestamp_bytes = np.frombuffer(data[timestamp_start:timestamp_end], dtype=np.uint32)[0]

        payload_length_index = timestamp_end
        payload_length = int(data[payload_length_index])

        crc_start = payload_length_index + PAYLOAD_LENGTH_SIZE
        crc_end = crc_start + CRC16_SIZE
        payload_crc16 = np.frombuffer(data[crc_start:crc_end], dtype=np.uint16)[0]

        payload_field = np.frombuffer(data[HEADER_SIZE:], dtype=np.uint8).copy()

        if payload_length > PAYLOAD_SIZE:
            raise ValueError(
                f"Payload length field exceeds maximum payload size: {payload_length} > {PAYLOAD_SIZE}."
            )

        padding = payload_field[payload_length:]
        if padding.size and not np.all(padding == PAD_BYTE):
            raise ValueError("Datagram payload padding is not zero-filled.")

        payload = payload_field[:payload_length]
        expected_crc16 = cls._compute_crc16(
            msg_id=msg_id,
            msg_type=msg_type,
            timestamp_ms=timestamp_bytes,
            payload_length=np.uint8(payload_length),
            payload=payload,
        )
        if np.uint16(payload_crc16) != expected_crc16:
            raise ValueError(
                f"Datagram CRC mismatch: expected 0x{int(expected_crc16):04X}, "
                f"got 0x{int(np.uint16(payload_crc16)):04X}."
            )

        # Route through __init__ so validation/padding rules stay centralized
        return cls(
            msg_id=msg_id,
            msg_type=msg_type,
            timestamp_ms=timestamp_bytes,
            payload=payload,
        )

    @property
    def get_payload(self) ->  np.ndarray:
        """Get payload data."""
        return self._payload

    @property
    def get_payload_without_padding(self) -> np.ndarray:
        """Backward-compatible alias for the logical payload bytes."""
        return self._payload

    @property
    def get_payload_length(self) -> np.uint8:
        """Length of the logical payload in bytes."""
        return self._payload_length

    @property
    def get_payload_crc16(self) -> np.uint16:
        """CRC-16 over the logical payload-carrying datagram fields."""
        return self._payload_crc16

    @property
    def get_msg_id(self) -> np.uint8:
        """Get the message ID used in the datagram."""
        return self._msg_id
    
    @property
    def get_msg_type(self) -> msgType:
        """Get message type."""
        return self._msg_type
    
    @property
    def get_timestamp_ms(self) -> np.uint32:
        """Get the timestamp of when the datagram was created."""
        return self._timestamp_ms

    def payload_bytes(self, trim_padding: bool = False) -> bytes:
        return self._payload.tobytes()

    def payload_text(
        self,
        encoding: str = "utf-8",
        errors: str = "replace",
        trim_padding: bool = True,
    ) -> str:
        return self.payload_bytes(trim_padding=trim_padding).decode(encoding, errors=errors)
    
    def __repr__(self) -> str:
        return (f"\n\tmsg_ID:\t\t{self._msg_id},\n"
                f"\tmsg_type:\t{self._msg_type.name},\n"
                f"\ttimestamp_ms:\t{self._timestamp_ms},\n"
                f"\tpayload_len:\t{self._payload_length},\n"
                f"\tpayload_crc16:\t0x{int(self._payload_crc16):04X},\n"
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
