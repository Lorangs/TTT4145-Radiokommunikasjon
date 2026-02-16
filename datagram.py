import numpy as np
from dataclasses import dataclass


from barker_code import BARKER_BITS

from enum import Enum

class msgType(Enum):
    DATA = 0
    ACK = 1

@dataclass(init=False, repr=False)
class Datagram():
    """ 
    Messaage format:
        - barker_code: 2 bytes (13-bit Barker code as preamble, padded to 16 bits)
        - msg_type: 1 byte string (DATA or ACK)
        - msg_id: 1 byte np.uint8 (message ID for tracking)
        - size_payload: 1 byte np.uint8 (length of payload in bytes) (0 - 255)
        - payload: variable length np.uint8 array
        - crc16: 2 bytes (CRC16 checksum of header + payload)
    """
    _msg_id: np.uint8
    _msg_type: msgType
    _payload_size: np.uint8
    _payload: np.ndarray
    _crc16: np.uint16

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

        if len(payload) > 255:
            raise ValueError("Payload size exceeds maximum of 255 bytes.")
        
        self._msg_id = msg_id if msg_id is not None else np.random.randint(0, 256, dtype=np.uint8)
        self._msg_type = msg_type        
        self._payload_size = np.uint8(len(payload))
        self._payload = payload

        # compute CRC16 checksum over all fields
        self._crc16 = self.compute_crc16((
            bytes([self._msg_id]) +
            bytes([self._msg_type.value]) + 
            bytes([self._payload_size]) + 
            self._payload.tobytes())
            )
        

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
        return (self._msg_id.to_bytes(2, byteorder='big') +
                bytes([self._msg_type.value]) + 
                bytes([self._payload_size]) + 
                self._payload.tobytes() + 
                self._crc16.to_bytes(2, byteorder='big'))

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
        MIN_LENGTH = 5 # 1 byte msg_ID + 1 byte msg_type + 1 byte payload_size + 2 byte CRC16
        if len(data) < MIN_LENGTH:
            raise ValueError("Data length is too short to be a valid datagram.")

        instance = cls.__new__(cls)
        
        # parse fields      
        instance._msg_id = np.uint8(data[0])

        instance._msg_type = msgType(data[1])
        instance._payload_size = data[2]
        if len(data) != MIN_LENGTH + instance._payload_size:
            raise ValueError("Data length does not match expected length based on payload size.")

        
        # check if data length matches expected length based on payload size
        instance._payload = np.frombuffer(data[3:3+instance._payload_size], dtype=np.uint8)

        instance._crc16 = int.from_bytes(data[3+instance._payload_size:5+instance._payload_size], byteorder='big')

        # Verify CRC16
        computed_crc16 = instance.compute_crc16(
            bytes([instance._msg_id]) +
            bytes([instance._msg_type.value]) + 
            bytes([instance._payload_size]) + 
            instance._payload.tobytes()
            )
        
        if computed_crc16 != instance._crc16:
            raise ValueError("CRC16 checksum does not match, data may be corrupted.")

        return instance
    
   
    @staticmethod
    def compute_crc16(data: bytes) -> np.uint16:
        """Compute CRC16 checksum."""
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if (crc & 0x0001):
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc
        
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
    
    @property
    def get_crc16(self) -> np.uint16:
        """Get CRC16 checksum."""
        return self._crc16
    
    @property
    def get_payload_size(self) -> int:
        """Get payload size."""
        return int(self._payload_size)
    
    @property
    def get_total_size(self) -> int:
        """Get total size of the datagram in bytes."""
        return len(self.pack())
    
    def __repr__(self) -> str:
        return (f"\n\tmsg_ID:\t\t{self._msg_id},\n"
                f"\tmsg_type:\t{self._msg_type.name},\n"
                f"\tpayload_size:\t{self._payload_size},\n"
                f"\tpayload:\t{self._payload},\n"
                f"\tcrc16:\t\t{hex(self._crc16)}")

    
if __name__ == "__main__":
    # Example usage
    payload = np.array([1, 2, 3, 4], dtype=np.uint8)
    datagram = Datagram(msg_type=msgType.DATA, payload=payload)
    packed_data = datagram.pack()
    print(f"Packed data: {packed_data}")
    unpacked_datagram = Datagram.unpack(packed_data)
    print(f"Unpacked datagram: {unpacked_datagram}")

