import numpy as np

from datagram import Datagram, msgType


def test_datagram_is_fixed_size_and_zero_padded_on_wire():
    datagram = Datagram.as_string("A ", msg_id=np.uint8(5), msg_type=msgType.DATA)
    packed = np.frombuffer(datagram.pack(), dtype=np.uint8)
    payload_start = Datagram.HEADER_SIZE

    assert packed.size == Datagram.TOTAL_SIZE
    assert datagram.get_msg_id == np.uint8(5)
    assert datagram.get_payload.tobytes() == b"A "
    assert datagram.get_payload_length == np.uint8(2)
    assert packed[Datagram.MSG_ID_SIZE + Datagram.MSG_TYPE_SIZE + Datagram.TIMESTAMP_SIZE] == 2
    assert datagram.get_payload_crc16 != np.uint16(0)
    assert packed[payload_start : payload_start + 2].tobytes() == b"A "
    assert np.all(packed[payload_start + 2 :] == Datagram.PAD_BYTE)


def test_unpack_roundtrip_preserves_payload_and_length():
    payload = np.array([1, 2, 3, 0], dtype=np.uint8)
    datagram = Datagram(msg_id=np.uint8(9), msg_type=msgType.DATA, payload=payload)
    unpacked = Datagram.unpack(datagram.pack())

    assert unpacked.get_msg_id == np.uint8(9)
    assert unpacked.get_msg_type == msgType.DATA
    assert unpacked.get_payload_length == np.uint8(4)
    assert np.array_equal(unpacked.get_payload, payload)


def test_unpack_roundtrip_keeps_trailing_zero_bytes_inside_payload():
    payload = b"iot\x00"
    datagram = Datagram.as_bytes(payload, msg_id=np.uint8(3), msg_type=msgType.DATA)
    unpacked = Datagram.unpack(datagram.pack())

    assert unpacked.get_payload_length == np.uint8(len(payload))
    assert unpacked.payload_bytes() == payload


def test_unpack_rejects_crc_mismatch():
    datagram = Datagram.as_string("crc", msg_id=np.uint8(4), msg_type=msgType.DATA)
    packed = bytearray(datagram.pack())
    crc_index = Datagram.MSG_ID_SIZE + Datagram.MSG_TYPE_SIZE + Datagram.TIMESTAMP_SIZE + Datagram.PAYLOAD_LENGTH_SIZE
    packed[crc_index] ^= 0x01

    try:
        Datagram.unpack(bytes(packed))
    except ValueError as exc:
        assert "CRC mismatch" in str(exc)
    else:
        raise AssertionError("Expected CRC mismatch to be detected")
