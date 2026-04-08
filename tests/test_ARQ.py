import threading
import time

import numpy as np

from ARQ import StopAndWaitARQ
from datagram import Datagram, msgType


def make_data_datagram(msg_id: int, timestamp: int, payload: bytes) -> Datagram:
    return Datagram(
        msg_id=np.uint8(msg_id),
        msg_type=msgType.DATA,
        timestamp=np.uint32(timestamp),
        payload=np.frombuffer(payload, dtype=np.uint8),
    )


def test_outbound_ack_event_is_set_when_ack_arrives():
    arq = StopAndWaitARQ(ack_timeout_s=0.2, max_retries=2, duplicate_cache_size=8)
    ack_event = arq.register_outbound(np.uint8(7))

    def send_ack():
        time.sleep(0.02)
        arq.acknowledge(np.uint8(7))

    ack_thread = threading.Thread(target=send_ack)
    ack_thread.start()
    ack_received = ack_event.wait(timeout=arq.ack_timeout_s)
    ack_thread.join(timeout=1.0)

    assert ack_received


def test_outbound_ack_event_times_out_without_ack():
    arq = StopAndWaitARQ(ack_timeout_s=0.02, max_retries=2, duplicate_cache_size=8)
    ack_event = arq.register_outbound(np.uint8(8))

    assert not ack_event.wait(timeout=arq.ack_timeout_s)


def test_duplicate_inbound_data_is_suppressed_but_new_logical_message_is_allowed():
    arq = StopAndWaitARQ(ack_timeout_s=0.1, max_retries=2, duplicate_cache_size=8)
    first = make_data_datagram(9, 1234, b"hello")
    duplicate = make_data_datagram(9, 1234, b"hello")
    newer_same_id = make_data_datagram(9, 1235, b"hello")

    assert arq.mark_inbound_data(first)
    assert not arq.mark_inbound_data(duplicate)
    assert arq.mark_inbound_data(newer_same_id)


def test_transmit_with_retry_retries_until_ack_arrives():
    arq = StopAndWaitARQ(ack_timeout_s=0.02, max_retries=2, duplicate_cache_size=8)
    datagram = make_data_datagram(11, 2000, b"ack me")
    send_attempts = []
    retry_callbacks = []
    stop_event = threading.Event()

    def send_once(outbound: Datagram):
        send_attempts.append(int(outbound.get_msg_id))
        if len(send_attempts) == 2:
            arq.acknowledge(outbound.get_msg_id)

    result = arq.transmit_with_retry(
        datagram,
        send_once=send_once,
        stop_event=stop_event,
        on_retry_timeout=lambda outbound, next_attempt, max_attempts: retry_callbacks.append(
            (int(outbound.get_msg_id), next_attempt, max_attempts)
        ),
    )

    assert result.ack_received
    assert result.attempts == 2
    assert send_attempts == [11, 11]
    assert retry_callbacks == [(11, 2, 3)]
