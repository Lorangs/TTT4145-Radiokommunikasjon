from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
import threading
import time
from typing import Deque

from datagram import Datagram


@dataclass(frozen=True)
class ARQTransmitResult:
    ack_received: bool
    attempts: int


class StopAndWaitARQ:
    """Track ACK waits, retransmissions, and duplicate inbound DATA delivery."""

    def __init__(
        self,
        *,
        ack_timeout_s: float = 1.5,
        max_retries: int = 2,
        duplicate_cache_size: int = 128,
    ):
        self.ack_timeout_s = max(0.0, float(ack_timeout_s))
        self.max_retries = max(0, int(max_retries))
        self.duplicate_cache_size = max(1, int(duplicate_cache_size))

        self._lock = threading.Lock()
        self._pending_acks: dict[int, threading.Event] = {}
        self._seen_data_order: Deque[tuple[int, int, int]] = deque()
        self._seen_data_keys: set[tuple[int, int, int]] = set()

    @staticmethod
    def _normalize_msg_id(msg_id) -> int:
        return int(msg_id) & 0xFF

    @staticmethod
    def inbound_delivery_key(datagram: Datagram) -> tuple[int, int, int]:
        return (
            int(datagram.get_msg_id),
            int(datagram.get_timestamp),
            int(datagram.get_payload_crc16),
        )

    def register_outbound(self, msg_id) -> threading.Event:
        normalized_msg_id = self._normalize_msg_id(msg_id)
        event = threading.Event()
        with self._lock:
            self._pending_acks[normalized_msg_id] = event
        return event

    def get_outbound_event(self, msg_id) -> threading.Event | None:
        normalized_msg_id = self._normalize_msg_id(msg_id)
        with self._lock:
            return self._pending_acks.get(normalized_msg_id)

    def acknowledge(self, msg_id) -> bool:
        event = self.get_outbound_event(msg_id)
        if event is None:
            return False
        event.set()
        return True

    def clear_outbound(self, msg_id) -> None:
        normalized_msg_id = self._normalize_msg_id(msg_id)
        with self._lock:
            self._pending_acks.pop(normalized_msg_id, None)

    def wait_for_ack(
        self,
        msg_id,
        *,
        stop_event: threading.Event,
        drain_control_queue: Callable[[], None] | None = None,
        poll_interval_s: float = 0.05,
    ) -> bool:
        ack_event = self.get_outbound_event(msg_id)
        if ack_event is None:
            return False

        deadline = time.monotonic() + self.ack_timeout_s
        while not stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return ack_event.is_set()
            if ack_event.wait(timeout=min(max(0.0, poll_interval_s), remaining)):
                return True
            if drain_control_queue is not None:
                drain_control_queue()

        return ack_event.is_set()

    def transmit_with_retry(
        self,
        datagram: Datagram,
        *,
        send_once: Callable[[Datagram], None],
        stop_event: threading.Event,
        drain_control_queue: Callable[[], None] | None = None,
        on_retry_timeout: Callable[[Datagram, int, int], None] | None = None,
    ) -> ARQTransmitResult:
        max_attempts = self.max_retries + 1
        self.register_outbound(datagram.get_msg_id)

        try:
            for attempt in range(1, max_attempts + 1):
                send_once(datagram)
                ack_received = self.wait_for_ack(
                    datagram.get_msg_id,
                    stop_event=stop_event,
                    drain_control_queue=drain_control_queue,
                )
                if ack_received:
                    return ARQTransmitResult(ack_received=True, attempts=attempt)
                if attempt < max_attempts and on_retry_timeout is not None:
                    on_retry_timeout(datagram, attempt + 1, max_attempts)

            return ARQTransmitResult(ack_received=False, attempts=max_attempts)
        finally:
            self.clear_outbound(datagram.get_msg_id)

    def mark_inbound_data(self, datagram: Datagram) -> bool:
        key = self.inbound_delivery_key(datagram)
        with self._lock:
            if key in self._seen_data_keys:
                return False

            if len(self._seen_data_order) >= self.duplicate_cache_size:
                evicted = self._seen_data_order.popleft()
                self._seen_data_keys.discard(evicted)

            self._seen_data_order.append(key)
            self._seen_data_keys.add(key)
            return True
