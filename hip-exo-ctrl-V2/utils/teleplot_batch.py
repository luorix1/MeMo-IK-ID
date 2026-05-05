"""UDP teleplot batch sender (`name:ms:value|g` per channel), legacy State2Torque format."""

from __future__ import annotations

import socket
import time


class TeleplotBatch:
    def __init__(self, ip: str = "127.0.0.1", port: int = 47269):
        self.addr = (ip, int(port))
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, data_dict: dict) -> bool:
        now = time.time() * 1000
        try:
            for name, value in data_dict.items():
                msg = f"{name}:{now}:{value}|g"
                self._sock.sendto(msg.encode(), self.addr)
            return True
        except OSError as e:
            print(f"[TeleplotBatch] send error: {e}")
            return False

    def close(self):
        try:
            self._sock.close()
        except OSError:
            pass
