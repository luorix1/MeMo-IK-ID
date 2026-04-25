import os
import socket


class Teleplot:
    def __init__(self, ip: str = "127.0.0.1", port: int = 47269):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._debug = os.environ.get("TELEPLOT_DEBUG", "").strip().lower() in ("1", "true", "yes")
        self._send_error_printed = False

    def sendValue(self, name: str, value: float) -> None:
        try:
            message = f"{name}:{value}"
            self.sock.sendto(message.encode("utf-8"), (self.ip, self.port))
        except Exception as e:
            if self._debug and not self._send_error_printed:
                self._send_error_printed = True
                print(f"[Teleplot] sendto failed (further UDP errors suppressed): {type(e).__name__}: {e}")

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass
