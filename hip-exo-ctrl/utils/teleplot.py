import socket


class Teleplot:
    def __init__(self, ip: str = "127.0.0.1", port: int = 47269):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def sendValue(self, name: str, value: float) -> None:
        try:
            message = f"{name}:{value}"
            self.sock.sendto(message.encode("utf-8"), (self.ip, self.port))
        except Exception:
            pass

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass
