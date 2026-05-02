import socket

class Teleplot:
    def __init__(self, ip="127.0.0.1", port=47269):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def sendValue(self, name, value):
        try:
            message = f"{name}:{value}"
            self.sock.sendto(message.encode('utf-8'), (self.ip, self.port))
        except Exception:
            pass

    def close(self):
        try:
            self.sock.close()
        except:
            pass