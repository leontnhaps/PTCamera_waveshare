import socket
import threading
import json
import struct
import pathlib

class GuiCtrlClient(threading.Thread):
    def __init__(self, host, port, ui_q):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.ui_q = ui_q
        self.sock = None

    def run(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.connect((self.host, self.port))
            self.sock = s
            self.ui_q.put(("toast", f"CTRL connected {self.host}:{self.port}"))
            buf = b""
            while True:
                data = s.recv(4096)
                if not data: break
                buf += data
                while True:
                    nl = buf.find(b"\n")
                    if nl < 0: break
                    line = buf[:nl].decode("utf-8", "ignore").strip()
                    buf = buf[nl+1:]
                    if not line: continue
                    try:
                        evt = json.loads(line)
                    except:
                        continue
                    self.ui_q.put(("evt", evt))
        except Exception as e:
            self.ui_q.put(("toast", f"CTRL err: {e}"))

    def send(self, obj: dict):
        if not self.sock: return
        try:
            self.sock.sendall((json.dumps(obj, separators=(",", ":")) + "\n").encode())
        except Exception as e:
            print(f"[GuiCtrlClient] Send failed: {e}")

class GuiImgClient(threading.Thread):
    def __init__(self, host, port, outdir: pathlib.Path, ui_q):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.outdir = outdir
        self.ui_q = ui_q
        self.sock = None

    def run(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((self.host, self.port))
            self.sock = s
            self.ui_q.put(("toast", f"IMG connected {self.host}:{self.port}"))
            while True:
                hdr = s.recv(2)
                if not hdr: break
                (nlen,) = struct.unpack("<H", hdr)
                name = s.recv(nlen).decode("utf-8", "ignore")
                (dlen,) = struct.unpack("<I", s.recv(4))
                buf = bytearray()
                remain = dlen
                while remain > 0:
                    chunk = s.recv(min(65536, remain))
                    if not chunk: raise ConnectionError("img closed")
                    buf += chunk
                    remain -= len(chunk)
                data = bytes(buf)
                if name.startswith("_preview_"):
                    self.ui_q.put(("preview", data))
                else:
                    self.outdir.mkdir(parents=True, exist_ok=True)
                    with open(self.outdir / name, "wb") as f:
                        f.write(data)
                    self.ui_q.put(("saved", (name, data)))
        except Exception as e:
            self.ui_q.put(("toast", f"IMG err: {e}"))
