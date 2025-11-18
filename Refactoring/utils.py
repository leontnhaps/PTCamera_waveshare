# utils.py
import socket
import struct
import json

def irange(a, b, s):
    """
    범위 생성 함수 (Rasp_main.py에서 이동)
    step이 양수/음수에 관계없이 a에서 b까지 s 간격으로 생성
    """
    if s <= 0: 
        raise ValueError("step > 0")
    out = []
    v = a
    if a <= b:
        while v <= b: 
            out.append(v)
            v += s
    else:
        while v >= b: 
            out.append(v)
            v -= s
    return out

def push_image(sock: socket.socket, name: str, data: bytes):
    """
    이미지 전송 함수 (Rasp_main.py에서 이동)
    GUI 이미지 채널 프로토콜로 이미지 전송
    """
    header = struct.pack("<H", len(name.encode())) + name.encode() + struct.pack("<I", len(data))
    sock.sendall(header)
    sock.sendall(data)

def send_json_line(sock: socket.socket, obj: dict):
    """
    JSON 라인 전송 함수 (server_main.py에서 이동)
    """
    data = (json.dumps(obj, separators=(",", ":")) + "\n").encode()
    sock.sendall(data)