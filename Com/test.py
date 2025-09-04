#!/usr/bin/env python3
import json, socket, threading, os, struct, pathlib
from datetime import datetime

CTRL_HOST = "raspberrypi.local"
CTRL_PORT = 7500
IMG_PORT  = 7501
OUT_DIR   = f"captures_socket_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
pathlib.Path(OUT_DIR).mkdir(exist_ok=True, parents=True)

def send_line(sock: socket.socket, obj: dict):
    sock.sendall((json.dumps(obj, separators=(",",":"))+"\n").encode())

def ctrl_reader(sock: socket.socket):
    buf=b""
    while True:
        data = sock.recv(4096)
        if not data: break
        buf += data
        while True:
            nl = buf.find(b"\n")
            if nl < 0: break
            line = buf[:nl].decode("utf-8","ignore").strip()
            buf = buf[nl+1:]
            if not line: continue
            try:
                evt = json.loads(line)
            except Exception:
                print("CTRL> bad json")
                continue
            print("CTRL>", evt)

def img_reader(sock: socket.socket):
    while True:
        # [uint16 name_len][name][uint32 data_len][data]
        hdr = sock.recv(2)
        if not hdr: break
        (nlen,) = struct.unpack("<H", hdr)
        name = sock.recv(nlen).decode()
        (dlen,) = struct.unpack("<I", sock.recv(4))
        buf = bytearray()
        while len(buf) < dlen:
            chunk = sock.recv(min(65536, dlen - len(buf)))
            if not chunk: raise ConnectionError("img socket closed")
            buf += chunk
        out = os.path.join(OUT_DIR, name)
        with open(out, "wb") as f: f.write(buf)
        print(f"IMG > saved {name} ({len(buf)}B)")

def main():
    # 이미지 소켓 먼저 연결(서버가 바로 푸시 가능)
    img = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    img.connect((CTRL_HOST, IMG_PORT))
    threading.Thread(target=img_reader, args=(img,), daemon=True).start()

    # 제어 소켓 연결
    ctrl = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ctrl.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    ctrl.connect((CTRL_HOST, CTRL_PORT))
    threading.Thread(target=ctrl_reader, args=(ctrl,), daemon=True).start()

    # 한번의 명령으로 전체 스캔 시작 (필요 값만 바꿔)
    send_line(ctrl, {
        "cmd":"scan_run",
        "pan_min": -180, "pan_max": 180, "pan_step": 15,
        "tilt_min": -30, "tilt_max": 90, "tilt_step": 15,
        "speed": 100, "acc": 1.0, "settle": 0.25,
        "width": 2592, "height": 1944, "quality": 90,
        "session": datetime.now().strftime("scan_%Y%m%d_%H%M%S"),
        "hard_stop": False
    })

    # 필요하면 수동 이동/정지 예시:
    # send_line(ctrl, {"cmd":"move","pan":0,"tilt":0,"speed":100,"acc":1.0})
    # send_line(ctrl, {"cmd":"scan_stop"})

    # 대기
    try:
        while True: pass
    except KeyboardInterrupt:
        send_line(ctrl, {"cmd":"scan_stop"})

if __name__ == "__main__":
    main()
