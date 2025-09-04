#!/usr/bin/env python3
import os, io, json, time, glob, threading, datetime, socket, struct
import serial
from picamera2 import Picamera2

# ── 환경설정 ──

SERVER_HOST = os.getenv("192.168.0.9", "192.168.0.9")  # ← 노트북 IP/호스트로 설정
CTRL_PORT   = int(os.getenv("CTRL_PORT", "7500"))
IMG_PORT    = int(os.getenv("IMG_PORT",  "7501"))
BAUD = 115200

PAN_MIN, PAN_MAX   = -180.0, 180.0
TILT_MIN, TILT_MAX =  -30.0,  90.0
MAX_W, MAX_H       = 2592, 1944

# ── 시리얼 (ESP32) ──
def open_serial():
    cands = ['/dev/ttyUSB0','/dev/ttyUSB1','/dev/ttyAMA0','/dev/ttyS0'] + glob.glob('/dev/ttyUSB*')
    last=None
    for dev in cands:
        try:
            ser = serial.Serial(dev, BAUD, timeout=0.2)
            print(f"[SERIAL] Connected: {dev}")
            return ser
        except Exception as e: last=e
    raise RuntimeError(f"Serial open failed: {last}")

ser = open_serial()

def send_to_slave(obj: dict):
    ser.write((json.dumps(obj) + "\n").encode()); ser.flush()

# ── 카메라 ──
picam = Picamera2()
cam_lock = threading.Lock()

def to_still(w,h,q):
    with cam_lock:
        picam.stop()
        cfg = picam.create_still_configuration(main={"size": (w, h)})
        picam.configure(cfg)
        picam.options["quality"] = q
        picam.start(); time.sleep(0.12)

# ── 유틸 ──
def irange(a,b,s):
    if s<=0: raise ValueError("step>0")
    out=[]; v=a
    if a<=b:
        while v<=b: out.append(v); v+=s
    else:
        while v>=b: out.append(v); v-=s
    return out

def push_image(sock: socket.socket, name: str, data: bytes):
    header = struct.pack("<H", len(name.encode())) + name.encode() + struct.pack("<I", len(data))
    sock.sendall(header); sock.sendall(data)

# ── 스캔 제어 ──
scan_stop_evt = threading.Event()

def scan_worker(params, ctrl_sock: socket.socket, img_sock: socket.socket):
    try:
        w = int(params.get("width",MAX_W)); h = int(params.get("height",MAX_H)); q = int(params.get("quality",90))
        to_still(w,h,q)

        pans  = irange(int(params["pan_min"]),  int(params["pan_max"]),  int(params["pan_step"]))
        tilts = irange(int(params["tilt_min"]), int(params["tilt_max"]), int(params["tilt_step"]))
        total = len(pans)*len(tilts)

        speed  = int(params.get("speed",100))
        acc    = float(params.get("acc",1.0))
        settle = float(params.get("settle",0.25))
        hard_stop = bool(params.get("hard_stop", False))

        def send_evt(obj): ctrl_sock.sendall((json.dumps(obj,separators=(",",":"))+"\n").encode())

        send_evt({"event":"start","total":total})
        done = 0

        for i,t in enumerate(tilts):
            row = pans if i%2==0 else list(reversed(pans))  # 지그재그
            for p in row:
                if scan_stop_evt.is_set():
                    raise InterruptedError

                # 1) 이동
                send_to_slave({"T":133, "X": float(p), "Y": float(t), "SPD": speed, "ACC": acc})

                # 2) 안정 대기
                time.sleep(settle)

                # 3) (선택) 정지 펄스(잔류 제거)
                if hard_stop:
                    send_to_slave({"T":135})
                    time.sleep(0.02)

                # 4) 캡처 → 메모리 → 즉시 푸시
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                name = f"img_t{t:+03d}_p{p:+04d}_{ts}.jpg"
                bio = io.BytesIO()
                with cam_lock:
                    picam.capture_file(bio, format="jpeg")
                push_image(img_sock, name, bio.getvalue())

                done += 1
                send_evt({"event":"progress","done":done,"total":total,"name":name})

        send_to_slave({"T":135})
        send_evt({"event":"done"})
    except InterruptedError:
        send_to_slave({"T":135})
        try: ctrl_sock.sendall((json.dumps({"event":"aborted"})+"\n").encode())
        except: pass
    except Exception as e:
        try: ctrl_sock.sendall((json.dumps({"event":"error","message":str(e)})+"\n").encode())
        except: pass
    finally:
        scan_stop_evt.clear()

# ── 메인 루프: 서버로 접속 ──
def main():
    # 이미지 소켓 먼저(서버가 수신 준비)
    while True:
        try:
            img = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            img.connect((SERVER_HOST, IMG_PORT))
            print(f"[IMG] connected to {SERVER_HOST}:{IMG_PORT}")
            break
        except Exception as e:
            print(f"[IMG] connect retry: {e}"); time.sleep(1.0)

    # 제어 소켓
    while True:
        try:
            ctrl = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            ctrl.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            ctrl.connect((SERVER_HOST, CTRL_PORT))
            print(f"[CTRL] connected to {SERVER_HOST}:{CTRL_PORT}")
            break
        except Exception as e:
            print(f"[CTRL] connect retry: {e}"); time.sleep(1.0)

    # 제어 수신 루프
    buf=b""
    while True:
        data = ctrl.recv(4096)
        if not data:
            print("[CTRL] server closed")
            break
        buf += data
        while True:
            nl = buf.find(b"\n")
            if nl < 0:
                break
            line = buf[:nl].decode("utf-8","ignore").strip()
            buf = buf[nl+1:]
            if not line:
                continue
            try:
                cmd = json.loads(line)
            except Exception:
                continue

            c = cmd.get("cmd")
            if c == "scan_run":
                print("[CTRL] scan_run")
                threading.Thread(target=scan_worker, args=(cmd, ctrl, img), daemon=True).start()
            elif c == "scan_stop":
                scan_stop_evt.set()
            elif c == "move":
                send_to_slave({"T":133, "X": float(cmd.get("pan",0.0)), "Y": float(cmd.get("tilt",0.0)),
                               "SPD": int(cmd.get("speed",100)), "ACC": float(cmd.get("acc",1.0))})
            elif c == "led":
                val = int(cmd.get("value",0))
                send_to_slave({"T":132, "IO4": val, "IO5": val})
            else:
                pass

if __name__ == "__main__":
    main()
