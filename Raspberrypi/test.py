#!/usr/bin/env python3
import os, io, json, time, glob, threading, datetime, socket, struct
import serial
from picamera2 import Picamera2

# ==== 환경 ====
S_host, Y_host = "192.168.0.9","192.168.0.9"

SERVER_HOST = os.getenv(S_host, Y_host)  # ← 노트북 IP/호스트
CTRL_PORT   = int(os.getenv("CTRL_PORT", "7500"))
IMG_PORT    = int(os.getenv("IMG_PORT",  "7501"))
BAUD = 115200
MAX_W, MAX_H = 2592, 1944

# ==== 시리얼(ESP32) ====
def open_serial():
    cands = ['/dev/ttyUSB0','/dev/ttyUSB1','/dev/ttyAMA0','/dev/ttyS0'] + glob.glob('/dev/ttyUSB*')
    last=None
    for dev in cands:
        try:
            s = serial.Serial(dev, BAUD, timeout=0.2)
            print(f"[SERIAL] Connected: {dev}")
            return s
        except Exception as e: last=e
    raise RuntimeError(f"Serial open failed: {last}")
ser = open_serial()
def send_to_slave(obj: dict):
    ser.write((json.dumps(obj) + "\n").encode()); ser.flush()

# ==== 카메라 ====
picam = Picamera2()
cam_lock = threading.Lock()

def to_still(w,h,q):
    with cam_lock:
        picam.stop()
        cfg = picam.create_still_configuration(main={"size": (w, h)})
        picam.configure(cfg)
        picam.options["quality"] = q
        picam.start(); time.sleep(0.12)

def to_preview(w,h,q):
    with cam_lock:
        picam.stop()
        cfg = picam.create_video_configuration(main={"size": (w, h)})
        picam.configure(cfg)
        picam.options["quality"] = q
        picam.start(); time.sleep(0.10)

# ==== 공통 유틸 ====
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

# ==== 프리뷰 스레드 ====
preview_thread = None
preview_stop = threading.Event()
preview_running = threading.Event()  # 상태 표시 용

def preview_worker(img_sock: socket.socket, w=640, h=360, fps=5, q=70):
    try:
        to_preview(w,h,q)
        preview_running.set()
        interval = 1.0/max(1,fps)
        while not preview_stop.is_set():
            bio = io.BytesIO()
            with cam_lock:
                picam.capture_file(bio, format="jpeg")
            push_image(img_sock, f"_preview_{int(time.time()*1000)}.jpg", bio.getvalue())
            time.sleep(interval)
    except Exception as e:
        pass
    finally:
        preview_running.clear()

def preview_start(img_sock, w=640, h=360, fps=5, q=70):
    global preview_thread
    preview_stop.set();  # 혹시 돌던 스레드 있으면 종료 지시
    if preview_thread and preview_thread.is_alive():
        preview_thread.join(timeout=0.5)
    preview_stop.clear()
    preview_thread = threading.Thread(target=preview_worker, args=(img_sock,w,h,fps,q), daemon=True)
    preview_thread.start()

def preview_stop_now():
    preview_stop.set()

# ==== 스캔 ====
scan_stop_evt = threading.Event()

def scan_worker(params, ctrl_sock: socket.socket, img_sock: socket.socket):
    try:
        # 프리뷰가 켜져 있으면 일단 멈춤(카메라 경합 방지)
        preview_stop_now()
        if preview_thread and preview_thread.is_alive():
            preview_thread.join(timeout=0.7)

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
        done=0

        for i,t in enumerate(tilts):
            row = pans if i%2==0 else list(reversed(pans))
            for p in row:
                if scan_stop_evt.is_set():
                    raise InterruptedError
                # move
                send_to_slave({"T":133, "X": float(p), "Y": float(t), "SPD": speed, "ACC": acc})
                time.sleep(settle)
                if hard_stop:
                    send_to_slave({"T":135}); time.sleep(0.02)

                bio = io.BytesIO()
                with cam_lock:
                    picam.capture_file(bio, format="jpeg")
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                name = f"img_t{t:+03d}_p{p:+04d}_{ts}.jpg"
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

# ==== 메인: 노트북 서버에 접속 ====
def main():
    # 이미지 소켓
    while True:
        try:
            img = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            img.connect((SERVER_HOST, IMG_PORT))
            print(f"[IMG] connected {SERVER_HOST}:{IMG_PORT}")
            break
        except Exception as e:
            print(f"[IMG] retry: {e}"); time.sleep(1.0)
    # 제어 소켓
    while True:
        try:
            ctrl = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            ctrl.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            ctrl.connect((SERVER_HOST, CTRL_PORT))
            print(f"[CTRL] connected {SERVER_HOST}:{CTRL_PORT}")
            break
        except Exception as e:
            print(f"[CTRL] retry: {e}"); time.sleep(1.0)

    # 제어 수신 루프
    buf=b""
    while True:
        data = ctrl.recv(4096)
        if not data: break
        buf += data
        while True:
            nl = buf.find(b"\n")
            if nl < 0: break
            line = buf[:nl].decode("utf-8","ignore").strip()
            buf = buf[nl+1:]
            if not line: continue
            try:
                cmd = json.loads(line)
            except: 
                continue
            c = cmd.get("cmd")
            if c == "scan_run":
                threading.Thread(target=scan_worker, args=(cmd, ctrl, img), daemon=True).start()
            elif c == "scan_stop":
                scan_stop_evt.set()
            elif c == "move":
                send_to_slave({"T":133, "X": float(cmd.get("pan",0.0)), "Y": float(cmd.get("tilt",0.0)),
                               "SPD": int(cmd.get("speed",100)), "ACC": float(cmd.get("acc",1.0))})
            elif c == "led":
                val = int(cmd.get("value",0))
                send_to_slave({"T":132, "IO4": val, "IO5": val})
            elif c == "preview":
                enable = bool(cmd.get("enable", True))
                if enable:
                    w=int(cmd.get("width",640)); h=int(cmd.get("height",360))
                    fps=int(cmd.get("fps",5)); q=int(cmd.get("quality",70))
                    preview_start(img, w,h,fps,q)
                else:
                    preview_stop_now()
            else:
                pass

if __name__ == "__main__":
    main()
