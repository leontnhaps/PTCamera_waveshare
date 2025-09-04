#!/usr/bin/env python3
import os, io, json, time, glob, threading, datetime, socket, socketserver, struct
from picamera2 import Picamera2
import serial

# ====== 기본 설정 ======
CTRL_PORT = int(os.getenv("CTRL_PORT", "7500"))   # 제어
IMG_PORT  = int(os.getenv("IMG_PORT",  "7501"))   # 이미지 푸시
BAUD = 115200

PAN_MIN, PAN_MAX   = -180.0, 180.0
TILT_MIN, TILT_MAX =  -30.0,  90.0

MAX_W, MAX_H = 2592, 1944
STILL_Q = 90

# 저장은 선택(디스크 부하 줄이려면 False)
SAVE_TO_DISK = False
CAP_BASE = os.getenv("CAP_BASE", os.path.expanduser("~/captures_socket"))
os.makedirs(CAP_BASE, exist_ok=True)

# ====== 시리얼(ESP32) ======
def open_serial():
    cands = ['/dev/ttyUSB0','/dev/ttyUSB1','/dev/ttyAMA0','/dev/ttyS0'] + glob.glob('/dev/ttyUSB*')
    last = None
    for dev in cands:
        try:
            ser = serial.Serial(dev, BAUD, timeout=0.2)
            print(f"[SERIAL] Connected: {dev}")
            return ser
        except Exception as e: last = e
    raise RuntimeError(f"Serial open failed: {last}")

ser = open_serial()

def send_to_slave(obj: dict):
    ser.write((json.dumps(obj) + "\n").encode()); ser.flush()

# ====== 카메라 ======
picam = Picamera2()
cam_lock = threading.Lock()
def cam_to_still(w, h, q):
    with cam_lock:
        picam.stop()
        cfg = picam.create_still_configuration(main={"size": (w, h)})
        picam.configure(cfg)
        picam.options["quality"] = q
        picam.start(); time.sleep(0.12)

# ====== 이미지 푸시 (port 7501) ======
image_clients = set()
image_clients_lock = threading.Lock()

class ImageClientHandler(socketserver.BaseRequestHandler):
    def handle(self):
        sock = self.request
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with image_clients_lock:
            image_clients.add(sock)
        try:
            # 읽을 건 없고, 연결만 유지
            while True:
                data = sock.recv(1)
                if not data:
                    break
        except Exception:
            pass
        finally:
            with image_clients_lock:
                if sock in image_clients:
                    image_clients.remove(sock)
            try: sock.close()
            except: pass

class ImageServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

def push_image(name: str, data: bytes):
    """
    프레이밍: [uint16 name_len][name bytes][uint32 data_len][data]
    """
    header = struct.pack("<H", len(name.encode())) + name.encode() + struct.pack("<I", len(data))
    dead = []
    with image_clients_lock:
        for sock in list(image_clients):
            try:
                sock.sendall(header)
                sock.sendall(data)
            except Exception:
                dead.append(sock)
        for d in dead:
            image_clients.discard(d)
            try: d.close()
            except: pass

# ====== 제어(스캔 스레드 포함) ======
scan_thread = None
scan_stop_evt = threading.Event()
scan_state = {
    "running": False, "session": None, "total": 0, "done": 0, "error": None
}
state_lock = threading.Lock()

def irange(a,b,s):
    if s<=0: raise ValueError("step>0")
    out=[]; v=a
    if a<=b:
        while v<=b: out.append(v); v+=s
    else:
        while v>=b: out.append(v); v-=s
    return out

def now_session(): return datetime.datetime.now().strftime("scan_%Y%m%d_%H%M%S")

def scan_worker(params, progress_send):
    """
    params: dict( pan_min, pan_max, pan_step, tilt_min, tilt_max, tilt_step,
                  speed, acc, settle, width, height, quality, session, hard_stop )
    progress_send: callable(dict) -> 서버가 제어소켓으로 이벤트 푸시
    """
    try:
        w,h,q = params["width"], params["height"], params["quality"]
        cam_to_still(w,h,q)

        pans  = irange(params["pan_min"],  params["pan_max"],  params["pan_step"])
        tilts = irange(params["tilt_min"], params["tilt_max"], params["tilt_step"])
        total = len(pans)*len(tilts)
        with state_lock:
            scan_state.update({"running": True, "total": total, "done": 0, "session": params["session"], "error": None})

        for i,t in enumerate(tilts):
            row = pans if i%2==0 else list(reversed(pans))  # 지그재그
            for p in row:
                if scan_stop_evt.is_set(): raise InterruptedError

                # Move
                send_to_slave({"T":133, "X": float(p), "Y": float(t), "SPD": int(params["speed"]), "ACC": float(params["acc"])})
                # settle
                time.sleep(float(params["settle"]))
                # 선택: 이동 후 정지(취소용이 아니라 잔류 제거용, 약간의 지연만)
                if params.get("hard_stop", False):
                    send_to_slave({"T":135})
                    time.sleep(0.02)

                # Capture to memory
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                name = f"img_t{t:+03d}_p{p:+04d}_{ts}.jpg"
                bio = io.BytesIO()
                with cam_lock:
                    picam.capture_file(bio, format="jpeg")
                data = bio.getvalue()

                # (선택) 저장
                if SAVE_TO_DISK:
                    path = os.path.join(CAP_BASE, params["session"]); os.makedirs(path, exist_ok=True)
                    with open(os.path.join(path, name), "wb") as f: f.write(data)

                # push to clients
                push_image(name, data)

                with state_lock:
                    scan_state["done"] += 1
                    done = scan_state["done"]

                # 제어 소켓으로 진행률 이벤트
                try:
                    progress_send({"event":"progress","done":done,"total":total,"name":name})
                except Exception:
                    pass

        send_to_slave({"T":135})
    except InterruptedError:
        send_to_slave({"T":135})
    except Exception as e:
        with state_lock: scan_state["error"] = str(e)
        try: progress_send({"event":"error","message":str(e)})
        except: pass
    finally:
        with state_lock: scan_state["running"] = False
        scan_stop_evt.clear()
        try: progress_send({"event":"done"})
        except: pass

class ControlHandler(socketserver.BaseRequestHandler):
    def handle(self):
        global scan_thread
        sock = self.request
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        def send_line(obj):
            line = (json.dumps(obj, separators=(",",":"))+"\n").encode()
            sock.sendall(line)

        # 간단 헬스
        send_line({"event":"hello","msg":"PT socket server ready"})

        buf = b""
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
                    cmd = json.loads(line)
                except Exception:
                    send_line({"event":"error","message":"bad json"})
                    continue

                c = cmd.get("cmd")
                if c == "scan_run":
                    if scan_thread and scan_thread.is_alive():
                        send_line({"event":"error","message":"scan already running"})
                        continue
                    par = {
                        "pan_min": int(cmd["pan_min"]), "pan_max": int(cmd["pan_max"]), "pan_step": int(cmd["pan_step"]),
                        "tilt_min": int(cmd["tilt_min"]), "tilt_max": int(cmd["tilt_max"]), "tilt_step": int(cmd["tilt_step"]),
                        "speed": int(cmd.get("speed",100)), "acc": float(cmd.get("acc",1.0)), "settle": float(cmd.get("settle",0.25)),
                        "width": int(cmd.get("width",MAX_W)), "height": int(cmd.get("height",MAX_H)), "quality": int(cmd.get("quality",STILL_Q)),
                        "session": cmd.get("session") or now_session(),
                        "hard_stop": bool(cmd.get("hard_stop", False)),
                    }
                    scan_stop_evt.clear()
                    scan_thread = threading.Thread(target=scan_worker, args=(par, send_line), daemon=True)
                    scan_thread.start()
                    send_line({"event":"ack","ok":True,"session":par["session"]})

                elif c == "scan_stop":
                    scan_stop_evt.set()
                    send_line({"event":"ack","ok":True})

                elif c == "move":
                    send_to_slave({"T":133, "X": float(cmd.get("pan",0.0)), "Y": float(cmd.get("tilt",0.0)),
                                   "SPD": int(cmd.get("speed",100)), "ACC": float(cmd.get("acc",1.0))})
                    send_line({"event":"ack","ok":True})

                elif c == "led":
                    val = int(cmd.get("value",0))
                    payload = {"T":132, "IO4": val, "IO5": val}
                    send_to_slave(payload)
                    send_line({"event":"ack","ok":True})

                elif c == "status":
                    with state_lock:
                        st = dict(scan_state)
                    send_line({"event":"status", **st})

                else:
                    send_line({"event":"error","message":"unknown cmd"})

class ControlServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

def main():
    # 카메라 처음 1회 초기화(스틸 모드 진입)
    cam_to_still(MAX_W, MAX_H, STILL_Q)

    ctrl_srv = ControlServer(("0.0.0.0", CTRL_PORT), ControlHandler)
    img_srv  = ImageServer(("0.0.0.0", IMG_PORT), ImageClientHandler)

    t1 = threading.Thread(target=ctrl_srv.serve_forever, daemon=True); t1.start()
    t2 = threading.Thread(target=img_srv.serve_forever,  daemon=True); t2.start()
    print(f"[CTRL] TCP {CTRL_PORT}  [IMG] TCP {IMG_PORT}")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        ctrl_srv.shutdown(); img_srv.shutdown()

if __name__ == "__main__":
    main()
