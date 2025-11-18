#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pi Agent — 라즈베리파이 카메라 제어 + PC 브로커(pc_server.py)와 소켓 통신
- 프리뷰(연속 전송), 스캔(팬틸트 그리드 캡처), 한장 캡처(snap) 지원
- 프리뷰/스캔/스냅 모두 Picamera2의 JPEG 경로 사용 → 색상/노출 일관
"""

import os, io, json, time, glob, threading, datetime, socket, struct
import serial
from picamera2 import Picamera2

# ===================== 환경 설정 =====================
DEFAULT_SERVER_HOST = "192.168.0.9" # 711a
#DEFAULT_SERVER_HOST = "172.30.1.100" # 602a
SERVER_HOST = os.getenv("SERVER_HOST", DEFAULT_SERVER_HOST)   # 노트북/PC 서버 IP
CTRL_PORT   = int(os.getenv("CTRL_PORT", "7500"))
IMG_PORT    = int(os.getenv("IMG_PORT",  "7501"))
BAUD        = 115200

MAX_W, MAX_H = 2592, 1944  # 센서 최대(모듈에 따라 조정)

# ===================== 시리얼(ESP32) =====================
def open_serial():
    cands = ['/dev/ttyUSB0','/dev/ttyUSB1','/dev/ttyAMA0','/dev/ttyS0'] + glob.glob('/dev/ttyUSB*')
    last=None
    for dev in cands:
        try:
            s = serial.Serial(dev, BAUD, timeout=0.2)
            print(f"[SERIAL] Connected: {dev}")
            return s
        except Exception as e:
            last=e
    raise RuntimeError(f"Serial open failed: {last}")

ser = open_serial()

def send_to_slave(obj: dict):
    ser.write((json.dumps(obj) + "\n").encode()); ser.flush()

# ===================== 카메라 =====================
picam = Picamera2()
cam_lock = threading.Lock()

def to_still(w,h,q):
    """정지 촬영용 모드로 전환 + 3A 수렴 대기"""
    with cam_lock:
        picam.stop()
        cfg = picam.create_still_configuration(main={"size": (w, h)})
        picam.configure(cfg)
        picam.options["quality"] = int(q)
        picam.start()
        time.sleep(0.6)  # AE/AWB 수렴 대기 (환경 따라 0.4~0.8 조절)

def to_preview(w,h,q):
    """프리뷰 모드로 전환"""
    with cam_lock:
        picam.stop()
        cfg = picam.create_video_configuration(main={"size": (w, h)})
        picam.configure(cfg)
        picam.options["quality"] = int(q)
        picam.start()
        time.sleep(0.10)

# ===================== 공통 유틸 =====================
def irange(a,b,s):
    if s<=0: raise ValueError("step>0")
    out=[]; v=a
    if a<=b:
        while v<=b: out.append(v); v+=s
    else:
        while v>=b: out.append(v); v-=s
    return out

def push_image(sock: socket.socket, name: str, data: bytes):
    """GUI 이미지 채널 프로토콜로 이미지 전송"""
    header = struct.pack("<H", len(name.encode())) + name.encode() + struct.pack("<I", len(data))
    sock.sendall(header); sock.sendall(data)

# ===================== 프리뷰 스레드 =====================
preview_thread = None
preview_stop = threading.Event()
preview_running = threading.Event()

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
        print("[PREVIEW] err:", e)
    finally:
        preview_running.clear()

def preview_start(img_sock, w=640, h=360, fps=5, q=70):
    global preview_thread
    preview_stop.set()  # 혹시 돌던 스레드 종료 지시
    if preview_thread and preview_thread.is_alive():
        preview_thread.join(timeout=0.5)
    preview_stop.clear()
    preview_thread = threading.Thread(target=preview_worker, args=(img_sock,w,h,fps,q), daemon=True)
    preview_thread.start()

def preview_stop_now():
    preview_stop.set()
    if preview_thread and preview_thread.is_alive():
        preview_thread.join(timeout=0.7)

# ===================== 스캔 =====================
scan_stop_evt = threading.Event()

def scan_worker(params, ctrl_sock: socket.socket, img_sock: socket.socket):
    try:
        # 프리뷰 멈춤(자원 경합 방지)
        preview_stop_now()

        w = int(params.get("width",MAX_W))
        h = int(params.get("height",MAX_H))
        q = int(params.get("quality",90))
        to_still(w,h,q)

        pans  = irange(int(params["pan_min"]),  int(params["pan_max"]),  int(params["pan_step"]))
        tilts = irange(int(params["tilt_min"]), int(params["tilt_max"]), int(params["tilt_step"]))
        total = len(pans)*len(tilts)
        speed  = int(params.get("speed",100))
        acc    = float(params.get("acc",1.0))
        settle = float(params.get("settle",0.25))
        hard_stop = bool(params.get("hard_stop", False))

        def send_evt(obj):
            try: ctrl_sock.sendall((json.dumps(obj,separators=(",",":"))+"\n").encode())
            except: pass

        send_evt({"event":"start","total":total})
        done=0

        for i,t in enumerate(tilts):
            row = pans if i%2==0 else list(reversed(pans))
            for p in row:
                if scan_stop_evt.is_set():
                    raise InterruptedError
                # 이동
                send_to_slave({"T":133, "X": float(p), "Y": float(t), "SPD": speed, "ACC": acc})
                time.sleep(settle)
                if hard_stop:
                    send_to_slave({"T":135}); time.sleep(0.02)

                # 캡처(JPEG)
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

# ===================== 스냅(한 장 캡처) =====================
def snap_once(cmd: dict, img_sock: socket.socket, ctrl_sock: socket.socket):
    """
    {"cmd":"snap","width","height","quality","save","hard_stop"} 처리.
    스캔과 동일한 경로( ISP JPEG )로 캡처해서 확실히 전송.
    """
    try:
        # 프리뷰 완전 정지
        preview_stop.set()
        global preview_thread
        if preview_thread and preview_thread.is_alive():
            preview_thread.join(timeout=0.8)

        # 파라미터
        W = int(cmd.get("width",  MAX_W))
        H = int(cmd.get("height", MAX_H))
        Q = int(cmd.get("quality", 90))
        fname = cmd.get("save") or datetime.datetime.now().strftime("snap_%Y%m%d_%H%M%S.jpg")
        if bool(cmd.get("hard_stop", False)):
            send_to_slave({"T":135}); time.sleep(0.02)

        # 스캔과 동일한 경로: 스틸 모드 전환 + 3A 대기 후 capture_file
        to_still(W, H, Q)

        # (안정성) 워밍업 1장 버리기 -> 조명/화이트밸런스 과민한 환경에서 유효
        with cam_lock:
            _warm = io.BytesIO()
            picam.capture_file(_warm, format="jpeg")
        time.sleep(0.05)

        # 실제 스냅
        bio = io.BytesIO()
        with cam_lock:
            picam.capture_file(bio, format="jpeg")

        # IMG 채널로 전송
        push_image(img_sock, fname, bio.getvalue())

        # 완료 이벤트(옵션)
        try:
            ctrl_sock.sendall((json.dumps({"event":"snap_done","name":fname,"size":bio.tell()})+"\n").encode())
        except:
            pass

    except Exception as e:
        try:
            ctrl_sock.sendall((json.dumps({"event":"error","where":"snap","message":str(e)})+"\n").encode())
        except:
            pass


# ===================== 메인: PC 서버에 접속 =====================
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
            elif c == "snap":
                snap_once(cmd, img, ctrl)
            else:
                # 알 수 없는 명령은 무시
                pass

if __name__ == "__main__":
    main()
