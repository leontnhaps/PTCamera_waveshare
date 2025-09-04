#!/usr/bin/env python3
import os, io, json, time, glob, base64, threading
from flask import Flask, request, jsonify, Response
import serial
from picamera2 import Picamera2

# ===== 설정 =====
PORT = int(os.getenv("PORT", "7000"))
BAUD = int(os.getenv("SERIAL_BAUD", "115200"))
PAN_MIN, PAN_MAX = -180, 180
TILT_MIN, TILT_MAX = -30, 90

app = Flask(__name__)

def clamp(v, lo, hi): return max(lo, min(hi, v))

def open_serial():
    cands = ['/dev/ttyUSB0','/dev/ttyUSB1','/dev/ttyAMA0','/dev/ttyS0'] + glob.glob('/dev/ttyUSB*')
    last = None
    for dev in cands:
        try:
            ser = serial.Serial(dev, BAUD, timeout=0.2)
            print(f"[SERIAL] Connected: {dev}")
            return ser
        except Exception as e:
            last = e
    raise RuntimeError(f"Serial open failed: {last}")

ser = open_serial()

# ===== 카메라 구성 =====
# 프리뷰는 빠른 비디오 구성(640x480)으로 10~15fps 목표
picam = Picamera2()
preview_size = (640, 480)
picam.configure(picam.create_video_configuration(main={"size": preview_size}))
picam.options["quality"] = 70  # MJPEG 인코딩 부담 감소
picam.start(); time.sleep(0.2)

cam_lock = threading.Lock()  # 스트림/캡처 동시 접근 보호

def send_to_slave(obj: dict):
    line = json.dumps(obj) + "\n"  # 한 줄 단위
    ser.write(line.encode("utf-8"))
    ser.flush()

@app.get("/health")
def health():
    return jsonify(ok=True, preview=list(preview_size))

@app.post("/move")
def move():
    d = request.get_json(force=True)
    x = float(d.get("pan", 0)); y = float(d.get("tilt", 0))
    x = max(PAN_MIN, min(PAN_MAX, x))
    y = max(TILT_MIN, min(TILT_MAX, y))
    spd = max(0, int(d.get("speed", 0)))
    acc = max(0.0, float(d.get("acc", 0)))
    send_to_slave({"T":133, "X":x, "Y":y, "SPD":spd, "ACC":acc})
    return jsonify(ok=True, pan=x, tilt=y)

@app.post("/stop")
def stop():
    send_to_slave({"T":135})
    return jsonify(ok=True)

@app.post("/led")
def led():
    d = request.get_json(force=True)
    val = clamp(int(d.get("value", 0)), 0, 255)
    send_to_slave({"T":132, "IO4":val, "IO5":val})
    return jsonify(ok=True, value=val)

# 단일 프레임(폴링용) - 필요시 유지
@app.get("/frame")
def frame():
    with cam_lock:
        buf = io.BytesIO()
        picam.capture_file(buf, format="jpeg")
        data = buf.getvalue()
    return Response(data, mimetype="image/jpeg")

# 실시간 MJPEG 스트림(권장)
@app.get("/stream")
def stream():
    def gen():
        while True:
            with cam_lock:
                buf = io.BytesIO()
                picam.capture_file(buf, format="jpeg")
                frame = buf.getvalue()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.066)  # ~15 fps
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

# 고해상도 캡처(요청 시에만 리컨피그)
@app.post("/capture")
def capture():
    d = request.get_json(silent=True) or {}
    size = d.get("size", [2592, 1944])  # OV5647 최대
    w, h = int(size[0]), int(size[1])
    with cam_lock:
        picam.stop()
        picam.configure(picam.create_still_configuration(main={"size": (w, h)}))
        picam.start(); time.sleep(0.15)
        buf = io.BytesIO()
        picam.capture_file(buf, format="jpeg")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        # 프리뷰 모드로 즉시 복귀
        picam.stop()
        picam.configure(picam.create_video_configuration(main={"size": (640, 480)}))
        picam.options["quality"] = 70
        picam.start(); time.sleep(0.1)
    return jsonify(ok=True, image_base64=b64, size=[w, h])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, threaded=True)
