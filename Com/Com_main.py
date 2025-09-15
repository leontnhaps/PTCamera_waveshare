#!/usr/bin/env python3
# pc_gui.py — GUI client connecting to pc_server.py (not to Pi agent)

import json, socket, struct, threading, queue, pathlib, io
from datetime import datetime
from tkinter import Tk, Label, Button, Scale, HORIZONTAL, IntVar, DoubleVar, Frame, Checkbutton, BooleanVar, filedialog
from tkinter import ttk
from PIL import Image, ImageTk

# === NEW ===
import numpy as np
import cv2

SERVER_HOST = "127.0.0.1"   # pc_server.py가 실행 중인 호스트
GUI_CTRL_PORT = 7600        # pc_server.py의 GUI 포트
GUI_IMG_PORT  = 7601

DEFAULT_OUT_DIR = pathlib.Path(f"captures_gui_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)

ui_q: "queue.Queue[tuple[str,object]]" = queue.Queue()

# ---- client sockets ----
class GuiCtrlClient(threading.Thread):
    def __init__(self, host, port):
        super().__init__(daemon=True); self.host=host; self.port=port; self.sock=None
    def run(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.connect((self.host, self.port)); self.sock=s
            ui_q.put(("toast", f"CTRL connected {self.host}:{self.port}"))
            buf=b""
            while True:
                data = s.recv(4096)
                if not data: break
                buf += data
                while True:
                    nl = buf.find(b"\n")
                    if nl<0: break
                    line = buf[:nl].decode("utf-8","ignore").strip()
                    buf = buf[nl+1:]
                    if not line: continue
                    try: evt = json.loads(line)
                    except: continue
                    ui_q.put(("evt", evt))
        except Exception as e:
            ui_q.put(("toast", f"CTRL err: {e}"))
    def send(self, obj: dict):
        if not self.sock: return
        self.sock.sendall((json.dumps(obj, separators=(",",":"))+"\n").encode())

class GuiImgClient(threading.Thread):
    def __init__(self, host, port, outdir: pathlib.Path):
        super().__init__(daemon=True); self.host=host; self.port=port; self.outdir=outdir; self.sock=None
    def run(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((self.host, self.port)); self.sock=s
            ui_q.put(("toast", f"IMG connected {self.host}:{self.port}"))
            while True:
                hdr = s.recv(2)
                if not hdr: break
                (nlen,) = struct.unpack("<H", hdr)
                name = s.recv(nlen).decode("utf-8","ignore")
                (dlen,) = struct.unpack("<I", s.recv(4))
                buf = bytearray(); remain=dlen
                while remain>0:
                    chunk = s.recv(min(65536, remain))
                    if not chunk: raise ConnectionError("img closed")
                    buf+=chunk; remain-=len(chunk)
                data = bytes(buf)
                if name.startswith("_preview_"):
                    ui_q.put(("preview", data))
                else:
                    self.outdir.mkdir(parents=True, exist_ok=True)
                    with open(self.outdir / name, "wb") as f: f.write(data)
                    ui_q.put(("saved", (name, data)))
        except Exception as e:
            ui_q.put(("toast", f"IMG err: {e}"))

# ---- GUI ----
class App:
    def __init__(self, root: Tk):
        self.root = root
        root.title("Pan-Tilt Socket GUI (Client)")
        root.geometry("980x820")

        # connections
        self.ctrl = GuiCtrlClient(SERVER_HOST, GUI_CTRL_PORT); self.ctrl.start()
        self.img  = GuiImgClient (SERVER_HOST, GUI_IMG_PORT, DEFAULT_OUT_DIR); self.img.start()

        # state
        self.tkimg=None
        self._resume_preview_after_snap = False

        # === NEW: undistort state ===
        self.ud_enable    = BooleanVar(value=False)   # 프리뷰 보정 on/off
        self.ud_save_copy = BooleanVar(value=False)   # 저장시 보정본 추가 저장
        self.ud_alpha     = DoubleVar(value=0.0)      # pinhole alpha / fisheye balance (0~1)

        self._ud_model = None           # "pinhole" | "fisheye"
        self._ud_K = self._ud_D = None  # np.ndarray(float32)
        self._ud_img_size = None        # (Wc,Hc) — 캘리브 기준 해상도
        self._ud_src_size = None        # 현재 맵 생성된 입력 프레임 크기 (w,h)
        self._ud_m1 = self._ud_m2 = None  # remap map

        # CUDA 경로(있으면 사용)
        self._use_cuda = False
        try:
            self._use_cuda = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
        except Exception:
            self._use_cuda = False
        self._ud_gm1 = self._ud_gm2 = None  # CUDA용 맵
        self._gpu_tmp = None                # CUDA용 임시 버퍼

        # top bar
        top = Frame(root); top.pack(fill="x", padx=10, pady=6)
        Button(top, text="한장 찍기 (Snap)", command=self.snap_one).pack(side="left", padx=(0,8))
        Button(top, text="출력 폴더", command=self.choose_outdir).pack(side="right")

        # preview fixed box
        center = Frame(root); center.pack(fill="x", padx=10)
        self.PREV_W, self.PREV_H = 800, 450
        self.preview_box = Frame(center, width=self.PREV_W, height=self.PREV_H,
                                 bg="#111", highlightthickness=1, highlightbackground="#333")
        self.preview_box.pack(); self.preview_box.pack_propagate(False)
        self.preview = Label(self.preview_box, bg="#222"); self.preview.pack(fill="both", expand=True)

        # bottom tabs
        nb = ttk.Notebook(root); nb.pack(fill="x", padx=10, pady=(6,10))
        tab_scan   = Frame(nb); nb.add(tab_scan, text="Scan")
        tab_manual = Frame(nb); nb.add(tab_manual, text="Manual / LED")
        tab_misc   = Frame(nb); nb.add(tab_misc, text="Preview & Settings")

        # scan params
        self.pan_min=IntVar(value=-180); self.pan_max=IntVar(value=180); self.pan_step=IntVar(value=15)
        self.tilt_min=IntVar(value=-30); self.tilt_max=IntVar(value=90);  self.tilt_step=IntVar(value=15)
        self.width=IntVar(value=2592);   self.height=IntVar(value=1944); self.quality=IntVar(value=90)
        self.speed=IntVar(value=100);    self.acc=DoubleVar(value=1.0);  self.settle=DoubleVar(value=0.25)
        self.hard_stop = BooleanVar(value=False)

        self._row(tab_scan, 0, "Pan min/max/step", self.pan_min, self.pan_max, self.pan_step)
        self._row(tab_scan, 1, "Tilt min/max/step", self.tilt_min, self.tilt_max, self.tilt_step)
        self._row(tab_scan, 2, "Resolution (w×h)", self.width, self.height, None, ("W","H",""))
        self._entry(tab_scan, 3, "Quality(%)", self.quality)
        self._entry(tab_scan, 4, "Speed", self.speed)
        self._entry(tab_scan, 5, "Accel", self.acc)
        self._entry(tab_scan, 6, "Settle(s)", self.settle)
        Checkbutton(tab_scan, text="Hard stop(정지 펄스)", variable=self.hard_stop)\
            .grid(row=7, column=1, sticky="w", padx=4, pady=2)

        ops = Frame(tab_scan); ops.grid(row=8, column=0, columnspan=4, sticky="w", pady=6)
        Button(ops, text="Start Scan", command=self.start_scan).pack(side="left", padx=4)
        Button(ops, text="Stop Scan",  command=self.stop_scan).pack(side="left", padx=4)
        self.prog = ttk.Progressbar(ops, orient=HORIZONTAL, length=280, mode="determinate"); self.prog.pack(side="left", padx=10)
        self.prog_lbl = Label(ops, text="0 / 0"); self.prog_lbl.pack(side="left")
        self.last_lbl = Label(ops, text="Last: -"); self.last_lbl.pack(side="left", padx=10)
        self.dl_lbl = Label(ops, text="DL 0"); self.dl_lbl.pack(side="left")

        # manual tab
        self.mv_pan=DoubleVar(value=0.0); self.mv_tilt=DoubleVar(value=0.0)
        self.mv_speed=IntVar(value=100);  self.mv_acc=DoubleVar(value=1.0)
        self.led=IntVar(value=0)
        self._slider(tab_manual,0,"Pan",-180,180,self.mv_pan,0.5)
        self._slider(tab_manual,1,"Tilt",-30,90,self.mv_tilt,0.5)
        self._slider(tab_manual,2,"Speed",0,100,self.mv_speed,1)
        self._slider(tab_manual,3,"Accel",0,1,self.mv_acc,0.1)
        Button(tab_manual, text="Center (0,0)", command=self.center).grid(row=4,column=0,sticky="w",pady=4)
        Button(tab_manual, text="Apply Move", command=self.apply_move).grid(row=4,column=1,sticky="e",pady=4)
        self._slider(tab_manual,5,"LED",0,255,self.led,1)
        Button(tab_manual, text="Set LED", command=self.set_led).grid(row=6,column=1,sticky="e",pady=4)

        # preview settings
        self.preview_enable=BooleanVar(value=True)
        self.preview_w=IntVar(value=640); self.preview_h=IntVar(value=360)
        self.preview_fps=IntVar(value=5); self.preview_q=IntVar(value=70)
        Checkbutton(tab_misc, text="Live Preview", variable=self.preview_enable, command=self.toggle_preview)\
            .grid(row=0,column=0,sticky="w",pady=2)
        self._row(tab_misc,1,"Preview w/h/-", self.preview_w, self.preview_h, None, ("W","H",""))
        self._entry(tab_misc,2,"Preview fps", self.preview_fps)
        self._entry(tab_misc,3,"Preview quality", self.preview_q)
        Button(tab_misc, text="Apply Preview Size", command=self.apply_preview_size)\
            .grid(row=4,column=1,sticky="w",pady=4)

        # === NEW: Undistort UI ===
        row = 5
        ttk.Separator(tab_misc, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", pady=(8,6)); row+=1
        Checkbutton(tab_misc, text="Undistort preview (use calib.npz)", variable=self.ud_enable)\
            .grid(row=row, column=0, sticky="w"); row+=1
        Button(tab_misc, text="Load calib.npz", command=self.load_npz)\
            .grid(row=row, column=0, sticky="w", pady=2)
        Checkbutton(tab_misc, text="Also save undistorted copy", variable=self.ud_save_copy)\
            .grid(row=row, column=1, sticky="w", pady=2); row+=1
        Label(tab_misc, text="Alpha/Balance (0~1)").grid(row=row, column=0, sticky="w")
        Scale(tab_misc, from_=0.0, to=1.0, orient=HORIZONTAL, resolution=0.01, length=200,
              variable=self.ud_alpha, command=lambda v: setattr(self, "_ud_src_size", None))\
            .grid(row=row, column=1, sticky="w"); row+=1

        # loop
        self.root.after(60, self._poll)

    # ========= Undistort helpers =========
    def load_npz(self):
        path = filedialog.askopenfilename(filetypes=[("NPZ","*.npz")])
        if not path: return
        cal = np.load(path, allow_pickle=True)
        # 필수 키: model, K, D, img_size  (우리가 만든 npz와 동일)
        self._ud_model = str(cal["model"])
        self._ud_K = cal["K"].astype(np.float32)
        self._ud_D = cal["D"].astype(np.float32)
        self._ud_img_size = tuple(int(x) for x in cal["img_size"])
        self._ud_src_size = None
        self._ud_m1 = self._ud_m2 = None
        self._ud_gm1 = self._ud_gm2 = None
        print(f"[UD] loaded calib: model={self._ud_model}, img_size={self._ud_img_size}, cuda={self._use_cuda}")

    def _scale_K(self, K, sx, sy):
        K2 = K.copy()
        K2[0,0]*=sx; K2[1,1]*=sy
        K2[0,2]*=sx; K2[1,2]*=sy
        K2[2,2]=1.0
        return K2

    def _ensure_ud_maps(self, w:int, h:int):
        if self._ud_K is None or self._ud_D is None or self._ud_model is None:
            return
        if self._ud_src_size == (w,h) and self._ud_m1 is not None:
            return
        Wc,Hc = self._ud_img_size
        sx, sy = w/float(Wc), h/float(Hc)
        K = self._scale_K(self._ud_K, sx, sy)
        D = self._ud_D
        a = float(self.ud_alpha.get())

        if self._ud_model == "pinhole":
            newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w,h), alpha=a, newImgSize=(w,h))
            m1,m2 = cv2.initUndistortRectifyMap(K, D, None, newK, (w,h), cv2.CV_16SC2)
        else:  # fisheye
            R = np.eye(3, dtype=np.float32)
            newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D, (w,h), R, balance=a, new_size=(w,h)
            )
            m1,m2 = cv2.fisheye.initUndistortRectifyMap(K, D, R, newK, (w,h), cv2.CV_16SC2)

        self._ud_m1, self._ud_m2 = m1, m2
        self._ud_src_size = (w,h)

        # CUDA로도 준비(가능하면)
        if self._use_cuda:
            try:
                self._ud_gm1 = cv2.cuda_GpuMat(); self._ud_gm1.upload(self._ud_m1)
                self._ud_gm2 = cv2.cuda_GpuMat(); self._ud_gm2.upload(self._ud_m2)
            except Exception as e:
                print("[UD][CUDA] map upload failed:", e)
                self._ud_gm1 = self._ud_gm2 = None

    # helpers

    def resume_preview(self):
        if self.preview_enable.get():
            self.ctrl.send({
                "cmd":"preview", "enable": True,
                "width":  self.preview_w.get(),
                "height": self.preview_h.get(),
                "fps":    self.preview_fps.get(),
                "quality":self.preview_q.get(),
            })

    def _row(self,parent,r,label,v1,v2,v3=None,caps=("min","max","step")):
        Label(parent,text=label).grid(row=r,column=0,sticky="w",padx=4,pady=2)
        ttk.Entry(parent,width=8,textvariable=v1).grid(row=r,column=1,sticky="w",padx=4)
        ttk.Entry(parent,width=8,textvariable=v2).grid(row=r,column=2,sticky="w",padx=4)
        if v3 is not None:
            ttk.Entry(parent,width=8,textvariable=v3).grid(row=r,column=3,sticky="w",padx=4)
    def _entry(self,parent,r,label,var):
        Label(parent,text=label).grid(row=r,column=0,sticky="w",padx=4,pady=2)
        ttk.Entry(parent,width=8,textvariable=var).grid(row=r,column=1,sticky="w",padx=4)
    def _slider(self,parent,r,label,a,b,var,res):
        Label(parent,text=label).grid(row=r,column=0,sticky="w",padx=4,pady=2)
        Scale(parent,from_=a,to=b,orient=HORIZONTAL,resolution=res,length=360,variable=var)\
            .grid(row=r,column=1,padx=6)

    def choose_outdir(self):
        d = filedialog.askdirectory()
        if d:
            global DEFAULT_OUT_DIR
            DEFAULT_OUT_DIR = pathlib.Path(d)

    # actions
    def start_scan(self):
        if self.preview_enable.get():
            self.ctrl.send({"cmd":"preview","enable": False})
        self.ctrl.send({
            "cmd":"scan_run",
            "pan_min":self.pan_min.get(),"pan_max":self.pan_max.get(),"pan_step":self.pan_step.get(),
            "tilt_min":self.tilt_min.get(),"tilt_max":self.tilt_max.get(),"tilt_step":self.tilt_step.get(),
            "speed":self.speed.get(),"acc":float(self.acc.get()),"settle":float(self.settle.get()),
            "width":self.width.get(),"height":self.height.get(),"quality":self.quality.get(),
            "session":datetime.now().strftime("scan_%Y%m%d_%H%M%S"),
            "hard_stop":self.hard_stop.get()
        })
    def stop_scan(self): self.ctrl.send({"cmd":"scan_stop"})
    def center(self): self.ctrl.send({"cmd":"move","pan":0.0,"tilt":0.0,"speed":self.speed.get(),"acc":float(self.acc.get())})
    def apply_move(self): self.ctrl.send({"cmd":"move","pan":float(self.mv_pan.get()),"tilt":float(self.mv_tilt.get()),
                                          "speed":self.mv_speed.get(),"acc":float(self.mv_acc.get())})
    def set_led(self): self.ctrl.send({"cmd":"led","value":int(self.led.get())})

    def toggle_preview(self):
        if self.preview_enable.get():
            self.ctrl.send({"cmd":"preview","enable": True, "width": self.preview_w.get(), "height": self.preview_h.get(),
                            "fps": self.preview_fps.get(), "quality": self.preview_q.get()})
        else:
            self.ctrl.send({"cmd":"preview","enable": False})

    def apply_preview_size(self):
        w = max(160, min(1280, self.preview_w.get()))
        h = max(120, min( 720, self.preview_h.get()))
        self.PREV_W, self.PREV_H = w, h
        self.preview_box.config(width=w, height=h)

    # NEW: one-shot capture
    def snap_one(self):
        """프리뷰 잠깐 끄고 고해상도 1장 요청 → 수신 후 프리뷰 재개"""
        self._resume_preview_after_snap = False
        if self.preview_enable.get():
            self.ctrl.send({"cmd":"preview","enable": False})
            self._resume_preview_after_snap = True
        fname = datetime.now().strftime("snap_%Y%m%d_%H%M%S.jpg")
        self.ctrl.send({
            "cmd":"snap",
            "width":  self.width.get(),
            "height": self.height.get(),
            "quality":self.quality.get(),
            "save":   fname,
            "hard_stop": self.hard_stop.get()
        })

    # event loop
    def _poll(self):
        try:
            while True:
                tag, payload = ui_q.get_nowait()
                if tag == "evt":
                    evt = payload; et = evt.get("event")
                    if et == "hello":
                        if self.preview_enable.get() and evt.get("agent_state")=="connected":
                            self.toggle_preview()
                    elif et == "agent":
                        if evt.get("state")=="connected" and self.preview_enable.get():
                            self.toggle_preview()
                    elif et == "start":
                        total = int(evt.get("total",0))
                        self.prog.configure(maximum=max(1,total), value=0)
                        self.prog_lbl.config(text=f"0 / {total}"); self.dl_lbl.config(text="DL 0"); self.last_lbl.config(text="Last: -")
                    elif et == "progress":
                        done=int(evt.get("done",0)); total=int(evt.get("total",0))
                        self.prog.configure(value=done); self.prog_lbl.config(text=f"{done} / {total}")
                        name = evt.get("name",""); 
                        if name: self.last_lbl.config(text=f"Last: {name}")
                    elif et == "done":
                        if self.preview_enable.get():
                            self.toggle_preview()  # resume preview for scan
                elif tag == "preview":
                    self._set_preview(payload)
                elif tag == "saved":
                    name, data = payload
                    self.dl_lbl.config(text=f"DL {int(self.dl_lbl.cget('text').split()[-1])+1}")
                    self.last_lbl.config(text=f"Last: {name}")
                    self._set_preview(data)

                    # === NEW: 저장 시 보정본도 추가 저장 (옵션) ===
                    if self.ud_save_copy.get() and self._ud_K is not None:
                        try:
                            arr = np.frombuffer(data, np.uint8)
                            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if bgr is not None:
                                h,w = bgr.shape[:2]
                                self._ensure_ud_maps(w,h)
                                ubgr = cv2.remap(bgr, self._ud_m1, self._ud_m2, cv2.INTER_LINEAR)
                                stem, dot, ext = name.partition(".")
                                out = DEFAULT_OUT_DIR / f"{stem}_ud.{ext or 'jpg'}"
                                cv2.imwrite(str(out), ubgr)
                        except Exception as e:
                            print("[save_ud] err:", e)

                    # snap 이후 자동 프리뷰 재개
                    if self._resume_preview_after_snap:
                        self._resume_preview_after_snap = False
                        self.resume_preview()
                elif tag == "toast":
                    print(payload)
        except queue.Empty:
            pass
        self.root.after(60, self._poll)

    def _set_preview(self, img_bytes: bytes):
        """수신된 JPEG을 표시. (옵션) npz 기반 undistort 적용"""
        try:
            arr = np.frombuffer(img_bytes, np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None: return

            if self.ud_enable.get() and self._ud_K is not None:
                h,w = bgr.shape[:2]
                self._ensure_ud_maps(w,h)

                if self._use_cuda and self._ud_gm1 is not None and self._ud_gm2 is not None:
                    # CUDA 경로 (OpenCV CUDA가 있을 때만)
                    try:
                        if self._gpu_tmp is None or self._gpu_tmp.size() != (h, w):
                            self._gpu_tmp = cv2.cuda_GpuMat(h, w, bgr.dtype.num)  # allocate once
                        gsrc = cv2.cuda_GpuMat(); gsrc.upload(bgr)
                        gout = cv2.cuda.remap(gsrc, self._ud_gm1, self._ud_gm2,
                                              interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        bgr = gout.download()
                    except Exception as e:
                        # 오류시 CPU 경로로 폴백
                        print("[preview][CUDA] remap failed, fallback CPU:", e)
                        bgr = cv2.remap(bgr, self._ud_m1, self._ud_m2, cv2.INTER_LINEAR)
                else:
                    # CPU 경로
                    bgr = cv2.remap(bgr, self._ud_m1, self._ud_m2, cv2.INTER_LINEAR)

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(rgb)
            im.thumbnail((self.PREV_W, self.PREV_H))
            self.tkimg = ImageTk.PhotoImage(im)
            self.preview.configure(image=self.tkimg)
        except Exception as e:
            print("[preview] err:", e)

def main():
    root = Tk()
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
