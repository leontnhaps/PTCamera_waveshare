#!/usr/bin/env python3
# pc_gui.py — GUI client connecting to pc_server.py (not to Pi agent)

import json, socket, struct, threading, queue, pathlib, io
from datetime import datetime
from tkinter import Tk, Label, Button, Scale, HORIZONTAL, IntVar, DoubleVar, Frame, Checkbutton, BooleanVar, filedialog, StringVar
from tkinter import ttk
from PIL import Image, ImageTk
import tkinter as tk  # ← 추가
import os, re, csv   # ← 추가

import numpy as np
import cv2

# ==== [NEW] Optional PyTorch (for CUDA remap acceleration) ====
try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    F = None
    _TORCH_AVAILABLE = False
# =============================================================

# ==== [NEW] Optional Ultralytics YOLO (GPU inference if available) ====
try:
    from ultralytics import YOLO
    _YOLO_OK = True
except Exception:
    YOLO = None
    _YOLO_OK = False
# =====================================================================

SERVER_HOST = "127.0.0.1"
GUI_CTRL_PORT = 7600
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

class ScrollFrame(Frame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.body = Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.body, anchor="nw")

        # 내용 바뀌면 스크롤영역 갱신
        self.body.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        # 부모 크기 바뀌면 내부 프레임 폭 맞춤
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfigure(self._win, width=e.width)
        )
        # 마우스 휠 스크롤
        self.canvas.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self._on_wheel))
        self.canvas.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

    def _on_wheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

# ---- GUI ----
class App:
    def __init__(self, root: Tk):
        self.root = root
        root.title("Pan-Tilt Socket GUI (Client)")
        root.geometry("980x820")
        root.minsize(980, 820)  # 창 최소 크기 고정

        # connections
        self.ctrl = GuiCtrlClient(SERVER_HOST, GUI_CTRL_PORT); self.ctrl.start()
        self.img  = GuiImgClient (SERVER_HOST, GUI_IMG_PORT, DEFAULT_OUT_DIR); self.img.start()

        # state
        self.tkimg=None
        self._resume_preview_after_snap = False

        # undistort state
        self.ud_enable    = BooleanVar(value=False)
        self.ud_save_copy = BooleanVar(value=False)
        self.ud_alpha     = DoubleVar(value=0.0)

        self._ud_model = None
        self._ud_K = self._ud_D = None
        self._ud_img_size = None
        self._ud_src_size = None
        self._ud_m1 = self._ud_m2 = None

        # cv2 CUDA 가능 여부
        self._use_cv2_cuda = False
        try:
            self._use_cv2_cuda = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
        except Exception:
            self._use_cv2_cuda = False
        self._ud_gm1 = self._ud_gm2 = None

        # ==== Torch 가속 관련 멤버 ====
        self._torch_available = _TORCH_AVAILABLE
        self._torch_cuda = bool(_TORCH_AVAILABLE and torch.cuda.is_available())
        self._torch_device = torch.device("cuda") if self._torch_cuda else torch.device("cpu") if _TORCH_AVAILABLE else None
        # 미리보기/저장 용도는 FP16로 충분. 안전하게 FP32로 시작하고, 성능 더 뽑고 싶으면 True.
        self._torch_use_fp16 = False
        self._torch_dtype = (torch.float16 if (self._torch_cuda and self._torch_use_fp16) else torch.float32) if _TORCH_AVAILABLE else None

        self._ud_torch_grid = None      # 1xHxWx2
        self._ud_torch_grid_wh = None   # (w,h)
        # ===================================

        # ==== YOLO overlay state ====
        self.yolo_enable = BooleanVar(value=False)
        self.yolo_conf   = DoubleVar(value=0.25)
        self.yolo_iou    = DoubleVar(value=0.55)
        self.yolo_imgsz  = IntVar(value=832)     # 32배수 권장 (640/704/768/832/896/960 ...)
        self.yolo_stride = IntVar(value=2)       # N프레임마다만 추론
        self.yolo_wpath  = StringVar(value="")   # best.pt 경로

        # [NEW] 가시성 옵션
        self.yolo_box_thick = IntVar(value=4)      # ← 박스 테두리 두께
        self.yolo_text_scale = DoubleVar(value=0.7) # ← 텍스트 스케일
        self.yolo_text_thick = IntVar(value=2)      # ← 텍스트 두께

        self._yolo_model   = None
        self._yolo_last    = None   # (boxes, confs, clses) 캐시
        self._yolo_idx     = 0
        self._yolo_device  = (0 if (torch is not None and torch.cuda.is_available()) else "cpu")

        print(f"[INFO] cv2.cuda={self._use_cv2_cuda}, torch_cuda={self._torch_cuda}, yolo_device={self._yolo_device}")

        # top bar
        top = Frame(root); top.pack(fill="x", padx=10, pady=6)
        Button(top, text="한장 찍기 (Snap)", command=self.snap_one).pack(side="left", padx=(0,8))
        Button(top, text="출력 폴더", command=self.choose_outdir).pack(side="right")

        # ---------- 프리뷰: 고정 박스 + Label(place) 절대 크기 ----------
        center = Frame(root); center.pack(fill="x", padx=10)
        self.PREV_W, self.PREV_H = 800, 450
        self.preview_box = Frame(center, width=self.PREV_W, height=self.PREV_H,
                                 bg="#111", highlightthickness=1, highlightbackground="#333")
        self.preview_box.pack()
        self.preview_box.pack_propagate(False)  # 자식 크기로 커지지 않게

        self.preview_label = Label(self.preview_box, bg="#111")
        self.preview_label.place(x=0, y=0, width=self.PREV_W, height=self.PREV_H)
        # -------------------------------------------------------------------

        # bottom tabs
        nb = ttk.Notebook(root); nb.pack(fill="x", padx=10, pady=(6,10))
        tab_scan   = Frame(nb); nb.add(tab_scan, text="Scan")
        tab_manual = Frame(nb); nb.add(tab_manual, text="Manual / LED")
        tab_misc = Frame(nb); nb.add(tab_misc, text="Preview & Settings")
        tab_point  = Frame(nb); nb.add(tab_point, text="Pointing")
        
        
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

        # ---- Pointing tab (CSV → 가중평균 타깃 → Move) ----
        self.point_csv_path = StringVar(value="")
        self.point_conf_min = DoubleVar(value=0.50)  # CSV 필터용
        self.point_min_samples = IntVar(value=2)     # 각 선형피팅 최소 샘플 수
        self.point_pan_target  = DoubleVar(value=0.0)
        self.point_tilt_target = DoubleVar(value=0.0)
        self.point_speed  = IntVar(value=self.speed.get())  # Scan 속도 기본 재사용
        self.point_acc    = DoubleVar(value=self.acc.get())

        Label(tab_point, text="CSV 경로").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(tab_point, width=52, textvariable=self.point_csv_path)\
            .grid(row=0, column=1, columnspan=2, sticky="we", padx=4)
        Button(tab_point, text="CSV 열기", command=self.pointing_choose_csv)\
            .grid(row=0, column=3, sticky="e", padx=4)

        Label(tab_point, text="conf≥").grid(row=1, column=0, sticky="e")
        ttk.Entry(tab_point, width=8, textvariable=self.point_conf_min).grid(row=1, column=1, sticky="w")
        Label(tab_point, text="min samples/fit").grid(row=1, column=2, sticky="e")
        ttk.Entry(tab_point, width=8, textvariable=self.point_min_samples).grid(row=1, column=3, sticky="w")

        Button(tab_point, text="가중평균 계산", command=self.pointing_compute)\
            .grid(row=2, column=0, sticky="w", padx=4, pady=6)

        Label(tab_point, text="pan target (deg)").grid(row=3, column=0, sticky="e")
        ttk.Entry(tab_point, width=10, textvariable=self.point_pan_target, state="readonly")\
            .grid(row=3, column=1, sticky="w")
        Label(tab_point, text="tilt target (deg)").grid(row=3, column=2, sticky="e")
        ttk.Entry(tab_point, width=10, textvariable=self.point_tilt_target, state="readonly")\
            .grid(row=3, column=3, sticky="w")

        Label(tab_point, text="Speed").grid(row=4, column=0, sticky="e")
        ttk.Entry(tab_point, width=8, textvariable=self.point_speed).grid(row=4, column=1, sticky="w")
        Label(tab_point, text="Accel").grid(row=4, column=2, sticky="e")
        ttk.Entry(tab_point, width=8, textvariable=self.point_acc).grid(row=4, column=3, sticky="w")

        Button(tab_point, text="Move to Target", command=self.pointing_move)\
            .grid(row=5, column=0, columnspan=4, pady=8)

        for c in range(4):
            tab_point.grid_columnconfigure(c, weight=1)
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
        misc_sf = ScrollFrame(tab_misc)
        misc_sf.pack(fill="both", expand=True)
        misc = misc_sf.body  # ← 앞으로 이걸 parent로 써요

        self.preview_enable=BooleanVar(value=True)
        self.preview_w=IntVar(value=640); self.preview_h=IntVar(value=360)
        self.preview_fps=IntVar(value=5); self.preview_q=IntVar(value=70)

        Checkbutton(misc, text="Live Preview", variable=self.preview_enable, command=self.toggle_preview)\
            .grid(row=0,column=0,sticky="w",pady=2)
        self._row(misc,1,"Preview w/h/-", self.preview_w, self.preview_h, None, ("W","H",""))
        self._entry(misc,2,"Preview fps", self.preview_fps)
        self._entry(misc,3,"Preview quality", self.preview_q)
        Button(misc, text="Apply Preview Size", command=self.apply_preview_size)\
            .grid(row=4,column=1,sticky="w",pady=4)

        row = 5
        ttk.Separator(misc, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", pady=(8,6)); row+=1
        Checkbutton(misc, text="Undistort preview (use calib.npz)", variable=self.ud_enable)\
            .grid(row=row, column=0, sticky="w"); row+=1
        Button(misc, text="Load calib.npz", command=self.load_npz)\
            .grid(row=row, column=0, sticky="w", pady=2)
        Checkbutton(misc, text="Also save undistorted copy", variable=self.ud_save_copy)\
            .grid(row=row, column=1, sticky="w", pady=2); row+=1
        Label(misc, text="Alpha/Balance (0~1)").grid(row=row, column=0, sticky="w")
        Scale(misc, from_=0.0, to=1.0, orient=HORIZONTAL, resolution=0.01, length=200,
            variable=self.ud_alpha, command=lambda v: setattr(self, "_ud_src_size", None))\
            .grid(row=row, column=1, sticky="w"); row+=1

        # ==== YOLO UI ====  ← 'tab_misc'가 아니라 'misc'를 parent로!
        ttk.Separator(misc, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", pady=(8,6)); row+=1

        Checkbutton(
            misc, text="YOLO overlay (preview에 결과 그리기)",
            variable=self.yolo_enable, command=self._on_toggle_yolo
        ).grid(row=row, column=0, sticky="w"); row+=1

        Button(misc, text="Load YOLO weights (.pt)", command=self.load_yolo_weights)\
            .grid(row=row, column=0, sticky="w", pady=2)

        self.lbl_yolo_w = Label(misc, textvariable=self.yolo_wpath, anchor="w")
        self.lbl_yolo_w.grid(row=row, column=1, columnspan=3, sticky="we"); row+=1

        Label(misc, text="conf").grid(row=row, column=0, sticky="w")
        ttk.Entry(misc, width=8, textvariable=self.yolo_conf).grid(row=row, column=1, sticky="w", padx=4)
        Label(misc, text="iou").grid(row=row, column=2, sticky="w")
        ttk.Entry(misc, width=8, textvariable=self.yolo_iou).grid(row=row, column=3, sticky="w", padx=4); row+=1

        Label(misc, text="imgsz").grid(row=row, column=0, sticky="w")
        ttk.Entry(misc, width=8, textvariable=self.yolo_imgsz).grid(row=row, column=1, sticky="w", padx=4)
        Label(misc, text="stride(N프레임)").grid(row=row, column=2, sticky="w")
        ttk.Entry(misc, width=8, textvariable=self.yolo_stride).grid(row=row, column=3, sticky="w", padx=4); row+=1
        # YOLO UI 아래쪽 어딘가에 추가
        self.yolo_show_centroid = BooleanVar(value=True)
        self.yolo_show_center_cross = BooleanVar(value=True)

        Checkbutton(misc, text="Show centroid dot (avg of detections)", 
                    variable=self.yolo_show_centroid).grid(row=row, column=0, sticky="w"); row+=1
        Checkbutton(misc, text="Show image center crosshair", 
                    variable=self.yolo_show_center_cross).grid(row=row, column=0, sticky="w"); row+=1

        # [NEW] 가시성 옵션 UI
        Label(misc, text="box thickness").grid(row=row, column=0, sticky="w")
        ttk.Entry(misc, width=8, textvariable=self.yolo_box_thick).grid(row=row, column=1, sticky="w", padx=4)
        Label(misc, text="text scale").grid(row=row, column=2, sticky="w")
        ttk.Entry(misc, width=8, textvariable=self.yolo_text_scale).grid(row=row, column=3, sticky="w", padx=4); row+=1
        Label(misc, text="text thickness").grid(row=row, column=0, sticky="w")
        ttk.Entry(misc, width=8, textvariable=self.yolo_text_thick).grid(row=row, column=1, sticky="w", padx=4); row+=1

        # (있으면) 이 줄도 추가해두면 너비 늘어날 때 경로 라벨이 자연스럽게 늘어남
        for c in range(4):
            misc.grid_columnconfigure(c, weight=1)

        # ==================

        self.root.after(60, self._poll)
                # ===== [SCAN CSV 로깅 상태] =====
        self._scan_csv_path = None
        self._scan_csv_file = None
        self._scan_csv_writer = None

        # 파일명에서 pan/tilt 파싱 (예: img_t+00_p+001_....jpg)
        self._fname_re = re.compile(r"img_t(?P<tilt>[+\-]\d{2,3})_p(?P<pan>[+\-]\d{2,3})_.*\.(jpg|jpeg|png)$", re.IGNORECASE)

        # 스캔 중 YOLO 강제 적용(overlay와 무관)
        self._scan_yolo_conf = 0.50   # 스캔용 conf (원하면 UI 변수와 같게 써도 OK)
        self._scan_yolo_imgsz = 832   # 스캔용 imgsz

        # === Pointing 좌표 로깅 상태 ===
        self._pointing_log_fp = None
        self._pointing_log_writer = None
        self._pointing_logging = False

        # (선택) 현재 명령 각도 기억
        self._curr_pan = 0.0
        self._curr_tilt = 0.0
        
        self._fits_h = {}
        self._fits_v = {}
        # Pointing 탭에 추가 UI
        # centering state
        self._centering_ok_frames = 0
        self._centering_last_ms = 0

        ttk.Separator(tab_point, orient="horizontal").grid(row=15, column=0, columnspan=4, sticky="ew", pady=(8,6))

        self.centering_enable   = BooleanVar(value=False)
        self.centering_px_tol   = IntVar(value=5)      # 중앙 판정 오차(px)
        self.centering_min_frames = IntVar(value=4)    # 연속 N프레임 만족 시 종료
        self.centering_max_step = DoubleVar(value=1.0) # 한번에 움직일 최대 각도(°)
        self.centering_cooldown = IntVar(value=250)    # 명령 간 최소 간격(ms)

        Checkbutton(tab_point, text="Centering mode (live refine)", variable=self.centering_enable)\
            .grid(row=16, column=0, sticky="w")

        Label(tab_point, text="px tol").grid(row=16, column=1, sticky="e")
        ttk.Entry(tab_point, width=6, textvariable=self.centering_px_tol).grid(row=16, column=2, sticky="w")

        Label(tab_point, text="max step(°)").grid(row=17, column=1, sticky="e")
        ttk.Entry(tab_point, width=6, textvariable=self.centering_max_step).grid(row=17, column=2, sticky="w")

        Label(tab_point, text="cooldown(ms)").grid(row=17, column=0, sticky="e")
        ttk.Entry(tab_point, width=8, textvariable=self.centering_cooldown).grid(row=17, column=1, sticky="w")

        Label(tab_point, text="stable frames").grid(row=17, column=3, sticky="e")
        ttk.Entry(tab_point, width=6, textvariable=self.centering_min_frames).grid(row=17, column=3, sticky="w")

        for c in range(4):
            tab_point.grid_columnconfigure(c, weight=1)

    def _ensure_yolo_model_for_scan(self) -> bool:
        """스캔 중 CSV 기록용 YOLO 모델 준비(overlay 사용 여부와 무관)."""
        # 이미 overlay에서 로드돼 있으면 재사용
        if self._yolo_model is not None:
            return True
        # overlay가 꺼져 있어도 경로 지정돼 있으면 로드
        wpath = self.yolo_wpath.get().strip()
        if not _YOLO_OK or not wpath:
            ui_q.put(("toast", "YOLO 가중치(.pt)을 로드하지 않아 CSV에 감지값을 기록할 수 없습니다."))
            return False
        try:
            self._yolo_model = YOLO(wpath)
            # 워밍업
            dummy = np.zeros((self._scan_yolo_imgsz, self._scan_yolo_imgsz, 3), dtype=np.uint8)
            self._yolo_model.predict(dummy, imgsz=self._scan_yolo_imgsz,
                                     conf=self._scan_yolo_conf,
                                     iou=float(self.yolo_iou.get()),
                                     device=self._yolo_device, verbose=False)
            return True
        except Exception as e:
            self._yolo_model = None
            ui_q.put(("toast", f"[SCAN] YOLO 로드 실패: {e}"))
            return False

    
    
    # ========= Undistort helpers =========
    def load_npz(self):
        path = filedialog.askopenfilename(filetypes=[("NPZ","*.npz")])
        if not path: return
        cal = np.load(path, allow_pickle=True)
        self._ud_model = str(cal["model"])
        self._ud_K = cal["K"].astype(np.float32)
        self._ud_D = cal["D"].astype(np.float32)
        self._ud_img_size = tuple(int(x) for x in cal["img_size"])
        self._ud_src_size = None
        self._ud_m1 = self._ud_m2 = None
        self._ud_gm1 = self._ud_gm2 = None
        self._ud_torch_grid = None
        self._ud_torch_grid_wh = None
        print(f"[UD] loaded calib: model={self._ud_model}, img_size={self._ud_img_size}, cv2.cuda={self._use_cv2_cuda}, torch_cuda={self._torch_cuda}")

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
        else:
            R = np.eye(3, dtype=np.float32)
            newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D, (w,h), R, balance=a, new_size=(w,h)
            )
            m1,m2 = cv2.fisheye.initUndistortRectifyMap(K, D, R, newK, (w,h), cv2.CV_16SC2)

        self._ud_m1, self._ud_m2 = m1, m2
        self._ud_src_size = (w,h)

        # cv2.cuda 맵 업로드 (가능하면)
        if self._use_cv2_cuda:
            try:
                self._ud_gm1 = cv2.cuda_GpuMat(); self._ud_gm1.upload(self._ud_m1)
                self._ud_gm2 = cv2.cuda_GpuMat(); self._ud_gm2.upload(self._ud_m2)
            except Exception as e:
                print("[UD][cv2.cuda] map upload failed:", e)
                self._ud_gm1 = self._ud_gm2 = None

        # [NEW] Torch grid 초기화 무효화 (재생성 필요)
        self._ud_torch_grid = None
        self._ud_torch_grid_wh = None

    # [NEW] OpenCV 맵 -> Torch grid(-1~1 정규화)로 변환/캐시
    def _ensure_torch_grid(self, w:int, h:int):
        if not (self._torch_cuda and self._ud_m1 is not None):
            return
        if self._ud_torch_grid is not None and self._ud_torch_grid_wh == (w,h):
            return

        mx, my = cv2.convertMaps(self._ud_m1, self._ud_m2, cv2.CV_32F)  # HxW float32
        H, W = mx.shape
        gx = (mx / max(W-1,1)) * 2.0 - 1.0
        gy = (my / max(H-1,1)) * 2.0 - 1.0
        grid = np.stack([gx, gy], axis=-1)  # HxWx2

        dtype = self._torch_dtype
        dev   = self._torch_device
        self._ud_torch_grid = torch.from_numpy(grid).unsqueeze(0).to(device=dev, dtype=dtype)
        self._ud_torch_grid_wh = (w,h)

    # [NEW] 단일 프레임 왜곡보정 (우선순위: Torch→cv2.cuda→CPU)
    def _undistort_bgr(self, bgr: np.ndarray) -> np.ndarray:
        h,w = bgr.shape[:2]
        self._ensure_ud_maps(w,h)

        # Torch CUDA 경로
        if self._torch_cuda and self._ud_m1 is not None:
            try:
                self._ensure_torch_grid(w,h)
                if self._ud_torch_grid is not None:
                    # np -> torch (CHW, [0,1] float)
                    t_cpu = torch.from_numpy(bgr).permute(2,0,1).contiguous()
                    # pinned memory 전송(속도 미세 향상)
                    try:
                        t_cpu = t_cpu.pin_memory()
                    except Exception:
                        pass
                    t = t_cpu.to(self._torch_device, dtype=self._torch_dtype, non_blocking=True).unsqueeze(0) / 255.0
                    out = F.grid_sample(t, self._ud_torch_grid, mode="bilinear", align_corners=True)
                    bgr = (out.squeeze(0).permute(1,2,0) * 255.0).clamp(0,255).byte().cpu().numpy()
                    return np.ascontiguousarray(bgr)
            except Exception as e:
                print("[UD][torch] remap failed → fallback:", e)

        # cv2 CUDA 경로
        if self._use_cv2_cuda and self._ud_gm1 is not None and self._ud_gm2 is not None:
            try:
                gsrc = cv2.cuda_GpuMat(); gsrc.upload(bgr)
                gout = cv2.cuda.remap(gsrc, self._ud_gm1, self._ud_gm2,
                                      interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                return gout.download()
            except Exception as e:
                print("[UD][cv2.cuda] remap failed → CPU:", e)

        # CPU 경로
        return cv2.remap(bgr, self._ud_m1, self._ud_m2, cv2.INTER_LINEAR)

    # ===== YOLO helpers =====
    def load_yolo_weights(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch Weights","*.pt")])
        if not path:
            return
        self.yolo_wpath.set(path)
        self._yolo_model = None  # 다음 사용 시 재로드

    def _on_toggle_yolo(self):
        if self.yolo_enable.get():
            ok = self._ensure_yolo_model()
            if not ok:
                self.yolo_enable.set(False)

    def _ensure_yolo_model(self) -> bool:
        if not _YOLO_OK:
            ui_q.put(("toast", "Ultralytics가 설치되어 있지 않습니다: pip install ultralytics"))
            return False
        wpath = self.yolo_wpath.get().strip()
        if not wpath:
            ui_q.put(("toast", "YOLO 가중치(.pt)를 먼저 로드하세요."))
            return False
        if self._yolo_model is None:
            try:
                self._yolo_model = YOLO(wpath)
                # 워밍업 (초기 지연 방지)
                dummy = np.zeros((int(self.yolo_imgsz.get()), int(self.yolo_imgsz.get()), 3), dtype=np.uint8)
                self._yolo_model.predict(dummy, imgsz=int(self.yolo_imgsz.get()),
                                         conf=float(self.yolo_conf.get()),
                                         iou=float(self.yolo_iou.get()),
                                         device=self._yolo_device, verbose=False)
                ui_q.put(("toast", f"YOLO ready: {wpath} (device={self._yolo_device})"))
            except Exception as e:
                self._yolo_model = None
                ui_q.put(("toast", f"YOLO 로드 실패: {e}"))
                return False
        return True

    def _run_yolo_and_draw(self, bgr: np.ndarray) -> np.ndarray:
        """N프레임마다 추론, 매 프레임 박스 + 평균점(centroid) 그리기."""
        if not self.yolo_enable.get():
            self._yolo_last = None
            return bgr
        bgr = np.ascontiguousarray(bgr, dtype=np.uint8)
        if not self._ensure_yolo_model():
            return bgr

        try:
            N = max(1, int(self.yolo_stride.get()))
            run_infer = (self._yolo_idx % N) == 0

            if run_infer:
                res = self._yolo_model.predict(
                    bgr,
                    imgsz=int(self.yolo_imgsz.get()),
                    conf=float(self.yolo_conf.get()),
                    iou=float(self.yolo_iou.get()),
                    device=self._yolo_device,
                    verbose=False
                )[0]
                if len(res.boxes) > 0:
                    boxes = res.boxes.xyxy.detach().cpu().numpy().astype(int)
                    confs = res.boxes.conf.detach().cpu().numpy()
                    clses = res.boxes.cls.detach().cpu().numpy().astype(int)
                    self._yolo_last = (boxes, confs, clses)
                else:
                    self._yolo_last = (np.empty((0,4), int), np.array([]), np.array([]))

            # draw from cache
            if self._yolo_last is not None:
                boxes, confs, clses = self._yolo_last
                th = max(1, int(self.yolo_box_thick.get()))
                ts = max(0.3, float(self.yolo_text_scale.get()))
                tth = max(1, int(self.yolo_text_thick.get()))

                H, W = bgr.shape[:2]

                # 1) 박스들 렌더
                for (x1,y1,x2,y2), c, k in zip(boxes, confs, clses):
                    cv2.rectangle(bgr, (x1,y1), (x2,y2), (0,255,0), th, lineType=cv2.LINE_AA)
                    label = f"{c:.2f}"
                    org = (x1, max(15, y1-6))
                    cv2.putText(bgr, label, org, cv2.FONT_HERSHEY_SIMPLEX, ts, (0,0,0), tth+2, cv2.LINE_AA)
                    cv2.putText(bgr, label, org, cv2.FONT_HERSHEY_SIMPLEX, ts, (0,255,0), tth,   cv2.LINE_AA)

                # 2) (옵션) 화면 중앙 십자
                if getattr(self, "yolo_show_center_cross", True) and self.yolo_show_center_cross.get():
                    cx0, cy0 = int(W/2), int(H/2)
                    cv2.drawMarker(bgr, (cx0, cy0), (255,255,255), markerType=cv2.MARKER_CROSS, 
                                markerSize=14, thickness=1, line_type=cv2.LINE_AA)

                # 3) (옵션) 검출들의 중심점 평균(centroid) 그리기
                if getattr(self, "yolo_show_centroid", True) and self.yolo_show_centroid.get():
                    # conf 필터는 이미 predict에 들어가 있으므로 전체 사용
                    if boxes.shape[0] > 0:
                        centers = []
                        for (x1,y1,x2,y2) in boxes:
                            cx = 0.5*(x1+x2); cy = 0.5*(y1+y2)
                            centers.append((cx, cy))
                        # 평균
                        m_cx = float(np.mean([c[0] for c in centers]))
                        m_cy = float(np.mean([c[1] for c in centers]))

                        # 점/원 + 텍스트
                        cv2.circle(bgr, (int(round(m_cx)), int(round(m_cy))), 5, (0,200,255), -1, lineType=cv2.LINE_AA)
                        err_x = (W/2.0 - m_cx); err_y = (H/2.0 - m_cy)
                        self._centering_on_centroid(m_cx, m_cy, W, H)
                        if self._pointing_logging and (self._pointing_log_writer is not None):
                            from datetime import datetime
                            try:
                                self._pointing_log_writer.writerow([
                                    datetime.now().isoformat(timespec="milliseconds"),
                                    f"{self._curr_pan:.3f}", f"{self._curr_tilt:.3f}",
                                    f"{m_cx:.3f}", f"{m_cy:.3f}",
                                    f"{err_x:.3f}", f"{err_y:.3f}",
                                    int(W), int(H),
                                    int(boxes.shape[0])
                                ])
                                # 즉시 파일에 내리고 싶으면 다음 줄 주석 해제
                                # self._pointing_log_fp.flush()
                            except Exception as e:
                                print("[Point] write err:", e)
                        txt = f"mean ({m_cx:.1f},{m_cy:.1f})  err ({err_x:+.1f},{err_y:+.1f}) px"
                        # 텍스트 배경
                        cv2.putText(bgr, txt, (10, max(20, H-15)), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.55, (0,0,0), 3, cv2.LINE_AA)
                        cv2.putText(bgr, txt, (10, max(20, H-15)), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.55, (0,255,255), 1, cv2.LINE_AA)
                        

            self._yolo_idx += 1
        except Exception as e:
            print("[YOLO] err:", e)
        return bgr


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
    # 보정 강제: calib.npz가 로드되지 않았으면 스캔 시작 금지
        if self._ud_K is None or self._ud_D is None:
            ui_q.put(("toast", "❌ 스캔은 보정 이미지만 허용합니다. 먼저 'Load calib.npz'를 해주세요."))
            return
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
        # 1) 입력값 정리 (스트림 해상도만)
        w = max(160, min(2592, self.preview_w.get()))
        h = max(120,  min(1944, self.preview_h.get()))
        self.preview_w.set(w); self.preview_h.set(h)

        # 2) 창/프리뷰 박스 크기 절대 변경 금지 !!!

        # 3) 토글과 동일하게 '중지→새 파라미터로 재시작'
        if self.preview_enable.get():
            self.ctrl.send({"cmd": "preview", "enable": False})
            self.root.after(80, lambda: self.ctrl.send({
                "cmd": "preview", "enable": True,
                "width": w, "height": h,
                "fps": self.preview_fps.get(),
                "quality": self.preview_q.get(),
            }))
        else:
            self.ctrl.send({"cmd": "preview", "enable": False,
                            "width": w, "height": h,
                            "fps": self.preview_fps.get(),
                            "quality": self.preview_q.get()})

    # NEW: one-shot capture
    def snap_one(self):
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
                    elif et == "start":
                        total = int(evt.get("total",0))
                        self.prog.configure(maximum=max(1,total), value=0)
                        self.prog_lbl.config(text=f"0 / {total}"); self.dl_lbl.config(text="DL 0"); self.last_lbl.config(text="Last: -")

                        # === CSV 오픈 ===
                        # 서버에서 session을 내려주면 쓰고, 없으면 타임스탬프
                        sess = evt.get("session") or datetime.now().strftime("scan_%Y%m%d_%H%M%S")
                        self._scan_csv_path = DEFAULT_OUT_DIR / f"{sess}_detections.csv"
                        try:
                            self._scan_csv_file = open(self._scan_csv_path, "w", newline="", encoding="utf-8")
                            self._scan_csv_writer = csv.writer(self._scan_csv_file)
                            self._scan_csv_writer.writerow(["file","pan_deg","tilt_deg","cx","cy","w","h","conf","cls","W","H"])
                            print(f"[SCAN] CSV → {self._scan_csv_path}")
                        except Exception as e:
                            self._scan_csv_file = None
                            self._scan_csv_writer = None
                            ui_q.put(("toast", f"CSV 오픈 실패: {e}"))

                    elif et == "start":
                        total = int(evt.get("total",0))
                        self.prog.configure(maximum=max(1,total), value=0)
                        self.prog_lbl.config(text=f"0 / {total}"); self.dl_lbl.config(text="DL 0"); self.last_lbl.config(text="Last: -")
                    elif et == "progress":
                        done=int(evt.get("done",0)); total=int(evt.get("total",0))
                        self.prog.configure(value=done); self.prog_lbl.config(text=f"{done} / {total}")
                        name = evt.get("name","")
                        if name: self.last_lbl.config(text=f"Last: {name}")
                    elif et == "done":
                        # === CSV 닫기 ===
                        if self._scan_csv_file:
                            try:
                                self._scan_csv_file.flush()
                                self._scan_csv_file.close()
                            except Exception:
                                pass
                            finally:
                                self._scan_csv_file = None
                                self._scan_csv_writer = None
                                ui_q.put(("toast", f"CSV 저장 완료: {self._scan_csv_path}"))
                        if self.preview_enable.get():
                            self.toggle_preview()

                elif tag == "preview":
                    self._set_preview(payload)
                elif tag == "saved":
                    name, data = payload
                    self.dl_lbl.config(text=f"DL {int(self.dl_lbl.cget('text').split()[-1])+1}")
                    self.last_lbl.config(text=f"Last: {name}")
                    self._set_preview(data)

                    if self.ud_save_copy.get() and self._ud_K is not None:
                        try:
                            arr = np.frombuffer(data, np.uint8)
                            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if bgr is not None:
                                ubgr = self._undistort_bgr(bgr)
                                stem, dot, ext = name.partition(".")
                                out = DEFAULT_OUT_DIR / f"{stem}_ud.{ext or 'jpg'}"
                                cv2.imwrite(str(out), ubgr)
                        except Exception as e:
                            print("[save_ud] err:", e)
                    # === [핵심] 스캔 중 CSV에 YOLO 결과 기록 ===
                    if self._scan_csv_writer is not None:
                            try:
                                # 파일명에서 pan/tilt 추출
                                m = self._fname_re.search(name)
                                pan_deg = float(m.group("pan")) if m else None
                                tilt_deg = float(m.group("tilt")) if m else None

                                # 원본 디코드
                                arr = np.frombuffer(data, np.uint8)
                                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                                if bgr is None:
                                    raise RuntimeError("cv2.imdecode 실패")

                                # ★★★ 스캔 CSV/YOLO는 '항상 보정' ★★★
                                if self._ud_K is None or self._ud_D is None:
                                    # 안전장치: 혹시 start_scan 체크를 우회했을 때 대비
                                    ui_q.put(("toast", "❌ 보정 파라미터 없음 → 스캔 감지/기록 중단"))
                                    return
                                bgr = self._undistort_bgr(bgr)   # ← 여기서 반드시 언디스토트
                                H, W = bgr.shape[:2]             # ← 보정 이후의 W,H로 교체

                                # YOLO 보장 & 추론
                                if self._ensure_yolo_model_for_scan():
                                    res = self._yolo_model.predict(
                                        bgr,
                                        imgsz=self._scan_yolo_imgsz,
                                        conf=self._scan_yolo_conf,
                                        iou=float(self.yolo_iou.get()),
                                        device=self._yolo_device,
                                        verbose=False
                                    )[0]

                                    if res is not None and res.boxes is not None and len(res.boxes) > 0:
                                        for b in res.boxes:
                                            conf = float(b.conf.cpu().item() or 0.0)
                                            if conf < self._scan_yolo_conf: 
                                                continue
                                            cls = int(b.cls.cpu().item() or -1)
                                            x1,y1,x2,y2 = b.xyxy[0].cpu().numpy().tolist()
                                            cx = 0.5*(x1+x2); cy = 0.5*(y1+y2)
                                            # ★ CSV는 '보정 좌표계'로 기록됨
                                            self._scan_csv_writer.writerow([name, pan_deg, tilt_deg, cx, cy, x2-x1, y2-y1, conf, cls, W, H])
                            except Exception as e:
                                print("[SCAN][CSV] write err:", e)
                    if self._resume_preview_after_snap:
                        self._resume_preview_after_snap = False
                        self.resume_preview()

                elif tag == "toast":
                    print(payload)
        except queue.Empty:
            pass
        self.root.after(60, self._poll)

    # ---------- 고정 박스 안에 '레터박스(contain)'로 그리기 ----------
    def _draw_preview_to_label(self, pil_image: Image.Image):
        W, H = int(self.PREV_W), int(self.PREV_H)
        iw, ih = pil_image.size
        if iw <= 0 or ih <= 0 or W <= 0 or H <= 0:
            return
        scale = min(W / iw, H / ih)
        nw = max(1, int(round(iw * scale)))
        nh = max(1, int(round(ih * scale)))
        img = pil_image.resize((nw, nh), Image.LANCZOS)
        bg = Image.new("RGB", (W, H), (17, 17, 17))
        x = (W - nw) // 2
        y = (H - nh) // 2
        bg.paste(img, (x, y))
        self.tkimg = ImageTk.PhotoImage(bg)
        self.preview_label.configure(image=self.tkimg)
    # -----------------------------------------------------------------------

    def _set_preview(self, img_bytes: bytes):
        try:
            arr = np.frombuffer(img_bytes, np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None: return

            if self.ud_enable.get() and self._ud_K is not None:
                bgr = self._undistort_bgr(bgr)

            # YOLO overlay
            bgr = self._run_yolo_and_draw(bgr)

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(rgb)
            self._draw_preview_to_label(im)  # 레이아웃 불변
        except Exception as e:
            print("[preview] err:", e)
        # ===== Pointing helpers =====
    def pointing_choose_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV","*.csv")])
        if path:
            self.point_csv_path.set(path)

    @staticmethod
    def _linfit_xy(x, y):
        import numpy as np
        x = np.asarray(x, float); y = np.asarray(y, float)
        if len(x) < 2:
            return None
        A = np.vstack([x, np.ones_like(x)]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(a), float(b)

    def pointing_compute(self):
        """
        CSV를 읽어:
          1) tilt별 cx= a*pan + b → pan_center = (W/2 - b)/a
          2) pan별  cy= e*tilt+ f → tilt_center= (H/2 - f)/e
        를 구하고, 각 bin의 샘플 수 N으로 가중평균하여 최종 타깃 pan/tilt 계산.
        """
        path = self.point_csv_path.get().strip()
        if not path:
            ui_q.put(("toast", "CSV를 선택하세요."))
            return

        try:
            import numpy as np, csv
            rows = []
            W_frame = H_frame = None
            conf_min = float(self.point_conf_min.get())
            min_samples = int(self.point_min_samples.get())

            with open(path, newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for d in r:
                    if d.get("conf","")=="":
                        continue
                    conf = float(d["conf"])
                    if conf < conf_min:
                        continue
                    pan  = d.get("pan_deg"); tilt = d.get("tilt_deg")
                    if pan in ("",None) or tilt in ("",None):
                        continue
                    pan = float(pan); tilt = float(tilt)
                    cx = float(d["cx"]); cy = float(d["cy"])
                    W  = int(d["W"]) if d.get("W") else None
                    H  = int(d["H"]) if d.get("H") else None
                    if W_frame is None and W: W_frame = W
                    if H_frame is None and H: H_frame = H
                    rows.append((pan, tilt, cx, cy))

            if not rows:
                ui_q.put(("toast", "CSV에서 조건을 만족하는 행이 없습니다. conf/min_samples 확인."))
                return
            if W_frame is None or H_frame is None:
                ui_q.put(("toast", "CSV에 W/H 정보가 없습니다. (W,H 열 필요)"))
                return

            # --- tilt별 수평 피팅: cx vs pan
            from collections import defaultdict
            # ---- tilt별: cx = a*pan + b → pan_center = (W/2 - b)/a
            by_tilt = defaultdict(list)
            for pan, tilt, cx, cy in rows:
                by_tilt[round(tilt, 3)].append((pan, cx))

            fits_h = {}  # tilt -> dict
            for tkey, arr in by_tilt.items():
                if len(arr) < min_samples: 
                    continue
                arr.sort(key=lambda v: v[0])
                pans = np.array([p for p,_ in arr], float)
                cxs  = np.array([c for _,c in arr], float)
                A = np.vstack([pans, np.ones_like(pans)]).T
                a, b = np.linalg.lstsq(A, cxs, rcond=None)[0]
                # R^2
                yhat = a*pans + b
                ss_res = float(np.sum((cxs - yhat)**2))
                ss_tot = float(np.sum((cxs - np.mean(cxs))**2)) + 1e-9
                R2 = 1.0 - ss_res/ss_tot
                pan_center = (W_frame/2.0 - b)/a if abs(a) > 1e-9 else np.nan
                fits_h[float(tkey)] = {
                    "a": float(a), "b": float(b), "R2": float(R2),
                    "N": int(len(arr)), "pan_center": float(pan_center),
                }

            # ---- pan별: cy = e*tilt + f → tilt_center = (H/2 - f)/e
            by_pan = defaultdict(list)
            for pan, tilt, cx, cy in rows:
                by_pan[round(pan, 3)].append((tilt, cy))

            fits_v = {}  # pan -> dict
            for pkey, arr in by_pan.items():
                if len(arr) < min_samples:
                    continue
                arr.sort(key=lambda v: v[0])
                tilts = np.array([t for t,_ in arr], float)
                cys   = np.array([c for _,c in arr], float)
                A = np.vstack([tilts, np.ones_like(tilts)]).T
                e, f = np.linalg.lstsq(A, cys, rcond=None)[0]
                yhat = e*tilts + f
                ss_res = float(np.sum((cys - yhat)**2))
                ss_tot = float(np.sum((cys - np.mean(cys))**2)) + 1e-9
                R2 = 1.0 - ss_res/ss_tot
                tilt_center = (H_frame/2.0 - f)/e if abs(e) > 1e-9 else np.nan
                fits_v[float(pkey)] = {
                    "e": float(e), "f": float(f), "R2": float(R2),
                    "N": int(len(arr)), "tilt_center": float(tilt_center),
                }

            # ---- 전역 저장 (센터링/보간에서 사용)
            self._fits_h = fits_h
            self._fits_v = fits_v

            # ---- (기존처럼) 가중평균 타깃 계산해서 UI에 표시
            def wavg_center(fits: dict, center_key: str):
                if not fits: return None
                vals = np.array([fits[k][center_key] for k in fits], float)
                w    = np.array([fits[k]["N"]          for k in fits], float)
                return float(np.sum(vals*w)/np.sum(w))

            pan_target  = wavg_center(fits_h, "pan_center")
            tilt_target = wavg_center(fits_v, "tilt_center")
            if pan_target is not None:  self.point_pan_target.set(round(pan_target, 3))
            if tilt_target is not None: self.point_tilt_target.set(round(tilt_target, 3))

            ui_q.put(("toast",
                f"[Pointing] pan={self.point_pan_target.get()}°, "
                f"tilt={self.point_tilt_target.get()}°  "
                f"(fits: H={len(fits_h)}, V={len(fits_v)})"))

        except Exception as e:
            ui_q.put(("toast", f"[Pointing] 계산 실패: {e}"))

    def pointing_move(self):
        try:
            pan_t  = float(self.point_pan_target.get())
            tilt_t = float(self.point_tilt_target.get())
        except Exception:
            ui_q.put(("toast", "먼저 '가중평균 계산'으로 타깃을 구하세요."))
            return
        spd = int(self.point_speed.get()); acc = float(self.point_acc.get())

        # 현재 명령 각도 기억
        self._curr_pan, self._curr_tilt = pan_t, tilt_t

        # 이동
        self.ctrl.send({"cmd":"move","pan":pan_t,"tilt":tilt_t,"speed":spd,"acc":acc})
        ui_q.put(("toast", f"→ Move to (pan={pan_t}°, tilt={tilt_t}°)"))

        # ==== 여기서 좌표 로깅 시작 ====
        try:
            from datetime import datetime
            import csv, os
            log_dir = DEFAULT_OUT_DIR
            os.makedirs(log_dir, exist_ok=True)
            fname = f"point_xy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            path  = log_dir / fname
            # 열려있던 거 있으면 닫기
            if self._pointing_log_fp:
                try: self._pointing_log_fp.close()
                except: pass
            self._pointing_log_fp = open(path, "w", newline="", encoding="utf-8")
            self._pointing_log_writer = csv.writer(self._pointing_log_fp)
            self._pointing_log_writer.writerow(
                ["ts","pan_cmd_deg","tilt_cmd_deg","mean_cx","mean_cy","err_x_px","err_y_px","W","H","n_dets"]
            )
            self._pointing_logging = True
            ui_q.put(("toast", f"[Point] logging → {path} (preview 켜고 YOLO ON 하면 기록)"))
        except Exception as e:
            self._pointing_logging = False
            ui_q.put(("toast", f"[Point] 로그 시작 실패: {e}"))
    def _interp_fit(self, fmap: dict, q: float, key_slope: str, k: int = 2):
        """근처 k개 키로 1/d 가중 평균 보간 (기울기 a 또는 e 추정)."""
        import numpy as np
        if not fmap:
            return np.nan
        ks = np.array(list(fmap.keys()), float)
        vs = np.array([fmap[k][key_slope] for k in fmap], float)
        order = np.argsort(np.abs(ks - q))[:max(1, min(k, len(ks)))]
        sel_k, sel_v = ks[order], vs[order]
        d = np.abs(sel_k - q) + 1e-6
        w = 1.0 / d
        return float(np.sum(sel_v * w) / np.sum(w))

    def _centering_on_centroid(self, m_cx: float, m_cy: float, W: int, H: int):
        """프리뷰에서 평균점 얻을 때마다 호출 → 작은 각도 스텝으로 중앙 수렴."""
        import time, numpy as np
        if not self.centering_enable.get():
            self._centering_ok_frames = 0
            return

        # 중앙 오차(px)
        ex = (W/2.0) - float(m_cx)
        ey = (H/2.0) - float(m_cy)
        tol = int(self.centering_px_tol.get())

        # 안정 프레임 카운트
        if abs(ex) <= tol and abs(ey) <= tol:
            self._centering_ok_frames += 1
        else:
            self._centering_ok_frames = 0

        # 충분히 안정되면 종료 메시지(선택)
        if self._centering_ok_frames >= int(self.centering_min_frames.get()):
            return

        # 쿨다운(명령 과다 방지)
        now_ms = int(time.time() * 1000)
        if now_ms - self._centering_last_ms < int(self.centering_cooldown.get()):
            return

        # px/deg 기울기 추정: a=∂cx/∂pan (tilt근방), e=∂cy/∂tilt (pan근방)
        a = self._interp_fit(getattr(self, "_fits_h", {}), self._curr_tilt, "a", k=2)
        e = self._interp_fit(getattr(self, "_fits_v", {}), self._curr_pan,  "e", k=2)

        # 기울기 없으면 보수적으로 스킵
        if not np.isfinite(a) or abs(a) < 1e-6 or not np.isfinite(e) or abs(e) < 1e-6:
            return

        # 각도 보정량(°)
        dpan  = float(np.clip(ex / a, -float(self.centering_max_step.get()), float(self.centering_max_step.get())))
        dtilt = float(np.clip(ey / e, -float(self.centering_max_step.get()), float(self.centering_max_step.get())))

        # 현재 명령 각도 업데이트
        self._curr_pan  = float(self._curr_pan  + dpan)
        self._curr_tilt = float(self._curr_tilt + dtilt)

        # 이동 명령
        self.ctrl.send({
            "cmd":"move",
            "pan":  self._curr_pan,
            "tilt": self._curr_tilt,
            "speed": int(self.point_speed.get()),
            "acc":   float(self.point_acc.get())
        })
        self._centering_last_ms = now_ms

def main():
    root = Tk()
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
