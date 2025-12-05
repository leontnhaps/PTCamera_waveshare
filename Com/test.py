#!/usr/bin/env python3
# pc_gui.py — GUI client connecting to pc_server.py (not to Pi agent)

import json, socket, struct, threading, queue, pathlib, io
from datetime import datetime
from tkinter import Tk, Label, Button, Scale, HORIZONTAL, IntVar, DoubleVar, Frame, Checkbutton, BooleanVar, filedialog, StringVar
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
import os, re, csv, time
from datetime import datetime
import numpy as np
import cv2

# Import from modules
from network import GuiCtrlClient, GuiImgClient, ui_q
from image_utils import ImageProcessor
from yolo_utils import YOLOProcessor, predict_with_tiling, non_max_suppression
from scan_utils import ScanController
from gui_parts import ScrollFrame
from app_helpers import AppHelpersMixin
from pointing_handler import PointingHandlerMixin
from event_handlers import EventHandlersMixin
from app_ui import AppUIMixin

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

# ==== YOLO (for LED difference detection) ====
try:
    from ultralytics import YOLO
    _YOLO_OK = True
except Exception:
    YOLO = None
    _YOLO_OK = False

# ========== Configuration Constants ==========
# Hardware Limits
PAN_MIN = -180
PAN_MAX = 180
TILT_MIN = -30
TILT_MAX = 90

# Control Parameters
CENTERING_GAIN_PAN = 0.02
CENTERING_GAIN_TILT = 0.02
ANGLE_DELTA_MAX = 5.0
POLL_INTERVAL_MS = 60

# YOLO Parameters
YOLO_CONF_THRESHOLD = 0.20
YOLO_IOU_THRESHOLD = 0.45
YOLO_TILE_ROWS = 2
YOLO_TILE_COLS = 3
YOLO_TILE_OVERLAP = 0.15

# LED/Laser Timing
LED_SETTLE_DEFAULT = 0.5  # seconds
# ============================================

SERVER_HOST = "127.0.0.1"
GUI_CTRL_PORT = 7600
GUI_IMG_PORT  = 7601

DEFAULT_OUT_DIR = pathlib.Path(f"captures_gui_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)



# ---- GUI ----
class App(AppHelpersMixin, PointingHandlerMixin, EventHandlersMixin, AppUIMixin):
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


        # Image processing (undistortion, loading)
        self.image_processor = ImageProcessor()
        self.ud_enable    = BooleanVar(value=True)
        self.ud_save_copy = BooleanVar(value=True)
        self.ud_alpha     = DoubleVar(value=0.0)

        # YOLO processing (model loading, caching)
        self.yolo_processor = YOLOProcessor()
        self.yolo_wpath = StringVar(value="yolov11m_diff.pt")
        self._scan_yolo_conf = 0.50
        self._scan_yolo_imgsz = 832
        
        # Scan processing (real-time YOLO detection)
        self.scan_controller = ScanController(self.image_processor, self.yolo_processor, DEFAULT_OUT_DIR)

        print(f"[INFO] cv2.cuda={self.image_processor._use_cv2_cuda}, torch_cuda={self.image_processor._torch_cuda}")

        # top bar
        top = Frame(root); top.pack(fill="x", padx=10, pady=6)
        self.outdir = StringVar(value=str(DEFAULT_OUT_DIR))
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
        self.notebook = nb # [NEW] Save reference
        tab_scan   = Frame(nb); nb.add(tab_scan, text="Scan")
        tab_manual = Frame(nb); nb.add(tab_manual, text="Manual / LED")
        tab_misc = Frame(nb); nb.add(tab_misc, text="Preview & Settings")
        # tab_point removed (replaced by new Scrollable tab later)
        
        # Pointing variables moved below Scan params

        
        
        # scan params
        self.pan_min=IntVar(value=-180); self.pan_max=IntVar(value=180); self.pan_step=IntVar(value=15)
        self.tilt_min=IntVar(value=-30); self.tilt_max=IntVar(value=90);  self.tilt_step=IntVar(value=15)
        self.width=IntVar(value=2592);   self.height=IntVar(value=1944); self.quality=IntVar(value=90)
        self.speed=IntVar(value=100);    self.acc=DoubleVar(value=1.0);  self.settle=DoubleVar(value=0.6)
        self.led_settle=DoubleVar(value=0.4)
        self.hard_stop = BooleanVar(value=False)

        # Pointing variables (Moved here to fix AttributeError)
        self.point_csv_path = StringVar(value="")
        self.point_conf_min = DoubleVar(value=0.50)
        self.point_min_samples = IntVar(value=2)
        self.point_pan_target  = DoubleVar(value=0.0)
        self.point_tilt_target = DoubleVar(value=0.0)
        self.point_speed  = IntVar(value=self.speed.get())
        self.point_acc    = DoubleVar(value=self.acc.get())

        self._row(tab_scan, 0, "Pan min/max/step", self.pan_min, self.pan_max, self.pan_step)
        self._row(tab_scan, 1, "Tilt min/max/step", self.tilt_min, self.tilt_max, self.tilt_step)
        self._row(tab_scan, 2, "Resolution (w×h)", self.width, self.height, None, ("W","H",""))
        self._entry(tab_scan, 3, "Quality(%)", self.quality)
        self._entry(tab_scan, 4, "Speed", self.speed)
        self._entry(tab_scan, 5, "Accel", self.acc)
        self._entry(tab_scan, 6, "Settle(s)", self.settle)
        self._entry(tab_scan, 7, "LED Settle(s)", self.led_settle)
        Checkbutton(tab_scan, text="Hard stop(정지 펄스)", variable=self.hard_stop)\
            .grid(row=8, column=1, sticky="w", padx=4, pady=2)

        ops = Frame(tab_scan); ops.grid(row=9, column=0, columnspan=4, sticky="w", pady=6)
        Button(ops, text="Start Scan", command=self.start_scan).pack(side="left", padx=4)
        Button(ops, text="Stop Scan",  command=self.stop_scan).pack(side="left", padx=4)
        self.prog = ttk.Progressbar(ops, orient=HORIZONTAL, length=280, mode="determinate"); self.prog.pack(side="left", padx=10)
        self.prog_lbl = Label(ops, text="0 / 0"); self.prog_lbl.pack(side="left")
        self.last_lbl = Label(ops, text="Last: -"); self.last_lbl.pack(side="left", padx=10)
        self.dl_lbl   = Label(ops, text="DL 0");    self.dl_lbl.pack(side="left", padx=10)

        # Old Pointing Tab code removed

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
        self.laser_on = BooleanVar(value=False)
        Button(tab_manual, text="Laser ON/OFF", command=self.toggle_laser).grid(row=6,column=2,sticky="w",padx=4,pady=4)

        # preview settings
        misc_sf = ScrollFrame(tab_misc)
        misc_sf.pack(fill="both", expand=True)
        misc = misc_sf.body  # ← 앞으로 이걸 parent로 써요

        self.preview_enable=BooleanVar(value=True)
        self.preview_w=IntVar(value=2592); self.preview_h=IntVar(value=1944)
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

        # ==== YOLO UI ====
        ttk.Separator(misc, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", pady=(8,6)); row+=1
        Label(misc, text="YOLO 가중치 (.pt)").grid(row=row, column=0, sticky="w")
        Button(misc, text="Load YOLO", command=self.load_yolo_weights).grid(row=row, column=1, sticky="w", pady=2); row+=1
        # ==================

        # (있으면) 이 줄도 추가해두면 너비 늘어날 때 경로 라벨이 자연스럽게 늘어남
        for c in range(4):
            misc.grid_columnconfigure(c, weight=1)

        # ==================

        self.root.after(POLL_INTERVAL_MS, self._poll)
                # ===== [SCAN CSV 로깅 상태] =====
        self._scan_csv_path = None
        self._scan_csv_file = None
        self._scan_csv_writer = None

        # 파일명에서 pan/tilt 파싱 (예: img_t+00_p+001_....jpg)
        self._fname_re = re.compile(r"img_t(?P<tilt>[+\-]\d{2,3})_p(?P<pan>[+\-]\d{2,3})_.*\.(jpg|jpeg|png)$", re.IGNORECASE)


        # === Pointing 좌표 로깅 상태 ===
        self._pointing_log_fp = None
        self._pointing_log_writer = None
        self._pointing_logging = False
        
        # Pointing State
        self._pointing_state = 0
        self._pointing_last_ts = 0
        self._pointing_stable_cnt = 0

        # (선택) 현재 명령 각도 기억
        self._curr_pan = 0.0
        self._curr_tilt = 0.00
        
        self._fits_h = {}
        self._fits_v = {}
        
        # Pointing mode settings
        self.pointing_interval = DoubleVar(value=3.0)  # seconds between cycles
        self.pointing_px_tol = IntVar(value=7)
        self.pointing_min_frames = IntVar(value=4)
        self.pointing_max_step = DoubleVar(value=5.0)
        self.pointing_cooldown = IntVar(value=250)


        # ---------------------------------------------------------------------
        # 4. Pointing Tab (Scrollable)
        # ---------------------------------------------------------------------
        self.tab_point = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_point, text="Pointing")
        
        # Create Canvas & Scrollbar
        self.point_canvas = tk.Canvas(self.tab_point)
        self.point_scroll = ttk.Scrollbar(self.tab_point, orient="vertical", command=self.point_canvas.yview)
        self.point_scroll_frame = ttk.Frame(self.point_canvas)
        
        self.point_scroll_frame.bind(
            "<Configure>",
            lambda e: self.point_canvas.configure(scrollregion=self.point_canvas.bbox("all"))
        )
        self.point_canvas.create_window((0, 0), window=self.point_scroll_frame, anchor="nw")
        self.point_canvas.configure(yscrollcommand=self.point_scroll.set)
        
        # [NEW] Mouse Wheel Binding
        # [NEW] Mouse Wheel Binding (Improved)
        def _on_mousewheel(event):
            self.point_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_mousewheel(event):
            self.point_canvas.bind_all("<MouseWheel>", _on_mousewheel)
            
        def _unbind_mousewheel(event):
            self.point_canvas.unbind_all("<MouseWheel>")

        # Bind to both canvas and scroll frame to ensure it catches hover
        self.point_canvas.bind("<Enter>", _bind_mousewheel)
        self.point_canvas.bind("<Leave>", _unbind_mousewheel)
        self.point_scroll_frame.bind("<Enter>", _bind_mousewheel)
        self.point_scroll_frame.bind("<Leave>", _unbind_mousewheel)
        
        self.point_canvas.pack(side="left", fill="both", expand=True)
        self.point_scroll.pack(side="right", fill="y")
        
        # --- Pointing Mode Controls (Inside Scroll Frame) ---
        # Use grid layout with 3 columns (all side-by-side)
        
        # Column 1: Pointing Settings
        col1_frame = ttk.Frame(self.point_scroll_frame)
        col1_frame.grid(row=0, column=0, padx=5, pady=10, sticky="nsew")
        
        # Column 2: Pointing Control
        col2_frame = ttk.Frame(self.point_scroll_frame)
        col2_frame.grid(row=0, column=1, padx=5, pady=10, sticky="nsew")
        
        # Column 3: CSV Analysis
        col3_frame = ttk.Frame(self.point_scroll_frame)
        col3_frame.grid(row=0, column=2, padx=5, pady=10, sticky="nsew")
        
        # Configure column weights for proper resizing
        self.point_scroll_frame.grid_columnconfigure(0, weight=1)
        self.point_scroll_frame.grid_columnconfigure(1, weight=1)
        self.point_scroll_frame.grid_columnconfigure(2, weight=1)
        
        # --- Column 1: Pointing Settings ---
        point_set_frame = ttk.LabelFrame(col1_frame, text="Pointing Settings")
        point_set_frame.pack(padx=5, pady=5, fill="both", expand=True)
        
        def add_entry(parent, label, var, row):
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(parent, textvariable=var, width=10).grid(row=row, column=1, sticky="w", padx=5, pady=2)

        self.pointing_roi_size = tk.IntVar(value=200)
        add_entry(point_set_frame, "Laser ROI Size (px):", self.pointing_roi_size, 0)
        
        ttk.Label(point_set_frame, text="--- Pointing Settings ---").grid(row=1, column=0, columnspan=2, pady=5)
        add_entry(point_set_frame, "Tolerance (px):", self.pointing_px_tol, 2)
        add_entry(point_set_frame, "Min Stable Frames:", self.pointing_min_frames, 3)
        add_entry(point_set_frame, "Max Step (deg):", self.pointing_max_step, 4)
        add_entry(point_set_frame, "Cooldown (ms):", self.pointing_cooldown, 5)
        add_entry(point_set_frame, "LED Settle (s):", self.led_settle, 6)
        
        # --- Column 2: Pointing Control ---
        point_ctrl_frame = ttk.LabelFrame(col2_frame, text="Pointing Control")
        point_ctrl_frame.pack(padx=5, pady=5, fill="both", expand=True)
        
        self.pointing_enable = tk.BooleanVar(value=False)
        ttk.Checkbutton(point_ctrl_frame, text="Enable Pointing Mode", variable=self.pointing_enable, command=self.on_pointing_toggle).pack(anchor="w", padx=5, pady=5)

        # --- Column 3: CSV Analysis ---
        point_csv_frame = ttk.LabelFrame(col3_frame, text="CSV Analysis (Legacy)")
        point_csv_frame.pack(padx=5, pady=5, fill="both", expand=True)
        
        self.point_csv_path = tk.StringVar()
        ttk.Label(point_csv_frame, textvariable=self.point_csv_path, wraplength=200).pack(anchor="w", padx=5, pady=2)
        
        ttk.Label(point_csv_frame, text="Conf Min:").pack(anchor="w", padx=5)
        self.point_conf_min = tk.StringVar(value="0.6")
        ttk.Entry(point_csv_frame, textvariable=self.point_conf_min, width=15).pack(anchor="w", padx=5)
        
        ttk.Label(point_csv_frame, text="Min Samples:").pack(anchor="w", padx=5)
        self.point_min_samples = tk.StringVar(value="2")
        ttk.Entry(point_csv_frame, textvariable=self.point_min_samples, width=15).pack(anchor="w", padx=5)
        
        self.point_result_lbl = ttk.Label(point_csv_frame, text="Result: -")
        self.point_result_lbl.pack(anchor="w", padx=5, pady=5)
        
        # CSV 파일 선택 버튼 ← 여기 추가!
        ttk.Button(point_csv_frame, text="Load CSV", 
        command=self.pointing_choose_csv).pack(anchor="w", padx=5, pady=2)

        # [RESTORED] Move to Target Button
        ttk.Button(point_csv_frame, text="Move to Target", command=self.pointing_move).pack(anchor="w", padx=5, pady=5)
        
        # Debug Preview Label (on right side of tab)
        debug_frame = Frame(self.tab_point, bg="#111", width=420, height=420, relief="solid", borderwidth=2)
        debug_frame.pack(side="right", padx=10, pady=10)
        debug_frame.pack_propagate(False)
        
        Label(debug_frame, text="Debug Target Preview", bg="#111", fg="white", font=("", 10, "bold")).pack(pady=5)
        self.debug_preview_label = Label(debug_frame, bg="#111", fg="#666", text="(Waiting for detection...)")
        self.debug_preview_label.pack(fill="both", expand=True, padx=10, pady=10)
        self.debug_preview_img = None
     




        # [NEW] Auto-load calib.npz if exists
        if pathlib.Path("calib.npz").exists():
            self.load_npz("calib.npz")

        # [NEW] Auto-load YOLO model if exists
        yolo_path = pathlib.Path(self.yolo_wpath.get())
        if yolo_path.exists():
            print(f"[YOLO] 자동 로드 시작: {yolo_path}")
            self._get_yolo_model()  # 미리 캐싱

    def run(self):
        self.root.mainloop()

    def load_npz(self, path=None):
        """Load calibration file and delegate to ImageProcessor"""
        if path is None:
            path = filedialog.askopenfilename(filetypes=[("NPZ","*.npz")])
        if not path:
            return
        
        # Update ImageProcessor's alpha from UI
        self.image_processor.alpha = float(self.ud_alpha.get())
        
        # Delegate to ImageProcessor
        success = self.image_processor.load_calibration(path)
        if success:
            print(f"[App] Calibration loaded successfully")
        else:
            print(f"[App] Calibration load failed")


    def _undistort_bgr(self, bgr: np.ndarray) -> np.ndarray:
        """Undistort BGR image - delegates to ImageProcessor"""
        # Update alpha before undistortion
        self.image_processor.alpha = float(self.ud_alpha.get())
        # Use Torch acceleration if available for best performance
        return self.image_processor.undistort(bgr, use_torch=True)

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

    def load_yolo_weights(self):
        """YOLO 가중치 파일 (.pt) 로드"""
        path = filedialog.askopenfilename(filetypes=[("YOLO weights", "*.pt"), ("All files", "*.*")])
        if path:
            self.yolo_wpath.set(path)
            ui_q.put(("toast", f"YOLO 가중치 로드: {pathlib.Path(path).name}"))

    # actions
    def start_scan(self):
    # 보정 강제: calib.npz가 로드되지 않았으면 스캔 시작 금지
        if not self.image_processor.has_calibration():
            ui_q.put(("toast", "❌ 스캔은 보정 이미지만 허용합니다. 먼저 'Load calib.npz'를 해주세요."))
            return
        if self.preview_enable.get():
            self.ctrl.send({"cmd":"preview","enable": False})
        self.ctrl.send({
            "cmd":"scan_run",
            "pan_min":self.pan_min.get(),"pan_max":self.pan_max.get(),"pan_step":self.pan_step.get(),
            "tilt_min":self.tilt_min.get(),"tilt_max":self.tilt_max.get(),"tilt_step":self.tilt_step.get(),
            "speed":self.speed.get(),"acc":float(self.acc.get()),"settle":float(self.settle.get()),
            "led_settle":float(self.led_settle.get()),
            "width":self.width.get(),"height":self.height.get(),"quality":self.quality.get(),
            "session":datetime.now().strftime("scan_%Y%m%d_%H%M%S"),
            "hard_stop":self.hard_stop.get()
        })
    def stop_scan(self):
        self.ctrl.send({"cmd":"scan_stop"})
        
        # Get scan results from ScanController
        result = self.scan_controller.stop_scan()
        print(f"[DEBUG stop_scan] result = {result}")
        
        # Auto-load CSV to Pointing tab if available
        if result and result.get('csv_path'):
            csv_path = result['csv_path']
            print(f"[DEBUG stop_scan] CSV path found: {csv_path}")
            self.point_csv_path.set(str(csv_path))
            print(f"[DEBUG stop_scan] point_csv_path set to: {self.point_csv_path.get()}")
            ui_q.put(("toast", f"✅ Scan 완료! CSV 자동 로드됨: {csv_path.name}"))
            self.pointing_compute()
        else:
            print(f"[DEBUG stop_scan] No CSV path in result!")
        
        self.root.after(500, lambda: ui_q.put(("preview_on", None)))


    def on_pointing_toggle(self):
        if self.pointing_enable.get():
            ui_q.put(("preview_on", None))
            # Laser OFF when stopping
            self.ctrl.send({"cmd":"laser", "value": 0})
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
        else:
            self.laser_on.set(False)
            # CSV 종료 추가
            if self._pointing_log_fp:
                try:
                    self._pointing_log_fp.close()
                    self._pointing_log_fp = None
                    self._pointing_log_writer = None
                    self._pointing_logging = False
                    ui_q.put(("toast", "📄 Pointing log 종료"))
                except Exception as e:
                    ui_q.put(("toast", f"❌ log 종료 실패: {e}"))
                
    def center(self): self.ctrl.send({"cmd":"move","pan":0.0,"tilt":0.0,"speed":self.speed.get(),"acc":float(self.acc.get())})
    def apply_move(self): self.ctrl.send({"cmd":"move","pan":float(self.mv_pan.get()),"tilt":float(self.mv_tilt.get()),
                                          "speed":self.mv_speed.get(),"acc":float(self.mv_acc.get())})
    def set_led(self): self.ctrl.send({"cmd":"led","value":int(self.led.get())})
    def toggle_laser(self):
        val = 1 if not self.laser_on.get() else 0
        self.laser_on.set(bool(val))
        self.ctrl.send({"cmd":"laser", "value": val})

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
        self._send_snap_cmd(fname, self.hard_stop.get())

    # event loop
    # ==== [NEW] Centering Mode Logic ====
    # (_start_centering_cycle and _snap_center_on are defined later - removed duplicate)

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

def main():
    root = Tk()
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()

