#!/usr/bin/env python3
# pc_gui.py — GUI client connecting to pc_server.py (not to Pi agent)

import json, socket, struct, threading, queue, pathlib, io
from datetime import datetime
from tkinter import Tk, Label, Button, Scale, HORIZONTAL, IntVar, DoubleVar, Frame, Checkbutton, BooleanVar, filedialog, StringVar
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
import os, re, csv, time
import numpy as np
import cv2

# ==== [NEW] Import Refactored Modules ====
from network import GuiCtrlClient, GuiImgClient
from image_utils import ImageProcessor
from yolo_utils import YOLOProcessor, predict_with_tiling
from scan_utils import ScanController
from gui_parts import ScrollFrame
# =========================================

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

# YOLO Parameters (Used in Pointing Mode as well)
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

ui_q: "queue.Queue[tuple[str,object]]" = queue.Queue()

# ---- GUI ----
class App:
    def __init__(self, root: Tk):
        self.root = root
        root.title("Pan-Tilt Socket GUI (Client)")
        root.geometry("980x820")
        root.minsize(980, 820)  # 창 최소 크기 고정

        # connections
        self.ctrl = GuiCtrlClient(SERVER_HOST, GUI_CTRL_PORT, ui_q); self.ctrl.start()
        self.img  = GuiImgClient (SERVER_HOST, GUI_IMG_PORT, DEFAULT_OUT_DIR, ui_q); self.img.start()

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

    # ========== Helper Methods (Refactoring Phase 1) ==========
    
    def _send_snap_cmd(self, save_name: str, hard_stop: bool = False):
        """Snap 명령 전송 헬퍼"""
        self.ctrl.send({
            "cmd": "snap",
            "width": self.width.get(),
            "height": self.height.get(),
            "quality": self.quality.get(),
            "save": save_name,
            "hard_stop": hard_stop
        })

    def _get_yolo_model(self):
        """YOLO 모델 캐싱 - delegates to YOLOProcessor"""
        wpath = self.yolo_wpath.get().strip()
        if not wpath:
            return None
        return self.yolo_processor.get_model(wpath)

    def _undistort_pair(self, img_on, img_off):
        """이미지 쌍 Undistort 헬퍼 - delegates to ImageProcessor"""
        self.image_processor.alpha = float(self.ud_alpha.get())
        return self.image_processor.undistort_pair(img_on, img_off, use_torch=True)

    def _calculate_angle_delta(self, err_x: float, err_y: float, 
                               k_pan: float = CENTERING_GAIN_PAN, k_tilt: float = CENTERING_GAIN_TILT):
        """픽셀 오차 → 각도 변환 (클램핑 포함)"""
        d_pan = err_x * k_pan
        d_tilt = -err_y * k_tilt
        max_step = self.pointing_max_step.get()
        d_pan = max(min(d_pan, max_step), -max_step)
        d_tilt = max(min(d_tilt, max_step), -max_step)
        return d_pan, d_tilt

    def _load_image_from_file(self, path):
        """파일에서 이미지 로드 - delegates to ImageProcessor"""
        return self.image_processor.load_image(path)

    def _load_image_pair(self, path_on, path_off):
        """ON/OFF 이미지 쌍 로드 - delegates to ImageProcessor"""
        return self.image_processor.load_image_pair(path_on, path_off)

    def _get_device(self):
        """YOLO/Torch 디바이스 반환 - delegates to YOLOProcessor"""
        return self.yolo_processor.get_device()

    # ========== End of Helper Methods ==========

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

    def _find_laser_center(self, img_on, img_off):
        """
        Find laser center using brightness centroid from diff image.
        No ROI, no Contour, just moments of diff grayscale.
        """
        # ROI: 중앙 ±roi_size (가로) + 위로 200px 확장 (세로)
        # roi_size=200 → 400x600, roi_size=300 → 600x800
        H, W = img_on.shape[:2]
        cx, cy = W // 2, H // 2
        roi_size = self.pointing_roi_size.get()
        
        # 가로: cx ± roi_size
        x1 = max(0, cx - roi_size)
        x2 = min(W, cx + roi_size)
        
        # 세로: (cy - roi_size - 200) ~ (cy + roi_size)
        y1 = max(0, cy - roi_size - 200)  # 위로 200 확장
        y2 = min(H, cy + roi_size)
        
        roi_on = img_on[y1:y2, x1:x2]
        roi_off = img_off[y1:y2, x1:x2]
        
        # Calculate difference image
        diff = cv2.absdiff(roi_on, roi_off)
        
        # Convert to grayscale
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        cv_thresh = 30
        _, binary = cv2.threshold(gray, cv_thresh, 255, cv2.THRESH_BINARY)

        # Calculate brightness centroid using moments
        M = cv2.moments(binary)
        if M["m00"] == 0:
            return None
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        return (cx, cy)

    # ==== Pointing Mode Logic ====
    def _start_pointing_cycle(self):
        # 1. Laser ON
        self._pointing_state = 1 # WAIT_LASER_ON
        self.ctrl.send({"cmd":"laser", "value":1})
        wait_ms = int(self.led_settle.get() * 1000)
        self.root.after(wait_ms, lambda: self.ctrl.send({
            "cmd":"snap", "width":self.width.get(), "height":self.height.get(),
            "quality":self.quality.get(), "save":"pointing_laser_on.jpg", "hard_stop":False
        }))

    def _run_pointing_laser_logic(self, img_on, img_off):
        try:
            img_on, img_off = self._undistort_pair(img_on, img_off)
            
            laser_pos = self._find_laser_center(img_on, img_off)
            
            if laser_pos is None:
                # Laser not found -> Retry cycle (no movement)
                ui_q.put(("toast", "⚠️ Laser not found -> Retry"))
                self._pointing_state = 0
                self._pointing_last_ts = time.time() * 1000
                return

            # Laser Found -> Proceed to Object Detection
            self._laser_px = laser_pos
            ui_q.put(("toast", f"✅ Laser Found: {laser_pos}"))
            
            # [DEBUG] Save laser visualization (UD applied!)
            
            diff_laser = cv2.absdiff(img_on, img_off)  # img_on, img_off는 이미 UD 적용됨!
            debug_laser = cv2.cvtColor(diff_laser, cv2.COLOR_BGR2RGB) if len(diff_laser.shape) == 3 else cv2.cvtColor(diff_laser, cv2.COLOR_GRAY2BGR)
            cv2.circle(debug_laser, laser_pos, 10, (0, 255, 0), 3)  # 녹색 원
            cv2.drawMarker(debug_laser, laser_pos, (0, 255, 0), cv2.MARKER_CROSS, 40, 3)  # 십자 마커
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 밀리초 포함
            debug_path = DEFAULT_OUT_DIR / f"debug_laser_ud_{ts}.jpg"
            cv2.imwrite(str(debug_path), debug_laser)
            print(f"[DEBUG] Laser saved (UD): {debug_path}, pos={laser_pos}")
            # Trigger LED ON
            ui_q.put(("pointing_step_2", None))
            
        except Exception as e:
            ui_q.put(("toast", f"❌ Pointing Laser Error: {e}"))
            self._pointing_state = 0

    def _run_pointing_object_logic(self, img_on, img_off):
        try:
            img_on, img_off = self._undistort_pair(img_on, img_off)
            
            diff = cv2.absdiff(img_on, img_off)
            
            model = self._get_yolo_model()
            if model is None:
                ui_q.put(("toast", "❌ YOLO 없음"))
                self._pointing_state = 0; return

            device = self._get_device()
            
            boxes, scores, classes = predict_with_tiling(model, diff, rows=YOLO_TILE_ROWS, cols=YOLO_TILE_COLS, overlap=YOLO_TILE_OVERLAP, conf=YOLO_CONF_THRESHOLD, iou=YOLO_IOU_THRESHOLD, device=device)
            
            if not boxes:
                ui_q.put(("toast", "⚠️ Object not found -> Retry"))
                self._pointing_state = 0; return # Retry next cycle

            # Find closest to center
            H, W = diff.shape[:2]
            cx, cy = W/2, H/2
            best_idx = -1; min_dist = 999999
            
            for i, (x, y, w, h) in enumerate(boxes):
                obj_cx = x + w/2; obj_cy = y + h/2
                dist = (obj_cx - cx)**2 + (obj_cy - cy)**2
                if dist < min_dist:
                    min_dist = dist; best_idx = i
            
            x, y, w, h = boxes[best_idx]
            obj_cx = x + w/2; obj_cy = y + h/2
            
            # Target Calculation (5cm below center)
            # Assume object is 5cm x 5cm
            px_per_cm = w / 5.0
            target_y_offset = 5.0 * px_per_cm
            target_px = (obj_cx, obj_cy + target_y_offset)
            
            # Error (Target - Laser)
            # We want to move Camera so that Laser hits Target.
            # Actually, Laser is fixed to Camera.
            # So we want to move Camera so that Laser point (fixed in frame) overlaps with Target point (in frame).
            # Wait, if we move Camera, the Scene moves.
            # If we want Laser (fixed px) to be at Target (scene px), we need to move Camera.
            # If Target is at (100, 100) and Laser is at (200, 200).
            # We need to move Camera so that Target moves to (200, 200).
            # To move Scene Point (100,100) to (200,200) (Right, Down), we need to Pan Left, Tilt Up?
            # Let's check coordinate system.
            # Pan + -> Camera Right -> Image Left.
            # Tilt + -> Camera Up -> Image Down.
            # We want Image Point to move from (100,100) to (200,200). (+100, +100).
            # So we need Pan Left (Pan -) and Tilt Up (Tilt +)?
            # Error = Target - Laser = (100-200, 100-200) = (-100, -100).
            # If we use Error directly:
            # d_pan = -100 * k. (Pan -). Correct.
            # d_tilt = -100 * k. (Tilt -). Wait.
            # If Tilt - -> Camera Down -> Image Up.
            # We want Image Down. So we need Tilt +.
            # So d_tilt should be positive.
            # So d_tilt = -Error_y * k?
            # Error_y = -100. -(-100) = +100. Correct.
            
            err_x = target_px[0] - self._laser_px[0]
            err_y = target_px[1] - self._laser_px[1]
            # [DEBUG] Save target visualization (UD applied!)
            debug_target = diff.copy()  # diff는 이미 UD 적용된 img_on, img_off의 차분!
            debug_target = cv2.cvtColor(debug_target, cv2.COLOR_GRAY2BGR) if len(debug_target.shape) == 2 else debug_target
            # 타겟 위치 (빨간색)
            cv2.circle(debug_target, (int(target_px[0]), int(target_px[1])), 12, (0, 0, 255), 3)
            cv2.drawMarker(debug_target, (int(target_px[0]), int(target_px[1])), (0, 0, 255), cv2.MARKER_CROSS, 50, 3)
            cv2.putText(debug_target, "TARGET", (int(target_px[0])+15, int(target_px[1])-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # 레이저 위치 (녹색)
            cv2.circle(debug_target, self._laser_px, 12, (0, 255, 0), 3)
            cv2.drawMarker(debug_target, self._laser_px, (0, 255, 0), cv2.MARKER_CROSS, 50, 3)
            cv2.putText(debug_target, "LASER", (self._laser_px[0]+15, self._laser_px[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # 객체 BBox (노란색)
            cv2.rectangle(debug_target, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 255), 3)
            cv2.putText(debug_target, "OBJECT", (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            # 오차 표시
            cv2.putText(debug_target, f"Err: ({err_x:.1f}, {err_y:.1f})", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 밀리초 포함
            debug_path = DEFAULT_OUT_DIR / f"debug_target_ud_{ts}.jpg"
            cv2.imwrite(str(debug_path), debug_target)
            print(f"[DEBUG] Target saved (UD): {debug_path}, L={self._laser_px}, T={target_px}")
            ui_q.put(("toast", f"Err:({err_x:.1f}, {err_y:.1f}) L:{self._laser_px} T:{target_px}"))
            
            # Convergence
            tol = self.pointing_px_tol.get()
            if abs(err_x) <= tol and abs(err_y) <= tol:
                self._pointing_stable_cnt += 1
                ui_q.put(("toast", f"✅ Pointing Converging... {self._pointing_stable_cnt}/{self.pointing_min_frames.get()}"))
                if self._pointing_stable_cnt >= self.pointing_min_frames.get():
                    ui_q.put(("toast", "🎉 Pointing Complete!"))
                    self.pointing_enable.set(False); ui_q.put(("preview_on", None))
                    self.ctrl.send({"cmd":"laser", "value":0}); self.laser_on.set(False)
                    self._pointing_state = 0
                    return
            else:
                self._pointing_stable_cnt = 0
                
                # Move
                d_pan, d_tilt = self._calculate_angle_delta(err_x, err_y)
                
                next_pan = self._curr_pan + d_pan
                next_tilt = self._curr_tilt + d_tilt
                
                # Hardware limits
                next_pan = max(-180, min(180, next_pan))
                next_tilt = max(-30, min(90, next_tilt))
                
                self._curr_pan = next_pan
                self._curr_tilt = next_tilt
                
                self.ctrl.send({"cmd":"move", "pan":next_pan, "tilt":next_tilt, "speed":self.speed.get(), "acc":float(self.acc.get())})
            
            self._pointing_state = 0 # Cycle Done
            self._pointing_last_ts = time.time() * 1000

        except Exception as e:
            ui_q.put(("toast", f"❌ Pointing Object Error: {e}"))
            self._pointing_state = 0


    # ========== Event Handlers (Phase 2 Refactoring) ==========
    
    def _check_pointing_trigger(self):
        """Check and trigger pointing cycle if needed"""
        if self.pointing_enable.get() and self._pointing_state == 0:
            now = time.time() * 1000
            if now - self._pointing_last_ts > self.pointing_cooldown.get():
                self._start_pointing_cycle()
    
    def _handle_hello_event(self, evt):
        """Handle server hello event"""
        if self.preview_enable.get() and evt.get("agent_state") == "connected":
            self.toggle_preview()
    
    def _handle_scan_start(self, evt):
        """Handle scan start event - delegate to ScanController"""
        total = int(evt.get("total", 0))
        self.prog.configure(maximum=max(1, total), value=0)
        self.prog_lbl.config(text=f"0 / {total}")
        self.dl_lbl.config(text="DL 0")
        self.last_lbl.config(text="Last: -")
        
        # Start ScanController
        sess = evt.get("session") or datetime.now().strftime("scan_%Y%m%d_%H%M%S")
        yolo_path = self.yolo_wpath.get()
        success = self.scan_controller.start_scan(sess, yolo_path)
        
        if not success:
            ui_q.put(("toast", "❌ CSV creation failed"))
    
    def _handle_scan_progress(self, evt):
        """Handle scan progress update"""
        done = int(evt.get("done", 0))
        total = int(evt.get("total", 0))
        if total > 0:
            self.prog.configure(maximum=total)
        self.prog.configure(value=done)
        self.prog_lbl.config(text=f"{done} / {total}")
        name = evt.get("name", "")
        if name:
            self.last_lbl.config(text=f"Last: {name}")
    
    def _handle_scan_done(self, evt):
        """Handle scan completion - ScanController stops and reports results"""
        # Stop ScanController
        results = self.scan_controller.stop_scan()
        
        # Report results
        csv_path = results.get('csv_path', 'unknown')
        processed = results.get('processed', 0)
        detected = results.get('detected', 0)
        
        # Auto-load CSV to Pointing tab
        if csv_path and csv_path != 'unknown':
            self.point_csv_path.set(str(csv_path))
            print(f"[DEBUG scan_done] CSV auto-loaded to Pointing tab: {csv_path}")
            self.pointing_compute()
        ui_q.put(("toast", f"✅ 스캔 완료: {processed}개 처리, {detected}개 검출"))
        ui_q.put(("toast", f"📄 CSV 자동 로드됨: {csv_path}"))
        ui_q.put(("preview_on", None))
    
    def _handle_server_event(self, evt):
        """Route server events to specific handlers"""
        et = evt.get("event")
        if et == "hello":
            self._handle_hello_event(evt)
        elif et == "start":
            self._handle_scan_start(evt)
        elif et == "progress":
            self._handle_scan_progress(evt)
        elif et == "done":
            self._handle_scan_done(evt)
    
    
    def _handle_pointing_laser_on(self, name, data):
        """Handle pointing laser ON image"""
        if name == "pointing_laser_on.jpg":
            self._pointing_state = 2
            self._set_preview(data)
            self.ctrl.send({"cmd": "laser", "value": 0})
            wait_ms = int(self.led_settle.get() * 1000)
            self.root.after(wait_ms, lambda: self.ctrl.send({
                "cmd": "snap", "width": self.width.get(), "height": self.height.get(),
                "quality": self.quality.get(), "save": "pointing_laser_off.jpg", "hard_stop": False
            }))
    
    def _handle_pointing_laser_off(self, name, data):
        """Handle pointing laser OFF image"""
        if name == "pointing_laser_off.jpg":
            self._pointing_state = 3
            self._set_preview(data)
            path_on = DEFAULT_OUT_DIR / "pointing_laser_on.jpg"
            
            # Load pair using helper
            img_on, img_off = self._load_image_pair(path_on, DEFAULT_OUT_DIR / name)
            
            if img_on is None or img_off is None:
                ui_q.put(("toast", "❌ Failed to load pointing images"))
                self._pointing_state = 0
                return
            
            self._run_pointing_laser_logic(img_on, img_off)

    def _handle_pointing_step_2(self):
        """Handle pointing step 2: LED ON"""
        self._pointing_state = 4 # WAIT_LED_ON
        self.ctrl.send({"cmd":"led", "value":255})
        wait_ms = int(self.led_settle.get() * 1000)
        self.root.after(wait_ms, lambda: self.ctrl.send({
            "cmd":"snap", "width":self.width.get(), "height":self.height.get(),
            "quality":self.quality.get(), "save":"pointing_led_on.jpg", "hard_stop":False
        }))

    def _handle_pointing_led_on(self, name, data):
        """Handle pointing LED ON image"""
        if name == "pointing_led_on.jpg":
            self._pointing_state = 5
            self._set_preview(data)
            self.ctrl.send({"cmd":"led", "value":0})
            wait_ms = int(self.led_settle.get() * 1000)
            self.root.after(wait_ms, lambda: self.ctrl.send({
                "cmd":"snap", "width":self.width.get(), "height":self.height.get(),
                "quality":self.quality.get(), "save":"pointing_led_off.jpg", "hard_stop":False
            }))

    def _handle_pointing_led_off(self, name, data):
        """Handle pointing LED OFF image"""
        if name == "pointing_led_off.jpg":
            self._pointing_state = 6
            self._set_preview(data)
            path_on = DEFAULT_OUT_DIR / "pointing_led_on.jpg"
            
            img_on, img_off = self._load_image_pair(path_on, DEFAULT_OUT_DIR / name)
            
            if img_on is None or img_off is None:
                ui_q.put(("toast", "❌ Failed to load pointing images (LED)"))
                self._pointing_state = 0
                return
            
            self._run_pointing_object_logic(img_on, img_off)

    def _set_preview(self, data):
        """Set preview image from data"""
        try:
            img = Image.open(io.BytesIO(data))
            img.thumbnail((self.PREV_W, self.PREV_H))
            self.tkimg = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=self.tkimg)
        except Exception as e:
            print(f"Preview error: {e}")

    def _poll(self):
        try:
            while True:
                item = ui_q.get_nowait()
                kind = item[0]
                
                if kind == "toast":
                    print(f"[TOAST] {item[1]}")
                    
                elif kind == "evt":
                    self._handle_server_event(item[1])
                    
                elif kind == "preview":
                    self._set_preview(item[1])
                    
                elif kind == "preview_on":
                    if self.preview_enable.get():
                        self.ctrl.send({"cmd":"preview","enable": True})
                        
                elif kind == "saved":
                    name, data = item[1]
                    # Pass to ScanController
                    self.scan_controller.on_image_received(name, data)
                    
                    # Pointing Mode Handlers
                    if self.pointing_enable.get():
                        if name == "pointing_laser_on.jpg":
                            self._handle_pointing_laser_on(name, data)
                        elif name == "pointing_laser_off.jpg":
                            self._handle_pointing_laser_off(name, data)
                        elif name == "pointing_led_on.jpg":
                            self._handle_pointing_led_on(name, data)
                        elif name == "pointing_led_off.jpg":
                            self._handle_pointing_led_off(name, data)
                            
                elif kind == "pointing_step_2":
                    self._handle_pointing_step_2()
                    
        except queue.Empty:
            pass
            
        # Check pointing trigger
        self._check_pointing_trigger()
        
        self.root.after(POLL_INTERVAL_MS, self._poll)

    # Legacy methods for compatibility (if needed)
    def pointing_choose_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV","*.csv")])
        if path:
            self.point_csv_path.set(path)
            self.pointing_compute()

    def pointing_compute(self):
        # Placeholder for legacy compute if needed, or remove if fully replaced
        pass

    def pointing_move(self):
        # Placeholder
        pass

if __name__ == "__main__":
    root = Tk()
    app = App(root)
    app.run()
