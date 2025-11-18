#!/usr/bin/env python3
# pc_gui.py — GUI client connecting to pc_server.py (not to Pi agent)

import json, socket, struct, threading, queue, pathlib, io
from datetime import datetime
from tkinter import Tk, Label, Button, Scale, HORIZONTAL, IntVar, DoubleVar, Frame, Checkbutton, BooleanVar, filedialog, StringVar
from tkinter import ttk
from PIL import Image, ImageTk
import tkinter as tk 
import os, re, csv   

import numpy as np
import cv2
import config
from network import GuiCtrlClient, GuiImgClient
from gui_panels import ManualPanel, ScanPanel
from processors import UndistortProcessor, YOLOProcessor

SERVER_HOST = config.GUI_SERVER_HOST
GUI_CTRL_PORT = config.GUI_CTRL_PORT
GUI_IMG_PORT = config.GUI_IMG_PORT
DEFAULT_OUT_DIR = config.get_gui_output_dir()

DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)

ui_q: "queue.Queue[tuple[str,object]]" = queue.Queue()


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
        self.ctrl = GuiCtrlClient(SERVER_HOST, GUI_CTRL_PORT, ui_q); self.ctrl.start()  # ✅ ui_q 추가
        self.img  = GuiImgClient (SERVER_HOST, GUI_IMG_PORT, DEFAULT_OUT_DIR, ui_q); self.img.start()  # ✅ ui_q 추가

        # state
        self.tkimg=None
        self._resume_preview_after_snap = False
        
        # ===== 이미지 처리기들 =====
        self.undistort_processor = UndistortProcessor()
        self.yolo_processor = YOLOProcessor()

        # UI 상태 변수들
        self.ud_enable    = BooleanVar(value=False)
        self.ud_save_copy = BooleanVar(value=False)
        self.ud_alpha     = DoubleVar(value=0.0)

        self.yolo_enable = BooleanVar(value=False)
        self.yolo_conf   = DoubleVar(value=0.25)
        self.yolo_iou    = DoubleVar(value=0.55)
        self.yolo_imgsz  = IntVar(value=832)
        self.yolo_stride = IntVar(value=2)
        self.yolo_wpath  = StringVar(value="")

        # YOLO 가시성 옵션
        self.yolo_box_thick = IntVar(value=4)
        self.yolo_text_scale = DoubleVar(value=0.7)
        self.yolo_text_thick = IntVar(value=2)
        self.yolo_show_centroid = BooleanVar(value=True)
        self.yolo_show_center_cross = BooleanVar(value=True)

        print(f"[INFO] Processors initialized")
        print(f"[INFO] UndistortProcessor ready, YOLOProcessor ready")
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
        # ✅ 새로운 패널 방식
        # Scan 패널
        self.scan_panel = ScanPanel(nb, {
            'on_start_scan': self.start_scan,
            'on_stop_scan': self.stop_scan
        })
        nb.add(self.scan_panel, text="Scan")

        # Manual 패널  
        self.manual_panel = ManualPanel(nb, {
            'on_center': self.center,
            'on_apply_move': self.apply_move,
            'on_set_led': self.set_led
        })
        nb.add(self.manual_panel, text="Manual / LED")
        tab_misc = Frame(nb); nb.add(tab_misc, text="Preview & Settings")
        tab_point  = Frame(nb); nb.add(tab_point, text="Pointing")

        # ---- Pointing tab (CSV → 가중평균 타깃 → Move) ----
        self.point_csv_path = StringVar(value="")
        self.point_conf_min = DoubleVar(value=0.50)  # CSV 필터용
        self.point_min_samples = IntVar(value=2)     # 각 선형피팅 최소 샘플 수
        self.point_pan_target  = DoubleVar(value=0.0)
        self.point_tilt_target = DoubleVar(value=0.0)
        self.point_speed  = IntVar(value=100)     # ✅ 기본값 사용
        self.point_acc    = DoubleVar(value=1.0)  # ✅ 기본값 사용

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
            variable=self.ud_alpha).grid(row=row, column=1, sticky="w"); row+=1

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
    # ===== 언디스토트 관련 메서드들 =====
    def load_npz(self):
        """보정 파일 로드"""
        path = filedialog.askopenfilename(filetypes=[("NPZ","*.npz")])
        if not path: 
            return
        
        success = self.undistort_processor.load_calibration(path)
        if success:
            ui_q.put(("toast", f"보정 파일 로드 완료: {path}"))
        else:
            ui_q.put(("toast", f"보정 파일 로드 실패: {path}"))

    # ===== YOLO 관련 메서드들 =====
    def load_yolo_weights(self):
        """YOLO 가중치 파일 로드"""
        path = filedialog.askopenfilename(filetypes=[("PyTorch Weights","*.pt")])
        if not path:
            return
        
        success = self.yolo_processor.load_model(path)
        if success:
            self.yolo_wpath.set(path)
            ui_q.put(("toast", f"YOLO 모델 로드 완료: {path}"))
        else:
            ui_q.put(("toast", f"YOLO 모델 로드 실패: {path}"))

    def _on_toggle_yolo(self):
        """YOLO 토글 이벤트"""
        if self.yolo_enable.get():
            if not self.yolo_processor.is_loaded():
                ui_q.put(("toast", "YOLO 가중치(.pt)를 먼저 로드하세요."))
                self.yolo_enable.set(False)
            else:
                # 시각화 설정 업데이트
                self._update_yolo_visualization_settings()

    def _update_yolo_visualization_settings(self):
        """YOLO 시각화 설정 업데이트"""
        self.yolo_processor.update_visualization_settings(
            box_thickness=int(self.yolo_box_thick.get()),
            text_scale=float(self.yolo_text_scale.get()),
            text_thickness=int(self.yolo_text_thick.get()),
            show_centroid=self.yolo_show_centroid.get(),
            show_center_cross=self.yolo_show_center_cross.get()
        )

    def _ensure_yolo_model_for_scan(self) -> bool:
        """스캔용 YOLO 모델 확인"""
        if self.yolo_processor.is_loaded():
            return True
        
        # 경로가 있으면 로드 시도
        wpath = self.yolo_wpath.get().strip()
        if wpath:
            return self.yolo_processor.load_model(wpath)
        
        ui_q.put(("toast", "YOLO 가중치(.pt)을 로드하지 않아 CSV에 감지값을 기록할 수 없습니다."))
        return False
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

    # actions - 메서드 시그니처 수정
    def start_scan(self, params=None):
        # 기존 보정 체크
        if not self.undistort_processor.is_loaded():  # ✅ 수정
            ui_q.put(("toast", "❌ 스캔은 보정 이미지만 허용합니다. 먼저 'Load calib.npz'를 해주세요."))
            return
        if self.preview_enable.get():
            self.ctrl.send({"cmd":"preview","enable": False})
        
        # ✅ 패널에서 파라미터 가져오기 (params가 None이면 직접 접근)
        if params is None:
            params = {
                'pan_min': self.scan_panel.pan_min.get(),
                'pan_max': self.scan_panel.pan_max.get(),
                'pan_step': self.scan_panel.pan_step.get(),
                'tilt_min': self.scan_panel.tilt_min.get(),
                'tilt_max': self.scan_panel.tilt_max.get(),
                'tilt_step': self.scan_panel.tilt_step.get(),
                'width': self.scan_panel.width.get(),
                'height': self.scan_panel.height.get(),
                'quality': self.scan_panel.quality.get(),
                'speed': self.scan_panel.speed.get(),
                'acc': float(self.scan_panel.acc.get()),
                'settle': float(self.scan_panel.settle.get()),
                'hard_stop': self.scan_panel.hard_stop.get()
            }
        
        self.ctrl.send({
            "cmd":"scan_run",
            "pan_min": params['pan_min'], "pan_max": params['pan_max'], "pan_step": params['pan_step'],
            "tilt_min": params['tilt_min'], "tilt_max": params['tilt_max'], "tilt_step": params['tilt_step'],
            "speed": params['speed'], "acc": params['acc'], "settle": params['settle'],
            "width": params['width'], "height": params['height'], "quality": params['quality'],
            "session": datetime.now().strftime("scan_%Y%m%d_%H%M%S"),
            "hard_stop": params['hard_stop']
        })


    def stop_scan(self): 
        self.ctrl.send({"cmd":"scan_stop"})

    def center(self): 
        speed = getattr(self.scan_panel, 'speed', IntVar(value=100))  # 기본값 사용
        self.ctrl.send({"cmd":"move","pan":0.0,"tilt":0.0,"speed":speed.get(),"acc":1.0})

    def apply_move(self, params=None):
        if params is None:
            params = {
                'pan': float(self.manual_panel.mv_pan.get()),
                'tilt': float(self.manual_panel.mv_tilt.get()),
                'speed': int(self.manual_panel.mv_speed.get()),
                'acc': float(self.manual_panel.mv_acc.get())
            }
        self.ctrl.send({"cmd":"move","pan":params['pan'],"tilt":params['tilt'],
                        "speed":params['speed'],"acc":params['acc']})

    def set_led(self, value=None): 
        if value is None:
            value = int(self.manual_panel.led.get())
        self.ctrl.send({"cmd":"led","value":value})

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

    def snap_one(self):
        self._resume_preview_after_snap = False
        if self.preview_enable.get():
            self.ctrl.send({"cmd":"preview","enable": False})
            self._resume_preview_after_snap = True
        fname = datetime.now().strftime("snap_%Y%m%d_%H%M%S.jpg")
        self.ctrl.send({
            "cmd":"snap",
            "width":  self.scan_panel.width.get(),      # ✅ 패널에서 가져오기
            "height": self.scan_panel.height.get(),     # ✅ 패널에서 가져오기
            "quality":self.scan_panel.quality.get(),    # ✅ 패널에서 가져오기
            "save":   fname,
            "hard_stop": self.scan_panel.hard_stop.get() # ✅ 패널에서 가져오기
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
                        self.scan_panel.prog.configure(maximum=max(1,total), value=0)
                        self.scan_panel.prog_lbl.config(text=f"0 / {total}")
                        self.scan_panel.dl_lbl.config(text="DL 0")
                        self.scan_panel.last_lbl.config(text="Last: -")

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
                        done=int(evt.get("done",0)); total=int(evt.get("total",0))
                        self.scan_panel.prog.configure(value=done)
                        self.scan_panel.prog_lbl.config(text=f"{done} / {total}")
                        name = evt.get("name","")
                        if name: self.scan_panel.last_lbl.config(text=f"Last: {name}")
                    elif et == "progress":
                        done=int(evt.get("done",0)); total=int(evt.get("total",0))
                        self.scan_panel.prog.configure(value=done)
                        self.scan_panel.prog_lbl.config(text=f"{done} / {total}")  # ✅ 패널에서 가져오기
                        name = evt.get("name","")
                        if name: self.scan_panel.last_lbl.config(text=f"Last: {name}")  # ✅ 패널에서 가져오기
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
                    dl_count = int(self.scan_panel.dl_lbl.cget('text').split()[-1]) + 1
                    self.scan_panel.dl_lbl.config(text=f"DL {dl_count}")
                    self.scan_panel.last_lbl.config(text=f"Last: {name}")

                    if self.ud_save_copy.get() and self.undistort_processor.is_loaded():
                        try:
                            arr = np.frombuffer(data, np.uint8)
                            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if bgr is not None:
                                ubgr = self.undistort_processor.process(bgr, alpha=float(self.ud_alpha.get()))
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
                            if not self.undistort_processor.is_loaded():
                                ui_q.put(("toast", "❌ 보정 파라미터 없음 → 스캔 감지/기록 중단"))
                                return
                                
                            bgr = self.undistort_processor.process(bgr, alpha=float(self.ud_alpha.get()))
                            H, W = bgr.shape[:2]

                            # YOLO 감지 & CSV 기록
                            if self._ensure_yolo_model_for_scan():
                                # 스캔용 YOLO 처리 (시각화 없이 감지만)
                                detected_bgr = self.yolo_processor.process(
                                    bgr.copy(),  # 복사본 사용 (원본 보존)
                                    conf=0.50,   # 스캔용 고정 conf
                                    iou=float(self.yolo_iou.get()),
                                    imgsz=832,   # 스캔용 고정 imgsz
                                    stride=1     # 매 프레임 처리
                                )
                                
                                # 감지 결과 가져오기
                                centroid = self.yolo_processor.get_last_centroid()
                                if centroid is not None:
                                    # 실제로는 개별 박스들을 기록해야 하므로 processor 확장 필요
                                    # 지금은 간단히 중심점만 기록
                                    m_cx, m_cy, n_dets = centroid
                                    if n_dets > 0:
                                        # 임시로 중심점 기록 (실제로는 모든 박스 기록해야 함)
                                        self._scan_csv_writer.writerow([
                                            name, pan_deg, tilt_deg, m_cx, m_cy, 
                                            0, 0, 0.50, 0, W, H  # 임시값들
                                        ])
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
            if bgr is None: 
                return

            # 언디스토트 처리
            if self.ud_enable.get() and self.undistort_processor.is_loaded():
                bgr = self.undistort_processor.process(bgr, alpha=float(self.ud_alpha.get()))

            # YOLO 처리
            if self.yolo_enable.get() and self.yolo_processor.is_loaded():
                # 시각화 설정 업데이트
                self._update_yolo_visualization_settings()
                
                bgr = self.yolo_processor.process(
                    bgr,
                    conf=float(self.yolo_conf.get()),
                    iou=float(self.yolo_iou.get()),
                    imgsz=int(self.yolo_imgsz.get()),
                    stride=int(self.yolo_stride.get())
                )
                
                # 센터링 처리
                centroid = self.yolo_processor.get_last_centroid()
                if centroid is not None:
                    m_cx, m_cy, n_detections = centroid
                    H, W = bgr.shape[:2]
                    self._centering_on_centroid(m_cx, m_cy, W, H)
                    
                    # 포인팅 로그 기록
                    if self._pointing_logging and (self._pointing_log_writer is not None):
                        self._log_pointing_data(m_cx, m_cy, W, H, n_detections)

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(rgb)
            self._draw_preview_to_label(im)
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
    def _log_pointing_data(self, m_cx: float, m_cy: float, W: int, H: int, n_detections: int):
        """포인팅 좌표 로그 기록"""
        if not (self._pointing_logging and self._pointing_log_writer):
            return
            
        try:
            err_x = (W/2.0 - m_cx)
            err_y = (H/2.0 - m_cy)
            
            self._pointing_log_writer.writerow([
                datetime.now().isoformat(timespec="milliseconds"),
                f"{self._curr_pan:.3f}", f"{self._curr_tilt:.3f}",
                f"{m_cx:.3f}", f"{m_cy:.3f}",
                f"{err_x:.3f}", f"{err_y:.3f}",
                int(W), int(H), int(n_detections)
            ])
        except Exception as e:
            print("[Point] write err:", e)
def main():
    root = Tk()
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
