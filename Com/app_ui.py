#!/usr/bin/env python3
"""
GUI layout and initialization
Separates UI setup from business logic
"""

from tkinter import Tk, Label, Button, Frame, Checkbutton, ttk, StringVar, IntVar, DoubleVar, BooleanVar
from gui_parts import ScrollFrame
import pathlib


class AppUIMixin:
    """GUI layout and widget initialization"""
    
    def setup_ui(self):
        """Initialize all UI components"""
        # Notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=4, pady=4)
        
        # Create tabs
        self._setup_scan_tab()
        self._setup_manual_tab()
        self._setup_misc_tab()
        self._setup_pointing_tab()
        
        # Preview box (common)
        self._setup_preview_box()
    
    # ========== Scan Tab ==========
    
    def _setup_scan_tab(self):
        """Setup scan tab layout"""
        tab_scan = Frame(self.notebook)
        self.notebook.add(tab_scan, text="Scan")
        
        # ScrollFrame for scan controls
        sf = ScrollFrame(tab_scan)
        sf.pack(fill="both", expand=True)
        f = sf.body
        
        r = 0
        Label(f, text="=== Pan/Tilt Range ===", font=("",10,"bold")).grid(row=r, column=0, columnspan=4, sticky="w", padx=4, pady=4)
        r += 1
        self._row(f, r, "Pan (deg)", self.scan_pan_min, self.scan_pan_max, self.scan_pan_step); r += 1
        self._row(f, r, "Tilt (deg)", self.scan_tilt_min, self.scan_tilt_max, self.scan_tilt_step); r += 1
        
        r += 1
        Label(f, text="=== Capture Settings ===", font=("",10,"bold")).grid(row=r, column=0, columnspan=4, sticky="w", padx=4, pady=4)
        r += 1
        self._row(f, r, "Resolution", self.width, self.height); r += 1
        self._entry(f, r, "Quality (1-100)", self.quality); r += 1
        self._entry(f, r, "Speed", self.speed); r += 1
        self._entry(f, r, "Accel", self.acc); r += 1
        self._entry(f, r, "LED Settle (s)", self.led_settle); r += 1
        
        r += 1
        Label(f, text="=== YOLO Settings ===", font=("",10,"bold")).grid(row=r, column=0, columnspan=4, sticky="w", padx=4, pady=4)
        r += 1
        Label(f, text="Weights Path:").grid(row=r, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(f, width=40, textvariable=self.yolo_wpath).grid(row=r, column=1, columnspan=3, sticky="ew", padx=4)
        r += 1
        Button(f, text="Browse YOLO Weights...", command=self.load_yolo_weights).grid(row=r, column=0, columnspan=4, sticky="ew", padx=4, pady=4)
        r += 1
        
        r += 1
        Button(f, text="START SCAN", bg="green", fg="white", font=("",12,"bold"), command=self.start_scan).grid(row=r, column=0, columnspan=4, sticky="ew", padx=4, pady=8)
        r += 1
        Button(f, text="STOP SCAN", bg="red", fg="white", font=("",12,"bold"), command=self.stop_scan).grid(row=r, column=0, columnspan=4, sticky="ew", padx=4, pady=8)
        r += 1
        
        # Progress
        ops = Frame(f)
        ops.grid(row=r, column=0, columnspan=4, sticky="ew", padx=4, pady=4)
        self.prog_lbl = Label(ops, text="0 / 0")
        self.prog_lbl.pack(side="left")
        self.last_lbl = Label(ops, text="Last: -")
        self.last_lbl.pack(side="left", padx=10)
    
    # ========== Manual Tab ==========
    
    def _setup_manual_tab(self):
        """Setup manual control tab"""
        tab_man = Frame(self.notebook)
        self.notebook.add(tab_man, text="Manual")
        
        sf = ScrollFrame(tab_man)
        sf.pack(fill="both", expand=True)
        f = sf.body
        
        r = 0
        Label(f, text="=== Manual Controls ===", font=("",10,"bold")).grid(row=r, column=0, columnspan=4, sticky="w", padx=4, pady=4)
        r += 1
        
        self._entry(f, r, "Pan (deg)", self.man_pan); r += 1
        self._entry(f, r, "Tilt (deg)", self.man_tilt); r += 1
        self._entry(f, r, "Speed", self.man_speed); r += 1
        self._entry(f, r, "Accel", self.man_acc); r += 1
        Button(f, text="MOVE", bg="blue", fg="white", command=self.apply_move).grid(row=r, column=0, columnspan=4, sticky="ew", padx=4, pady=4)
        r += 1
        
        Button(f, text="CENTER (0,0)", command=self.center).grid(row=r, column=0, columnspan=4, sticky="ew", padx=4, pady=4)
        r += 1
        
        r += 1
        Label(f, text="=== LED/Laser ===", font=("",10,"bold")).grid(row=r, column=0, columnspan=4, sticky="w", padx=4, pady=4)
        r += 1
        self._entry(f, r, "LED Value (0-255)", self.led_val); r += 1
        Button(f, text="SET LED", command=self.set_led).grid(row=r, column=0, columnspan=4, sticky="ew", padx=4, pady=4)
        r += 1
        
        laser_frame = Frame(f)
        laser_frame.grid(row=r, column=0, columnspan=4, sticky="ew", padx=4, pady=4)
        Checkbutton(laser_frame, text="Laser ON", variable=self.laser_on).pack(side="left")
        Button(laser_frame, text="Toggle Laser", command=self.toggle_laser).pack(side="left", padx=10)
        r += 1
        
        r += 1
        Label(f, text="=== Single Snap ===", font=("",10,"bold")).grid(row=r, column=0, columnspan=4, sticky="w", padx=4, pady=4)
        r += 1
        Button(f, text="SNAP ONE", bg="orange", command=self.snap_one).grid(row=r, column=0, columnspan=4, sticky="ew", padx=4, pady=4)
    
    # ========== Misc Tab ==========
    
    def _setup_misc_tab(self):
        """Setup misc settings tab"""
        tab_misc = Frame(self.notebook)
        self.notebook.add(tab_misc, text="Misc")
        
        sf = ScrollFrame(tab_misc)
        sf.pack(fill="both", expand=True)
        f = sf.body
        
        r = 0
        Label(f, text="=== Output Directory ===", font=("",10,"bold")).grid(row=r, column=0, columnspan=4, sticky="w", padx=4, pady=4)
        r += 1
        ttk.Entry(f, width=40, textvariable=self.outdir).grid(row=r, column=0, columnspan=3, sticky="ew", padx=4)
        Button(f, text="Browse", command=self.choose_outdir).grid(row=r, column=3, padx=4)
        r += 1
        
        r += 1
        Label(f, text="=== Calibration (Undistort) ===", font=("",10,"bold")).grid(row=r, column=0, columnspan=4, sticky="w", padx=4, pady=4)
        r += 1
        Button(f, text="Load calib.npz", command=lambda: self.load_npz()).grid(row=r, column=0, columnspan=4, sticky="ew", padx=4, pady=4)
        r += 1
        Checkbutton(f, text="Enable Undistort", variable=self.ud_enable).grid(row=r, column=0, columnspan=4, sticky="w", padx=4, pady=2)
        r += 1
        self._slider(f, r, "Alpha (0=crop, 1=keep all)", 0.0, 1.0, self.ud_alpha, 0.01); r += 1
        
        r += 1
        Label(f, text="=== Preview ===", font=("",10,"bold")).grid(row=r, column=0, columnspan=4, sticky="w", padx=4, pady=4)
        r += 1
        Checkbutton(f, text="Enable Preview", variable=self.preview_enable, command=self.toggle_preview).grid(row=r, column=0, columnspan=4, sticky="w", padx=4, pady=2)
        r += 1
        self._row(f, r, "Preview Size", self.preview_w, self.preview_h); r += 1
        self._entry(f, r, "FPS", self.preview_fps); r += 1
        self._entry(f, r, "Quality", self.preview_q); r += 1
        Button(f, text="Apply Preview Settings", command=self.apply_preview_size).grid(row=r, column=0, columnspan=4, sticky="ew", padx=4, pady=4)
    
    # ========== Pointing Tab ==========
    
    def _setup_pointing_tab(self):
        """Setup pointing mode tab"""
        tab_point = Frame(self.notebook)
        self.notebook.add(tab_point, text="Pointing")
        
        sf = ScrollFrame(tab_point)
        sf.pack(fill="both", expand=True)
        f = sf.body
        
        r = 0
        Label(f, text="=== Pointing Mode ===", font=("",10,"bold")).grid(row=r, column=0, columnspan=4, sticky="w", padx=4, pady=4)
        r += 1
        Checkbutton(f, text="Enable Pointing Mode", variable=self.pointing_enable, command=self.on_pointing_toggle).grid(row=r, column=0, columnspan=4, sticky="w", padx=4, pady=2)
        r += 1
        
        self._entry(f, r, "Interval (s)", self.pointing_interval); r += 1
        self._entry(f, r, "Pixel Tolerance", self.pointing_px_tol); r += 1
        self._entry(f, r, "Min Stable Frames", self.pointing_min_frames); r += 1
        self._entry(f, r, "Max Step (deg)", self.pointing_max_step); r += 1
        self._entry(f, r, "Cooldown (s)", self.pointing_cooldown); r += 1
        self._entry(f, r, "ROI Size (px)", self.pointing_roi_size); r += 1
        
        r += 1
        Label(f, text="=== CSV Analysis ===", font=("",10,"bold")).grid(row=r, column=0, columnspan=4, sticky="w", padx=4, pady=4)
        r += 1
        ttk.Entry(f, width=40, textvariable=self.point_csv_path).grid(row=r, column=0, columnspan=3, sticky="ew", padx=4)
        Button(f, text="Browse", command=self.pointing_choose_csv).grid(row=r, column=3, padx=4)
        r += 1
        
        self._entry(f, r, "Conf Min", self.point_conf_min); r += 1
        self._entry(f, r, "Min Samples", self.point_min_samples); r += 1
        Button(f, text="Compute Target", command=self.pointing_compute).grid(row=r, column=0, columnspan=4, sticky="ew", padx=4, pady=4)
        r += 1
        
        self._entry(f, r, "Pan Target", self.point_pan_target); r += 1
        self._entry(f, r, "Tilt Target", self.point_tilt_target); r += 1
        self._entry(f, r, "Move Speed", self.point_speed); r += 1
        self._entry(f, r, "Move Accel", self.point_acc); r += 1
        Button(f, text="Move to Target", bg="green", fg="white", command=self.pointing_move).grid(row=r, column=0, columnspan=4, sticky="ew", padx=4, pady=4)
        r += 1
        
        self.point_result_lbl = Label(f, text="", fg="blue")
        self.point_result_lbl.grid(row=r, column=0, columnspan=4, sticky="w", padx=4, pady=4)
        
        # Debug Preview (right side)
        debug_frame = Frame(tab_point, bg="#111", width=420, height=420, relief="solid", borderwidth=2)
        debug_frame.pack(side="right", padx=10, pady=10)
        debug_frame.pack_propagate(False)
        
        Label(debug_frame, text="Debug Target Preview", bg="#111", fg="white", font=("", 10, "bold")).pack(pady=5)
        self.debug_preview_label = Label(debug_frame, bg="#111", fg="#666", text="(Waiting for detection...)")
        self.debug_preview_label.pack(fill="both", expand=True, padx=10, pady=10)
        self.debug_preview_img = None
    
    # ========== Preview Box ==========
    
    def _setup_preview_box(self):
        """Setup preview display box"""
        preview_frame = Frame(self.root, bg="black", width=self.PREV_W, height=self.PREV_H)
        preview_frame.pack(side="bottom", fill="both", expand=False, padx=4, pady=4)
        preview_frame.pack_propagate(False)
        
        self.preview_label = Label(preview_frame, bg="black")
        self.preview_label.pack(fill="both", expand=True)
