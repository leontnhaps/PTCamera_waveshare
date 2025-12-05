#!/usr/bin/env python3
"""
GUI layout and initialization
Separates UI setup from business logic
"""

from tkinter import Tk, Label, Button, Frame, Checkbutton, ttk, StringVar, IntVar, DoubleVar, BooleanVar, Scale, HORIZONTAL
import tkinter as tk
from gui_parts import ScrollFrame
import pathlib

class AppUIMixin:
    """GUI layout and widget initialization"""
    
    def setup_ui(self):
        """Initialize all UI components"""
        self._setup_scan_tab()
        self._setup_manual_tab()
        self._setup_misc_tab()
        self._setup_pointing_tab()
    
    def _setup_scan_tab(self):
        for i in range(self.notebook.index("end")):
            if self.notebook.tab(i, "text") == "Scan":
                tab_scan = self.notebook.nametowidget(self.notebook.tabs()[i])
                break
        
        for widget in tab_scan.winfo_children(): widget.destroy()
        
        r = 0
        self._row(tab_scan, r, "Pan min/max/step", self.pan_min, self.pan_max, self.pan_step); r += 1
        self._row(tab_scan, r, "Tilt min/max/step", self.tilt_min, self.tilt_max, self.tilt_step); r += 1
        self._row(tab_scan, r, "Resolution (w×h)", self.width, self.height, None, ("W","H","")); r += 1
        self._entry(tab_scan, r, "Quality(%)", self.quality); r += 1
        self._entry(tab_scan, r, "Speed", self.speed); r += 1
        self._entry(tab_scan, r, "Accel", self.acc); r += 1
        self._entry(tab_scan, r, "Settle(s)", self.settle); r += 1
        self._entry(tab_scan, r, "LED Settle(s)", self.led_settle); r += 1
        Checkbutton(tab_scan, text="Hard stop(정지 펄스)", variable=self.hard_stop).grid(row=r, column=1, sticky="w", padx=4, pady=2); r += 1
        
        ops = Frame(tab_scan); ops.grid(row=r, column=0, columnspan=4, sticky="w", pady=6)
        Button(ops, text="Start Scan", command=self.start_scan).pack(side="left", padx=4)
        Button(ops, text="Stop Scan", command=self.stop_scan).pack(side="left", padx=4)
        self.prog = ttk.Progressbar(ops, orient=HORIZONTAL, length=280, mode="determinate"); self.prog.pack(side="left", padx=10)
        self.prog_lbl = Label(ops, text="0 / 0"); self.prog_lbl.pack(side="left")
        self.last_lbl = Label(ops, text="Last: -"); self.last_lbl.pack(side="left", padx=10)
        self.dl_lbl = Label(ops, text="DL 0"); self.dl_lbl.pack(side="left", padx=10)
    
    def _setup_manual_tab(self):
        for i in range(self.notebook.index("end")):
            if self.notebook.tab(i, "text") == "Manual / LED":
                tab_manual = self.notebook.nametowidget(self.notebook.tabs()[i])
                break
        
        for widget in tab_manual.winfo_children(): widget.destroy()
        
        self._slider(tab_manual, 0, "Pan", -180, 180, self.mv_pan, 0.5)
        self._slider(tab_manual, 1, "Tilt", -30, 90, self.mv_tilt, 0.5)
        self._slider(tab_manual, 2, "Speed", 0, 100, self.mv_speed, 1)
        self._slider(tab_manual, 3, "Accel", 0, 1, self.mv_acc, 0.1)
        Button(tab_manual, text="Center (0,0)", command=self.center).grid(row=4, column=0, sticky="w", pady=4)
        Button(tab_manual, text="Apply Move", command=self.apply_move).grid(row=4, column=1, sticky="e", pady=4)
        self._slider(tab_manual, 5, "LED", 0, 255, self.led, 1)
        Button(tab_manual, text="Set LED", command=self.set_led).grid(row=6, column=1, sticky="e", pady=4)
        Button(tab_manual, text="Laser ON/OFF", command=self.toggle_laser).grid(row=6, column=2, sticky="w", padx=4, pady=4)
    
    def _setup_misc_tab(self):
        for i in range(self.notebook.index("end")):
            if self.notebook.tab(i, "text") == "Preview & Settings":
                tab_misc = self.notebook.nametowidget(self.notebook.tabs()[i])
                break
        
        for widget in tab_misc.winfo_children(): widget.destroy()
        
        misc_sf = ScrollFrame(tab_misc)
        misc_sf.pack(fill="both", expand=True)
        misc = misc_sf.body
        
        row = 0
        Checkbutton(misc, text="Live Preview", variable=self.preview_enable, command=self.toggle_preview).grid(row=row, column=0, sticky="w", pady=2); row += 1
        self._row(misc, row, "Preview w/h/-", self.preview_w, self.preview_h, None, ("W","H","")); row += 1
        self._entry(misc, row, "Preview fps", self.preview_fps); row += 1
        self._entry(misc, row, "Preview quality", self.preview_q); row += 1
        Button(misc, text="Apply Preview Size", command=self.apply_preview_size).grid(row=row, column=1, sticky="w", pady=4); row += 1
        
        ttk.Separator(misc, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", pady=(8,6)); row += 1
        Checkbutton(misc, text="Undistort preview (use calib.npz)", variable=self.ud_enable).grid(row=row, column=0, sticky="w"); row += 1
        Button(misc, text="Load calib.npz", command=self.load_npz).grid(row=row, column=0, sticky="w", pady=2)
        Checkbutton(misc, text="Also save undistorted copy", variable=self.ud_save_copy).grid(row=row, column=1, sticky="w", pady=2); row += 1
        Label(misc, text="Alpha/Balance (0~1)").grid(row=row, column=0, sticky="w")
        Scale(misc, from_=0.0, to=1.0, orient=HORIZONTAL, resolution=0.01, length=200,
              variable=self.ud_alpha, command=lambda v: setattr(self, "_ud_src_size", None)).grid(row=row, column=1, sticky="w"); row += 1
        
        ttk.Separator(misc, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", pady=(8,6)); row += 1
        Label(misc, text="YOLO 가중치 (.pt)").grid(row=row, column=0, sticky="w")
        Button(misc, text="Load YOLO", command=self.load_yolo_weights).grid(row=row, column=1, sticky="w", pady=2); row += 1
        
        for c in range(4): misc.grid_columnconfigure(c, weight=1)

    def _setup_pointing_tab(self):
        """Pointing Tab Layout (Restored from Com_main.py)"""
        for i in range(self.notebook.index("end")):
            if self.notebook.tab(i, "text") == "Pointing":
                self.tab_point = self.notebook.nametowidget(self.notebook.tabs()[i])
                break
        
        for widget in self.tab_point.winfo_children(): widget.destroy()

        # Canvas & Scrollbar
        self.point_canvas = tk.Canvas(self.tab_point)
        self.point_scroll = ttk.Scrollbar(self.tab_point, orient="vertical", command=self.point_canvas.yview)
        self.point_scroll_frame = ttk.Frame(self.point_canvas)
        
        self.point_scroll_frame.bind("<Configure>", lambda e: self.point_canvas.configure(scrollregion=self.point_canvas.bbox("all")))
        self.point_canvas.create_window((0, 0), window=self.point_scroll_frame, anchor="nw")
        self.point_canvas.configure(yscrollcommand=self.point_scroll.set)
        
        # Mouse Wheel
        def _on_mousewheel(event): self.point_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        def _bind_mousewheel(event): self.point_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        def _unbind_mousewheel(event): self.point_canvas.unbind_all("<MouseWheel>")
        self.point_canvas.bind("<Enter>", _bind_mousewheel); self.point_canvas.bind("<Leave>", _unbind_mousewheel)
        self.point_scroll_frame.bind("<Enter>", _bind_mousewheel); self.point_scroll_frame.bind("<Leave>", _unbind_mousewheel)
        
        self.point_canvas.pack(side="left", fill="both", expand=True)
        self.point_scroll.pack(side="right", fill="y")
        
        # Grid Layout
        col1_frame = ttk.Frame(self.point_scroll_frame); col1_frame.grid(row=0, column=0, padx=5, pady=10, sticky="nsew")
        col2_frame = ttk.Frame(self.point_scroll_frame); col2_frame.grid(row=0, column=1, padx=5, pady=10, sticky="nsew")
        col3_frame = ttk.Frame(self.point_scroll_frame); col3_frame.grid(row=0, column=2, padx=5, pady=10, sticky="nsew")
        
        self.point_scroll_frame.grid_columnconfigure(0, weight=1)
        self.point_scroll_frame.grid_columnconfigure(1, weight=1)
        self.point_scroll_frame.grid_columnconfigure(2, weight=1)
        
        # 1. Pointing Settings
        point_set_frame = ttk.LabelFrame(col1_frame, text="Pointing Settings")
        point_set_frame.pack(padx=5, pady=5, fill="both", expand=True)
        def add_entry(parent, label, var, r):
            ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(parent, textvariable=var, width=10).grid(row=r, column=1, sticky="w", padx=5, pady=2)
        
        add_entry(point_set_frame, "Laser ROI Size (px):", self.pointing_roi_size, 0)
        ttk.Label(point_set_frame, text="--- Pointing Settings ---").grid(row=1, column=0, columnspan=2, pady=5)
        add_entry(point_set_frame, "Tolerance (px):", self.pointing_px_tol, 2)
        add_entry(point_set_frame, "Min Stable Frames:", self.pointing_min_frames, 3)
        add_entry(point_set_frame, "Max Step (deg):", self.pointing_max_step, 4)
        add_entry(point_set_frame, "Cooldown (ms):", self.pointing_cooldown, 5)
        add_entry(point_set_frame, "LED Settle (s):", self.led_settle, 6)
        
        # 2. Pointing Control
        point_ctrl_frame = ttk.LabelFrame(col2_frame, text="Pointing Control")
        point_ctrl_frame.pack(padx=5, pady=5, fill="both", expand=True)
        ttk.Checkbutton(point_ctrl_frame, text="Enable Pointing Mode", variable=self.pointing_enable, command=self.on_pointing_toggle).pack(anchor="w", padx=5, pady=5)
        
        # 3. CSV Analysis
        point_csv_frame = ttk.LabelFrame(col3_frame, text="CSV Analysis (Legacy)")
        point_csv_frame.pack(padx=5, pady=5, fill="both", expand=True)
        ttk.Label(point_csv_frame, textvariable=self.point_csv_path, wraplength=200).pack(anchor="w", padx=5, pady=2)
        
        ttk.Label(point_csv_frame, text="Conf Min:").pack(anchor="w", padx=5)
        ttk.Entry(point_csv_frame, textvariable=self.point_conf_min, width=15).pack(anchor="w", padx=5)
        ttk.Label(point_csv_frame, text="Min Samples:").pack(anchor="w", padx=5)
        ttk.Entry(point_csv_frame, textvariable=self.point_min_samples, width=15).pack(anchor="w", padx=5)
        
        self.point_result_lbl = ttk.Label(point_csv_frame, text="Result: -")
        self.point_result_lbl.pack(anchor="w", padx=5, pady=5)
        ttk.Button(point_csv_frame, text="Load CSV", command=self.pointing_choose_csv).pack(anchor="w", padx=5, pady=2)
        ttk.Button(point_csv_frame, text="Move to Target", command=self.pointing_move).pack(anchor="w", padx=5, pady=5)
        
        # Debug Preview
        debug_frame = Frame(self.tab_point, bg="#111", width=420, height=420, relief="solid", borderwidth=2)
        debug_frame.pack(side="right", padx=10, pady=10)
        debug_frame.pack_propagate(False)
        Label(debug_frame, text="Debug Target Preview", bg="#111", fg="white", font=("", 10, "bold")).pack(pady=5)
        self.debug_preview_label = Label(debug_frame, bg="#111", fg="#666", text="(Waiting for detection...)")
        self.debug_preview_label.pack(fill="both", expand=True, padx=10, pady=10)