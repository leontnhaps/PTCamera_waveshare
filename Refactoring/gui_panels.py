# gui_panels.py
from tkinter import Frame, Label, Button, Scale, HORIZONTAL, IntVar, DoubleVar
from tkinter import ttk

class ManualPanel(Frame):
    """수동 제어 패널 (Manual / LED 탭)"""
    
    def __init__(self, parent, callbacks):
        super().__init__(parent)
        
        # 콜백 함수들 저장
        self.on_center = callbacks['on_center']
        self.on_apply_move = callbacks['on_apply_move'] 
        self.on_set_led = callbacks['on_set_led']
        
        # 변수들
        self.mv_pan = DoubleVar(value=0.0)
        self.mv_tilt = DoubleVar(value=0.0)
        self.mv_speed = IntVar(value=100)
        self.mv_acc = DoubleVar(value=1.0)
        self.led = IntVar(value=0)
        
        # UI 생성
        self._create_widgets()
    
    def _create_widgets(self):
        """위젯들 생성"""
        # Pan 슬라이더
        self._slider(0, "Pan", -180, 180, self.mv_pan, 0.5)
        # Tilt 슬라이더  
        self._slider(1, "Tilt", -30, 90, self.mv_tilt, 0.5)
        # Speed 슬라이더
        self._slider(2, "Speed", 0, 100, self.mv_speed, 1)
        # Accel 슬라이더
        self._slider(3, "Accel", 0, 1, self.mv_acc, 0.1)
        
        # 버튼들
        Button(self, text="Center (0,0)", command=self.on_center)\
            .grid(row=4, column=0, sticky="w", pady=4)
        Button(self, text="Apply Move", command=self._handle_apply_move)\
            .grid(row=4, column=1, sticky="e", pady=4)
            
        # LED 슬라이더
        self._slider(5, "LED", 0, 255, self.led, 1)
        Button(self, text="Set LED", command=self._handle_set_led)\
            .grid(row=6, column=1, sticky="e", pady=4)
    
    def _slider(self, row, label, min_val, max_val, var, resolution):
        """슬라이더 생성 헬퍼"""
        Label(self, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=2)
        Scale(self, from_=min_val, to=max_val, orient=HORIZONTAL, resolution=resolution, 
              length=360, variable=var).grid(row=row, column=1, padx=6)
    
    def _handle_apply_move(self):
        """Apply Move 버튼 클릭 처리"""
        self.on_apply_move({
            'pan': float(self.mv_pan.get()),
            'tilt': float(self.mv_tilt.get()),
            'speed': int(self.mv_speed.get()),
            'acc': float(self.mv_acc.get())
        })
    
    def _handle_set_led(self):
        """Set LED 버튼 클릭 처리"""
        self.on_set_led(int(self.led.get()))


class ScanPanel(Frame):
    """스캔 제어 패널"""
    
    def __init__(self, parent, callbacks):
        super().__init__(parent)
        
        # 콜백 함수들
        self.on_start_scan = callbacks['on_start_scan']
        self.on_stop_scan = callbacks['on_stop_scan']
        
        # 스캔 파라미터 변수들
        self.pan_min = IntVar(value=-180)
        self.pan_max = IntVar(value=180)
        self.pan_step = IntVar(value=15)
        self.tilt_min = IntVar(value=-30)
        self.tilt_max = IntVar(value=90)
        self.tilt_step = IntVar(value=15)
        self.width = IntVar(value=2592)
        self.height = IntVar(value=1944)
        self.quality = IntVar(value=90)
        self.speed = IntVar(value=100)
        self.acc = DoubleVar(value=1.0)
        self.settle = DoubleVar(value=0.25)
        self.hard_stop = BooleanVar(value=False)
        
        # UI 생성
        self._create_widgets()
        
    def _create_widgets(self):
        """위젯들 생성"""
        # 파라미터 입력들
        self._row(0, "Pan min/max/step", self.pan_min, self.pan_max, self.pan_step)
        self._row(1, "Tilt min/max/step", self.tilt_min, self.tilt_max, self.tilt_step)  
        self._row(2, "Resolution (w×h)", self.width, self.height, None, ("W","H",""))
        self._entry(3, "Quality(%)", self.quality)
        self._entry(4, "Speed", self.speed)
        self._entry(5, "Accel", self.acc)
        self._entry(6, "Settle(s)", self.settle)
        
        # Hard stop 체크박스
        from tkinter import Checkbutton
        Checkbutton(self, text="Hard stop(정지 펄스)", variable=self.hard_stop)\
            .grid(row=7, column=1, sticky="w", padx=4, pady=2)
        
        # 버튼들과 프로그레스바
        ops = Frame(self)
        ops.grid(row=8, column=0, columnspan=4, sticky="w", pady=6)
        
        Button(ops, text="Start Scan", command=self._handle_start_scan)\
            .pack(side="left", padx=4)
        Button(ops, text="Stop Scan", command=self.on_stop_scan)\
            .pack(side="left", padx=4)
            
        # 프로그레스바 (나중에 App에서 접근할 수 있도록 self로 저장)
        self.prog = ttk.Progressbar(ops, orient=HORIZONTAL, length=280, mode="determinate")
        self.prog.pack(side="left", padx=10)
        self.prog_lbl = Label(ops, text="0 / 0")
        self.prog_lbl.pack(side="left")
        self.last_lbl = Label(ops, text="Last: -") 
        self.last_lbl.pack(side="left", padx=10)
        self.dl_lbl = Label(ops, text="DL 0")
        self.dl_lbl.pack(side="left")
        
    def _row(self, r, label, v1, v2, v3=None, caps=("min","max","step")):
        """3개 입력 필드 행 생성"""
        Label(self, text=label).grid(row=r, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(self, width=8, textvariable=v1).grid(row=r, column=1, sticky="w", padx=4)
        ttk.Entry(self, width=8, textvariable=v2).grid(row=r, column=2, sticky="w", padx=4)
        if v3 is not None:
            ttk.Entry(self, width=8, textvariable=v3).grid(row=r, column=3, sticky="w", padx=4)
    
    def _entry(self, r, label, var):
        """단일 입력 필드 행 생성"""
        Label(self, text=label).grid(row=r, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(self, width=8, textvariable=var).grid(row=r, column=1, sticky="w", padx=4)
        
    def _handle_start_scan(self):
        """Start Scan 버튼 클릭 처리"""
        params = {
            'pan_min': self.pan_min.get(),
            'pan_max': self.pan_max.get(), 
            'pan_step': self.pan_step.get(),
            'tilt_min': self.tilt_min.get(),
            'tilt_max': self.tilt_max.get(),
            'tilt_step': self.tilt_step.get(),
            'width': self.width.get(),
            'height': self.height.get(),
            'quality': self.quality.get(),
            'speed': self.speed.get(),
            'acc': float(self.acc.get()),
            'settle': float(self.settle.get()),
            'hard_stop': self.hard_stop.get()
        }
        self.on_start_scan(params)