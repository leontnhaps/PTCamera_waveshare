#!/usr/bin/env python3
# pc_gui.py — GUI client connecting to pc_server.py (not to Pi agent)

import json, socket, threading, queue, pathlib, struct
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from tkinter import Tk, Label, Button, Scale, HORIZONTAL, IntVar, DoubleVar, Frame, Checkbutton, BooleanVar, filedialog, StringVar
from tkinter import ttk
from PIL import Image, ImageTk
import tkinter as tk  # ← 추가
import os, re, csv, time   # ← 추가

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

# ==== YOLO (for LED difference detection) ====
try:
    from ultralytics import YOLO
    _YOLO_OK = True
except Exception:
    YOLO = None
    _YOLO_OK = False
# =============================================

# ==== [NEW] Tiling Helper Functions ====
def non_max_suppression(boxes, scores, iou_threshold):
    # OpenCV NMS 사용
    if len(boxes) == 0:
        return []
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou_threshold)
    if len(indices) > 0:
        return indices.flatten()
    return []

# ==== [NEW] 배치 크기 캐싱 (스캔 시 효율성 향상) ====
_SAHI_OPTIMAL_BATCH_SIZE = None  # 최적 배치 크기 캐시
# ==========================================================

def predict_with_tiling(model, img, rows=2, cols=3, overlap=0.15, conf=0.25, iou=0.45, device='cuda'):
    """
    이미지를 타일로 쪼개서 예측 후 결과 병합
    rows, cols: 행/열 개수 (2x3 = 6등분)
    overlap: 타일 간 겹치는 비율 (0.15 = 15%)
    [NEW] 🚀 적응형 배치 + 캐싱! (한 번 찾으면 계속 재사용)
    """
    global _SAHI_OPTIMAL_BATCH_SIZE
    
    H, W = img.shape[:2]
    
    # 타일 크기 계산 (겹침 포함)
    tile_h = int(H / rows)
    tile_w = int(W / cols)
    
    # 겹침 크기
    ov_h = int(tile_h * overlap)
    ov_w = int(tile_w * overlap)
    
    # 실제 타일 크기 (겹침 포함)
    step_h = tile_h - ov_h
    step_w = tile_w - ov_w
    
    # 타일 좌표 생성
    tiles = []
    for y in range(0, H, step_h):
        for x in range(0, W, step_w):
            y2 = min(y + tile_h, H)
            x2 = min(x + tile_w, W)
            y1 = max(0, y2 - tile_h)
            x1 = max(0, x2 - tile_w)
            tiles.append((x1, y1, x2, y2))
            if x2 >= W: break
        if y2 >= H: break

    # 타일 이미지 미리 추출
    tile_images = [img[ty1:ty2, tx1:tx2] for (tx1, ty1, tx2, ty2) in tiles]
    
    # 적응형 배치 크기 (캐시된 값부터 시작)
    all_batch_sizes = [6, 3, 2, 1]
    
    # 캐시된 최적 배치 크기가 있으면 그것부터 시작
    if _SAHI_OPTIMAL_BATCH_SIZE is not None:
        # 캐시된 크기부터 그 이하만 시도
        idx = all_batch_sizes.index(_SAHI_OPTIMAL_BATCH_SIZE)
        batch_sizes = all_batch_sizes[idx:]
    else:
        # 캐시 없으면 전체 시도
        batch_sizes = all_batch_sizes
    
    batch_results = None
    
    for batch_size in batch_sizes:
        try:
            # GPU 캐시 정리
            if device == 'cuda':
                import torch
                torch.cuda.empty_cache()
            
            # 배치로 나누어 추론
            all_results = []
            for i in range(0, len(tile_images), batch_size):
                batch = tile_images[i:i+batch_size]
                results = model.predict(batch, conf=conf, iou=iou, device=device, verbose=False)
                all_results.extend(results)
            
            batch_results = all_results
            
            # 성공하면 캐시에 저장
            if _SAHI_OPTIMAL_BATCH_SIZE != batch_size:
                _SAHI_OPTIMAL_BATCH_SIZE = batch_size
                print(f"[SAHI] ✅ 배치 크기 {batch_size}로 성공! (캐시에 저장)")
            else:
                print(f"[SAHI] ✅ 배치 크기 {batch_size}로 성공!")
            
            break  # 성공하면 루프 종료
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                if batch_size == 1:
                    # 1개도 안 되면 진짜 문제
                    print(f"[SAHI] ❌ GPU 메모리 심각 부족! (배치 크기 1도 실패)")
                    raise
                else:
                    print(f"[SAHI] ⚠️ 배치 크기 {batch_size} OOM, {batch_sizes[batch_sizes.index(batch_size)+1]}로 재시도...")
                    # 캐시 무효화 (메모리 상황이 바뀜)
                    _SAHI_OPTIMAL_BATCH_SIZE = None
                    continue
            else:
                # 다른 에러는 그대로 raise
                raise
    
    # 결과 처리
    all_boxes = []
    all_scores = []
    all_classes = []
    
    for i, (results, (tx1, ty1, tx2, ty2)) in enumerate(zip(batch_results, tiles)):
        if results.boxes:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                c = float(box.conf.cpu().numpy().item())
                cls = int(box.cls.cpu().numpy().item())
                
                # 글로벌 좌표로 변환
                gx1 = x1 + tx1
                gy1 = y1 + ty1
                gx2 = x2 + tx1
                gy2 = y2 + ty1
                
                w = gx2 - gx1
                h = gy2 - gy1
                
                all_boxes.append([int(gx1), int(gy1), int(w), int(h)])
                all_scores.append(c)
                all_classes.append(cls)

    # 전체 결과에 대해 NMS 수행
    if not all_boxes:
        return [], [], []

    indices = non_max_suppression(all_boxes, all_scores, iou_threshold=0.3)
    
    final_boxes = []
    final_scores = []
    final_classes = []
    
    for idx in indices:
        final_boxes.append(all_boxes[idx])
        final_scores.append(all_scores[idx])
        final_classes.append(all_classes[idx])
        
    return final_boxes, final_scores, final_classes
# =======================================

SERVER_HOST = "127.0.0.1"
GUI_CTRL_PORT = 7600
GUI_IMG_PORT  = 7601

DEFAULT_OUT_DIR = pathlib.Path(f"captures_gui_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)

ui_q: "queue.Queue[tuple[str,object]]" = queue.Queue()

# ==== [NEW] 실시간 YOLO 파이프라인 ====
_yolo_model = None  # 전역 YOLO 모델 (App에서 로드)
_yolo_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="YOLO")
_scan_led_pairs = {}  # {session: {(pan, tilt): {'off': data, 'on': data}}}
_scan_csv_files = {}  # {session: csv_writer}
_scan_lock = threading.Lock()
# =========================================

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
    
    def _process_scan_image(self, name, data):
        """실시간 스캔 이미지 처리 (LED ON/OFF 쌍 감지 및 YOLO)"""
        # 파일명 파싱: scan_20241203_120000_pan-30_tilt15_off.jpg
        import re
        match = re.match(r'(scan_\d{8}_\d{6})_pan([\-\d]+)_tilt([\-\d]+)_(off|on)\.jpg', name)
        if not match:
            return False  # 스캔 이미지 아님
        
        session, pan, tilt, led_state = match.groups()
        pan, tilt = int(pan), int(tilt)
        
        with _scan_lock:
            # 세션별 LED 쌍 딕셔너리 초기화
            if session not in _scan_led_pairs:
                _scan_led_pairs[session] = {}
                # CSV 파일 생성
                csv_path = self.outdir / f"{session}_results.csv"
                csv_f = open(csv_path, 'w', newline='', encoding='utf-8')
                csv_writer = csv.writer(csv_f)
                csv_writer.writerow(['pan', 'tilt', 'class', 'confidence', 'x', 'y', 'w', 'h'])
                _scan_csv_files[session] = (csv_f, csv_writer)
            
            # 위치별 LED 쌍 저장
            pos_key = (pan, tilt)
            if pos_key not in _scan_led_pairs[session]:
                _scan_led_pairs[session][pos_key] = {}
            
            _scan_led_pairs[session][pos_key][led_state] = data
            
            # LED OFF/ON 쌍이 완성되었는지 확인
            pair = _scan_led_pairs[session][pos_key]
            if 'off' in pair and 'on' in pair:
                # 쌍 완성! 백그라운드에서 YOLO 처리
                off_data = pair['off']
                on_data = pair['on']
                csv_writer = _scan_csv_files[session][1]
                
                # 백그라운드 YOLO 처리 제출
                _yolo_executor.submit(
                    self._process_led_pair,
                    session, pan, tilt, off_data, on_data, csv_writer
                )
                
                # 처리된 쌍 제거 (메모리 절약)
                del _scan_led_pairs[session][pos_key]
                
                print(f"[SCAN] LED 쌍 수신: pan={pan}, tilt={tilt} → YOLO 처리 중...")
        
        return True
    
    def _process_led_pair(self, session, pan, tilt, off_data, on_data, csv_writer):
        """LED ON/OFF 쌍에 대해 YOLO 처리 (백그라운드)"""
        global _yolo_model
        
        try:
            # YOLO 모델 체크
            if _yolo_model is None:
                print(f"[SCAN] YOLO 모델 미로드, 스킵: pan={pan}, tilt={tilt}")
                return
            
            # 이미지 디코딩
            import numpy as np
            import cv2
            
            off_arr = np.frombuffer(off_data, dtype=np.uint8)
            on_arr = np.frombuffer(on_data, dtype=np.uint8)
            
            off_img = cv2.imdecode(off_arr, cv2.IMREAD_COLOR)
            on_img = cv2.imdecode(on_arr, cv2.IMREAD_COLOR)
            
            # 차분 이미지 계산
            diff_img = cv2.absdiff(on_img, off_img)
            
            # 🚀 SAHI 타일링으로 YOLO 처리
            boxes, scores, classes = predict_with_tiling(
                _yolo_model, diff_img,
                rows=2, cols=3,
                overlap=0.15,
                conf=0.25,
                iou=0.45,
                device='cuda' if _TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
            )
            
            # CSV에 결과 기록
            with _scan_lock:
                csv_f, writer = _scan_csv_files[session]
                for box, score, cls in zip(boxes, scores, classes):
                    x, y, w, h = box
                    writer.writerow([pan, tilt, cls, f"{score:.4f}", x, y, w, h])
                csv_f.flush()  # 즉시 디스크에 기록
            
            print(f"[SCAN] YOLO 완료: pan={pan}, tilt={tilt}, 검출={len(boxes)}개")
            
        except Exception as e:
            import traceback
            print(f"[SCAN] YOLO 에러 (pan={pan}, tilt={tilt}): {e}")
            traceback.print_exc()
    
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
                    # 스캔 이미지인지 확인 및 실시간 처리
                    is_scan = self._process_scan_image(name, data)
                    
                    # 일반 저장
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
        self.ud_enable    = BooleanVar(value=True)
        self.ud_save_copy = BooleanVar(value=True)
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

        # ==== YOLO 관련 변수 ====
        self.yolo_wpath = StringVar(value="yolov11m_diff.pt")  # YOLO 가중치 경로
        self._scan_yolo_conf = 0.50  # YOLO confidence threshold
        self._scan_yolo_imgsz = 832  # YOLO image size
        # ========================

        print(f"[INFO] cv2.cuda={self._use_cv2_cuda}, torch_cuda={self._torch_cuda}")

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

        self.root.after(60, self._poll)
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

        # (선택) 현재 명령 각도 기억
        self._curr_pan = 0.0
        self._curr_tilt = 0.0
        
        self._fits_h = {}
        self._fits_v = {}
        # Pointing 탭에 추가 UI
        # centering state
        self._centering_state = 0 # 0:IDLE, 1:WAIT_ON, 2:WAIT_OFF
        self._centering_on_img = None
        self._centering_off_img = None
        self._centering_stable_cnt = 0
        self._centering_last_ts = 0
        self._centering_ok_frames = 0
        self._centering_last_ms = 0
        
        # Pointing state
        self._pointing_state = 0 # 0:IDLE, 1:LASER_ON, 2:LASER_OFF, 3:LED_ON, 4:LED_OFF
        self._pointing_laser_on_img = None
        self._pointing_laser_off_img = None
        self._pointing_led_on_img = None
        self._pointing_led_off_img = None
        self._pointing_stable_cnt = 0
        self._pointing_last_ts = 0

        # [MOVED] Centering variables definition
        self.centering_enable   = BooleanVar(value=False)
        self.centering_px_tol   = IntVar(value=5)      # 중앙 판정 오차(px)
        self.centering_min_frames = IntVar(value=4)    # 연속 N프레임 만족 시 종료
        self.centering_max_step = DoubleVar(value=1.0) # 한번에 움직일 최대 각도(°)
        self.centering_cooldown = IntVar(value=250)    # 명령 간 최소 간격(ms)
        self.show_center_marker = BooleanVar(value=False)


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
        point_ctrl_frame = ttk.LabelFrame(self.point_scroll_frame, text="Pointing Control")
        point_ctrl_frame.pack(padx=10, pady=10, fill="x")
        
        self.pointing_enable = tk.BooleanVar(value=False)
        ttk.Checkbutton(point_ctrl_frame, text="Enable Pointing Mode", variable=self.pointing_enable, command=self.on_pointing_toggle).pack(anchor="w", padx=5, pady=5)
        
        # Pointing Settings (Editable)
        point_set_frame = ttk.LabelFrame(self.point_scroll_frame, text="Pointing Settings")
        point_set_frame.pack(padx=10, pady=10, fill="x")
        
        def add_entry(parent, label, var, row):
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(parent, textvariable=var, width=10).grid(row=row, column=1, sticky="w", padx=5, pady=2)

        self.pointing_roi_size = tk.IntVar(value=200)
        add_entry(point_set_frame, "Laser ROI Size (px):", self.pointing_roi_size, 0)
        
        ttk.Label(point_set_frame, text="--- Shared Settings ---").grid(row=1, column=0, columnspan=2, pady=5)
        add_entry(point_set_frame, "Tolerance (px):", self.centering_px_tol, 2)
        add_entry(point_set_frame, "Min Stable Frames:", self.centering_min_frames, 3)
        add_entry(point_set_frame, "Max Step (deg):", self.centering_max_step, 4)
        add_entry(point_set_frame, "Cooldown (ms):", self.centering_cooldown, 5)
        add_entry(point_set_frame, "Cooldown (ms):", self.centering_cooldown, 5)
        add_entry(point_set_frame, "LED Settle (s):", self.led_settle, 6)
        
        # [NEW] Centering & Marker Toggles in Settings
        ttk.Checkbutton(point_set_frame, text="Centering Mode (Live Refine)", variable=self.centering_enable, command=self.on_centering_toggle).grid(row=7, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        ttk.Checkbutton(point_set_frame, text="Show Center Marker", variable=self.show_center_marker).grid(row=8, column=0, columnspan=2, sticky="w", padx=5, pady=2)

        # CSV Analysis (Existing)
        point_csv_frame = ttk.LabelFrame(self.point_scroll_frame, text="CSV Analysis (Legacy)")
        point_csv_frame.pack(padx=10, pady=10, fill="x")
        
        ttk.Button(point_csv_frame, text="Select CSV", command=self.pointing_choose_csv).pack(anchor="w", padx=5, pady=2)
        self.point_csv_path = tk.StringVar()
        ttk.Label(point_csv_frame, textvariable=self.point_csv_path, wraplength=300).pack(anchor="w", padx=5, pady=2)
        
        ttk.Label(point_csv_frame, text="Conf Min:").pack(anchor="w", padx=5)
        self.point_conf_min = tk.StringVar(value="0.5")
        ttk.Entry(point_csv_frame, textvariable=self.point_conf_min, width=10).pack(anchor="w", padx=5)
        
        ttk.Label(point_csv_frame, text="Min Samples:").pack(anchor="w", padx=5)
        self.point_min_samples = tk.StringVar(value="5")
        ttk.Entry(point_csv_frame, textvariable=self.point_min_samples, width=10).pack(anchor="w", padx=5)
        
        ttk.Button(point_csv_frame, text="Compute Target", command=self.pointing_compute).pack(anchor="w", padx=5, pady=5)
        self.point_result_lbl = ttk.Label(point_csv_frame, text="Result: -")
        self.point_result_lbl.pack(anchor="w", padx=5, pady=5)
        
        # [RESTORED] Move to Target Button
        ttk.Button(point_csv_frame, text="Move to Target", command=self.pointing_move).pack(anchor="w", padx=5, pady=5)


        # [NEW] Auto-load calib.npz if exists
        if pathlib.Path("calib.npz").exists():
            self.load_npz("calib.npz")

    def run(self):
        self.root.mainloop()

    def load_npz(self, path=None):
        if path is None:
            path = filedialog.askopenfilename(filetypes=[("NPZ","*.npz")])
        if not path: return
        try:
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
        except Exception as e:
            print(f"[UD] Load failed: {e}")

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
        global _yolo_model
        
        path = filedialog.askopenfilename(filetypes=[("YOLO weights", "*.pt"), ("All files", "*.*")])
        if path:
            self.yolo_wpath.set(path)
            
            # YOLO 모델 로드
            if _YOLO_OK:
                try:
                    _yolo_model = YOLO(path)
                    ui_q.put(("toast", f"✅ YOLO 모델 로드 완료: {pathlib.Path(path).name}"))
                    print(f"[YOLO] 모델 로드 완료, 실시간 스캔 준비됨!")
                except Exception as e:
                    ui_q.put(("toast", f"❌ YOLO 로드 실패: {e}"))
                    _yolo_model = None
            else:
                ui_q.put(("toast", f"⚠️ YOLO 라이브러리 미설치"))

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
            "led_settle":float(self.led_settle.get()),
            "width":self.width.get(),"height":self.height.get(),"quality":self.quality.get(),
            "session":datetime.now().strftime("scan_%Y%m%d_%H%M%S"),
            "hard_stop":self.hard_stop.get()
        })
    def stop_scan(self):
        self.ctrl.send({"cmd":"scan_stop"})
        self.root.after(500, lambda: ui_q.put(("preview_on", None)))

    def on_centering_toggle(self):
        if self.centering_enable.get():
            ui_q.put(("toast", "🚀 Centering Mode Started"))
            self._centering_state = 0
            self._centering_stable_cnt = 0
            self._snap_center_on()
        else:
            ui_q.put(("preview_on", None))

    def on_pointing_toggle(self):
        if not self.pointing_enable.get():
            ui_q.put(("preview_on", None))
            # Laser OFF when stopping
            self.ctrl.send({"cmd":"laser", "value": 0})
            self.laser_on.set(False)
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
        self.ctrl.send({
            "cmd":"snap",
            "width":  self.width.get(),
            "height": self.height.get(),
            "quality":self.quality.get(),
            "save":   fname,
            "hard_stop": self.hard_stop.get()
        })

    # event loop
    # ==== [NEW] Centering Mode Logic ====
    def _start_centering_cycle(self):
        # 1. LED ON
        self._centering_state = 1 # WAIT_ON
        self.ctrl.send({"cmd":"led", "value":255})
        # Settle time wait -> Snap
        wait_ms = int(self.led_settle.get() * 1000)
        self.root.after(wait_ms, self._snap_center_on)

    def _snap_center_on(self):
        # 2. Snap ON image
        # save="center_on.jpg"로 요청하여 _poll에서 식별
        self.ctrl.send({
            "cmd":"snap",
            "width":  self.width.get(),
            "height": self.height.get(),
            "quality":self.quality.get(),
            "save":   "center_on.jpg",
            "hard_stop": False
        })

    def _snap_center_off(self):
        # 4. Snap OFF image
        self.ctrl.send({
            "cmd":"snap",
            "width":  self.width.get(),
            "height": self.height.get(),
            "quality":self.quality.get(),
            "save":   "center_off.jpg",
            "hard_stop": False
        })

    def _run_centering_logic(self, img_on, img_off):
        """백그라운드 스레드에서 실행되는 Centering 핵심 로직"""
        try:
            # 1. Undistort
            if self._ud_K is not None:
                img_on = self._undistort_bgr(img_on)
                img_off = self._undistort_bgr(img_off)
            
            # 2. Diff
            diff = cv2.absdiff(img_on, img_off)
            
            # 3. YOLO (Tiling)
            if not _YOLO_OK:
                ui_q.put(("toast", "❌ YOLO 없음"))
                return
            
            yolo_wpath = self.yolo_wpath.get().strip()
            if not yolo_wpath:
                ui_q.put(("toast", "⚠️ YOLO 가중치 없음"))
                return
                
            # 모델 로드 (매번 로드하면 느리지만, 스레드 안전성을 위해.. 
            # 혹은 self.yolo_model을 캐싱해서 써야 함. 여기서는 매번 로드하거나 캐싱 고려)
            # 성능을 위해 전역/멤버 변수로 모델을 유지하는게 좋음.
            # 하지만 간단히 하기 위해 여기서 로드 (또는 App에 캐싱된거 사용)
            # App에 캐싱된게 없으므로 로드. (속도 문제시 개선 필요)
            model = YOLO(yolo_wpath) 
            device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
            
            # conf=0.20, iou=0.45
            boxes, scores, classes = predict_with_tiling(
                model, diff, rows=2, cols=3, overlap=0.15, 
                conf=0.20, iou=0.45, device=device
            )
            
            if not boxes:
                ui_q.put(("toast", "[Center] ⚠️ YOLO 객체 없음 (No boxes)"))
                self._centering_stable_cnt = 0
                # [DEBUG] Save images for inspection
                cv2.imwrite("debug_center_on.jpg", img_on)
                cv2.imwrite("debug_center_off.jpg", img_off)
                cv2.imwrite("debug_center_diff.jpg", diff)
                return

            # 4. 최고 conf 객체 찾기
            best_idx = np.argmax(scores)
            x, y, w, h = boxes[best_idx]
            conf = scores[best_idx]
            
            # 중심 좌표
            obj_cx = x + w / 2.0
            obj_cy = y + h / 2.0
            
            # 5. 오차 계산
            H, W = diff.shape[:2]
            center_x, center_y = W / 2.0, H / 2.0
            err_x = obj_cx - center_x
            err_y = obj_cy - center_y
            
            ui_q.put(("toast", f"[Center] err=({err_x:.1f}, {err_y:.1f}) conf={conf:.2f}"))
            
            # 6. 안정성 판단
            tol = self.centering_px_tol.get()
            if abs(err_x) <= tol and abs(err_y) <= tol:
                self._centering_stable_cnt += 1
                ui_q.put(("toast", f"✅ 수렴 중... {self._centering_stable_cnt}/{self.centering_min_frames.get()}"))
                
                if self._centering_stable_cnt >= self.centering_min_frames.get():
                    final_pan = round(self._curr_pan, 2)
                    final_tilt = round(self._curr_tilt, 2)
                    ui_q.put(("toast", f"🎉 Centering 완료! Final: (P={final_pan}, T={final_tilt})"))
                    self.centering_enable.set(False); ui_q.put(("preview_on", None)) # 종료 및 프리뷰 복구
                    return
                
                # [FIX] Not yet finished, schedule next check
                self.root.after(self.centering_cooldown.get(), self._snap_center_on)
            else:
                self._centering_stable_cnt = 0
                
                # 7. 이동 (Move)
                # 픽셀 오차 -> 각도 변환
                # _fits_h, _fits_v 데이터가 있으면 사용, 없으면 대략적인 비례상수 사용
                # 대략: 2592px ~= 60도? (FOV에 따라 다름)
                # 일단 단순 비례 제어 (P-control)
                # FOV가 약 60도라고 가정하면, 1px ~= 0.023도
                # 하지만 정확히 하기 위해 fits 데이터가 있으면 좋음.
                
                # 여기서는 간단히 고정 게인 사용 (사용자가 max_step으로 제한하므로 안전)
                # err_x > 0 이면 객체가 오른쪽에 있음 -> 카메라를 오른쪽(Pan +)으로 돌려야 함
                # err_y > 0 이면 객체가 아래쪽에 있음 -> 카메라를 아래쪽(Tilt -)으로 돌려야 함 (Tilt 좌표계 확인 필요)
                # 보통 Tilt +가 위쪽이면, 아래에 있는 객체를 보려면 Tilt를 줄여야 함.
                
                # 게인 (튜닝 필요)
                k_pan = 0.02 
                k_tilt = 0.02 
                
                d_pan = err_x * k_pan
                d_tilt = -err_y * k_tilt # Tilt 방향 주의
                
                # Max step 제한
                max_step = self.centering_max_step.get()
                d_pan = max(min(d_pan, max_step), -max_step)
                d_tilt = max(min(d_tilt, max_step), -max_step)
                
                # 현재 위치 추정 (명령 기준)
                # self._curr_pan, self._curr_tilt 사용
                next_pan = self._curr_pan + d_pan
                next_tilt = self._curr_tilt + d_tilt
                
                # [NEW] Round to nearest integer (no accumulation)
                # next_pan = float(round(next_pan))
                # next_tilt = float(round(next_tilt))
                
                # Revert to accumulation (User request)
                # self._curr_pan, self._curr_tilt are floats and accumulate small changes.
                # Hardware will take the integer part when sending commands, but we keep the float state.
                
                # 범위 제한 (Centering Mode는 스캔 범위가 아닌 전체 하드웨어 범위를 사용해야 함)
                # Hardware limits: Pan -180~180, Tilt -30~90 (Defaults)
                next_pan = max(-180, min(180, next_pan))
                next_tilt = max(-30, min(90, next_tilt))
                
                ui_q.put(("toast", f"→ Move: Cur({self._curr_pan:.2f}, {self._curr_tilt:.2f}) + d({d_pan:.2f}, {d_tilt:.2f}) -> Next({next_pan:.2f}, {next_tilt:.2f})"))

                self._curr_pan = next_pan
                self._curr_tilt = next_tilt
                
                self.ctrl.send({
                    "cmd": "move",
                    "pan": next_pan,
                    "tilt": next_tilt,
                    "speed": self.speed.get(),
                    "acc": float(self.acc.get())
                })
                # ui_q.put(("toast", f"→ Adjust: dP={d_pan:.2f}, dT={d_tilt:.2f}"))
                
                # [FIX] Schedule next cycle
                self.root.after(self.centering_cooldown.get(), self._snap_center_on)

        except Exception as e:
            ui_q.put(("toast", f"❌ Centering Error: {e}"))
            import traceback
            traceback.print_exc()

    # [NEW] Helper to start centering cycle
    def _snap_center_on(self):
        if not self.centering_enable.get(): return
        self._centering_state = 1 # WAIT_ON
        self.ctrl.send({"cmd":"led", "value":255})
        wait_ms = int(self.led_settle.get() * 1000)
        self.root.after(wait_ms, lambda: self.ctrl.send({
            "cmd":"snap", "width":self.width.get(), "height":self.height.get(),
            "quality":self.quality.get(), "save":"center_on.jpg", "hard_stop":False
        }))

    def _snap_center_off(self):
        if not self.centering_enable.get(): return
        self.ctrl.send({
            "cmd":"snap", "width":self.width.get(), "height":self.height.get(),
            "quality":self.quality.get(), "save":"center_off.jpg", "hard_stop":False
        })

    def _find_laser_center(self, img_on, img_off, roi_size=200):
        h, w = img_on.shape[:2]
        cx, cy = w // 2, h // 2
        half = roi_size // 2
        x1 = max(0, cx - half); y1 = max(0, cy - half)
        x2 = min(w, cx + half); y2 = min(h, cy + half)
        
        roi_on = img_on[y1:y2, x1:x2]
        roi_off = img_off[y1:y2, x1:x2]
        
        g1 = cv2.cvtColor(roi_on, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(roi_off, cv2.COLOR_BGR2GRAY)
        g1 = cv2.GaussianBlur(g1, (5,5), 0)
        g2 = cv2.GaussianBlur(g2, (5,5), 0)
        diff = cv2.absdiff(g1, g2)
        _, bin_img = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0: return None
        
        lcx = int(M["m10"] / M["m00"])
        lcy = int(M["m01"] / M["m00"])
        return (lcx + x1, lcy + y1)

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
            if self._ud_K is not None:
                img_on = self._undistort_bgr(img_on)
                img_off = self._undistort_bgr(img_off)
            
            laser_pos = self._find_laser_center(img_on, img_off, self.pointing_roi_size.get())
            
            if laser_pos is None:
                # Blind Search: Tilt Down 1 deg
                ui_q.put(("toast", "⚠️ Laser not found -> Blind Search (Tilt -1°)"))
                next_tilt = self._curr_tilt - 1.0
                next_tilt = max(-30, min(90, next_tilt)) # Limit
                self._curr_tilt = next_tilt
                self.ctrl.send({"cmd":"move", "pan":self._curr_pan, "tilt":next_tilt, "speed":self.speed.get(), "acc":float(self.acc.get())})
                
                # End cycle, wait for cooldown
                self._pointing_state = 0
                self._pointing_last_ts = time.time() * 1000
                return

            # Laser Found -> Proceed to Object Detection
            self._laser_px = laser_pos
            ui_q.put(("toast", f"✅ Laser Found: {laser_pos}"))
            
            # Trigger LED ON
            ui_q.put(("pointing_step_2", None))
            
        except Exception as e:
            ui_q.put(("toast", f"❌ Pointing Laser Error: {e}"))
            self._pointing_state = 0

    def _run_pointing_object_logic(self, img_on, img_off):
        try:
            if self._ud_K is not None:
                img_on = self._undistort_bgr(img_on)
                img_off = self._undistort_bgr(img_off)
            
            diff = cv2.absdiff(img_on, img_off)
            
            if not _YOLO_OK:
                ui_q.put(("toast", "❌ YOLO 없음"))
                self._pointing_state = 0; return

            yolo_wpath = self.yolo_wpath.get().strip()
            model = YOLO(yolo_wpath)
            device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
            
            boxes, scores, classes = predict_with_tiling(model, diff, rows=2, cols=3, overlap=0.15, conf=0.20, iou=0.45, device=device)
            
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
            
            ui_q.put(("toast", f"Err:({err_x:.1f}, {err_y:.1f}) L:{self._laser_px} T:{target_px}"))
            
            # Convergence
            tol = self.centering_px_tol.get()
            if abs(err_x) <= tol and abs(err_y) <= tol:
                self._pointing_stable_cnt += 1
                ui_q.put(("toast", f"✅ Pointing Converging... {self._pointing_stable_cnt}/{self.centering_min_frames.get()}"))
                if self._pointing_stable_cnt >= self.centering_min_frames.get():
                    ui_q.put(("toast", "🎉 Pointing Complete!"))
                    self.pointing_enable.set(False); ui_q.put(("preview_on", None))
                    self.ctrl.send({"cmd":"laser", "value":0}); self.laser_on.set(False)
                    self._pointing_state = 0
                    return
            else:
                self._pointing_stable_cnt = 0
                
                # Move
                k_pan = 0.02; k_tilt = 0.02
                d_pan = err_x * k_pan
                d_tilt = -err_y * k_tilt
                
                max_step = self.centering_max_step.get()
                d_pan = max(min(d_pan, max_step), -max_step)
                d_tilt = max(min(d_tilt, max_step), -max_step)
                
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

    def _poll(self):
        # [NEW] Centering Trigger Check
        if self.centering_enable.get() and self._centering_state == 0:
            now = time.time() * 1000
            if now - self._centering_last_ts > self.centering_cooldown.get():
                self._start_centering_cycle()

        # [NEW] Pointing Trigger Check
        if self.pointing_enable.get() and self._pointing_state == 0:
            now = time.time() * 1000
            if now - self._pointing_last_ts > self.centering_cooldown.get():
                self._start_pointing_cycle()

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
                        sess = evt.get("session") or datetime.now().strftime("scan_%Y%m%d_%H%M%S")
                        self._scan_csv_path = DEFAULT_OUT_DIR / f"{sess}_detections.csv"
                        try:
                            self._scan_csv_file = open(self._scan_csv_path, "w", newline="", encoding="utf-8")
                            self._scan_csv_writer = csv.writer(self._scan_csv_file)
                            self._scan_csv_writer.writerow(["pan_deg","tilt_deg","cx","cy","w","h","conf","cls","W","H"])
                            print(f"[SCAN] CSV → {self._scan_csv_path}")
                        except Exception as e:
                            self._scan_csv_file = None
                            self._scan_csv_writer = None
                            ui_q.put(("toast", f"CSV 오픈 실패: {e}"))

                    elif et == "progress":
                        done=int(evt.get("done",0)); total=int(evt.get("total",0))
                        if total > 0: self.prog.configure(maximum=total)
                        self.prog.configure(value=done); self.prog_lbl.config(text=f"{done} / {total}")
                        name = evt.get("name","")
                        if name: self.last_lbl.config(text=f"Last: {name}")
                    elif et == "done":
                        ui_q.put(("toast", "[SCAN] 스캔 완료! LED ON/OFF 차분 이미지 처리 시작..."))
                        
                        def process_diff_and_yolo():
                            try:
                                import glob
                                from collections import defaultdict
                                led_on_files = sorted(glob.glob(str(DEFAULT_OUT_DIR / "*_led_on.jpg")))
                                led_off_files = sorted(glob.glob(str(DEFAULT_OUT_DIR / "*_led_off.jpg")))
                                pairs = defaultdict(dict)
                                fname_re = re.compile(r"img_t(?P<tilt>[+\-]\d{2,3})_p(?P<pan>[+\-]\d{2,3})_.*_led_(?P<state>on|off)\.jpg$", re.IGNORECASE)
                                for fpath in led_on_files + led_off_files:
                                    fname = os.path.basename(fpath)
                                    m = fname_re.search(fname)
                                    if m:
                                        pairs[(int(m.group("pan")), int(m.group("tilt")))][m.group("state")] = fpath
                                ui_q.put(("toast", f"[DIFF] {len(pairs)}개 위치의 LED ON/OFF 쌍 발견"))
                                
                                # 2. CSV 파일 생성
                                # sess = evt.get("session") or datetime.now().strftime("scan_%Y%m%d_%H%M%S")
                                # csv_path = DEFAULT_OUT_DIR / f"{sess}_detections.csv"
                                
                                # 이미 _poll 시작 부분에서 생성된 self._scan_csv_file 사용
                                if self._scan_csv_file is None:
                                     sess = evt.get("session") or datetime.now().strftime("scan_%Y%m%d_%H%M%S")
                                     csv_path = DEFAULT_OUT_DIR / f"{sess}_detections.csv"
                                     # ... (open logic if needed, but usually opened at 'start')
                                else:
                                     csv_path = self._scan_csv_path

                                # 만약 'start' 이벤트에서 열린 파일이 있다면 닫고 새로 열거나, 이어서 쓰거나.
                                # 기존 로직: 'start'에서 열고 헤더 씀.
                                # 여기서 또 열면 2개가 되거나 덮어씀.
                                # 'start'에서 만든 파일에 이어서 쓰는게 맞음.
                                
                                # 하지만 여기서 'with open'으로 새로 열고 있음 -> 이게 문제.
                                # self._scan_csv_writer를 사용해야 함.
                                
                                writer = self._scan_csv_writer
                                if writer is None:
                                    # fallback
                                    f = open(csv_path, "a", newline="", encoding="utf-8")
                                    writer = csv.writer(f)
                                
                                # 3. YOLO 모델 로드 (GPU 사용)
                                if not _YOLO_OK:
                                    ui_q.put(("toast", "❌ YOLO 미설치"))
                                    return
                                yolo_wpath = self.yolo_wpath.get().strip()
                                if not yolo_wpath:
                                    ui_q.put(("toast", "⚠️ YOLO 가중치 없음"))
                                    return
                                yolo_model = YOLO(yolo_wpath)
                                device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
                                ui_q.put(("toast", f"[YOLO] Device: {device}"))
                                total_pairs = len(pairs); processed = 0; detected_count = 0
                                for (pan, tilt), files in sorted(pairs.items()):
                                    if "on" not in files or "off" not in files: continue
                                    img_on = cv2.imread(files["on"])
                                    img_off = cv2.imread(files["off"])
                                    if img_on is None or img_off is None: continue
                                    if self._ud_K is not None:
                                        img_on = self._undistort_bgr(img_on)
                                        img_off = self._undistort_bgr(img_off)
                                    diff = cv2.absdiff(img_on, img_off)
                                    H, W = diff.shape[:2]
                                    boxes, scores, classes = predict_with_tiling(yolo_model, diff, rows=2, cols=3, overlap=0.15, conf=0.20, iou=0.45, device=device)
                                    if boxes:
                                        for i, (x, y, w, h) in enumerate(boxes):
                                            writer.writerow([pan, tilt, x+w/2, y+h/2, w, h, float(scores[i]), int(classes[i]), W, H])
                                            detected_count += 1
                                    processed += 1
                                    # [NEW] Update progress bar
                                    ui_q.put(("evt", {"event": "progress", "done": processed, "total": total_pairs, "name": f"YOLO {processed}/{total_pairs}"}))
                                    if processed % 10 == 0: ui_q.put(("toast", f"[DIFF] {processed}/{total_pairs}"))
                                
                                # [NEW] Flush and close CSV
                                if self._scan_csv_file:
                                    self._scan_csv_file.flush()
                                    self._scan_csv_file.close()
                                    self._scan_csv_file = None
                                    self._scan_csv_writer = None
                                    
                                ui_q.put(("toast", f"✅ 완료: {csv_path} ({detected_count}개)")); ui_q.put(("preview_on", None))
                            except Exception as e:
                                ui_q.put(("toast", f"❌ 에러: {e}"))
                                import traceback; traceback.print_exc()
                        threading.Thread(target=process_diff_and_yolo, daemon=True).start()

                elif tag == "preview":
                    self._set_preview(payload)

                elif tag == "saved":
                    name, data = payload
                    if name == "center_on.jpg" and self._centering_state == 1:
                        try:
                            nparr = np.frombuffer(data, np.uint8)
                            self._centering_on_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            self._set_preview(data) # [NEW] Show captured image
                            self._centering_state = 2
                            self.ctrl.send({"cmd":"led", "value":0})
                            self.root.after(int(self.led_settle.get()*1000), self._snap_center_off)
                        except: self._centering_state = 0
                    elif name == "center_off.jpg" and self._centering_state == 2:
                        try:
                            nparr = np.frombuffer(data, np.uint8)
                            self._centering_off_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            self._set_preview(data) # [NEW] Show captured image
                            
                            # [FIX] Run Centering Logic
                            # [FIX] Run Centering Logic
                            if self._centering_on_img is not None and self._centering_off_img is not None:
                                ui_q.put(("toast", "🚀 Centering Logic Start"))
                                threading.Thread(target=self._run_centering_logic, args=(self._centering_on_img, self._centering_off_img), daemon=True).start()
                            else:
                                ui_q.put(("toast", "❌ Centering Images Missing"))
                                self._centering_state = 0
                                self.resume_preview(); self._resume_preview_after_snap = False
                        except Exception as e:
                            print(f"[Centering] Error: {e}")
                            self._centering_state = 0

                    # [NEW] Pointing Mode Handlers
                    elif name == "pointing_laser_on.jpg":
                        self._pointing_state = 2 # WAIT_LASER_OFF
                        self._set_preview(data) # [NEW] Preview
                        self.ctrl.send({"cmd":"laser", "value":0})
                        wait_ms = int(self.led_settle.get() * 1000)
                        self.root.after(wait_ms, lambda: self.ctrl.send({
                            "cmd":"snap", "width":self.width.get(), "height":self.height.get(),
                            "quality":self.quality.get(), "save":"pointing_laser_off.jpg", "hard_stop":False
                        }))
                        
                    elif name == "pointing_laser_off.jpg":
                        self._pointing_state = 3 # PROCESSING_LASER
                        self._set_preview(data) # [NEW] Preview
                        path_on = DEFAULT_OUT_DIR / "pointing_laser_on.jpg"
                        path_off = DEFAULT_OUT_DIR / "pointing_laser_off.jpg"
                        try:
                            nparr_on = np.fromfile(path_on, np.uint8)
                            img_on = cv2.imdecode(nparr_on, cv2.IMREAD_COLOR)
                            nparr_off = np.fromfile(path_off, np.uint8)
                            img_off = cv2.imdecode(nparr_off, cv2.IMREAD_COLOR)
                            if img_on is not None and img_off is not None:
                                threading.Thread(target=self._run_pointing_laser_logic, args=(img_on, img_off), daemon=True).start()
                        except Exception as e:
                            print(f"[Pointing] Laser Load Error: {e}")
                            self._pointing_state = 0

                    elif name == "pointing_led_on.jpg":
                        self._pointing_state = 5 # WAIT_LED_OFF
                        self._set_preview(data) # [NEW] Preview
                        self.ctrl.send({"cmd":"led", "value":0})
                        wait_ms = int(self.led_settle.get() * 1000)
                        self.root.after(wait_ms, lambda: self.ctrl.send({
                            "cmd":"snap", "width":self.width.get(), "height":self.height.get(),
                            "quality":self.quality.get(), "save":"pointing_led_off.jpg", "hard_stop":False
                        }))

                    elif name == "pointing_led_off.jpg":
                        self._pointing_state = 6 # PROCESSING_OBJECT
                        self._set_preview(data) # [NEW] Preview
                        path_on = DEFAULT_OUT_DIR / "pointing_led_on.jpg"
                        path_off = DEFAULT_OUT_DIR / "pointing_led_off.jpg"
                        try:
                            nparr_on = np.fromfile(path_on, np.uint8)
                            img_on = cv2.imdecode(nparr_on, cv2.IMREAD_COLOR)
                            nparr_off = np.fromfile(path_off, np.uint8)
                            img_off = cv2.imdecode(nparr_off, cv2.IMREAD_COLOR)
                            if img_on is not None and img_off is not None:
                                threading.Thread(target=self._run_pointing_object_logic, args=(img_on, img_off), daemon=True).start()
                        except Exception as e:
                            print(f"[Pointing] Object Load Error: {e}")
                            self._pointing_state = 0







                    else:
                        self.dl_lbl.config(text=f"DL {len(data)}")
                        # [NEW] Show scanned image in preview
                        self._set_preview(data)
                        
                        # [RESTORED] Save undistorted copy if enabled
                        if self.ud_save_copy.get() and self._ud_K is not None:
                             try:
                                 nparr = np.frombuffer(data, np.uint8)
                                 bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                 if bgr is not None:
                                     ud = self._undistort_bgr(bgr)
                                     # name is like "img_t..._p..._....jpg"
                                     # save as "img_t..._p..._....ud.jpg"
                                     base, ext = os.path.splitext(name)
                                     ud_name = f"{base}.ud{ext}"
                                     ud_path = DEFAULT_OUT_DIR / ud_name
                                     cv2.imwrite(str(ud_path), ud)
                             except Exception as e:
                                 print(f"[UD Save] Error: {e}")

                        if self._resume_preview_after_snap:
                            self.resume_preview(); self._resume_preview_after_snap = False

                elif tag == "toast":
                    print(f"[TOAST] {payload}")

                elif tag == "pointing_step_2":
                    self._pointing_state = 4 # WAIT_LED_ON
                    self.ctrl.send({"cmd":"led", "value":255})
                    wait_ms = int(self.led_settle.get() * 1000)
                    self.root.after(wait_ms, lambda: self.ctrl.send({
                        "cmd":"snap", "width":self.width.get(), "height":self.height.get(),
                        "quality":self.quality.get(), "save":"pointing_led_on.jpg", "hard_stop":False
                    }))

                elif tag == "preview_on":
                    self.preview_enable.set(True)
                    self.toggle_preview()

        except queue.Empty:
            pass
        self.root.after(60, self._poll)

    # ---------- 고정 박스 안에 '레터박스(contain)'로 그리기 ----------
    def _draw_preview_to_label(self, pil_image: Image.Image):
        W, H = int(self.PREV_W), int(self.PREV_H)
        iw, ih = pil_image.size
        if iw <= 0 or ih <= 0 or W <= 0 or H <= 0:
            return
        
        # [NEW] Centering Mode Marker
        if self.centering_enable.get() or self.show_center_marker.get():
            draw = ImageDraw.Draw(pil_image)
            cx, cy = iw / 2, ih / 2
            r = 5
            # Red circle
            draw.ellipse((cx-r, cy-r, cx+r, cy+r), outline="red", width=2)
            # Crosshair
            draw.line((cx-10, cy, cx+10, cy), fill="red", width=2)
            draw.line((cx, cy-10, cx, cy+10), fill="red", width=2)

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



            # YOLO 및 Laser tracking 제거됨


            # (필요 시) 화면 중앙 십자 등 유지
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(rgb)
            self._draw_preview_to_label(im)

        except Exception as e:
            print("[preview] err:", e)

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



def main():
    root = Tk()
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
