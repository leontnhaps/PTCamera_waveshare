#!/usr/bin/env python3
# pc_gui.py — GUI client connecting to pc_server.py (not to Pi agent)

import json, socket, struct, threading, queue, pathlib, io
from datetime import datetime
from tkinter import Tk, Label, Button, Scale, HORIZONTAL, IntVar, DoubleVar, Frame, Checkbutton, BooleanVar, filedialog, StringVar
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw
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

def predict_with_tiling(model, img, rows=2, cols=3, overlap=0.15, conf=0.25, iou=0.45, device='cuda', use_full_image=True):
    """
    이미지를 타일로 쪼개서 예측 후 결과 병합
    rows, cols: 행/열 개수 (2x3 = 6등분)
    overlap: 타일 간 겹치는 비율 (0.15 = 15%)
    use_full_image: 전체 이미지도 함께 검출 (큰 객체 검출용)
    
    배치 처리: 전체 이미지 + 타일 6개를 한 번에!
    """
    H, W = img.shape[:2]
    
    # 타일 크기 계산 (겹침 포함)
    tile_h = int(H / rows)
    tile_w = int(W / cols)
    
    # 겹침 크기
    ov_h = int(tile_h * overlap)
    ov_w = int(tile_w * overlap)
    
    # 실제 타일 크기 (겹침 포함)
    step_h = tile_h - ov_h
    
    # 타일 좌표 생성 (정확히 rows x cols 개수)
    tiles = []
    base_tile_h = H // rows
    base_tile_w = W // cols
    
    # 오버랩 크기 계산
    ov_h = int(base_tile_h * overlap)
    ov_w = int(base_tile_w * overlap)
    
    for row_idx in range(rows):
        for col_idx in range(cols):
            # 기본 타일 영역
            y1 = row_idx * base_tile_h
            y2 = (row_idx + 1) * base_tile_h if row_idx < rows - 1 else H
            x1 = col_idx * base_tile_w
            x2 = (col_idx + 1) * base_tile_w if col_idx < cols - 1 else W
            
            # 오버랩 확장 (경계 체크)
            y1 = max(0, y1 - ov_h)
            y2 = min(H, y2 + ov_h)
            x1 = max(0, x1 - ov_w)
            x2 = min(W, x2 + ov_w)
            
            tiles.append((x1, y1, x2, y2))
    
    print(f"[DEBUG] 타일 개수: {len(tiles)}, 이미지 크기: {W}x{H}, rows={rows}, cols={cols}")

    # 모든 이미지 수집 (배치 처리용)
    all_boxes = []
    all_scores = []
    all_classes = []
    
    # 배치 크기 자동 조정
    # 전체 이미지 포함 시: 7개, 3개, 1개
    # 타일만: 6개, 3개, 1개
    total_images = len(tiles) + (1 if use_full_image else 0)
    batch_sizes = [total_images, 3, 1]
    
    for batch_size in batch_sizes:
        try:
            # 배치 이미지 준비
            all_batch_images = []
            image_info = []  # (type, tx1, ty1, tx2, ty2) - type: 'full' or 'tile'
            
            # 1. 전체 이미지 먼저 (큰 객체 검출용)
            if use_full_image:
                all_batch_images.append(img)
                image_info.append(('full', 0, 0, W, H))
            
            # 2. 타일 이미지들
            for tx1, ty1, tx2, ty2 in tiles:
                tile_img = img[ty1:ty2, tx1:tx2]
                all_batch_images.append(tile_img)
                image_info.append(('tile', tx1, ty1, tx2, ty2))
            
            # 배치로 처리
            for i in range(0, len(all_batch_images), batch_size):
                batch_images = all_batch_images[i:i + batch_size]
                batch_info = image_info[i:i + batch_size]
                
                # YOLO 배치 추론
                if len(batch_images) == 1:
                    results_list = [model.predict(batch_images[0], conf=conf, iou=iou, device=device, verbose=False)[0]]
                else:
                    results_list = model.predict(batch_images, conf=conf, iou=iou, device=device, verbose=False)
                
                # 결과 처리
                for j, (img_type, tx1, ty1, tx2, ty2) in enumerate(batch_info):
                    results = results_list[j]
                    
                    if results.boxes:
                        for box in results.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            c = float(box.conf.cpu().numpy().item())
                            cls = int(box.cls.cpu().numpy().item())
                            
                            # 글로벌 좌표로 변환
                            if img_type == 'full':
                                # 전체 이미지는 그대로
                                gx1, gy1, gx2, gy2 = x1, y1, x2, y2
                            else:
                                # 타일은 오프셋 추가
                                gx1 = x1 + tx1
                                gy1 = y1 + ty1
                                gx2 = x2 + tx1
                                gy2 = y2 + ty1
                            
                            w = gx2 - gx1
                            h = gy2 - gy1
                            
                            all_boxes.append([int(gx1), int(gy1), int(w), int(h)])
                            all_scores.append(c)
                            all_classes.append(cls)
            
            # 성공하면 빠져나오기
            if batch_size == total_images:
                if use_full_image:
                    print(f"[YOLO] 전체+타일 배치 처리 성공: {total_images}개 동시 처리")
                else:
                    print(f"[YOLO] 타일 배치 처리 성공: {batch_size}개 동시 처리")
            elif batch_size == 3:
                print(f"[YOLO] 배치 크기 감소: {batch_size}개씩 처리")
            break
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() or 'cuda' in str(e).lower():
                # 메모리 부족 - 다음 배치 크기 시도
                if batch_size == batch_sizes[-1]:
                    print(f"[YOLO] 메모리 부족: 모든 배치 크기 실패")
                    raise
                else:
                    print(f"[YOLO] 메모리 부족: 배치 크기 {batch_size} 실패, {batch_sizes[batch_sizes.index(batch_size)+1]}로 재시도...")
                    all_boxes = []
                    all_scores = []
                    all_classes = []
                    continue
            else:
                raise

    # 전체 결과에 대해 NMS 수행 (중복 제거)
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

# ---- Image Processing ----
class ImageProcessor:
    """Handles image loading and undistortion with CUDA/Torch acceleration"""
    
    def __init__(self):
        # Calibration data
        self._ud_model = None
        self._ud_K = None
        self._ud_D = None
        self._ud_img_size = None
        self._ud_src_size = None
        
        # CPU undistortion maps
        self._ud_m1 = None
        self._ud_m2 = None
        
        # CUDA support
        self._use_cv2_cuda = False
        try:
            self._use_cv2_cuda = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
        except Exception:
            self._use_cv2_cuda = False
        self._ud_gm1 = None
        self._ud_gm2 = None
        
        # Torch support
        self._torch_available = _TORCH_AVAILABLE
        self._torch_cuda = bool(_TORCH_AVAILABLE and torch.cuda.is_available())
        self._torch_device = torch.device("cuda") if self._torch_cuda else torch.device("cpu") if _TORCH_AVAILABLE else None
        self._torch_use_fp16 = False
        self._torch_dtype = (torch.float16 if (self._torch_cuda and self._torch_use_fp16) else torch.float32) if _TORCH_AVAILABLE else None
        self._ud_torch_grid = None
        self._ud_torch_grid_wh = None
        
        # Alpha for getOptimalNewCameraMatrix
        self.alpha = 0.0
    
    def load_calibration(self, path):
        """Load camera calibration from npz file"""
        try:
            cal = np.load(str(path), allow_pickle=True)
            self._ud_model = str(cal["model"])
            self._ud_K = cal["K"].astype(np.float32)
            self._ud_D = cal["D"].astype(np.float32)
            self._ud_img_size = tuple(int(x) for x in cal["img_size"])
            self._ud_src_size = None
            self._ud_m1 = self._ud_m2 = None
            self._ud_gm1 = self._ud_gm2 = None
            self._ud_torch_grid = None
            self._ud_torch_grid_wh = None
            print(f"[ImageProcessor] Loaded: model={self._ud_model}, img_size={self._ud_img_size}, cv2.cuda={self._use_cv2_cuda}, torch={self._torch_cuda}")
            return True
        except Exception as e:
            print(f"[ImageProcessor] Load failed: {e}")
            return False
    
    def has_calibration(self):
        """Check if calibration is loaded"""
        return self._ud_K is not None
    
    def _scale_K(self, K, sx, sy):
        """Scale camera matrix K"""
        K2 = K.copy()
        K2[0,0] *= sx
        K2[1,1] *= sy
        K2[0,2] *= sx
        K2[1,2] *= sy
        K2[2,2] = 1.0
        return K2
    
    def _ensure_ud_maps(self, w: int, h: int):
        """Ensure undistortion maps are created for given size"""
        if self._ud_K is None or self._ud_D is None or self._ud_model is None:
            return
        if self._ud_src_size == (w, h) and self._ud_m1 is not None:
            return
        
        Wc, Hc = self._ud_img_size
        sx, sy = w / float(Wc), h / float(Hc)
        K = self._scale_K(self._ud_K, sx, sy)
        D = self._ud_D
        a = float(self.alpha)
        
        if self._ud_model == "pinhole":
            newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=a, newImgSize=(w, h))
            self._ud_m1, self._ud_m2 = cv2.initUndistortRectifyMap(
                K, D, None, newK, (w, h), cv2.CV_32FC1
            )
        elif self._ud_model == "fisheye":
            newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D, (w, h), None, balance=a
            )
            self._ud_m1, self._ud_m2 = cv2.fisheye.initUndistortRectifyMap(
                K, D, None, newK, (w, h), cv2.CV_32FC1
            )
        
        self._ud_src_size = (w, h)
        
        # CUDA maps
        if self._use_cv2_cuda and self._ud_m1 is not None:
            try:
                self._ud_gm1 = cv2.cuda_GpuMat()
                self._ud_gm2 = cv2.cuda_GpuMat()
                self._ud_gm1.upload(self._ud_m1)
                self._ud_gm2.upload(self._ud_m2)
            except Exception as e:
                print(f"[ImageProcessor] CUDA upload failed: {e}")
                self._ud_gm1 = self._ud_gm2 = None
    
    def _ensure_torch_grid(self, h: int, w: int):
        """Ensure torch grid for GPU undistortion"""
        if not self._torch_available:
            return
        if self._ud_torch_grid is not None and self._ud_torch_grid_wh == (w, h):
            return
        
        self._ensure_ud_maps(w, h)
        if self._ud_m1 is None:
            return
        
        import torch
        import torch.nn.functional as F
        
        m1_t = torch.from_numpy(self._ud_m1).float()
        m2_t = torch.from_numpy(self._ud_m2).float()
        
        grid_x = (m1_t / (w - 1)) * 2 - 1
        grid_y = (m2_t / (h - 1)) * 2 - 1
        self._ud_torch_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        
        if self._torch_cuda:
            self._ud_torch_grid = self._ud_torch_grid.to(self._torch_device)
        
        self._ud_torch_grid_wh = (w, h)
    
    def undistort(self, img, use_torch=False):
        """Undistort a BGR image using best available method"""
        if self._ud_K is None or img is None:
            return img
        
        h, w = img.shape[:2]
        
        # Torch acceleration (fastest)
        if use_torch and self._torch_available and self._torch_cuda:
            return self._undistort_torch(img, h, w)
        
        # CUDA acceleration
        if self._use_cv2_cuda:
            return self._undistort_cuda(img, w, h)
        
        # CPU fallback
        self._ensure_ud_maps(w, h)
        if self._ud_m1 is None:
            return img
        return cv2.remap(img, self._ud_m1, self._ud_m2, cv2.INTER_LINEAR)
    
    def _undistort_cuda(self, img, w, h):
        """Undistort using CUDA"""
        self._ensure_ud_maps(w, h)
        if self._ud_gm1 is None:
            return img
        
        try:
            gpu_src = cv2.cuda_GpuMat()
            gpu_src.upload(img)
            gpu_dst = cv2.cuda.remap(gpu_src, self._ud_gm1, self._ud_gm2, cv2.INTER_LINEAR)
            return gpu_dst.download()
        except Exception as e:
            print(f"[ImageProcessor] CUDA remap failed: {e}")
            return cv2.remap(img, self._ud_m1, self._ud_m2, cv2.INTER_LINEAR)
    
    def _undistort_torch(self, img, h, w):
        """Undistort using Torch"""
        import torch
        import torch.nn.functional as F
        
        self._ensure_torch_grid(h, w)
        if self._ud_torch_grid is None:
            return img
        
        try:
            # BGR -> RGB, HWC -> CHW
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(
                dtype=self._torch_dtype, device=self._torch_device
            )
            img_t = img_t / 255.0
            
            out_t = F.grid_sample(img_t, self._ud_torch_grid, mode='bilinear', 
                                  padding_mode='border', align_corners=True)
            
            out_t = (out_t * 255.0).clamp(0, 255).squeeze(0).permute(1, 2, 0)
            out_np = out_t.cpu().to(torch.uint8).numpy()
            out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
            return out_bgr
        except Exception as e:
            print(f"[ImageProcessor] Torch undistort failed: {e}")
            return self.undistort(img, use_torch=False)
    
    def undistort_pair(self, img_on, img_off, use_torch=False):
        """Undistort a pair of images"""
        if self._ud_K is None:
            return img_on, img_off
        img_on = self.undistort(img_on, use_torch=use_torch)
        img_off = self.undistort(img_off, use_torch=use_torch)
        return img_on, img_off
    
    def load_image(self, path):
        """Load image from file"""
        try:
            nparr = np.fromfile(str(path), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"[ImageProcessor] Image load error: {e}")
            return None
    
    def load_image_pair(self, path_on, path_off):
        """Load a pair of images"""
        img_on = self.load_image(path_on)
        img_off = self.load_image(path_off)
        return img_on, img_off

# ---- YOLO Processing ----
class YOLOProcessor:
    """Handles YOLO model loading and caching"""
    
    def __init__(self):
        self._cached_model = None
        self._cached_path = None
    
    def get_model(self, weights_path):
        """Get YOLO model with caching"""
        if not _YOLO_OK:
            return None
        
        # Return cached model if path matches
        if self._cached_model is not None and self._cached_path == weights_path:
            return self._cached_model
        
        # Load new model
        try:
            self._cached_model = YOLO(weights_path)
            self._cached_path = weights_path
            print(f"[YOLOProcessor] Loaded model: {weights_path}")
            return self._cached_model
        except Exception as e:
            print(f"[YOLOProcessor] Load failed: {e}")
            return None
    
    def get_device(self):
        """Get available device for inference"""
        if not _TORCH_AVAILABLE:
            return "cpu"
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

# ---- Scan Processing ----
class ScanController:
    """실시간 스캔 처리 관리자 - Real-time scan processing with immediate YOLO detection"""
    
    def __init__(self, image_processor, yolo_processor, output_dir):
        self.image_processor = image_processor
        self.yolo_processor = yolo_processor
        self.output_dir = output_dir
        
        # Scan state
        self.is_scanning = False
        self.yolo_weights_path = None
        
        # Real-time processing buffer: (pan, tilt) -> {'on': img_ud, 'off': img_ud}
        self.image_pairs = {}
        
        # CSV writer
        self.csv_writer = None
        self.csv_file = None
        self.csv_path = None
        
        # Statistics
        self.processed_count = 0
        self.detected_count = 0
    
    def start_scan(self, session_name, yolo_path):
        """Start scan - create CSV file"""
        self.is_scanning = True
        self.yolo_weights_path = yolo_path
        self.image_pairs.clear()
        self.processed_count = 0
        self.detected_count = 0
        
        # Create CSV file
        self.csv_path = self.output_dir / f"{session_name}_detections.csv"
        try:
            self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["pan_deg", "tilt_deg", "cx", "cy", "w", "h", "conf", "cls", "W", "H"])
            print(f"[ScanController] CSV created: {self.csv_path}")
            return True
        except Exception as e:
            print(f"[ScanController] CSV creation failed: {e}")
            self.csv_file = None
            self.csv_writer = None
            return False
    
    def on_image_received(self, name, data):
        """Process received image"""
        # Save to file (existing feature)
        file_path = self.output_dir / name
        try:
            with open(file_path, 'wb') as f:
                f.write(data)
        except Exception as e:
            print(f"[ScanController] File save failed: {e}")
        
        # Parse pan/tilt from filename: "img_t045_p090_..._led_on.jpg"
        match = re.search(r't([+-]?\d+)_p([+-]?\d+)', name)
        if not match:
            return data  # Return for preview
        
        tilt = int(match.group(1))
        pan = int(match.group(2))
        
        # Real-time processing (if scanning)
        if self.is_scanning:
            self._process_realtime(name, data, pan, tilt)
        
        return data  # Return for preview
    
    def _process_realtime(self, name, data, pan, tilt):
        """Real-time processing: undistort → buffer → YOLO when pair complete"""
        # Decode image
        try:
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return
        except Exception as e:
            print(f"[ScanController] Image decode failed: {e}")
            return
        
        # Undistort
        img_ud = self.image_processor.undistort(img, use_torch=True)
        
        # Store in buffer
        key = (pan, tilt)
        if 'led_on' in name:
            self.image_pairs.setdefault(key, {})['on'] = img_ud
        elif 'led_off' in name:
            self.image_pairs.setdefault(key, {})['off'] = img_ud
        
        # Check if pair is complete
        pair = self.image_pairs.get(key, {})
        if 'on' in pair and 'off' in pair:
            self._process_pair(pan, tilt, pair)
            del self.image_pairs[key]  # Free memory
    
    def _process_pair(self, pan, tilt, pair):
        """Process complete pair: diff → YOLO → CSV"""
        try:
            # Calculate difference
            diff = cv2.absdiff(pair['on'], pair['off'])
            H, W = diff.shape[:2]
            
            # YOLO detection
            model = self.yolo_processor.get_model(self.yolo_weights_path)
            if model is None:
                return
            
            device = self.yolo_processor.get_device()
            boxes, scores, classes = predict_with_tiling(
                model, diff,
                rows=YOLO_TILE_ROWS, cols=YOLO_TILE_COLS,
                overlap=YOLO_TILE_OVERLAP,
                conf=YOLO_CONF_THRESHOLD, iou=YOLO_IOU_THRESHOLD,
                device=device
            )
            
            # Write to CSV
            if boxes and self.csv_writer:
                for i, (x, y, w, h) in enumerate(boxes):
                    self.csv_writer.writerow([
                        pan, tilt, x+w/2, y+h/2, w, h,
                        float(scores[i]), int(classes[i]), W, H
                    ])
                    self.detected_count += 1
                self.csv_file.flush()  # Immediate write to disk
            
            self.processed_count += 1
            
        except Exception as e:
            print(f"[ScanController] Pair processing failed ({pan}, {tilt}): {e}")
    
    def stop_scan(self):
        """Stop scan - close CSV"""
        self.is_scanning = False
        
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
        
        print(f"[ScanController] Scan stopped. Processed: {self.processed_count}, Detected: {self.detected_count}")
        
        # Clear buffer
        self.image_pairs.clear()
        
        return {
            'csv_path': self.csv_path,
            'processed': self.processed_count,
            'detected': self.detected_count
        }

# ---- GUI Components ----
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
        max_step = self.centering_max_step.get()
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
        self._send_snap_cmd(fname, self.hard_stop.get())

    # event loop
    # ==== [NEW] Centering Mode Logic ====
    # (_start_centering_cycle and _snap_center_on are defined later - removed duplicate)

    def _snap_center_off(self):
        # 4. Snap OFF image
        self._send_snap_cmd("center_off.jpg", False)

    def _run_centering_logic(self, img_on, img_off):
        """백그라운드 스레드에서 실행되는 Centering 핵심 로직"""
        try:
            # 1. Undistort (use helper)
            img_on, img_off = self._undistort_pair(img_on, img_off)
            
            # 2. Diff
            diff = cv2.absdiff(img_on, img_off)
            
            # 3. YOLO (Tiling) - use cached model
            model = self._get_yolo_model()
            if model is None:
                ui_q.put(("toast", "❌ YOLO 모델 없음"))
                return
            
            device = self._get_device()
            
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
                
                # 7. 이동 (Move) - use helper for angle calculation
                d_pan, d_tilt = self._calculate_angle_delta(err_x, err_y)
                
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
        self.root.after(wait_ms, lambda: self._send_snap_cmd("center_on.jpg", False))

    def _snap_center_off(self):
        if not self.centering_enable.get(): return
        self._send_snap_cmd("center_off.jpg", False)

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
            img_on, img_off = self._undistort_pair(img_on, img_off)
            
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
    
    def _check_centering_trigger(self):
        """Check and trigger centering cycle if needed"""
        if self.centering_enable.get() and self._centering_state == 0:
            now = time.time() * 1000
            if now - self._centering_last_ts > self.centering_cooldown.get():
                self._start_centering_cycle()
    
    def _check_pointing_trigger(self):
        """Check and trigger pointing cycle if needed"""
        if self.pointing_enable.get() and self._pointing_state == 0:
            now = time.time() * 1000
            if now - self._pointing_last_ts > self.centering_cooldown.get():
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
        
        ui_q.put(("toast", f"✅ 스캔 완료: {processed}개 처리, {detected}개 검출"))
        ui_q.put(("toast", f"📄 CSV: {csv_path}"))
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
    
    def _handle_center_on_image(self, name, data):
        """Handle centering ON image capture"""
        if name == "center_on.jpg" and self._centering_state == 1:
            try:
                nparr = np.frombuffer(data, np.uint8)
                self._centering_on_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                self._set_preview(data)
                self._centering_state = 2
                self.ctrl.send({"cmd": "led", "value": 0})
                self.root.after(int(self.led_settle.get() * 1000), self._snap_center_off)
            except:
                self._centering_state = 0
    
    def _handle_center_off_image(self, name, data):
        """Handle centering OFF image capture"""
        if name == "center_off.jpg" and self._centering_state == 2:
            try:
                nparr = np.frombuffer(data, np.uint8)
                self._centering_off_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                self._set_preview(data)
                
                if self._centering_on_img is not None and self._centering_off_img is not None:
                    ui_q.put(("toast", "🚀 Centering Logic Start"))
                    threading.Thread(target=self._run_centering_logic, args=(self._centering_on_img, self._centering_off_img), daemon=True).start()
                else:
                    ui_q.put(("toast", "❌ Centering Images Missing"))
                    self._centering_state = 0
                    self.resume_preview()
                    self._resume_preview_after_snap = False
            except Exception as e:
                print(f"[Centering] Error: {e}")
                self._centering_state = 0
    
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
    
    def _handle_pointing_led_on(self, name, data):
        """Handle pointing LED ON image"""
        if name == "pointing_led_on.jpg":
            self._pointing_state = 5
            self._set_preview(data)
            self.ctrl.send({"cmd": "led", "value": 0})
            wait_ms = int(self.led_settle.get() * 1000)
            self.root.after(wait_ms, lambda: self.ctrl.send({
                "cmd": "snap", "width": self.width.get(), "height": self.height.get(),
                "quality": self.quality.get(), "save": "pointing_led_off.jpg", "hard_stop": False
            }))
    
    def _handle_pointing_led_off(self, name, data):
        """Handle pointing LED OFF image"""
        if name == "pointing_led_off.jpg":
            self._pointing_state = 6
            self._set_preview(data)
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
    
    def _handle_generic_saved_image(self, name, data):
        """Handle generic saved image - delegate to ScanController if scanning"""
        # Process through ScanController (handles file save and real-time processing)
        data = self.scan_controller.on_image_received(name, data)
        
        # Update GUI
        self.dl_lbl.config(text=f"DL {len(data)}")
        self._set_preview(data)
        
        # Save undistorted copy if enabled (existing feature)
        if self.ud_save_copy.get() and self.image_processor.has_calibration():
            try:
                nparr = np.frombuffer(data, np.uint8)
                bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if bgr is not None:
                    ud = self.image_processor.undistort(bgr, use_torch=True)
                    base, ext = os.path.splitext(name)
                    ud_name = f"{base}.ud{ext}"
                    ud_path = DEFAULT_OUT_DIR / ud_name
                    cv2.imwrite(str(ud_path), ud)
            except Exception as e:
                print(f"[UD Save] Error: {e}")
        
        if self._resume_preview_after_snap:
            self.resume_preview()
            self._resume_preview_after_snap = False
    
    def _handle_saved_image(self, payload):
        """Route saved image events to specific handlers"""
        name, data = payload
        
        self._handle_center_on_image(name, data)
        self._handle_center_off_image(name, data)
        self._handle_pointing_laser_on(name, data)
        self._handle_pointing_laser_off(name, data)
        self._handle_pointing_led_on(name, data)
        self._handle_pointing_led_off(name, data)
        
        if name not in ["center_on.jpg", "center_off.jpg", "pointing_laser_on.jpg", 
                        "pointing_laser_off.jpg", "pointing_led_on.jpg", "pointing_led_off.jpg"]:
            self._handle_generic_saved_image(name, data)
    
    def _handle_pointing_step2(self):
        """Handle pointing step 2 (LED ON)"""
        self._pointing_state = 4
        self.ctrl.send({"cmd": "led", "value": 255})
        wait_ms = int(self.led_settle.get() * 1000)
        self.root.after(wait_ms, lambda: self.ctrl.send({
            "cmd": "snap", "width": self.width.get(), "height": self.height.get(),
            "quality": self.quality.get(), "save": "pointing_led_on.jpg", "hard_stop": False
        }))
    
    def _handle_preview_on(self):
        """Handle preview enable request"""
        self.preview_enable.set(True)
        self.toggle_preview()
    
    # ========== End of Event Handlers ==========

    def _poll(self):
        """Main event loop - check triggers and process events"""
        self._check_centering_trigger()
        self._check_pointing_trigger()
        
        try:
            while True:
                tag, payload = ui_q.get_nowait()
                
                if tag == "evt":
                    self._handle_server_event(payload)
                elif tag == "preview":
                    self._set_preview(payload)
                elif tag == "saved":
                    self._handle_saved_image(payload)
                elif tag == "toast":
                    print(f"[TOAST] {payload}")
                elif tag == "pointing_step_2":
                    self._handle_pointing_step2()
                elif tag == "preview_on":
                    self._handle_preview_on()
        except queue.Empty:
            pass
        
        self.root.after(POLL_INTERVAL_MS, self._poll)

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

            if self.ud_enable.get() and self.image_processor.has_calibration():
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

    def _centering_on_laser(self, lx: float, ly: float, W: int, H: int):
        """
        레이저 점(lx,ly) 기준 정밀 보정.
        pan: 중앙(W/2) 기준
        tilt: (H/2 + Δy) 기준  ← Δy = self.laser_target_y_offset_px
        """
        import time, numpy as np
        if not self.centering_enable.get():
            self._centering_ok_frames = 0
            return

        # 목표 좌표
        target_x = W/2.0
        target_y = H/2.0 + float(self.laser_target_y_offset_px.get())

        # 오차(px)
        ex = (target_x - float(lx))
        ey = (target_y - float(ly))
        tol = int(self.centering_px_tol.get())

        # 안정 판정
        if abs(ex) <= tol and abs(ey) <= tol:
            self._centering_ok_frames += 1
        else:
            self._centering_ok_frames = 0
        if self._centering_ok_frames >= int(self.centering_min_frames.get()):
            return

        # 쿨다운
        now_ms = int(time.time() * 1000)
        if now_ms - self._centering_last_ms < int(self.centering_cooldown.get()):
            return

        # px/deg 추정: a=∂cx/∂pan, e=∂cy/∂tilt (CSV에서 회귀한 값 보간)
        a = self._interp_fit(getattr(self, "_fits_h", {}), self._curr_tilt, "a", k=2)
        e = self._interp_fit(getattr(self, "_fits_v", {}), self._curr_pan,  "e", k=2)
        if not np.isfinite(a) or abs(a) < 1e-6 or not np.isfinite(e) or abs(e) < 1e-6:
            return

        # 각도 보정량(°) 클램프
        dpan  = float(np.clip(ex / a, -float(self.centering_max_step.get()), float(self.centering_max_step.get())))
        dtilt = float(np.clip(ey / e, -float(self.centering_max_step.get()), float(self.centering_max_step.get())))

        # 누적/전송
        self._curr_pan  = float(self._curr_pan  + dpan)
        self._curr_tilt = float(self._curr_tilt + dtilt)
        self.ctrl.send({
            "cmd":"move",
            "pan":  self._curr_pan,
            "tilt": self._curr_tilt,
            "speed": int(self.point_speed.get()),
            "acc":   float(self.point_acc.get())
        })
        self._centering_last_ms = now_ms
    def _detect_red_laser(self, bgr: np.ndarray):
        """
        빨간 레이저 포인트를 HSV 두 구간(0~H1, H2~180) + S/V 임계로 마스크한 뒤
        연결요소에서 '점수 = 평균 V * 면적'이 가장 큰 blob의 subpixel 중심을 반환.
        반환: (found: bool, cx: float, cy: float, score: float)
        """
        try:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            H1_lo, H1_hi = int(self.laser_h_lo1.get()), int(self.laser_h_hi1.get())
            H2_lo, H2_hi = int(self.laser_h_lo2.get()), int(self.laser_h_hi2.get())
            S_min, V_min = int(self.laser_s_min.get()), int(self.laser_v_min.get())

            h, s, v = cv2.split(hsv)

            # 빨강 hue는 양끝단에 걸려서 두 구간 합침
            m1 = cv2.inRange(h, H1_lo, H1_hi)
            m2 = cv2.inRange(h, H2_lo, H2_hi)
            mh = cv2.bitwise_or(m1, m2)

            ms = cv2.inRange(s, S_min, 255)
            mv = cv2.inRange(v, V_min, 255)

            mask = cv2.bitwise_and(mh, cv2.bitwise_and(ms, mv))

            # 모폴로지 오픈으로 노이즈 제거
            ksz = max(0, int(self.laser_open_ksz.get()))
            if ksz > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*ksz+1, 2*ksz+1))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

            # 연결요소로 blob 스코어링
            num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if num <= 1:
                return (False, 0.0, 0.0, 0.0)

            minA = int(self.laser_min_area.get())
            maxA = int(self.laser_max_area.get())

            best = (-1, -1.0)  # (idx, score)
            for i in range(1, num):
                x,y,w,h,a = stats[i]
                if a < minA or a > maxA: 
                    continue

                # blob 평균 V를 계산해서 포화/광원과 구분(채도가 이미 크지만 보정)
                mask_i = (labels == i).astype(np.uint8)
                mean_v = float(cv2.mean(v, mask=mask_i)[0])
                score = mean_v * float(a)  # 간단한 점수 함수

                if score > best[1]:
                    best = (i, score)

            if best[0] < 0:
                return (False, 0.0, 0.0, 0.0)

            # subpixel 무게중심(밝기 가중치; V를 가중치로 사용)
            i = best[0]
            mask_i = (labels == i).astype(np.uint8)
            ys, xs = np.nonzero(mask_i)
            if xs.size == 0:
                return (False, 0.0, 0.0, 0.0)
            weights = v[ys, xs].astype(np.float32) + 1.0
            cx = float(np.sum(xs * weights) / np.sum(weights))
            cy = float(np.sum(ys * weights) / np.sum(weights))
            return (True, cx, cy, float(best[1]))
        except Exception as e:
            print("[laser] detect err:", e)
            return (False, 0.0, 0.0, 0.0)
    def _align_laser_to_film(self, lx: float, ly: float, tx: float, ty: float, W: int, H: int):
        """
        레이저 (lx, ly)를 타깃 (tx, ty) = '필름 중심'으로 정렬.
        px 오차 → a,e로 각도 환산 → 쿨다운/클램프 → 이동
        """
        import time, numpy as np
        if not self.centering_enable.get():
            return

        ex = float(tx) - float(lx)
        ey = float(ty) - float(ly)

        # 쿨다운
        now_ms = int(time.time() * 1000)
        if now_ms - self._centering_last_ms < int(self.centering_cooldown.get()):
            return

        # px/deg 추정
        a = self._interp_fit(getattr(self, "_fits_h", {}), self._curr_tilt, "a", k=2)
        e = self._interp_fit(getattr(self, "_fits_v", {}), self._curr_pan,  "e", k=2)
        if not np.isfinite(a) or abs(a) < 1e-6 or not np.isfinite(e) or abs(e) < 1e-6:
            return

        dpan  = float(np.clip(ex / a, -float(self.centering_max_step.get()), float(self.centering_max_step.get())))
        dtilt = float(np.clip(ey / e, -float(self.centering_max_step.get()), float(self.centering_max_step.get())))

        self._curr_pan  = float(self._curr_pan  + dpan)
        self._curr_tilt = float(self._curr_tilt + dtilt)

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
