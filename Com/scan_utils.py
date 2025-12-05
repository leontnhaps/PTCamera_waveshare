#!/usr/bin/env python3
"""
Scan processing controller
Real-time YOLO detection and CSV logging during scans
"""

import cv2
import numpy as np
import csv
import re
import pathlib
import threading
import queue  # ← Worker Thread용 Queue 추가


class ScanController:
    """실시간 스캔 처리 관리자 - Worker Thread pattern for async processing"""
    
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
        
        # ===== Worker Thread (핵심 최적화!) =====
        self.processing_queue = queue.Queue(maxsize=50)  # 이미지 처리 큐
        self.worker_thread = None
        self.worker_running = False
        
        # CSV writer
        self.csv_writer = None
        self.csv_file = None
        self.csv_path = None
        
        # Statistics
        self.processed_count = 0
        self.detected_count = 0
        # ===== Worker Thread (핵심 최적화!) =====
        self.processing_queue = queue.Queue(maxsize=50)  # 이미지 처리 큐
        self.worker_thread = None
        self.worker_running = False
    
    def _start_worker_thread(self):
        """Start background worker thread for heavy processing"""
        if self.worker_thread and self.worker_thread.is_alive():
            return
        
        self.worker_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        print("[ScanController] Worker thread started")
    
    def _stop_worker_thread(self):
        """Stop worker thread"""
        self.worker_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
        print("[ScanController] Worker thread stopped")
    
    def _worker_loop(self):
        """Worker thread loop - processes images asynchronously"""
        while self.worker_running:
            try:
                # Get task from queue (timeout to check worker_running)
                task = self.processing_queue.get(timeout=0.5)
                if task is None:  # Poison pill
                    break
                
                pan, tilt, pair = task
                self._process_pair(pan, tilt, pair)
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ScanController] Worker error: {e}")
    
    def start_scan(self, session_name, yolo_path):
        """Start scan - create CSV file and worker thread"""
        self.is_scanning = True
        self.yolo_weights_path = yolo_path
        self.image_pairs.clear()
        self.processed_count = 0
        self.detected_count = 0
        
        # Start worker thread
        self._start_worker_thread()
        
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
        """Real-time processing: undistort → buffer → enqueue when pair complete"""
        # Decode image
        try:
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return
        except Exception as e:
            print(f"[ScanController] Image decode failed: {e}")
            return
        
        # Undistort (여전히 메인 스레드에서 - 빠름)
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
            # ===== 핵심: Worker Thread로 전달 =====
            try:
                self.processing_queue.put_nowait((pan, tilt, pair))
                del self.image_pairs[key]  # Free memory immediately
            except queue.Full:
                print(f"[ScanController] Queue full, skipping ({pan}, {tilt})")
                del self.image_pairs[key]
    
    def _process_pair(self, pan, tilt, pair):
        """Process complete pair: diff → YOLO → CSV"""
        # Import here to avoid circular dependency at module load time
        from yolo_utils import predict_with_tiling
        
        # YOLO constants (copied from Com_main to avoid circular import)
        YOLO_TILE_ROWS = 2
        YOLO_TILE_COLS = 3
        YOLO_TILE_OVERLAP = 0.15
        YOLO_CONF_THRESHOLD = 0.20
        YOLO_IOU_THRESHOLD = 0.45
        
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
