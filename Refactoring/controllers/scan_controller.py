# controllers/scan_controller.py
"""스캔 진행 상황 관리 및 CSV 로깅 컨트롤러"""
import csv
import re
import numpy as np
import cv2
from pathlib import Path
from typing import Optional

from processors.undistort_processor import UndistortProcessor
from processors.yolo_processor import YOLOProcessor


class ScanController:
    """
    스캔 진행 상황 관리 + CSV 로깅
    
    역할:
    1. 스캔 시작 시 CSV 파일 생성
    2. 이미지 수신 → 언디스토트 → YOLO → CSV 기록
    3. 스캔 종료 시 CSV 닫기
    """
    
    def __init__(self, output_dir: Path, 
                 undistort_processor: UndistortProcessor,
                 yolo_processor: YOLOProcessor):
        self.output_dir = output_dir
        self.undistort = undistort_processor
        self.yolo = yolo_processor
        
        # CSV 상태
        self.csv_path: Optional[Path] = None
        self.csv_file = None
        self.csv_writer = None
        
        # 파일명 파싱용 정규식 (예: img_t+30_p-045_20250121_123456.jpg)
        self.filename_pattern = re.compile(
            r"img_t(?P<tilt>[+\-]\d{2,3})_p(?P<pan>[+\-]\d{2,3})_.*\.(jpg|jpeg|png)$",
            re.IGNORECASE
        )
        
        # YOLO 설정 (스캔용)
        self.scan_yolo_conf = 0.50
        self.scan_yolo_imgsz = 832
    
    def start_scan(self, session_id: str) -> bool:
        """
        스캔 시작: CSV 파일 생성 및 헤더 작성
        
        Args:
            session_id: 세션 ID (파일명에 사용)
        
        Returns:
            성공 여부
        """
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.csv_path = self.output_dir / f"{session_id}_detections.csv"
            
            self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                "file", "pan_deg", "tilt_deg", "cx", "cy", "w", "h", "conf", "cls", "W", "H"
            ])
            
            print(f"[ScanController] CSV 생성: {self.csv_path}")
            return True
            
        except Exception as e:
            print(f"[ScanController] CSV 생성 실패: {e}")
            self.csv_file = None
            self.csv_writer = None
            return False
    
    def process_image(self, image_data: bytes, filename: str, 
                     alpha: float = 0.0, yolo_iou: float = 0.55) -> bool:
        """
        이미지 처리: 디코드 → 언디스토트 → YOLO → CSV 기록
        
        Args:
            image_data: JPEG 바이트 데이터
            filename: 파일명 (pan/tilt 각도 추출용)
            alpha: 언디스토트 알파 값
            yolo_iou: YOLO IoU 임계값
        
        Returns:
            성공 여부
        """
        if self.csv_writer is None:
            return False
        
        try:
            # 파일명에서 pan/tilt 각도 추출
            match = self.filename_pattern.search(filename)
            if not match:
                print(f"[ScanController] 파일명 파싱 실패: {filename}")
                return False
            
            pan_deg = float(match.group("pan"))
            tilt_deg = float(match.group("tilt"))
            
            # 이미지 디코드
            arr = np.frombuffer(image_data, np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"[ScanController] 이미지 디코드 실패: {filename}")
                return False
            
            # ★ 언디스토트 (필수)
            if not self.undistort.is_loaded():
                print("[ScanController] ❌ 보정 파라미터 없음 → 스캔 중단")
                return False
            
            bgr = self.undistort.process(bgr, alpha=alpha)
            H, W = bgr.shape[:2]
            
            # YOLO 추론 (시각화 없이 감지만)
            if not self.yolo.is_loaded():
                print("[ScanController] YOLO 모델 없음 → CSV 기록 생략")
                return True  # 이미지 저장은 성공
            
            # YOLO 처리
            _ = self.yolo.process(
                bgr.copy(),
                conf=self.scan_yolo_conf,
                iou=yolo_iou,
                imgsz=self.scan_yolo_imgsz,
                stride=1  # 매 프레임 처리
            )
            
            # 마지막 감지 결과 가져오기
            if self.yolo._last_detection is None:
                return True
            
            boxes, confs, clses = self.yolo._last_detection
            
            # 각 박스를 CSV에 기록
            if boxes.shape[0] > 0:
                for (x1, y1, x2, y2), conf, cls in zip(boxes, confs, clses):
                    if conf < self.scan_yolo_conf:
                        continue
                    
                    cx = 0.5 * (x1 + x2)
                    cy = 0.5 * (y1 + y2)
                    box_w = x2 - x1
                    box_h = y2 - y1
                    
                    self.csv_writer.writerow([
                        filename, pan_deg, tilt_deg,
                        f"{cx:.3f}", f"{cy:.3f}",
                        f"{box_w:.1f}", f"{box_h:.1f}",
                        f"{conf:.3f}", int(cls),
                        W, H
                    ])
            
            return True
            
        except Exception as e:
            print(f"[ScanController] 이미지 처리 실패: {e}")
            return False
    
    def finish_scan(self) -> str:
        """
        스캔 종료: CSV 파일 닫기
        
        Returns:
            완료 메시지
        """
        if self.csv_file:
            try:
                self.csv_file.flush()
                self.csv_file.close()
                message = f"✅ CSV 저장 완료: {self.csv_path}"
            except Exception as e:
                message = f"⚠️ CSV 닫기 오류: {e}"
            finally:
                self.csv_file = None
                self.csv_writer = None
        else:
            message = "⚠️ 스캔 세션이 없습니다."
        
        return message
    
    def is_active(self) -> bool:
        """스캔이 진행 중인지 확인"""
        return self.csv_writer is not None
