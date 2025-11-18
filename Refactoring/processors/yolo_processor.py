# processors/yolo_processor.py
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    _YOLO_OK = True
except Exception:
    YOLO = None
    _YOLO_OK = False

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    _TORCH_AVAILABLE = False


class YOLOProcessor:
    """YOLO 객체 감지 및 결과 시각화 처리기"""
    
    def __init__(self, device="auto"):
        self.model = None
        self.model_path = ""
        self._last_detection = None
        self._detection_idx = 0
        
        # 디바이스 설정
        if device == "auto":
            self.device = (0 if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu")
        else:
            self.device = device
            
        # 시각화 옵션
        self.box_thickness = 4
        self.text_scale = 0.7
        self.text_thickness = 2
        self.show_centroid = True
        self.show_center_cross = True
        
        print(f"[YOLOProcessor] device={self.device}, available={_YOLO_OK}")
    
    def load_model(self, model_path: str) -> bool:
        """YOLO 모델 로드"""
        if not _YOLO_OK:
            print("[YOLOProcessor] Ultralytics not available")
            return False
            
        try:
            self.model = YOLO(model_path)
            self.model_path = model_path
            
            # 워밍업
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, imgsz=640, conf=0.25, iou=0.55, 
                             device=self.device, verbose=False)
            
            print(f"[YOLOProcessor] Model loaded: {model_path}")
            return True
        except Exception as e:
            print(f"[YOLOProcessor] Load failed: {e}")
            self.model = None
            return False
    
    def is_loaded(self) -> bool:
        """모델이 로드되었는지 확인"""
        return self.model is not None
    
    def process(self, image: np.ndarray, conf: float = 0.25, iou: float = 0.55, 
                imgsz: int = 832, stride: int = 2) -> np.ndarray:
        """
        이미지에서 YOLO 추론 및 결과 그리기
        stride: N프레임마다만 추론 (성능 최적화)
        """
        if not self.is_loaded():
            return image
            
        image = np.ascontiguousarray(image, dtype=np.uint8)
        
        try:
            # N프레임마다만 추론 실행
            run_inference = (self._detection_idx % max(1, stride)) == 0
            
            if run_inference:
                results = self.model.predict(
                    image,
                    imgsz=imgsz,
                    conf=conf,
                    iou=iou,
                    device=self.device,
                    verbose=False
                )[0]
                
                # 결과 저장
                if len(results.boxes) > 0:
                    boxes = results.boxes.xyxy.detach().cpu().numpy().astype(int)
                    confs = results.boxes.conf.detach().cpu().numpy()
                    clses = results.boxes.cls.detach().cpu().numpy().astype(int)
                    self._last_detection = (boxes, confs, clses)
                else:
                    self._last_detection = (np.empty((0, 4), int), np.array([]), np.array([]))
            
            # 캐시된 결과로 그리기
            annotated_image = self._draw_results(image)
            self._detection_idx += 1
            
            return annotated_image
            
        except Exception as e:
            print(f"[YOLOProcessor] Process error: {e}")
            return image
    
    def _draw_results(self, image: np.ndarray) -> np.ndarray:
        """감지 결과를 이미지에 그리기"""
        if self._last_detection is None:
            return image
            
        boxes, confs, clses = self._last_detection
        H, W = image.shape[:2]
        
        # 1) 바운딩 박스와 라벨 그리기
        for (x1, y1, x2, y2), conf, cls in zip(boxes, confs, clses):
            # 박스
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 
                         self.box_thickness, lineType=cv2.LINE_AA)
            
            # 라벨
            label = f"{conf:.2f}"
            org = (x1, max(15, y1 - 6))
            # 텍스트 배경 (검은색)
            cv2.putText(image, label, org, cv2.FONT_HERSHEY_SIMPLEX, 
                       self.text_scale, (0, 0, 0), self.text_thickness + 2, cv2.LINE_AA)
            # 텍스트 전경 (초록색)
            cv2.putText(image, label, org, cv2.FONT_HERSHEY_SIMPLEX, 
                       self.text_scale, (0, 255, 0), self.text_thickness, cv2.LINE_AA)
        
        # 2) 화면 중앙 십자가 (옵션)
        if self.show_center_cross:
            cx0, cy0 = int(W/2), int(H/2)
            cv2.drawMarker(image, (cx0, cy0), (255, 255, 255), 
                          markerType=cv2.MARKER_CROSS, markerSize=14, 
                          thickness=1, line_type=cv2.LINE_AA)
        
        # 3) 검출들의 중심점 평균 (centroid)
        if self.show_centroid and boxes.shape[0] > 0:
            centers = []
            for (x1, y1, x2, y2) in boxes:
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                centers.append((cx, cy))
            
            # 평균 계산
            m_cx = float(np.mean([c[0] for c in centers]))
            m_cy = float(np.mean([c[1] for c in centers]))
            
            # 중심점 그리기
            cv2.circle(image, (int(round(m_cx)), int(round(m_cy))), 5, 
                      (0, 200, 255), -1, lineType=cv2.LINE_AA)
            
            # 오차 텍스트
            err_x = (W/2.0 - m_cx)
            err_y = (H/2.0 - m_cy)
            txt = f"mean ({m_cx:.1f},{m_cy:.1f})  err ({err_x:+.1f},{err_y:+.1f}) px"
            
            # 텍스트 배경
            cv2.putText(image, txt, (10, max(20, H-15)), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, txt, (10, max(20, H-15)), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.55, (0, 255, 255), 1, cv2.LINE_AA)
        
        return image
    
    def get_last_centroid(self) -> tuple:
        """마지막 감지의 중심점 좌표 반환 (센터링 용도)"""
        if self._last_detection is None:
            return None
            
        boxes, confs, clses = self._last_detection
        if boxes.shape[0] == 0:
            return None
            
        centers = []
        for (x1, y1, x2, y2) in boxes:
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            centers.append((cx, cy))
        
        m_cx = float(np.mean([c[0] for c in centers]))
        m_cy = float(np.mean([c[1] for c in centers]))
        
        return (m_cx, m_cy, len(boxes))  # (중심_x, 중심_y, 감지수)
    
    def update_visualization_settings(self, box_thickness=None, text_scale=None, 
                                    text_thickness=None, show_centroid=None, 
                                    show_center_cross=None):
        """시각화 설정 업데이트"""
        if box_thickness is not None:
            self.box_thickness = max(1, int(box_thickness))
        if text_scale is not None:
            self.text_scale = max(0.3, float(text_scale))
        if text_thickness is not None:
            self.text_thickness = max(1, int(text_thickness))
        if show_centroid is not None:
            self.show_centroid = bool(show_centroid)
        if show_center_cross is not None:
            self.show_center_cross = bool(show_center_cross)