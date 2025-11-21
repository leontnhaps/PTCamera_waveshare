# controllers/centering_controller.py
"""실시간 YOLO centroid 기반 센터링 컨트롤러"""
import time
import numpy as np
from typing import Optional, Dict

from controllers.pointing_controller import PointingController


class CenteringController:
    """
    YOLO centroid를 화면 중앙으로 미세 조정
    
    알고리즘:
    1. YOLO의 평균 중심점과 화면 중심의 픽셀 오차 계산
    2. Pointing 피팅 데이터로부터 px/deg 기울기 추정
    3. 오차를 각도 보정량으로 변환하여 이동 명령 생성
    4. 연속 N프레임 안정 시 종료
    """
    
    def __init__(self, pointing_controller: PointingController):
        self.pointing = pointing_controller
        
        # 상태
        self.enabled = False
        self.current_pan = 0.0
        self.current_tilt = 0.0
        self.ok_frames = 0
        self.last_command_ms = 0
        
        # 설정 (기본값, GUI에서 오버라이드 가능)
        self.px_tolerance = 5
        self.min_stable_frames = 4
        self.max_step_deg = 1.0
        self.cooldown_ms = 250
    
    def set_current_position(self, pan: float, tilt: float):
        """현재 명령 각도 설정 (초기 위치)"""
        self.current_pan = pan
        self.current_tilt = tilt
        self.ok_frames = 0
    
    def update_settings(self, px_tol: int = None, min_frames: int = None,
                       max_step: float = None, cooldown: int = None):
        """센터링 설정 업데이트"""
        if px_tol is not None:
            self.px_tolerance = px_tol
        if min_frames is not None:
            self.min_stable_frames = min_frames
        if max_step is not None:
            self.max_step_deg = max_step
        if cooldown is not None:
            self.cooldown_ms = cooldown
    
    def process(self, centroid_x: float, centroid_y: float,
                image_w: int, image_h: int,
                speed: int = 100, acc: float = 1.0) -> Optional[Dict]:
        """
        센터링 업데이트 - YOLO centroid 받아서 보정 명령 생성
        
        Args:
            centroid_x: YOLO 평균 중심 x 좌표
            centroid_y: YOLO 평균 중심 y 좌표
            image_w: 이미지 너비
            image_h: 이미지 높이
            speed: 이동 속도
            acc: 가속도
        
        Returns:
            이동 명령 dict {"cmd": "move", "pan": ..., "tilt": ..., "speed": ..., "acc": ...}
            또는 None (명령 불필요)
        """
        if not self.enabled:
            self.ok_frames = 0
            return None
        
        # 픽셀 오차 계산
        err_x = (image_w / 2.0) - centroid_x
        err_y = (image_h / 2.0) - centroid_y
        
        # 안정성 체크
        if abs(err_x) <= self.px_tolerance and abs(err_y) <= self.px_tolerance:
            self.ok_frames += 1
        else:
            self.ok_frames = 0
        
        # 충분히 안정되면 종료
        if self.ok_frames >= self.min_stable_frames:
            return None
        
        # 쿨다운 체크 (명령 과다 방지)
        now_ms = int(time.time() * 1000)
        if now_ms - self.last_command_ms < self.cooldown_ms:
            return None
        
        # px/deg 기울기 추정 (Pointing 피팅 데이터 활용)
        a = self.pointing.get_slope_at(self.current_tilt, axis='pan')   # ∂cx/∂pan
        e = self.pointing.get_slope_at(self.current_pan, axis='tilt')   # ∂cy/∂tilt
        
        # 기울기 유효성 체크
        if not np.isfinite(a) or abs(a) < 1e-6 or not np.isfinite(e) or abs(e) < 1e-6:
            return None
        
        # 각도 보정량 계산 (픽셀 오차 → 각도)
        dpan = err_x / a
        dtilt = err_y / e
        
        # 최대 스텝 제한
        dpan = float(np.clip(dpan, -self.max_step_deg, self.max_step_deg))
        dtilt = float(np.clip(dtilt, -self.max_step_deg, self.max_step_deg))
        
        # 현재 각도 업데이트
        self.current_pan += dpan
        self.current_tilt += dtilt
        
        # 명령 생성
        self.last_command_ms = now_ms
        
        return {
            "cmd": "move",
            "pan": self.current_pan,
            "tilt": self.current_tilt,
            "speed": speed,
            "acc": acc
        }
    
    def reset(self):
        """센터링 상태 초기화"""
        self.ok_frames = 0
        self.last_command_ms = 0
    
    def enable(self):
        """센터링 활성화"""
        self.enabled = True
        self.reset()
    
    def disable(self):
        """센터링 비활성화"""
        self.enabled = False
        self.reset()
