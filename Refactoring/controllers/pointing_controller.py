# controllers/pointing_controller.py
"""Pointing 타겟 각도 계산 컨트롤러"""
import csv
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from collections import defaultdict

from utils.geometry import linear_fit, r_squared, weighted_average, interpolate_nearest_k


class PointingController:
    """
    스캔 CSV 데이터로부터 반사판 중심 타겟 각도 계산
    
    알고리즘:
    1. tilt별로 cx = a*pan + b 선형 피팅 → pan_center 계산
    2. pan별로 cy = e*tilt + f 선형 피팅 → tilt_center 계산
    3. 샘플 수 기반 가중 평균으로 최종 타겟 각도 결정
    """
    
    def __init__(self):
        self.fits_h: Dict[float, Dict] = {}  # tilt별 수평 피팅 결과
        self.fits_v: Dict[float, Dict] = {}  # pan별 수직 피팅 결과
        self.frame_size: Optional[Tuple[int, int]] = None  # (W, H)
    
    def compute_target(self, csv_path: str, conf_min: float = 0.5, 
                      min_samples: int = 2) -> Tuple[Optional[float], Optional[float], str]:
        """
        CSV 파일로부터 타겟 각도 계산
        
        Args:
            csv_path: 스캔 결과 CSV 경로
            conf_min: 최소 confidence 임계값
            min_samples: 각 피팅에 필요한 최소 샘플 수
        
        Returns:
            (pan_target, tilt_target, message): 타겟 각도 및 결과 메시지
        """
        try:
            # CSV 읽기 및 필터링
            rows, W_frame, H_frame = self._load_csv(csv_path, conf_min)
            
            if not rows:
                return None, None, "CSV에서 조건을 만족하는 행이 없습니다."
            
            if W_frame is None or H_frame is None:
                return None, None, "CSV에 W/H 정보가 없습니다."
            
            self.frame_size = (W_frame, H_frame)
            
            # 수평 피팅 (tilt별)
            self.fits_h = self._fit_horizontal(rows, W_frame, min_samples)
            
            # 수직 피팅 (pan별)
            self.fits_v = self._fit_vertical(rows, H_frame, min_samples)
            
            # 가중 평균으로 최종 타겟 계산
            pan_target = self._weighted_center(self.fits_h, 'pan_center')
            tilt_target = self._weighted_center(self.fits_v, 'tilt_center')
            
            message = (f"✅ Pointing 계산 완료: pan={pan_target:.3f}°, tilt={tilt_target:.3f}° "
                      f"(수평 피팅: {len(self.fits_h)}개, 수직 피팅: {len(self.fits_v)}개)")
            
            return pan_target, tilt_target, message
            
        except Exception as e:
            return None, None, f"❌ Pointing 계산 실패: {e}"
    
    def _load_csv(self, csv_path: str, conf_min: float) -> Tuple[list, Optional[int], Optional[int]]:
        """CSV 파일 로드 및 필터링"""
        rows = []
        W_frame = H_frame = None
        
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # confidence 필터
                if row.get("conf", "") == "":
                    continue
                conf = float(row["conf"])
                if conf < conf_min:
                    continue
                
                # pan/tilt 존재 확인
                pan_str = row.get("pan_deg")
                tilt_str = row.get("tilt_deg")
                if not pan_str or not tilt_str or pan_str == "" or tilt_str == "":
                    continue
                
                pan = float(pan_str)
                tilt = float(tilt_str)
                cx = float(row["cx"])
                cy = float(row["cy"])
                
                # 프레임 크기 저장
                if W_frame is None and row.get("W"):
                    W_frame = int(row["W"])
                if H_frame is None and row.get("H"):
                    H_frame = int(row["H"])
                
                rows.append((pan, tilt, cx, cy))
        
        return rows, W_frame, H_frame
    
    def _fit_horizontal(self, rows: list, W_frame: int, min_samples: int) -> Dict[float, Dict]:
        """tilt별 수평 피팅: cx = a*pan + b"""
        by_tilt = defaultdict(list)
        for pan, tilt, cx, cy in rows:
            by_tilt[round(tilt, 3)].append((pan, cx))
        
        fits = {}
        for tilt_key, data in by_tilt.items():
            if len(data) < min_samples:
                continue
            
            data.sort(key=lambda v: v[0])
            pans = np.array([p for p, _ in data], dtype=float)
            cxs = np.array([c for _, c in data], dtype=float)
            
            # 선형 피팅
            a, b = linear_fit(pans, cxs)
            
            # R² 계산
            y_pred = a * pans + b
            R2 = r_squared(cxs, y_pred)
            
            # pan_center 계산: cx = W/2일 때의 pan
            pan_center = (W_frame / 2.0 - b) / a if abs(a) > 1e-9 else np.nan
            
            fits[float(tilt_key)] = {
                'a': float(a),
                'b': float(b),
                'R2': float(R2),
                'N': len(data),
                'pan_center': float(pan_center)
            }
        
        return fits
    
    def _fit_vertical(self, rows: list, H_frame: int, min_samples: int) -> Dict[float, Dict]:
        """pan별 수직 피팅: cy = e*tilt + f"""
        by_pan = defaultdict(list)
        for pan, tilt, cx, cy in rows:
            by_pan[round(pan, 3)].append((tilt, cy))
        
        fits = {}
        for pan_key, data in by_pan.items():
            if len(data) < min_samples:
                continue
            
            data.sort(key=lambda v: v[0])
            tilts = np.array([t for t, _ in data], dtype=float)
            cys = np.array([c for _, c in data], dtype=float)
            
            # 선형 피팅
            e, f = linear_fit(tilts, cys)
            
            # R² 계산
            y_pred = e * tilts + f
            R2 = r_squared(cys, y_pred)
            
            # tilt_center 계산: cy = H/2일 때의 tilt
            tilt_center = (H_frame / 2.0 - f) / e if abs(e) > 1e-9 else np.nan
            
            fits[float(pan_key)] = {
                'e': float(e),
                'f': float(f),
                'R2': float(R2),
                'N': len(data),
                'tilt_center': float(tilt_center)
            }
        
        return fits
    
    def _weighted_center(self, fits: Dict[float, Dict], center_key: str) -> Optional[float]:
        """샘플 수 기반 가중 평균"""
        if not fits:
            return None
        
        values = np.array([fits[k][center_key] for k in fits], dtype=float)
        weights = np.array([fits[k]['N'] for k in fits], dtype=float)
        
        # NaN 제거
        valid_mask = np.isfinite(values)
        if not np.any(valid_mask):
            return None
        
        values = values[valid_mask]
        weights = weights[valid_mask]
        
        return weighted_average(values, weights)
    
    def get_slope_at(self, angle: float, axis: str = 'pan') -> float:
        """
        특정 각도에서의 px/deg 기울기 추정 (센터링용)
        
        Args:
            angle: 현재 각도 (pan or tilt)
            axis: 'pan' 또는 'tilt'
        
        Returns:
            기울기 (a 또는 e)
        """
        if axis == 'pan':
            # tilt 근방의 a (∂cx/∂pan) 추정
            return interpolate_nearest_k(self.fits_h, angle, 'a', k=2)
        else:
            # pan 근방의 e (∂cy/∂tilt) 추정
            return interpolate_nearest_k(self.fits_v, angle, 'e', k=2)
