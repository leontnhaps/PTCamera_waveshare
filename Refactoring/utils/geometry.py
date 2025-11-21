# utils/geometry.py
"""기하학 및 통계 유틸리티 함수들"""
import numpy as np
from typing import Dict, Optional


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    선형 회귀: y = ax + b
    
    Args:
        x: 독립 변수 배열
        y: 종속 변수 배열
    
    Returns:
        (a, b): 기울기와 절편
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    if len(x) < 2:
        return np.nan, np.nan
    
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R² 결정계수 계산
    
    Args:
        y_true: 실제 값
        y_pred: 예측 값
    
    Returns:
        R² 값 (0~1, 1에 가까울수록 좋은 피팅)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    ss_res = float(np.sum((y_true - y_pred)**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2)) + 1e-9
    return 1.0 - ss_res / ss_tot


def weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    """
    가중 평균 계산
    
    Args:
        values: 값들
        weights: 가중치들
    
    Returns:
        가중 평균값
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    if len(values) == 0 or np.sum(weights) == 0:
        return np.nan
    
    return float(np.sum(values * weights) / np.sum(weights))


def interpolate_nearest_k(data_dict: Dict[float, Dict], query: float, 
                         value_key: str, k: int = 2) -> float:
    """
    근처 k개 값으로 1/거리 가중 보간
    
    Args:
        data_dict: {키: {value_key: 값, ...}} 형태의 딕셔너리
        query: 보간할 위치
        value_key: 추출할 값의 키 이름
        k: 사용할 근처 점의 개수
    
    Returns:
        보간된 값
    
    Example:
        >>> data = {0.0: {'slope': 2.0}, 1.0: {'slope': 3.0}, 2.0: {'slope': 2.5}}
        >>> interpolate_nearest_k(data, 0.5, 'slope', k=2)
        2.33...  # 0.0과 1.0 사이 보간
    """
    if not data_dict:
        return np.nan
    
    keys = np.array(list(data_dict.keys()), dtype=float)
    values = np.array([data_dict[k][value_key] for k in data_dict], dtype=float)
    
    # 거리 기준 정렬하여 가장 가까운 k개 선택
    distances = np.abs(keys - query)
    order = np.argsort(distances)[:max(1, min(k, len(keys)))]
    
    selected_keys = keys[order]
    selected_values = values[order]
    
    # 거리 기반 가중치 (거리가 0이면 무한대 가중치 방지)
    d = np.abs(selected_keys - query) + 1e-6
    weights = 1.0 / d
    
    return float(np.sum(selected_values * weights) / np.sum(weights))
