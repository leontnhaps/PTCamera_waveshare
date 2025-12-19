"""
MOT Scan Simulation Test
기존 스캔 이미지 폴더로 전체 추적 알고리즘 테스트
"""
import cv2
import numpy as np
import sys
import os
import re
from pathlib import Path
from numpy.linalg import norm
from ultralytics import YOLO

# ---------------------------------------------------------
# 기존 모듈 로드
# ---------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Com'))

try:
    from yolo_utils import predict_with_tiling
    print("✅ yolo_utils 로드 성공!")
except ImportError:
    print("❌ 오류: Com/yolo_utils.py를 찾을 수 없습니다.")
    sys.exit()

# =========================================================
# [설정] 스캔 이미지 폴더 경로
# =========================================================
MODEL_PATH = "yolov11m_diff.pt"

# ⭐ 여기에 스캔 폴더 경로 입력! (예시)
SCAN_FOLDER = r"C:\Users\gmlwn\OneDrive\바탕 화면\ICon1학년\OpticalWPT\추계 이후자료\Diff YOLO Test\captures_gui_20251201_004045"

CONF_THRES = 0.50
IOU_THRES = 0.45
PADDING_RATIO = 2.0

# =========================================================
# 특징 추출 (Grayscale)
# =========================================================
def get_feature_vector(roi_bgr):
    """Grayscale 32-bin 히스토그램"""
    if roi_bgr is None or roi_bgr.size == 0:
        return None
    
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, 30, 255)
    hist = cv2.calcHist([gray], [0], mask, [32], [0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten()

def calc_cosine_similarity(vec_a, vec_b):
    """코사인 유사도"""
    if vec_a is None or vec_b is None:
        return 0.0
    dot = np.dot(vec_a, vec_b)
    n_a, n_b = norm(vec_a), norm(vec_b)
    if n_a == 0 or n_b == 0:
        return 0.0
    return dot / (n_a * n_b)

# =========================================================
# MOT Tracker (Pan + Tilt 양방향)
# =========================================================
class ObjectTracker:
    def __init__(self):
        self.next_id = 0
        # {(pan, tilt): [{'box', 'vec', 'track_id'}, ...]}
        self.frame_objects = {}
        
    def reset(self):
        self.next_id = 0
        self.frame_objects = {}
        
    def add_detections(self, boxes, scores, img_on, pan, tilt):
        """
        양방향 비교:
        1. 직전 Pan 프레임 (같은 Tilt)
        2. 이전 Tilt 프레임 (같은 Pan)
        """
        # 현재 프레임 특징 추출
        curr_objects = []
        H, W = img_on.shape[:2]
        
        for i, (x, y, w, h) in enumerate(boxes):
            pad_w = int(w * PADDING_RATIO)
            pad_h = int(h * PADDING_RATIO)
            x1 = max(0, int(x - pad_w))
            y1 = max(0, int(y - pad_h))
            x2 = min(W, int(x + w + pad_w))
            y2 = min(H, int(y + h + pad_h))
            
            roi = img_on[y1:y2, x1:x2]
            if roi.size == 0:
                continue
                
            vec = get_feature_vector(roi)
            curr_objects.append({
                'box': (x, y, w, h),
                'vec': vec,
                'idx': i
            })
        
        # 이전 프레임 후보 찾기
        pan_candidates = self._find_prev_pan_candidates(pan, tilt)
        tilt_candidates = self._find_prev_tilt_candidates(pan, tilt)
        
        # 매칭 수행
        track_ids = []
        for obj in curr_objects:
            best_id = self._match_object(obj, pan_candidates, tilt_candidates)
            obj['track_id'] = best_id
            track_ids.append(best_id)
        
        # 현재 프레임 저장
        self.frame_objects[(pan, tilt)] = curr_objects
        
        return track_ids
    
    def _find_prev_pan_candidates(self, pan, tilt):
        """직전 Pan 프레임 찾기 (같은 Tilt)"""
        # Pan은 보통 15도 간격 (-180, -165, -150, ...)
        prev_pan = pan - 15
        if prev_pan < -180:
            prev_pan = 180  # 순환
        return self.frame_objects.get((prev_pan, tilt), [])
    
    def _find_prev_tilt_candidates(self, pan, tilt):
        """이전 Tilt 프레임 찾기 (같은 Pan)"""
        # Tilt는 보통 15도 간격 (-15, 0, 15, ...)
        prev_tilt = tilt - 15
        return self.frame_objects.get((pan, prev_tilt), [])
    
    def _match_object(self, curr_obj, pan_candidates, tilt_candidates):
        """양쪽 후보와 비교하여 최적 매칭"""
        best_match_id = None
        best_sim = 0.8  # 임계값
        
        # Pan 후보들과 비교
        for candidate in pan_candidates:
            sim = calc_cosine_similarity(curr_obj['vec'], candidate['vec'])
            if sim > best_sim:
                best_sim = sim
                best_match_id = candidate['track_id']
        
        # Tilt 후보들과 비교
        for candidate in tilt_candidates:
            sim = calc_cosine_similarity(curr_obj['vec'], candidate['vec'])
            if sim > best_sim:
                best_sim = sim
                best_match_id = candidate['track_id']
        
        # 매칭 실패 시 새 ID 부여
        if best_match_id is None:
            best_match_id = self.next_id
            self.next_id += 1
        
        return best_match_id

# =========================================================
# 스캔 이미지 파싱 및 정렬
# =========================================================
def parse_scan_images(scan_folder):
    """
    스캔 폴더에서 이미지 파싱
    Returns: [(pan, tilt, 'on'/'off', filepath), ...]
    """
    folder = Path(scan_folder)
    images = []
    
    for img_file in folder.glob("*.jpg"):
        # 파일명 파싱: img_t+15_p-180_..._led_on_ud.jpg
        match = re.search(r't([+-]?\d+)_p([+-]?\d+).*_(led_on|led_off)', img_file.name)
        if not match:
            continue
        
        tilt = int(match.group(1))
        pan = int(match.group(2))
        led_type = 'on' if 'led_on' in match.group(3) else 'off'
        
        images.append((pan, tilt, led_type, str(img_file)))
    
    # 정렬: Tilt 오름차순 → Pan 오름차순
    images.sort(key=lambda x: (x[1], x[0]))
    return images

# =========================================================
# 메인 실행
# =========================================================
def main():
    if not os.path.exists(MODEL_PATH):
        print("❌ 모델 파일 없음")
        return
    
    model = YOLO(MODEL_PATH)
    tracker = ObjectTracker()
    tracker.reset()
    
    # 스캔 이미지 로드
    print(f"\n📂 스캔 폴더: {SCAN_FOLDER}")
    images = parse_scan_images(SCAN_FOLDER)
    print(f"✅ 총 {len(images)}개 이미지 발견\n")
    
    # ON/OFF 쌍 만들기
    pairs = {}
    for pan, tilt, led_type, filepath in images:
        key = (pan, tilt)
        if key not in pairs:
            pairs[key] = {}
        pairs[key][led_type] = filepath
    
    # 정렬된 키 (Tilt → Pan 순서)
    sorted_keys = sorted(pairs.keys(), key=lambda x: (x[1], x[0]))
    
    print("="*60)
    print("🚀 추적 시작!")
    print("="*60)
    
    total_detections = 0
    
    for pan, tilt in sorted_keys:
        pair = pairs[(pan, tilt)]
        
        # ON/OFF 모두 있는지 확인
        if 'on' not in pair or 'off' not in pair:
            continue
        
        # 이미지 로드
        img_on = cv2.imread(pair['on'])
        img_off = cv2.imread(pair['off'])
        
        if img_on is None or img_off is None:
            continue
        
        # Diff 계산
        diff = cv2.absdiff(img_on, img_off)
        
        # YOLO 검출
        boxes, scores, classes = predict_with_tiling(
            model, diff, rows=2, cols=3, overlap=0.15,
            conf=CONF_THRES, iou=IOU_THRES
        )
        
        if not boxes:
            print(f"[Pan={pan:+4d}, Tilt={tilt:+3d}] 검출 없음")
            continue
        
        # 추적
        track_ids = tracker.add_detections(boxes, scores, img_on, pan, tilt)
        
        # 결과 출력
        print(f"[Pan={pan:+4d}, Tilt={tilt:+3d}] {len(boxes)}개 검출 → track_ids: {track_ids}")
        total_detections += len(boxes)
    
    print("\n" + "="*60)
    print("✅ 추적 완료!")
    print(f"총 검출: {total_detections}개")
    print(f"부여된 고유 ID: 0 ~ {tracker.next_id - 1} ({tracker.next_id}개)")
    print("="*60)

if __name__ == "__main__":
    main()
