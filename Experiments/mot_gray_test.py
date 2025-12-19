import cv2
import numpy as np
import sys
import os
from numpy.linalg import norm
from ultralytics import YOLO

# ---------------------------------------------------------
# [1] 기존 모듈 로드
# ---------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), 'Com'))

try:
    from yolo_utils import predict_with_tiling
    print("✅ 기존 알고리즘(yolo_utils) 로드 성공!")
except ImportError:
    print("❌ 오류: Com/yolo_utils.py를 찾을 수 없습니다.")
    sys.exit()

# =========================================================
# [설정] 테스트할 이미지 및 모델 경로
# =========================================================
MODEL_PATH = "yolov11m_diff.pt"

# 이미지 경로

# IMG_PREV_ON = r"C:\Users\gmlwn\OneDrive\바탕 화면\ICon1학년\OpticalWPT\추계 이후자료\Diff YOLO Test\captures_gui_20251201_004045\img_t-15_p-075_20251201_004206_138_led_on_ud.jpg"
# IMG_PREV_OFF = r"C:\Users\gmlwn\OneDrive\바탕 화면\ICon1학년\OpticalWPT\추계 이후자료\Diff YOLO Test\captures_gui_20251201_004045\img_t-15_p-075_20251201_004206_588_led_off_ud.jpg"
# IMG_CURR_ON = r"C:\Users\gmlwn\OneDrive\바탕 화면\ICon1학년\OpticalWPT\추계 이후자료\Diff YOLO Test\captures_gui_20251201_004045\img_t-15_p-090_20251201_004204_348_led_on_ud.jpg"
# IMG_CURR_OFF = rC:\Users\gmlwn\OneDrive\바탕 화면\ICon1학년\OpticalWPT\추계 이후자료\Diff YOLO Test\captures_gui_20251201_004045\img_t-15_p-090_20251201_004204_803_led_off_ud.jpg


IMG_PREV_ON = r"C:\Users\gmlwn\OneDrive\바탕 화면\ICon1학년\OpticalWPT\추계 이후자료\Diff YOLO Dataset\젤먼거4\img_t+15_p-135_20251128_220759_113_led_on_ud.jpg"
IMG_PREV_OFF = r"C:\Users\gmlwn\OneDrive\바탕 화면\ICon1학년\OpticalWPT\추계 이후자료\Diff YOLO Dataset\젤먼거4\img_t+15_p-135_20251128_220759_817_led_off_ud.jpg"
IMG_CURR_ON = r"C:\Users\gmlwn\OneDrive\바탕 화면\ICon1학년\OpticalWPT\추계 이후자료\Diff YOLO Dataset\젤먼거4\img_t+15_p-150_20251128_220756_809_led_on_ud.jpg"
IMG_CURR_OFF = r"C:\Users\gmlwn\OneDrive\바탕 화면\ICon1학년\OpticalWPT\추계 이후자료\Diff YOLO Dataset\젤먼거4\img_t+15_p-150_20251128_220757_770_led_off_ud.jpg"

CONF_THRES = 0.50 
IOU_THRES = 0.45

# =========================================================
# 핵심 로직 (특징 추출 & 유사도) - GRAYSCALE 버전
# =========================================================
def get_feature_vector(roi_bgr):
    """
    [🆕 GRAYSCALE 방식]
    색상(H) 대신 밝기 패턴만으로 특징 추출!
    - 색상 정보 제거 → 명암 구조에 집중
    - HSV 15x8=120차원 → Gray 32차원으로 단순화
    """
    if roi_bgr is None or roi_bgr.size == 0: return None
    
    # 1. Grayscale 변환
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. [옵션] 마스크 생성 (너무 어두운 픽셀 제거)
    # V < 30 제거 효과 유지 (배경 그림자 무시)
    mask = cv2.inRange(gray, 30, 255)
    
    # 3. 히스토그램 계산 (1D, 32 bins)
    # 0~255 밝기 범위를 32개 구간으로 나눔
    hist = cv2.calcHist([gray], [0], mask, [32], [0, 256])
    
    # 4. 정규화 & 벡터화
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten()  # 32차원 벡터

def calc_cosine_similarity(vec_a, vec_b):
    if vec_a is None or vec_b is None: return 0.0
    dot = np.dot(vec_a, vec_b)
    n_a, n_b = norm(vec_a), norm(vec_b)
    if n_a == 0 or n_b == 0: return 0.0
    return dot / (n_a * n_b)

def process_step(model, img_on, img_off, step_name="Step"):
    print(f"\n--- Processing {step_name} ---")
    diff = cv2.absdiff(img_on, img_off)
    
    # YOLO 수행
    boxes, scores, classes = predict_with_tiling(
        model, diff, rows=2, cols=3, overlap=0.15, 
        conf=CONF_THRES, iou=IOU_THRES
    )
    print(f"   -> {len(boxes)}개 객체 검출됨.")
    
    objects = []
    H, W = img_on.shape[:2]
    
    # ★ [설정] 패딩 비율
    PADDING_RATIO = 2.0 
    
    for i, (x, y, w, h) in enumerate(boxes):
        # 1. 패딩 크기 계산
        pad_w = int(w * PADDING_RATIO)
        pad_h = int(h * PADDING_RATIO)
        
        # 2. 좌표 확장
        x1 = max(0, int(x - pad_w))
        y1 = max(0, int(y - pad_h))
        x2 = min(W, int(x + w + pad_w))
        y2 = min(H, int(y + h + pad_h))
        
        # 3. ROI 추출 (LED ON 원본에서)
        roi = img_on[y1:y2, x1:x2]
        
        if roi.size == 0: continue

        vec = get_feature_vector(roi)
        
        objects.append({
            'id': i,
            'box': (x1, y1, x2-x1, y2-y1),
            'roi': roi,
            'vec': vec
        })
    return objects

# =========================================================
# ★ 시각화 함수들
# =========================================================
def show_roi_grid(objects, window_name):
    """검출된 모든 ROI를 한 창에 모아서 보여줌"""
    if not objects: return

    display_h = 150
    images = []
    
    for obj in objects:
        roi = obj['roi']
        h, w = roi.shape[:2]
        if h == 0 or w == 0: continue
        
        # 비율 유지 리사이징
        scale = display_h / h
        new_w = int(w * scale)
        resized_roi = cv2.resize(roi, (new_w, display_h))
        
        # 테두리 및 ID 텍스트 추가
        vis = cv2.copyMakeBorder(resized_roi, 20, 2, 2, 2, cv2.BORDER_CONSTANT, value=(40, 40, 40))
        cv2.putText(vis, f"ID:{obj['id']}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        images.append(vis)
    
    if images:
        grid_img = np.hstack(images)
        cv2.imshow(window_name, grid_img)

def show_matched_pairs(matches):
    """매칭된 쌍(Prev <-> Curr)을 위아래로 나열해서 보여줌"""
    if not matches:
        print("매칭된 결과가 없습니다.")
        return

    display_h = 120
    pair_images = []

    for (p_obj, c_obj, sim) in matches:
        roi1 = p_obj['roi']
        roi2 = c_obj['roi']
        
        # 높이 맞춤
        h1, w1 = roi1.shape[:2]
        scale1 = display_h / h1
        roi1_vis = cv2.resize(roi1, (int(w1 * scale1), display_h))
        
        h2, w2 = roi2.shape[:2]
        scale2 = display_h / h2
        roi2_vis = cv2.resize(roi2, (int(w2 * scale2), display_h))
        
        # 가운데 연결 정보창
        info_w = 150
        info_panel = np.zeros((display_h, info_w, 3), dtype=np.uint8)
        
        color = (0, 255, 0)
        cv2.putText(info_panel, f"Sim: {sim:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(info_panel, "-------->", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 합치기
        pair_row = np.hstack((roi1_vis, info_panel, roi2_vis))
        pair_row = cv2.copyMakeBorder(pair_row, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        
        cv2.putText(pair_row, f"Prev ID:{p_obj['id']}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(pair_row, f"Curr ID:{c_obj['id']}", (pair_row.shape[1] - 100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        pair_images.append(pair_row)
    
    # 세로로 쌓기
    max_w = max([img.shape[1] for img in pair_images])
    
    final_stack = []
    for img in pair_images:
        h, w = img.shape[:2]
        if w < max_w:
            img = cv2.copyMakeBorder(img, 0, 0, 0, max_w - w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        final_stack.append(img)
        
    result_img = np.vstack(final_stack)
    cv2.imshow("Matched Pairs Result (GRAYSCALE)", result_img)

# =========================================================
# 메인 실행
# =========================================================
def main():
    if not os.path.exists(MODEL_PATH):
        print("❌ 모델 파일 없음")
        return
    model = YOLO(MODEL_PATH)

    prev_on = cv2.imread(IMG_PREV_ON)
    prev_off = cv2.imread(IMG_PREV_OFF)
    curr_on = cv2.imread(IMG_CURR_ON)
    curr_off = cv2.imread(IMG_CURR_OFF)

    if prev_on is None or curr_on is None:
        print("❌ 이미지 로드 실패")
        return

    # 1. 처리
    prev_objs = process_step(model, prev_on, prev_off, "PREV (Step 1)")
    curr_objs = process_step(model, curr_on, curr_off, "CURR (Step 2)")

    if not prev_objs or not curr_objs:
        print("❌ 객체 검출 실패")
        return

    # 2. ROI 확인창
    show_roi_grid(prev_objs, "Step 1: Prev ROIs (Grayscale)")
    show_roi_grid(curr_objs, "Step 2: Curr ROIs (Grayscale)")

    # 3. 매칭 수행
    print("\n=== 🤝 매칭 분석 (Cosine Similarity - GRAYSCALE) ===")
    print(f"특징 벡터 차원: 32 (밝기 히스토그램만 사용)")
    matches = []
    
    for p_obj in prev_objs:
        p_id = p_obj['id']
        best_sim = -1.0
        best_match_idx = -1
        
        for c_idx, c_obj in enumerate(curr_objs):
            c_id = c_obj['id']
            sim = calc_cosine_similarity(p_obj['vec'], c_obj['vec'])
            print(f"   👉 Prev[{p_id}] vs Curr[{c_id}] : Sim {sim:.4f}")
            if sim > best_sim:
                best_sim = sim
                best_match_idx = c_idx
        
        # 임계값 0.8
        if best_sim > 0.8:
            print(f"✅ MATCH: Prev[{p_obj['id']}] <==> Curr[{best_match_idx}] (Sim: {best_sim:.4f})")
            matches.append((p_obj, curr_objs[best_match_idx], best_sim))
        else:
            print(f"❌ NO MATCH for Prev[{p_obj['id']}] (Best Sim was {best_sim:.4f})")

    # 4. 매칭 결과 시각화
    if matches:
        show_matched_pairs(matches)
    else:
        print("⚠️ 시각화할 매칭 결과가 없습니다.")

    print("\n📸 모든 창이 떴습니다. 아무 키나 누르면 종료합니다.")
    print("\n🔬 [Grayscale 방식 특징]")
    print("  - 색상 정보 제거 (H, S 무시)")
    print("  - 밝기 패턴만 사용 (32-bin histogram)")
    print("  - HSV 120차원 → Gray 32차원 (75% 감소)")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
