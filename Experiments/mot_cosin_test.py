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

# 이미지 경로 (형님 파일명으로 수정!)
IMG_PREV_ON = r"C:\Users\gmlwn\OneDrive\바탕 화면\ICon1학년\OpticalWPT\추계 이후자료\Diff YOLO Test\captures_gui_20251201_004045\img_t-15_p-075_20251201_004206_138_led_on_ud.jpg"
IMG_PREV_OFF = r"C:\Users\gmlwn\OneDrive\바탕 화면\ICon1학년\OpticalWPT\추계 이후자료\Diff YOLO Test\captures_gui_20251201_004045\img_t-15_p-075_20251201_004206_588_led_off_ud.jpg"
IMG_CURR_ON = r"C:\Users\gmlwn\OneDrive\바탕 화면\ICon1학년\OpticalWPT\추계 이후자료\Diff YOLO Test\captures_gui_20251201_004045\img_t-15_p-090_20251201_004204_348_led_on_ud.jpg"
IMG_CURR_OFF = r"C:\Users\gmlwn\OneDrive\바탕 화면\ICon1학년\OpticalWPT\추계 이후자료\Diff YOLO Test\captures_gui_20251201_004045\img_t-15_p-090_20251201_004204_803_led_off_ud.jpg"

CONF_THRES = 0.50 
IOU_THRES = 0.45
# ⭐ 고정 ROI 크기 (중심 기준)
ROI_SIZE = 200  # 200x200 픽셀

# =========================================================
# 핵심 로직 (특징 추출 & 유사도)
# =========================================================
def get_feature_vector(roi_bgr):
    if roi_bgr is None or roi_bgr.size == 0: return None
    
    # 1. HSV 변환
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    
    # 2. [핵심] 마스크 생성 (배경 제거)
    # Saturation(채도)이 30보다 낮은 놈(회색/흰색 벽)은 무시!
    # Value(밝기)가 30보다 낮은 놈(검은 그림자)도 무시!
    mask = cv2.inRange(hsv, (0, 0, 30), (180, 255, 255))
    
    # 3. 히스토그램 계산 (Mask 적용)
    # Hue 구간을 30 -> 15로 줄임 (색상을 더 뭉뚱그려서 봄)
    # Saturation 구간도 32 -> 8로 줄임 (진하기 차이는 크게 신경 안 씀)
    hist = cv2.calcHist([hsv], [0, 1], mask, [15, 8], [0, 180, 0, 256])
    
    # 4. 정규화 & 벡터화
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten()

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
    
    # ★ [설정] 패딩 비율 (0.5 = 50% 더 넓게 봄)
    # 값을 키울수록 주변을 더 많이 봅니다. (0.3 ~ 0.5 추천)
    PADDING_RATIO = 2.0 
    
    for i, (x, y, w, h) in enumerate(boxes):
        # ⭐ 객체 중심 계산
        center_x = int(x + w / 2)
        center_y = int(y + h / 2)
        
        # ⭐ 중심 기준 고정 크기 ROI
        half_size = ROI_SIZE // 2
        x1 = max(0, center_x - half_size)
        y1 = max(0, center_y - half_size)
        x2 = min(W, center_x + half_size)
        y2 = min(H, center_y + half_size)
        
        # 3. 넓어진 좌표로 ROI 추출 (LED ON 원본에서)
        roi = img_on[y1:y2, x1:x2]
        
        # (혹시 너무 구석이라 ROI가 비어있으면 스킵)
        if roi.size == 0: continue

        vec = get_feature_vector(roi)
        
        objects.append({
            'id': i,
            'box': (x1, y1, x2-x1, y2-y1), # 박스 정보도 넓어진 걸로 저장
            'roi': roi,
            'vec': vec
        })
    return objects

# =========================================================
# ★ [신규] 시각화 함수들 (ROI 그리드 & 매칭 페어)
# =========================================================
def show_roi_grid(objects, window_name):
    """검출된 모든 ROI를 한 창에 모아서 보여줌"""
    if not objects: return

    # 보기 좋게 모든 ROI를 높이 100px로 리사이징해서 가로로 붙임
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

    display_h = 120 # 보여줄 높이
    pair_images = []

    for (p_obj, c_obj, sim) in matches:
        roi1 = p_obj['roi']
        roi2 = c_obj['roi']
        
        # 높이 맞춤 (Prev 기준)
        h1, w1 = roi1.shape[:2]
        scale1 = display_h / h1
        roi1_vis = cv2.resize(roi1, (int(w1 * scale1), display_h))
        
        # 높이 맞춤 (Curr 기준)
        h2, w2 = roi2.shape[:2]
        scale2 = display_h / h2
        roi2_vis = cv2.resize(roi2, (int(w2 * scale2), display_h))
        
        # 가운데 연결 정보창 (점수 표시)
        info_w = 150
        info_panel = np.zeros((display_h, info_w, 3), dtype=np.uint8)
        
        color = (0, 255, 0) # 매칭 성공 초록색
        cv2.putText(info_panel, f"Sim: {sim:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(info_panel, "-------->", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # [Prev ROI] + [점수판] + [Curr ROI] 합치기
        pair_row = np.hstack((roi1_vis, info_panel, roi2_vis))
        
        # 테두리
        pair_row = cv2.copyMakeBorder(pair_row, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        
        # 텍스트 추가 (왼쪽: Prev ID, 오른쪽: Curr ID)
        cv2.putText(pair_row, f"Prev ID:{p_obj['id']}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        # 오른쪽 끝 좌표 계산이 귀찮으니 중간쯤에 씀
        cv2.putText(pair_row, f"Curr ID:{c_obj['id']}", (pair_row.shape[1] - 100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        pair_images.append(pair_row)
    
    # 세로로 모든 매칭 결과 쌓기
    # 너비가 다를 수 있으므로 최대 너비에 맞춰서 패딩
    max_w = max([img.shape[1] for img in pair_images])
    
    final_stack = []
    for img in pair_images:
        h, w = img.shape[:2]
        if w < max_w:
            img = cv2.copyMakeBorder(img, 0, 0, 0, max_w - w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        final_stack.append(img)
        
    result_img = np.vstack(final_stack)
    cv2.imshow("Matched Pairs Result", result_img)

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

    # 2. ROI 확인창 띄우기 (잘 따졌나 눈으로 확인)
    show_roi_grid(prev_objs, "Step 1: Prev ROIs")
    show_roi_grid(curr_objs, "Step 2: Curr ROIs")

    # 3. 매칭 수행
    print("\n=== 🤝 매칭 분석 (Cosine Similarity) ===")
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

    # 4. 매칭 결과 시각화 (짝지어서 보여주기)
    if matches:
        show_matched_pairs(matches)
    else:
        print("⚠️ 시각화할 매칭 결과가 없습니다.")

    print("\n📸 모든 창이 떴습니다. 아무 키나 누르면 종료합니다.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()