#!/usr/bin/env python3
"""
LED 차분 이미지 최종 로직 튜너 (Auto Logic + Tuning)
- 자동 선택 로직(Universe -> RGB Two)을 시뮬레이션하며
- 오탐(False Positive)을 제거하기 위한 최적의 파라미터를 찾습니다.
"""

import cv2
import numpy as np
from tkinter import Tk, filedialog
import os
import sys

def nothing(x):
    pass

def load_image_with_hangul(image_path):
    try:
        with open(image_path, 'rb') as f:
            image_array = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"❌ 이미지 로드 실패: {e}")
        return None

def select_two_images():
    print("\n[시스템] 파일 선택창을 엽니다...")
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    initial_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(">> 1. LED ON (빛 받은) 이미지를 선택하세요.")
    path_on = filedialog.askopenfilename(
        initialdir=initial_dir, title="1. LED ON 이미지 선택",
        filetypes=(("이미지 파일", "*.jpg *.jpeg *.png *.bmp"), ("모든 파일", "*.*")),
        parent=root
    )
    if not path_on: root.destroy(); return None, None
    
    print(">> 2. LED OFF (빛 없는) 이미지를 선택하세요.")
    path_off = filedialog.askopenfilename(
        initialdir=initial_dir, title="2. LED OFF 이미지 선택",
        filetypes=(("이미지 파일", "*.jpg *.jpeg *.png *.bmp"), ("모든 파일", "*.*")),
        parent=root
    )
    if not path_off: root.destroy(); return None, None

    root.destroy()
    return path_on, path_off

def preprocess_image(img, boost_on):
    processed = img.copy()
    if boost_on == 1:
        processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX)
        processed = cv2.GaussianBlur(processed, (3, 3), 0)
    return processed

def apply_morphology(mask):
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask

# ==========================================
# 메인 실행부
# ==========================================
def main():
    path_on, path_off = select_two_images()
    if not path_on or not path_off: sys.exit()

    img_on = load_image_with_hangul(path_on)
    img_off = load_image_with_hangul(path_off)
    if img_on is None or img_off is None: sys.exit()

    # 차분 이미지
    diff_img_original = cv2.absdiff(img_on, img_off)
    height, width = diff_img_original.shape[:2]
    scale_ratio = 800 / width
    new_dim = (800, int(height * scale_ratio))
    diff_img_resized = cv2.resize(diff_img_original, new_dim)

    cv2.namedWindow('Final Logic Tuner')

    # === [공통] 노이즈 제거용 ===
    # 면적 필터: 이 값보다 작은 점은 무시 (매우 중요)
    cv2.createTrackbar('[Noise] Min Area', 'Final Logic Tuner', 5, 100, nothing)

    # === [Filter 1] Universe (Green) ===
    cv2.createTrackbar('F1 Boost', 'Final Logic Tuner', 0, 1, nothing)
    cv2.createTrackbar('F1 Min Bright', 'Final Logic Tuner', 30, 255, nothing)
    cv2.createTrackbar('F1 White Cut', 'Final Logic Tuner', 70, 100, nothing)
    cv2.createTrackbar('F1 Yellow Range', 'Final Logic Tuner', 60, 200, nothing)

    # === [Filter 2] RGB Two (Purple) ===
    cv2.createTrackbar('F2 Boost', 'Final Logic Tuner', 0, 1, nothing)
    cv2.createTrackbar('F2 Min Red', 'Final Logic Tuner', 40, 255, nothing)
    cv2.createTrackbar('F2 Diff G', 'Final Logic Tuner', 30, 100, nothing)
    cv2.createTrackbar('F2 Diff B', 'Final Logic Tuner', 30, 100, nothing)

    print("\n-------------------------------------------")
    print("🔥 오탐(False Positive) 잡는 법:")
    print("1. '[Noise] Min Area'를 10~20까지 올려보세요. (자잘한 점 제거)")
    print("2. F1(Green)이 잡힌다면 'F1 White Cut'이나 'F1 Min Bright'를 올리세요.")
    print("3. F2(Purple)가 잡힌다면 'F2 Min Red'를 올리세요.")
    print("-------------------------------------------\n")

    while True:
        # 파라미터 읽기
        min_area = cv2.getTrackbarPos('[Noise] Min Area', 'Final Logic Tuner')
        
        f1_boost = cv2.getTrackbarPos('F1 Boost', 'Final Logic Tuner')
        f1_bright = cv2.getTrackbarPos('F1 Min Bright', 'Final Logic Tuner')
        f1_white = cv2.getTrackbarPos('F1 White Cut', 'Final Logic Tuner')
        f1_yellow = cv2.getTrackbarPos('F1 Yellow Range', 'Final Logic Tuner')

        f2_boost = cv2.getTrackbarPos('F2 Boost', 'Final Logic Tuner')
        f2_min_red = cv2.getTrackbarPos('F2 Min Red', 'Final Logic Tuner')
        f2_diff_g = cv2.getTrackbarPos('F2 Diff G', 'Final Logic Tuner')
        f2_diff_b = cv2.getTrackbarPos('F2 Diff B', 'Final Logic Tuner')

        # ==========================
        # 1차 필터 시뮬레이션
        # ==========================
        img_f1 = preprocess_image(diff_img_resized, f1_boost)
        B, G, R = cv2.split(img_f1)
        R_int, G_int, B_int = R.astype(np.int16), G.astype(np.int16), B.astype(np.int16)

        mask1 = (R > f1_bright) & \
                ((R_int - B_int) > f1_white) & \
                ((R_int - G_int) > -f1_yellow)
        mask1 = mask1.astype(np.uint8) * 255
        mask1 = apply_morphology(mask1)

        # 1차 검출 확인
        contours_1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found_f1 = False
        valid_cnt_f1 = None
        
        if contours_1:
            max_cnt = max(contours_1, key=cv2.contourArea)
            if cv2.contourArea(max_cnt) > min_area: # 면적 필터 적용
                found_f1 = True
                valid_cnt_f1 = max_cnt

        # ==========================
        # 2차 필터 시뮬레이션 (1차 실패 시만)
        # ==========================
        found_f2 = False
        valid_cnt_f2 = None
        mask2_display = np.zeros_like(mask1)

        if not found_f1:
            img_f2 = preprocess_image(diff_img_resized, f2_boost)
            B2, G2, R2 = cv2.split(img_f2)
            R2_int, G2_int, B2_int = R2.astype(np.int16), G2.astype(np.int16), B2.astype(np.int16)

            mask2 = (R2 > f2_min_red) & \
                    ((R2_int - G2_int) > f2_diff_g) & \
                    ((R2_int - B2_int) > f2_diff_b)
            mask2 = mask2.astype(np.uint8) * 255
            mask2 = apply_morphology(mask2)
            mask2_display = mask2

            contours_2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_2:
                max_cnt = max(contours_2, key=cv2.contourArea)
                if cv2.contourArea(max_cnt) > min_area:
                    found_f2 = True
                    valid_cnt_f2 = max_cnt

        # ==========================
        # 시각화
        # ==========================
        result_view = diff_img_resized.copy()
        
        # 상태 표시
        if found_f1:
            status = "ACTIVE: Filter 1 (Green)"
            color = (0, 255, 0)
            x, y, w, h = cv2.boundingRect(valid_cnt_f1)
            cv2.rectangle(result_view, (x, y), (x+w, y+h), color, 2)
            display_mask = mask1
        elif found_f2:
            status = "ACTIVE: Filter 2 (Purple)"
            color = (255, 0, 255)
            x, y, w, h = cv2.boundingRect(valid_cnt_f2)
            cv2.rectangle(result_view, (x, y), (x+w, y+h), color, 2)
            display_mask = mask2_display
        else:
            status = "NO DETECTIONS"
            color = (100, 100, 100)
            display_mask = np.zeros_like(mask1)

        cv2.putText(result_view, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(result_view, f"Min Area: {min_area}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        mask_bgr = cv2.cvtColor(display_mask, cv2.COLOR_GRAY2BGR)
        stacked = np.hstack((result_view, mask_bgr))

        cv2.imshow('Final Logic Tuner', stacked)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()